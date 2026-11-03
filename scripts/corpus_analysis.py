#!/usr/bin/env python3
"""Train once on 200 prompts; evaluate frozen INT8 AiF policies on 100 prompts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

import aif_model as flash
import overhead_model as overhead
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256, validate_prompt, write_json

GROUPS = (1, 2, 4, 8, 16)
COMPONENTS = ('flash_ns', 'control_ns', 'host_ns', 'input_ns', 'output_ns')


def counts_for(routes, placement, groups):
    assigned = np.asarray(placement)[routes] % groups
    return np.stack([(assigned == g).sum(axis=-1) for g in range(groups)],axis=-1)


def profiles(dims, g, stacked):
    phases = [([dims['gate']],True),([dims['up']],True),([dims['down']],False)]
    if stacked:
        phases = [([(dims['gate'][0]+dims['up'][0],dims['up'][1])],True),([dims['down']],False)]
    result = []
    for projections, common in phases:
        demand = flash.phase_demand([0],projections,g)
        inputs,outputs,streams = overhead.phase_traffic([0],projections,g,common)
        result.append(dict(service=np.array([d.service_ns for d in demand[:g]],dtype=np.int64),
                           inputs=np.array(inputs[:g],dtype=np.int64),
                           outputs=np.array(outputs[:g],dtype=np.int64),
                           streams=np.array(streams[:g],dtype=np.int64),
                           input_dimension=projections[0][1],common=common))
    return result


def layer_costs(routes, placement, g, phase_profiles):
    """Exact vectorization of overhead.evaluate for homogeneous expert geometry.

    Each chip's page/cold/byte demand is linear in its expert count. Service is
    integral here (16 KiB page granularity), so multiplication preserves the
    full reference model's rounding. Channels still pair chips c and c+8.
    """
    original_shape = routes.shape[:-1]
    routes = routes.reshape(-1,routes.shape[-1])
    counts = counts_for(routes,placement,16//g)
    chip_counts = np.repeat(counts,g,axis=1)
    costs = {k:np.zeros(len(routes),dtype=np.int64) for k in COMPONENTS}
    active = routes.shape[-1]
    for p in phase_profiles:
        service = np.tile(p['service'],16//g)
        inputs = ((chip_counts > 0) if p['common'] else chip_counts)*np.tile(p['inputs'],16//g)
        outputs = chip_counts*np.tile(p['outputs'],16//g)
        streams = chip_counts*np.tile(p['streams'],16//g)
        host_input = p['input_dimension']*(1 if p['common'] else active)
        channel_in = inputs[:,:8]+inputs[:,8:]
        channel_out = outputs[:,:8]+outputs[:,8:]
        costs['flash_ns'] += (chip_counts*service).max(axis=1)
        costs['control_ns'] += 250*streams.sum(axis=1)
        costs['host_ns'] += 2000
        costs['input_ns'] += np.maximum((host_input+7)//8,(channel_in.max(axis=1)+1)//2)
        costs['output_ns'] += np.maximum((outputs.sum(axis=1)+7)//8,(channel_out.max(axis=1)+1)//2)
    return {k:v.reshape(original_shape) for k,v in costs.items()}


def evaluate_routes(routes, layers, dimensions, g, stacked, mappings=None):
    result = {k:np.zeros(routes.shape[:2],dtype=np.int64) for k in COMPONENTS}
    for li,layer in enumerate(layers):
        placement = mappings[str(layer)] if mappings is not None else np.arange(
            int(routes.max())+1)
        values = layer_costs(routes[:,:,li,:],placement,g,profiles(dimensions[layer],g,stacked))
        for k,v in values.items(): result[k] += v
    result['model_ns'] = sum(result.values())
    return result


def optimize_weighted(routes, expert_count, g, trials, seed):
    """Fixed-budget capacity-preserving swaps, weighted by real route frequency."""
    placement = list(range(expert_count))
    ngroups = 16//g
    if ngroups == 1:
        return placement,dict(attempted=0,accepted=0,unique_routes=0)
    unique,weights = np.unique(np.sort(routes.reshape(-1,routes.shape[-1]),axis=1),axis=0,return_counts=True)
    incidence = np.zeros((len(unique),expert_count),dtype=np.int16)
    incidence[np.arange(len(unique))[:,None],unique] = 1
    counts = np.stack([incidence[:,group::ngroups].sum(axis=1) for group in range(ngroups)],axis=1)
    def score(value):
        return (int(value.max(axis=1) @ weights),int(np.square(value).sum(axis=1) @ weights))
    initial = current = score(counts)
    accepted = 0; rng = random.Random(seed)
    for _ in range(trials):
        a,b = rng.sample(range(expert_count),2)
        ga,gb = placement[a]%ngroups,placement[b]%ngroups
        if ga == gb: continue
        delta = incidence[:,b]-incidence[:,a]
        counts[:,ga] += delta; counts[:,gb] -= delta
        candidate = score(counts)
        if candidate < current:
            placement[a],placement[b] = placement[b],placement[a]
            current = candidate; accepted += 1
        else:
            counts[:,ga] -= delta; counts[:,gb] += delta
    assert sorted(placement) == list(range(expert_count)) and current <= initial
    return placement,dict(attempted=trials,accepted=accepted,unique_routes=len(unique),
                          initial_peak_sum=initial[0],final_peak_sum=current[0],
                          initial_square_sum=initial[1],final_square_sum=current[1])


def load_split(run_dir, name, rows, metadata, split):
    selected = [r for r in rows if r['split'] == split]
    arrays=[]; validation=[]
    for row in selected:
        stem = run_dir/name/'traces'/row['id']
        result,routes = validate_prompt(stem,row,metadata,save_prefix=False)
        arrays.append(routes); validation.append(result)
    return selected,np.stack(arrays),validation


def validate_vector_model(routes, rows, layers, dimensions, experts, mappings_by_g):
    checks=[]
    # Include every task/length cell and both start and end of generation.
    selected={}
    for i,r in enumerate(rows): selected.setdefault((r['task'],r['length_bucket']),i)
    for g in GROUPS:
        for stacked in (False,True):
            for policy,mappings in (('modulo',None),('learned',mappings_by_g[f'g{g}'])):
                for li,layer in enumerate(layers):
                    event_routes = np.stack([routes[i,t,li] for i in selected.values() for t in (0,31,127)])
                    placement = list(range(experts)) if mappings is None else mappings[str(layer)]
                    fast = layer_costs(event_routes,placement,g,profiles(dimensions[layer],g,stacked))
                    events = [dict(prompt='reference',token_index=i,layer=layer,expert_ids=r.tolist())
                              for i,r in enumerate(event_routes)]
                    exact = overhead.evaluate(events,dimensions,g,stack_gate_up=stacked,
                                               mappings=None if mappings is None else {layer:placement})
                    for component in COMPONENTS:
                        if int(fast[component].sum()) != exact[component]:
                            raise ValueError(f'Vector model mismatch: {g}/{stacked}/{layer}/{component}')
                    checks.append(dict(group=g,stacked=stacked,policy=policy,layer=layer,routes=len(events)))
    return dict(checks=len(checks),routing_cases=sum(c['routes'] for c in checks),
                all_components_match_reference=True,components=list(COMPONENTS))


def fit(run_dir,name,rows,metadata,trials):
    output=run_dir/name/'analysis'; output.mkdir(exist_ok=True)
    frozen=output/'frozen_plan.json'
    if frozen.exists():
        plan=json.loads(frozen.read_text())
        assert plan['corpus_sha256']==sha256(run_dir/'corpus/prompts.jsonl')
        for r in plan['training_routing']:
            assert sha256(run_dir/name/'traces'/(r['id']+'.routing.jsonl'))==r['routing_sha256']
        print(f'{name}: reusing frozen plan {sha256(frozen)}',flush=True)
        return plan
    started=time.time()
    train,routes,validation=load_split(run_dir,name,rows,metadata,'train')
    assert len(train)==200 and routes.shape[1]==128
    layers,experts,active=geometry(metadata); dimensions=flash.tensor_dimensions(metadata)
    maps={}; stats={}
    for g in GROUPS:
        maps[f'g{g}']={}; stats[f'g{g}']={}
        for li,layer in enumerate(layers):
            placement,stat=optimize_weighted(routes[:,:,li,:],experts,g,trials,20261027+layer*131+g)
            maps[f'g{g}'][str(layer)]=placement; stats[f'g{g}'][str(layer)]=stat
        print(f'{name}: learned g={g} from 200 training prompts',flush=True)
    candidates=[]
    for policy in ('modulo','learned'):
        for stacked in (False,True):
            for g in GROUPS:
                values=evaluate_routes(routes,layers,dimensions,g,stacked,
                                      None if policy=='modulo' else maps[f'g{g}'])
                candidates.append(dict(policy=policy,group_chips=g,stack_gate_up=stacked,
                                       training_ffn_ms_per_token=float(values['model_ns'].mean()/1e6),
                                       training_components_ms={k:float(values[k].mean()/1e6) for k in COMPONENTS}))
    winners={}
    for policy in ('modulo','learned'):
        winners[policy]=min((r for r in candidates if r['policy']==policy),
                            key=lambda r:(r['training_ffn_ms_per_token'],r['group_chips'],r['stack_gate_up']))
    reference_checks=validate_vector_model(routes,train,layers,dimensions,experts,maps)
    plan=dict(schema_version=1,model=name,model_sha256=metadata['sha256'],
              corpus_sha256=sha256(run_dir/'corpus/prompts.jsonl'),training_prompts=200,
              training_steps=128,test_prompts=100,training_only_selection=True,
              training_routing=[dict(id=r['id'],routing_sha256=r['routing_sha256']) for r in validation],
              mapping_objective='Minimize frequency-weighted maximum coactivated experts per group, then squared occupancy; capacity-preserving expert-slot swaps.',
              selection_objective='Mean training routed-FFN latency including flash, controller fanout, chip commands and vector traffic.',
              search=dict(swap_trials_per_layer_per_g=trials,seed=20261027,groups=list(GROUPS),stacking=[False,True]),
              assumptions=dict(per_chip_control_ns=250,per_host_command_ns=2000,input_bytes=1,output_bytes=4,
                               chips=16,planes=4,page_bytes=16384,cold_ns=28000,cr_ns=9700,
                               pcie_bytes_per_ns=8,channel_bytes_per_ns=2,controller_fanout=True),
              candidates=candidates,winners=winners,mappings=maps,optimization_stats=stats,
              training_reference_checks=reference_checks,fit_seconds=time.time()-started,
              software_sha256={s:sha256(ROOT/'scripts'/s) for s in
                               ('corpus_analysis.py','aif_model.py','overhead_model.py')})
    write_json(frozen,plan)
    write_json(output/'freeze_receipt.json',dict(frozen_plan_sha256=sha256(frozen),
               frozen_unix=time.time(),analysis_test_routes_loaded_before_freeze=False,
               test_routes_used_for_training_or_selection=False))
    # Persist training data as a compact reproducibility artifact after freezing.
    np.savez_compressed(output/'train_routes.npz',routes=routes,
                        prompt_ids=np.array([r['id'] for r in train]),layers=np.array(layers))
    print(f'{name}: frozen {winners}',flush=True)
    return plan


def paired_interval(baseline, learned, seed=20261027):
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(baseline),(2000,len(baseline)))
    improvements=1-learned[indices].mean(axis=1)/baseline[indices].mean(axis=1)
    return [float(x) for x in np.quantile(improvements,[.025,.975])]


def summary_row(name,window,rows,values,indices):
    means={p:float(v['model_ns'][indices,:window].mean()/1e6) for p,v in values.items()}
    base=values['baseline_16_stacked']['model_ns'][indices,:window].mean(axis=1)
    learned=values['learned_selected']['model_ns'][indices,:window].mean(axis=1)
    separate=values['baseline_16_separate']['model_ns'][indices,:window].mean(axis=1)
    return dict(model=name,window=window,prompts=len(indices),means_ms=means,
                improvement_vs_16_stacked=1-means['learned_selected']/means['baseline_16_stacked'],
                improvement_vs_16_separate=1-means['learned_selected']/means['baseline_16_separate'],
                speedup_vs_16_separate=means['baseline_16_separate']/means['learned_selected'],
                placement_improvement=1-means['learned_selected']/means['modulo_at_learned_config'],
                improvement_vs_16_stacked_ci95=paired_interval(base,learned),
                improvement_vs_16_separate_ci95=paired_interval(separate,learned),
                fraction_prompts_improved_vs_16_stacked=float((learned<base).mean()),
                learned_token_p50_ms=float(np.quantile(values['learned_selected']['model_ns'][indices,:window],.5)/1e6),
                learned_token_p95_ms=float(np.quantile(values['learned_selected']['model_ns'][indices,:window],.95)/1e6),
                learned_components_ms={k:float(values['learned_selected'][k][indices,:window].mean()/1e6)
                                       for k in COMPONENTS})


def test(run_dir,name,rows,metadata,plan):
    output=run_dir/name/'analysis'; frozen=output/'frozen_plan.json'
    frozen_hash=sha256(frozen)
    receipt=json.loads((output/'freeze_receipt.json').read_text())
    assert receipt['frozen_plan_sha256']==frozen_hash
    assert not receipt['analysis_test_routes_loaded_before_freeze']
    assert not receipt['test_routes_used_for_training_or_selection']
    heldout,routes,validation=load_split(run_dir,name,rows,metadata,'test')
    assert len(heldout)==100
    layers,experts,active=geometry(metadata); dimensions=flash.tensor_dimensions(metadata)
    specs={'baseline_16_separate':dict(group_chips=16,stack_gate_up=False),
           'baseline_16_stacked':dict(group_chips=16,stack_gate_up=True),
           'modulo_selected':plan['winners']['modulo'],
           'learned_selected':plan['winners']['learned'],
           'modulo_at_learned_config':plan['winners']['learned']}
    values={}
    for policy,spec in specs.items():
        mappings=plan['mappings'][f'g{spec["group_chips"]}'] if policy=='learned_selected' else None
        values[policy]=evaluate_routes(routes,layers,dimensions,spec['group_chips'],spec['stack_gate_up'],mappings)
    checks=validate_vector_model(routes,heldout,layers,dimensions,experts,plan['mappings'])
    # Validation against the reference implementation is an invariant check,
    # never a held-out configuration comparison or an opportunity to reselect.
    rows_out=[]; strata=[]
    for window in (32,128):
        rows_out.append(summary_row(name,window,heldout,values,np.arange(len(heldout))))
        for field in ('task','length_bucket'):
            for label in sorted({r[field] for r in heldout}):
                indices=np.array([i for i,r in enumerate(heldout) if r[field]==label])
                item=summary_row(name,window,heldout,values,indices)
                item.update(stratum=field,label=label); strata.append(item)
    per_prompt=[]
    for i,r in enumerate(heldout):
        for window in (32,128):
            item=dict(id=r['id'],task=r['task'],length_bucket=r['length_bucket'],window=window,
                      prompt_tokens=validation[i]['prompt_tokens'])
            for p,v in values.items(): item[p]=float(v['model_ns'][i,:window].mean()/1e6)
            per_prompt.append(item)
    np.savez_compressed(output/'test_token_costs.npz',prompt_ids=np.array([r['id'] for r in heldout]),
                        **{f'{p}__{k}':v for p,components in values.items() for k,v in components.items()})
    np.savez_compressed(output/'test_routes.npz',routes=routes,
                        prompt_ids=np.array([r['id'] for r in heldout]),layers=np.array(layers))
    write_json(output/'test_validation.json',validation)
    write_json(output/'summary.json',rows_out)
    write_json(output/'strata.json',strata)
    write_json(output/'per_prompt.json',per_prompt)
    write_json(output/'evaluation_receipt.json',dict(frozen_plan_sha256=frozen_hash,
               test_evaluated_unix=time.time(),test_prompts=100,decode_steps=128,prefix_steps=32,
               fixed_same_plan_for_both_windows=True,prompt_level_bootstrap_resamples=2000,
               test_reference_checks=checks,test_routing=[dict(id=r['id'],routing_sha256=r['routing_sha256'])
                                                          for r in validation]))
    assert sha256(frozen)==frozen_hash,'Frozen plan changed during held-out evaluation'
    print(f'{name}: held-out evaluation complete {rows_out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DATA/'runs/corpus-300-v1')
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--stage',choices=('fit','test','both'),default='both')
    parser.add_argument('--swap-trials',type=int,default=512)
    args=parser.parse_args()
    rows=[json.loads(line) for line in (args.run_dir/'corpus/prompts.jsonl').read_text().splitlines()]
    assert Counter(r['split'] for r in rows)=={'train':200,'test':100}
    for name in args.models:
        metadata=json.loads((args.data/'models'/(name+'-metadata.json')).read_text())
        if args.stage in ('fit','both'):
            plan=fit(args.run_dir,name,rows,metadata,args.swap_trials)
        else:
            plan=json.loads((args.run_dir/name/'analysis/frozen_plan.json').read_text())
        if args.stage in ('test','both'):
            test(args.run_dir,name,rows,metadata,plan)


if __name__=='__main__':
    main()
