#!/usr/bin/env python3
"""Export all held-out inputs to v3 workloads, split into prefix32 and tail96."""
import argparse
import json
from pathlib import Path

import numpy as np

import aif_model as flash
import overhead_model as overhead
from corpus_analysis import COMPONENTS, counts_for, profiles, evaluate_routes
from materialize_plan import make_plan
from run_corpus_traces import DATA, MODELS, ROOT, sha256, write_json

COUNTERS = ('commands','model_ns','flash_ns','control_ns','host_ns','input_ns','output_ns',
            'useful_bytes','physical_pages','read_rounds','cold_reads','active_chip_commands',
            'chip_service_ns','vector_input_bytes','vector_output_bytes','chip_matrix_commands',
            'pcie_input_bytes','pcie_output_bytes')


def phase_bytes(slots,dims,g,stacked):
    phases=[([dims['gate']],True),([dims['up']],True),([dims['down']],False)]
    if stacked:
        phases=[([(dims['gate'][0]+dims['up'][0],dims['up'][1])],True),([dims['down']],False)]
    chunks=[]
    for projections,common in phases:
        demands=flash.phase_demand(slots,projections,g)
        inputs,outputs,streams=overhead.phase_traffic(slots,projections,g,common)
        host_input=projections[0][1]*(1 if common else len(slots))
        chunks.append(overhead.encode_v2(demands,inputs,outputs,streams,250,2000,host_input,sum(outputs)))
    return b''.join(chunks)


def export(routes,prompt_ids,layers,dimensions,g,stacked,mappings,path,start):
    n,steps=routes.shape[:2]; stride=1600 if stacked else 2400
    # At most ~1 GB for the largest test window. Group-count caching preserves
    # every real route's descriptor, order, multiplicity and completion delay.
    payload=np.empty((n,steps,len(layers),stride),dtype=np.uint8)
    total={k:0 for k in COUNTERS}; by_chip=np.zeros((2,16),dtype=np.int64)
    token_values={k:np.zeros((n,steps),dtype=np.int64) for k in ('flash','input','output','streams','commands')}
    cache={}
    for li,layer in enumerate(layers):
        logical=routes[:,:,li,:].reshape(-1,routes.shape[-1])
        slots=logical if mappings is None else np.asarray(mappings[str(layer)])[logical]
        counts=counts_for(slots,np.arange(int(slots.max())+1),16//g)
        unique,first,inverse,frequency=np.unique(counts,axis=0,return_index=True,return_inverse=True,return_counts=True)
        command_array=np.empty((len(unique),stride),dtype=np.uint8)
        coeff={k:np.empty(len(unique),dtype=np.int64) for k in token_values}
        for i,count in enumerate(unique):
            key=(tuple(dimensions[layer]['gate']),g,stacked,tuple(count))
            if key not in cache:
                real_slots=slots[first[i]].tolist()
                event=dict(prompt='cached_descriptor',token_index=0,layer=layer,expert_ids=real_slots)
                expected=overhead.evaluate([event],{layer:dimensions[layer]},g,stack_gate_up=stacked)
                cache[key]=(phase_bytes(real_slots,dimensions[layer],g,stacked),expected)
            blob,expected=cache[key]
            command_array[i]=np.frombuffer(blob,dtype=np.uint8)
            for k in COUNTERS: total[k] += int(frequency[i])*expected[k]
            by_chip[0] += int(frequency[i])*np.asarray(expected['chip_bytes'])
            by_chip[1] += int(frequency[i])*np.asarray(expected['chip_service_ns_by_chip'])
            for k in coeff: coeff[k][i]=expected['token_cost_coefficients'][0][k]
        payload[:,:,li]=command_array[inverse].reshape(n,steps,stride)
        for k in token_values: token_values[k] += coeff[k][inverse].reshape(n,steps)
    # Use normal buffered writes: ndarray.tofile can bypass Python buffering and
    # NFS metadata gets expensive with a write per command.
    with path.open('wb') as f:
        for i in range(n): f.write(payload[i].tobytes())
    latency=token_values['flash']+token_values['input']+token_values['output']+250*token_values['streams']+2000*token_values['commands']
    assert int(latency.sum())==total['model_ns']
    coefficients=[dict(prompt=str(prompt_ids[i]),token_index=start+t,
                       **{k:int(v[i,t]) for k,v in token_values.items()}) for i in range(n) for t in range(steps)]
    total.update(label=path.stem,decode_tokens=n*steps,prompts=n,routing_records=n*steps*len(layers),
                 group_chips=g,stack_gate_up=stacked,model='metadata_only_int8_v3',
                 chip_bytes=by_chip[0].tolist(),chip_service_ns_by_chip=by_chip[1].tolist(),
                 token_cost_coefficients=coefficients,ffn_ms_per_token=total['model_ns']/(n*steps)/1e6,
                 window_start=start,window_steps=steps,binary_sha256=sha256(path))
    return total


def export_model(run_dir,data,name):
    analysis=run_dir/name/'analysis'; output=run_dir/name/'replays'; output.mkdir(exist_ok=True)
    plan_path=analysis/'frozen_plan.json'; plan=json.loads(plan_path.read_text()); plan_hash=sha256(plan_path)
    receipt=json.loads((analysis/'evaluation_receipt.json').read_text())
    assert receipt['frozen_plan_sha256']==plan_hash
    metadata_path=data/'models'/(name+'-metadata.json'); metadata=json.loads(metadata_path.read_text())
    dimensions=flash.tensor_dimensions(metadata)
    with np.load(analysis/'test_routes.npz') as stored:
        routes=stored['routes']; prompt_ids=stored['prompt_ids']; layers=stored['layers'].tolist()
    with np.load(analysis/'test_token_costs.npz') as costs:
        primary_costs={p:{k:costs[f'{p}__{k}'] for k in COMPONENTS} for p in ('baseline_16_stacked','learned_selected')}
    specs={'baseline_16_stacked':dict(group_chips=16,stack_gate_up=True,mappings=None),
           'learned_selected':dict(group_chips=plan['winners']['learned']['group_chips'],
                                   stack_gate_up=plan['winners']['learned']['stack_gate_up'],
                                   mappings=plan['mappings'][f'g{plan["winners"]["learned"]["group_chips"]}'])}
    jobs=[]; physical=[]
    for policy,spec in specs.items():
        for label,start,end in (('prefix32',0,32),('tail96',32,128)):
            path=output/f'{policy}_{label}.bin'; expected_path=path.with_suffix('.expected.json')
            if expected_path.exists() and path.exists():
                expected=json.loads(expected_path.read_text())
                assert expected['frozen_plan_sha256']==plan_hash
                assert expected['binary_sha256']==sha256(path)
            else:
                expected=export(routes[:,start:end],prompt_ids,layers,dimensions,spec['group_chips'],
                                spec['stack_gate_up'],spec['mappings'],path,start)
                expected.update(frozen_plan_sha256=plan_hash,policy=policy)
                for k in COMPONENTS: assert expected[k]==int(primary_costs[policy][k][:,start:end].sum())
                write_json(expected_path,expected)
            jobs.append(dict(policy=policy,window=label,binary=str(path),expected=str(expected_path),
                             commands=expected['commands'],model_ns=expected['model_ns']))
            print(f'{name}: exported {path.name}, {expected["commands"]} real commands',flush=True)
        specification=dict(heldout_prompt='test100',policy=policy,**spec,
                           train_prompts=[r['id'] for r in plan['training_routing']])
        physical_plan=make_plan(metadata,specification,dict(frozen_plan=str(plan_path),
                                frozen_plan_sha256=plan_hash,metadata_sha256=sha256(metadata_path)))
        physical_path=output/f'{policy}.physical_plan.json'
        physical_path.write_text(json.dumps(physical_plan,separators=(',',':'))+'\n')
        physical.append(dict(policy=policy,path=str(physical_path),checks=physical_plan['checks'],
                             **physical_plan['totals']))
    write_json(output/'manifest.json',dict(model=name,frozen_plan_sha256=plan_hash,
               policy='All 100 held-out prompts, every routed layer and all 128 decode inputs; prefix32 + tail96 partition without duplicated decode inputs.',
               workloads=jobs,physical_plans=physical,export_script_sha256=sha256(Path(__file__))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DATA/'runs/corpus-300-v1')
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    a=p.parse_args()
    for name in a.models: export_model(a.run_dir,a.data,name)


if __name__=='__main__': main()
