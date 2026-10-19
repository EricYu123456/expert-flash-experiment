#!/usr/bin/env python3
"""Select group size on training prompts, replay held-out fixed placement.

Expert swaps were learned by optimize_placement.py using only training prompts.
This stage reselects chip count and gate/up layout using control/vector costs,
again only on training prompts. Held-out routes never influence either choice.
"""
import argparse
import hashlib
import json
from pathlib import Path
import aif_model as flash
import overhead_model as overhead


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-manifest',type=Path,required=True)
    p.add_argument('--mappings',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    source=json.loads(a.run_manifest.read_text())
    metadata=json.loads(Path(source['metadata']).read_text())
    dimensions=flash.tensor_dimensions(metadata)
    events=flash.load_events([Path(path) for path in source['traces']])
    fit=json.loads(a.mappings.read_text())
    selected_rows,all_rows,sweep_rows,plans=[],[],[],[]
    for fold in fit['folds']:
        heldout=fold['heldout_prompt']
        train=[e for e in events if e['prompt']!=heldout]
        test=[e for e in events if e['prompt']==heldout]
        assert sorted({e['prompt'] for e in train})==fold['train_prompts']
        for policy in ('modulo','learned'):
            candidates=[]
            for stacked in (False,True):
                for g in (1,2,4,8,16):
                    mappings=None if policy=='modulo' else {
                        int(layer):placement for layer,placement in fold['placements'][f'g{g}'].items()}
                    for permutation in (mappings or {}).values():
                        assert sorted(permutation)==list(range(len(permutation)))
                    train_result=overhead.evaluate(train,dimensions,g,stack_gate_up=stacked,mappings=mappings)
                    test_result=overhead.evaluate(test,dimensions,g,stack_gate_up=stacked,mappings=mappings)
                    candidate=dict(g=g,stacked=stacked,mappings=mappings,train=train_result,test=test_result)
                    candidates.append(candidate)
                    row={k:v for k,v in test_result.items() if not isinstance(v,(dict,list))}
                    row.update(heldout_prompt=heldout,policy=policy,training_model_ns=train_result['model_ns'])
                    all_rows.append(row)
            winner=min(candidates,key=lambda c:(c['train']['model_ns'],c['g']))
            g,stacked=winner['g'],winner['stacked']
            tag=f'{heldout}_{policy}_selected'
            expected=overhead.evaluate(test,dimensions,g,stack_gate_up=stacked,mappings=winner['mappings'],
                                      binary_path=a.output_dir/(tag+'.bin'))
            expected.update(label=tag,heldout_prompt=heldout,policy=policy,
                            train_prompts=fold['train_prompts'],selection='training_only_joint_group_and_stack')
            (a.output_dir/(tag+'.expected.json')).write_text(json.dumps(expected,indent=2)+'\n')
            row={k:v for k,v in expected.items() if not isinstance(v,(dict,list))}
            selected_rows.append(row)
            plans.append(dict(heldout_prompt=heldout,policy=policy,group_chips=g,stack_gate_up=stacked,
                              mappings=winner['mappings'],train_prompts=fold['train_prompts']))
            for ctrl in (0,100,250,500,1000,2000,5000):
                for host in (0,2000,10000):
                    chosen=min(candidates,key=lambda c:(overhead.sweep_cost(c['train'],ctrl,host)['model_ns'],c['g']))
                    result=overhead.sweep_cost(chosen['test'],ctrl,host)
                    result.update(heldout_prompt=heldout,policy=policy,selection='training_only_joint_group_and_stack')
                    sweep_rows.append(result)
    flash.write_csv(a.output_dir/'heldout_selected.csv',selected_rows)
    flash.write_csv(a.output_dir/'heldout_all_configurations.csv',all_rows)
    flash.write_csv(a.output_dir/'heldout_control_sensitivity.csv',sweep_rows)
    (a.output_dir/'plans.json').write_text(json.dumps(plans,indent=2)+'\n')
    summary=[]
    for policy in ('modulo','learned'):
        rows=[r for r in selected_rows if r['policy']==policy]
        summary.append(dict(policy=policy,mean_ffn_ms_per_token=sum(r['ffn_ms_per_token'] for r in rows)/len(rows),
                            mean_useful_bandwidth_fraction=sum(r['useful_bandwidth_fraction'] for r in rows)/len(rows),
                            folds=[dict(prompt=r['heldout_prompt'],group_chips=r['group_chips'],
                                        stack_gate_up=r['stack_gate_up'],ffn_ms_per_token=r['ffn_ms_per_token']) for r in rows]))
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest=dict(source_run_manifest=str(a.run_manifest.resolve()),mappings=str(a.mappings.resolve()),
                  mappings_sha256=hashlib.sha256(a.mappings.read_bytes()).hexdigest(),
                  per_chip_control_ns=250,per_host_command_ns=2000,
                  selection='Group size and stacking selected using only two training prompts; third prompt is held out.',
                  expert_mapping_objective='Flash-only coactivation collisions; group/layout selection includes vector/control cost.',
                  limitations=['Correlated three-fold experiment with only three prompts.',
                    'Heuristic fixed capacity-preserving mapping, not globally optimal.',
                    'Control latency and vector precision are exposed assumptions, not calibrated physical AiF costs.'])
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
