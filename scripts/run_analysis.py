#!/usr/bin/env python3
"""Reproduce all layout, control cost and held-out placement analyses."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--traces-dir',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/home/nfs_client/ericyu/expert_flash_experiment'))
    a=p.parse_args()
    scripts=Path(__file__).resolve().parent
    run=a.data/'runs'/(a.name+'-aif'); run.mkdir(parents=True,exist_ok=True)
    traces=sorted(a.traces_dir.glob('*.routing.jsonl'))
    if len(traces)!=3:
        raise ValueError('Need three complete real prompt traces for held-out validation')
    trace_args=[v for t in traces for v in ('--trace',str(t.resolve()))]
    common=['--metadata',str(a.metadata.resolve()),*trace_args]
    tasks=[('flash', 'aif_model.py', [*common,'--output-dir',str(run)]),
           ('permutations','placement_sensitivity.py',['--run-manifest',str(run/'manifest.json')]),
           ('fit','optimize_placement.py',['--run-manifest',str(run/'manifest.json')]),
           ('overhead','overhead_model.py',[*common,'--output-dir',str(run/'overhead_fanout')]),
           ('heldout','evaluate_learned_overhead.py',['--run-manifest',str(run/'manifest.json'),
              '--mappings',str(run/'learned_placement/mappings.json'),'--output-dir',str(run/'learned_overhead_fanout')]),
           ('physical','materialize_plan.py',['--run-manifest',str(run/'manifest.json'),
              '--plans',str(run/'learned_overhead_fanout/plans.json'),'--output-dir',str(run/'learned_overhead_fanout/physical_plans')])]
    commands=[]
    for label,script,arguments in tasks:
        cmd=[sys.executable,str(scripts/script),*arguments]
        commands.append(cmd)
        print(f'Running {a.name}: {label}',flush=True)
        with (run/(label+'.analysis.log')).open('w') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    (run/'analysis_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(run)


if __name__=='__main__':
    main()
