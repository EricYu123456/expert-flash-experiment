#!/usr/bin/env python3
"""Finish analysis/replays/figures as each model's 300-prompt collection completes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_corpus_traces import DATA, MODELS, ROOT, write_json, sha256


def command(args,log):
    with log.open('a') as f:
        subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                       env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))


def prepare(name,run_dir):
    log=run_dir/name/'postprocess.log'
    receipt=run_dir/name/'analysis/evaluation_receipt.json'
    replay_manifest=run_dir/name/'replays/manifest.json'
    if receipt.exists() and replay_manifest.exists():
        expected=sha256(run_dir/name/'analysis/frozen_plan.json')
        assert json.loads(receipt.read_text())['frozen_plan_sha256']==expected
        assert json.loads(replay_manifest.read_text())['frozen_plan_sha256']==expected
        return name
    command([sys.executable,'-u',str(ROOT/'scripts/corpus_analysis.py'),'--run-dir',str(run_dir),
             '--models',name],log)
    command([sys.executable,'-u',str(ROOT/'scripts/export_corpus_replays.py'),'--run-dir',str(run_dir),
             '--models',name],log)
    return name


def replay(name,run_dir):
    relative=run_dir.relative_to(DATA)
    guest=f'/mnt/experiment/{relative}/{name}/replays'
    command(['bash',str(ROOT/'scripts/vm_ssh.sh'),
             f'bash /mnt/workspace/scripts/run_corpus_replays.sh {guest}'],run_dir/name/'replay.log')
    return name


def collectors_alive(run_dir):
    for path in [run_dir/'collector.pid',*run_dir.glob('*.collector.pid')]:
        if path.exists():
            pid=path.read_text().strip()
            proc=Path('/proc')/pid/'stat'
            if proc.exists() and proc.read_text().split()[2]!='Z': return True
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DATA/'runs/corpus-300-v1')
    p.add_argument('--stop-vm',action='store_true')
    a=p.parse_args()
    status=dict(stage='waiting_for_collection',prepared=[],replayed=[],failures=[],updated_unix=time.time())
    futures={};submitted=set();ready=[];active=None
    with ThreadPoolExecutor(max_workers=2) as analysis_pool,ThreadPoolExecutor(max_workers=1) as replay_pool:
        while len(status['replayed'])<len(MODELS):
            for name in MODELS:
                if name not in submitted and (a.run_dir/name/'validation.json').exists():
                    print(f'{name}: 300 validated prompts ready; fitting and evaluating',flush=True)
                    submitted.add(name);futures[analysis_pool.submit(prepare,name,a.run_dir)]=name
            for future,name in list(futures.items()):
                if future.done():
                    future.result();del futures[future]
                    ready.append(name);status['prepared'].append(name)
                    print(f'{name}: frozen plans and all held-out workloads ready',flush=True)
            if active is not None and active.done():
                name=active.result();status['replayed'].append(name);active=None
                print(f'{name}: all 100 held-out prompts replayed and counters verified',flush=True)
            if active is None and ready:
                name=ready.pop(0);status['stage']='replaying_'+name
                active=replay_pool.submit(replay,name,a.run_dir)
                print(f'{name}: starting serial NVMeVirt queue replays',flush=True)
            elif active is None:
                status['stage']='waiting_for_collection'
            if len(submitted)<len(MODELS) and not collectors_alive(a.run_dir):
                missing=[n for n in MODELS if n not in submitted]
                raise RuntimeError(f'Collectors stopped before complete validation: {missing}')
            status['updated_unix']=time.time()
            write_json(a.run_dir/'workflow_status.json',status)
            if len(status['replayed'])==len(MODELS):break
            time.sleep(15)
    command([sys.executable,str(ROOT/'scripts/plot_corpus_results.py'),'--run-dir',str(a.run_dir)],
            a.run_dir/'figures.log')
    if a.stop_vm:
        command(['bash',str(ROOT/'scripts/vm_stop.sh')],a.run_dir/'vm_stop.log')
    status.update(stage='complete',updated_unix=time.time())
    write_json(a.run_dir/'workflow_status.json',status)
    print('Complete: all five models, full held-out replays, figures and report',flush=True)


if __name__=='__main__':main()
