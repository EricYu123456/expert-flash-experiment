#!/usr/bin/env python3
"""Fixed-six, original-16-token TRAIN/VALID matrix requested by 1108_2_spec."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/ericyu/expert_flash_experiment')
OLD=Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')
NEW=Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def run():
    out=NEW/'spec2-screen16';out.mkdir(exist_ok=False)
    freeze=out/'source-v1';freeze.mkdir()
    files=[ROOT/'scripts/research_20261109_shared_feature_fit.py',
           ROOT/'scripts/research_20261108_shared_feature_fit.py',Path(__file__).resolve(),
           ROOT/'docs/1108_2_spec.txt',OLD/'corpus/manifest.json']
    identities={str(p):sha(p) for p in files}
    for p in files[:3]:shutil.copy2(p,freeze/p.name)
    for split in ('train','validation'):
        for shard in range(4):
            p=OLD/f'captures/{split}/shard{shard}/capture_manifest.json';identities[str(p)]=sha(p)
    save(out/'protocol.frozen.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
         inputs_sha256=identities,TRAIN=600,VALID=200,max_tokens=16,
         fixed_experts={'1':[14,40],'13':[10,56],'26':[0,1]},
         bases=['raw_x_eq','shared_z_eq','hybrid_x_z128','hybrid_x_z256','hybrid_x_z512'],
         betas=[0,.5,.9,.99],lambdas=[1,3,10],CPU_only=True,
         TEST_opened=False,feature_order='Original full-TRAIN residual correlation; separate article-OOF diagnostic follows.',
         selection_scope='VALID; factor diagnostics retain every beta/lambda instead of only winners.'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',OMP_DYNAMIC='FALSE')
    active=[];started=time.monotonic()
    for layer,experts,cpu in [(1,[14,40],'0-7'),(13,[10,56],'8-15'),(26,[0,1],'16-23')]:
        cmd=['taskset','-c',cpu,'python3',str(freeze/'research_20261109_shared_feature_fit.py'),'screen',
             '--train',str(OLD/'captures/train'),'--valid',str(OLD/'captures/validation'),
             '--layer',str(layer),'--experts',*map(str,experts),'--output',str(out/f'layer{layer}'),
             '--bases','raw_x_eq','shared_z_eq','hybrid_x_z128','hybrid_x_z256','hybrid_x_z512',
             '--betas','0','.5','.9','.99','--lambdas','1','3','10','--threads','8']
        save(out/f'layer{layer}.command.json',dict(argv=cmd,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
        so=(out/f'layer{layer}.stdout.log').open('w');se=(out/f'layer{layer}.stderr.log').open('w')
        proc=subprocess.Popen(cmd,env=env,stdout=so,stderr=se);active.append((layer,proc,so,se))
        print(json.dumps(dict(event='launch',layer=layer,pid=proc.pid)),flush=True)
    done=[]
    while active:
        for layer,proc,so,se in active.copy():
            rc=proc.poll()
            if rc is None:continue
            so.close();se.close();done.append(dict(layer=layer,exit_code=rc))
            active.remove((layer,proc,so,se));print(json.dumps(dict(event='complete',layer=layer,exit_code=rc)),flush=True)
        save(out/'progress.json',dict(seconds=time.monotonic()-started,active=[l for l,p,s,e in active],done=done))
        if active:time.sleep(5)
    for p,h in identities.items():assert sha(p)==h
    status='complete' if all(x['exit_code']==0 for x in done) else 'failed'
    save(out/'execution.json',dict(status=status,seconds=time.monotonic()-started,batches=done,all_input_hashes_rechecked=True))
    if status!='complete':raise SystemExit(1)


if __name__=='__main__':run()
