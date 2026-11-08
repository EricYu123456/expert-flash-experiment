#!/usr/bin/env python3
"""Extend immutable calibration paragraphs to up to64 tokens, four CPU workers."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

REPO = Path('/home/ericyu/expert_flash_experiment')
OLD = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')
NEW = OLD.with_name('research-2026-11-09-shared-features')
GROUPS = [list(range(a, a+8)) for a in (0, 8, 32, 40)]


def save(p, x):
    p.write_text(json.dumps(x, indent=2)+'\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = NEW/'captures64'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'started.json').exists(), 'Preserve existing execution'
    binary = OLD/'build/shared-feature-capture'
    source = REPO/'scripts/research_20261108_shared_feature_capture.cpp'
    corpus = json.loads((OLD/'corpus/manifest.json').read_text())
    selected = [r for r in corpus['records'] if r['split'] in ('train', 'validation')]
    assert len(selected)==800
    for r in selected:
        assert sha(Path(r['path'])) == r['prompt_sha256']
    queue = [(s,b) for s in ('train','validation') for b in range(4)]
    protocol = {'CPU_only':True,'max_tokens':64,'fixed_old_TRAIN_VALID_IDs':True,
                'never_open_old_TEST':True,'candidate_decisions_do_not_use_new_TEST':True,
                'counts':{'train':600,'validation':200},'threads':8,'cpu_groups':GROUPS,
                'variable_lengths':'Keep all old600/200 paragraphs; min(native tokens,64), no input padding or output-dependent filtering.',
                'inputs_sha256':{str(p):sha(p) for p in (Path(__file__),binary,source,OLD/'corpus/manifest.json')},
                'source_records':selected}
    save(out/'protocol.frozen.json',protocol)
    save(out/'started.json',{'pid':os.getpid(),'protocol_sha256':sha(out/'protocol.frozen.json')})
    active, done = {},[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
    start=time.monotonic(); last=0; failed=False
    while queue or active:
        for slot in range(4):
            if slot in active or not queue or failed:continue
            split,shard=queue.pop(0); folder=out/split/f'shard{shard}';folder.mkdir(parents=True,exist_ok=False)
            jobs=OLD/f'corpus/{split}.shard{shard}.jobs.tsv'
            cmd=['taskset','-c',','.join(map(str,GROUPS[slot])),str(binary),
                 '/home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-v2-lite-chat-q8_0.gguf',
                 str(jobs),str(folder),'64','8','w8','1:14,40;13:10,56;26:0,1']
            save(folder/'command.json',{'argv':cmd,'CUDA_VISIBLE_DEVICES':''})
            stdout=(folder/'stdout.log').open('w');stderr=(folder/'stderr.log').open('w')
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            active[slot]=(proc,folder,split,shard,stdout,stderr,time.monotonic())
            print(json.dumps({'launch':split,'shard':shard,'pid':proc.pid}),flush=True)
        for slot,(p,folder,s,b,so,se,begin) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            so.close();se.close();result={'split':s,'shard':b,'exit_code':rc,'seconds':time.monotonic()-begin}
            if rc==0:
                m=json.loads((folder/'capture_manifest.json').read_text());assert m['jobs']==(150 if s=='train' else 50)
                scores=[json.loads(f.read_text()) for f in (folder/'scores').glob('*.summary.json')]
                actual_rows=sum(x['tokens'] for x in scores);assert len(scores)==m['jobs']
                assert all(x['rows']==actual_rows for x in m['layers'].values())
                result['actual_rows']=actual_rows
            else:failed=True;queue.clear()
            save(folder/'execution.json',result);done.append(result);del active[slot]
            print(json.dumps({'complete':result}),flush=True)
        if time.monotonic()-last>=30:
            progress=[{'split':s,'shard':b,'prompts':sum(x.startswith('COMPLETE ') for x in (f/'stdout.log').read_text().splitlines())}
                      for p,f,s,b,so,se,begin in active.values()]
            save(out/'progress.json',{'seconds':time.monotonic()-start,'active':progress,'done':done,'pending':len(queue)})
            print(json.dumps({'progress':progress,'seconds':time.monotonic()-start}),flush=True);last=time.monotonic()
        if active:time.sleep(2)
    assert not failed, 'Capture failed; partial outputs preserved'
    for p,h in protocol['inputs_sha256'].items():assert sha(Path(p))==h
    save(out/'execution.json',{'status':'complete','batches':done,'seconds':time.monotonic()-start})


if __name__=='__main__':main()
