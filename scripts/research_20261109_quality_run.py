#!/usr/bin/env python3
"""Run frozen CPU quality plans; TEST requires a frozen model-selection receipt."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/ericyu/expert_flash_experiment')
DATA=Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')
MODEL=Path('/home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-v2-lite-chat-q8_0.gguf')
DEFAULT_GROUPS=[list(range(a,a+8)) for a in (0,8,16,32,40,48)]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def save(p,x):
    p=Path(p);temporary=p.with_name(p.name+'.partial')
    temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temporary.replace(p)


def now():return datetime.now(timezone.utc).isoformat()


def model_identity():
    st=MODEL.stat();stat={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns}
    cache=DATA/'model_identity.json'
    if cache.exists():
        value=json.loads(cache.read_text())
        if value['stat']==stat:return value
    value={'path':str(MODEL),'stat':stat,'sha256':sha(MODEL),'verified_utc':now()}
    save(cache,value);return value


def registry_files(registry):
    if registry=='-':return []
    p=Path(registry);files=[p];seen=set()
    for line in p.read_text().splitlines():
        if not line or line.startswith('#'):continue
        cols=line.split('\t');assert len(cols)==6
        layer,expert=int(cols[0]),int(cols[1]);assert (layer,expert) not in seen
        seen.add((layer,expert));assert 1<=layer<=26 and 0<=expert<64 and 0<int(cols[3])<=2816
        transform,head=cols[4:]
        for suffix in ('.sources.u8','.indices.i32','.center.f32','.inv_scale.f32'):
            files.append(Path(transform+suffix))
        proto=Path(transform+'.prototypes.i32')
        if proto.exists():files.append(proto)
        factor=Path(head+'.factor.json')
        if factor.exists():
            cfg=json.loads(factor.read_text());width=int(cols[3])
            assert cfg['version']==1 and cfg['H']==2048 and cfg.get('input_width',cfg.get('width'))==width
            assert 'width' not in cfg or cfg['width']==width
            assert 0<int(cfg['rank'])<=1408 and not cfg.get('V_bias',False)
            assert 'input_transform_prefix' not in cfg or cfg['input_transform_prefix']==transform
            expected={head+s for s in ('.V.i8','.V.scales.f32','.latent.center.f32',
                '.latent.inv_scale.f32','.U.i8','.U.scales.f32','.bias.f32')}
            assert set(cfg['payload_sha256'])==expected
            files.append(factor)
            for path,expected_sha in cfg['payload_sha256'].items():
                assert sha(path)==expected_sha,path
                files.append(Path(path))
        else:
            for suffix in ('.i8','.scales.f32','.bias.f32'):files.append(Path(head+suffix))
    assert seen
    return files


def run(planpath):
    planpath=Path(planpath).resolve();plan=json.loads(planpath.read_text())
    stage=plan['stage'];assert stage in ('validation','test','legacy_sensitivity')
    corpus_path=DATA/'corpus103/manifest.json'
    if stage=='legacy_sensitivity':
        corpus_path=Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features/corpus/manifest.json')
    corpus=json.loads(corpus_path.read_text())
    split='quality_validation' if stage=='validation' else 'test'
    rows=[r for r in corpus['records'] if r['split']==split]
    n=plan.get('prompts',len(rows));assert 0<n<=len(rows)
    limit=plan.get('max_tokens',64)
    assert limit in (16,64)
    if stage=='legacy_sensitivity':assert n==1000 and limit==16 and plan['steps']==8
    if stage=='test':
        assert n==1000 and plan['steps']==16
        selection_path=Path(plan['selection_receipt']);selection=json.loads(selection_path.read_text())
        assert selection['status']=='frozen_before_new_TEST'
        assert selection['TEST_results_used_for_selection'] is False
        for path,expected in selection['files_sha256'].items():assert sha(path)==expected,path
    rows=rows[:n]
    assert len({r['id'] for r in rows})==n
    for r in rows:assert sha(r['path'])==r['prompt_sha256']
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    binary=Path(plan['binary']).resolve();groups=plan.get('cpu_groups',DEFAULT_GROUPS)
    threads=plan.get('threads_per_worker',8)
    assert 1<=threads<=16
    assert all(len(g)==threads for g in groups) and len(set(sum(groups,[])))==threads*len(groups)
    identities={str(p):sha(p) for p in (planpath,Path(__file__).resolve(),corpus_path,binary)}
    for source in plan.get('source_files',[]):identities[str(Path(source).resolve())]=sha(source)
    if stage=='test':identities[str(selection_path.resolve())]=sha(selection_path)
    for r in rows:identities[r['path']]=r['prompt_sha256']
    names=[a['name'] for a in plan['arms']];assert len(names)==len(set(names))
    for arm in plan['arms']:
        for p in registry_files(arm['registry']):
            path=str(p.resolve());identities[path]=sha(p)
            if stage=='test':assert selection['files_sha256'].get(path)==identities[path],f'Uncovered TEST arm payload: {path}'
    if stage=='test':
        # Also recheck cost policies, fitting receipts and all other frozen
        # selection inputs at completion, rather than only at initial launch.
        for path,expected in selection['files_sha256'].items():
            if path in identities:assert identities[path]==expected,path
            identities[path]=expected
    model=model_identity()
    save(out/'protocol.frozen.json',dict(plan=plan,at_utc=now(),inputs_sha256=identities,
         model_identity=model,CUDA_VISIBLE_DEVICES='',gpu_layers=0,threads_per_worker=threads))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
    shards=plan.get('shards_per_arm',4)
    assert 1<=shards<=n
    pending=[(a,b) for a in plan['arms'] for b in range(shards)]
    active,done={},[];start=time.monotonic();last=0;failed=False
    while pending or active:
        for slot in range(len(groups)):
            if slot in active or not pending or failed:continue
            arm,shard=pending.pop(0);folder=out/arm['name']/f'shard{shard}';folder.mkdir(parents=True)
            jobs=rows[shard::shards];tsv=folder/'jobs.tsv'
            tsv.write_text(''.join(f"{r['path']}\t{r['id']}\n" for r in jobs))
            cmd=['taskset','-c',','.join(map(str,groups[slot])),str(binary),str(MODEL),str(tsv),
                 str(folder/'data'),str(limit),str(threads),str(plan['steps']),arm['registry'],'w8']
            save(folder/'command.json',dict(argv=cmd,at_utc=now(),CPU_only=True))
            so=(folder/'stdout.log').open('w');se=(folder/'stderr.log').open('w')
            p=subprocess.Popen(cmd,stdout=so,stderr=se,env=env,start_new_session=True)
            active[slot]=(p,folder,arm,shard,jobs,so,se,time.monotonic())
            print(json.dumps(dict(event='launch',arm=arm['name'],shard=shard,pid=p.pid,slot=slot)),flush=True)
        for slot,(p,folder,arm,shard,jobs,so,se,begin) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            so.close();se.close();result=dict(arm=arm['name'],shard=shard,exit_code=rc,seconds=time.monotonic()-begin)
            if rc==0:
                manifest=json.loads((folder/'data/completion.json').read_text())
                assert manifest['status']=='complete' and manifest['jobs']==len(jobs) and manifest['gpu_layers']==0
                assert manifest['decode_steps']==plan['steps'] and manifest['max_tokens']==limit
                summaries=[json.loads(p.read_text()) for p in (folder/'data').glob('*.summary.json')]
                assert len(summaries)==len(jobs) and {r['prompt_id'] for r in summaries}=={r['id'] for r in jobs}
                assert all(r['input_tokens']==limit and r['scored_tokens']==limit-1 for r in summaries)
                result['completion']=manifest
            else:failed=True;pending.clear()
            save(folder/'execution.json',result);done.append(result);del active[slot]
            print(json.dumps(dict(event='complete',**result)),flush=True)
        if time.monotonic()-last>=30:
            running=[dict(arm=a['name'],shard=s,prompts=len(list((f/'data').glob('*.summary.json'))))
                     for p,f,a,s,jobs,so,se,begin in active.values()]
            save(out/'progress.json',dict(seconds=time.monotonic()-start,active=running,done=done,pending=len(pending)))
            print(json.dumps(dict(event='progress',active=running,seconds=time.monotonic()-start)),flush=True);last=time.monotonic()
        if active:time.sleep(2)
    result=dict(status='failed' if failed else 'complete',at_utc=now(),seconds=time.monotonic()-start,batches=done)
    if not failed:
        for p,expected in identities.items():assert sha(p)==expected,f'Changed frozen input: {p}'
        assert model_identity()==model
        if stage=='test':assert sha(MODEL)==model['sha256']
        result['all_input_hashes_rechecked']=True
    save(out/'execution.json',result)
    print(json.dumps(result),flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)
