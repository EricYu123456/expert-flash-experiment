#!/usr/bin/env python3
"""Stage2 build/gate launcher. Standard-library only; prepare is payload-free."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
ROOT=Path('/home/ericyu/expert_flash_experiment');ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine';OUT=ROUND/'stage2-quality'
PROTOCOL=ROUND/'stage2_quality_harness.protocol.frozen.json';EXPECTED='0e59b5cc4d3fad695d500a1b50897864fa048ba682d51702914d66b9075e57c6'
S0=ROUND/'stage0';F0=S0/'freezes/rev003';S1=ROUND/'stage1-controls';F1=S1/'freezes/rev003';PREFIX='research_20261103_local_affine_stage2'
UP=Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c');BLD=Path('/home/nfs_client/ericyu/expert_flash_experiment/build/llama-route-trace/llama')
SOURCES=[ROOT/'scripts'/(PREFIX+s) for s in ['_run.py','_worker.py','_wire.hpp','_activation.cpp','_harness.cpp','_synthetic.cpp','_synthetic.py']]
FLAGS=['-O3','-march=native','-fno-fast-math','-ffp-contract=off','-std=c++17','-fopenmp','-pthread']
LIBS=[str(BLD/'src/libllama.a'),str(BLD/'ggml/src/libggml.a'),'-ldl',str(BLD/'ggml/src/libggml-cpu.a'),str(BLD/'ggml/src/libggml-base.a'),'-lm']
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def stat(p):
    s=Path(p).stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'dev':s.st_dev,'inode':s.st_ino}
def cpu_topology(c):
    p=Path(f'/sys/devices/system/cpu/cpu{c}/topology');return tuple((p/k).read_text().strip() for k in ['physical_package_id','core_id'])
def bind(cpus):
    allowed=sorted(os.sched_getaffinity(0));before=allowed
    if cpus:chosen=[int(v) for v in cpus.split(',')]
    elif len(allowed)==8 and len({cpu_topology(c) for c in allowed})==8:chosen=allowed
    else:
        def sample():
            r={}
            for line in Path('/proc/stat').read_text().splitlines():
                k,*v=line.split()
                if re.fullmatch('cpu[0-9]+',k):
                    a=list(map(int,v));r[int(k[3:])]=(sum(a),a[3]+(a[4] if len(a)>4 else 0))
            return r
        a=sample();time.sleep(.25);b=sample();rank=sorted(allowed,key=lambda c:(1-(b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]),c));chosen=[];physical=set()
        for c in rank:
            key=cpu_topology(c)
            if key not in physical:chosen.append(c);physical.add(key)
            if len(chosen)==8:break
    assert len(chosen)==8 and len(set(chosen))==8 and set(chosen)<=set(allowed) and len({cpu_topology(c) for c in chosen})==8
    os.sched_setaffinity(0,chosen)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT']:os.environ[k]='8'
    os.environ.update(OMP_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    return {'at_utc':now(),'before':before,'after':sorted(os.sched_getaffinity(0)),'distinct_physical_cores':8,'all_children_inherit':True}
def manifest():return {
    'candidate_scope':'eight naturally selected partial operators only; other routed/shared/dense sourceGGUF pinned host functions',
    'arms':{'source':0,'observer':1,'copyoriginal':2,'B_pilotcanonical':3,'copycanonical':4,'C_local_pilot':5,'D_factor_pilot':6},
    'core':'Immutable Stage0 canonical quantize/dot/rescale/full-I GGML SiLU and local predictor; immutable Stage1 q512W8 host_floatlatent FMA predictor. No driver imports.',
    'arithmetic':'Canonical input full-H A8 once/layer request; exact DGEMM integer dot; F32 scale order; full-I pinned GGML SiLU/product; full-I A8; Down; actual calibrated separate F32 bias. Factor host FMA latent never A8.',
    'numeric_environment':'x87 RN and MXCSR roundbits0x6000=0/FTZ0/DAZ0; actual8 GGML threadpool checked, strict original activation wrapper checks all8 OMP workers, frozen FMA guard checks all8 and each numeric helper bracketed by strict pool checks. Exact integer DGEMM BLAS restricted to checked caller1 (independent BLAS workers are not inferred from OMP checks); FMA/OpenMP remain8, no candidate arithmetic change.',
    'wire':{'little_endian_header_bytes':116,'version':1,'max_payload_bytes':16<<20,'fields':'magic/version/kind/mode/request/run/context/subrun/arm/decode/micro/model/layer/T/H/K/slots/bytes/status/reserved/absolute_position/token_phase/reserved2','token_phase':{'prefix':0,'own_generated_decode':1,'fixed_replay_or_GT':2},'timeout_seconds':120,'stdout':'binary only','stderr':'diagnostics','rows':'owned F32 X + I32 IDs, reply only complete H2048 pilot slots (t,k,e,flags,y)','errors':'short/duplicate/unsolicited/misepoch/nonfinite/workerdeath abort whole prompt before graph write; no silent fallback'},
    'state':'Evaluated up owns X/IDs, Down same epoch/IDs and consumeonce even zero mask, final live alpha MUL and native sum unchanged. No crossrun scratch/KV; all subruns fresh context; metadata callback ask never reads payload.',
    'preprocessing':'Explicit approved registry operator preprocessing_path + payload manifest path/SHA; Root release binds registry SHA. No hardcoded old tree/P; future refresh registry may bind new tree/P/model hashes with same roles/numerics. No refresh/refit or collector enabled here.',
    'provenance_extension':'Worker observe(epoch,owned_X,owned_IDs) no-op extension point, with absolute input position and prefix/generated/fixed phase. No second capture pipeline or payload persistence.',
    'tooling':'first fixed FIT context/model, source nohook/observer/copyoriginal/repeat fresh fullvocab bits and IDs/alpha trace bits; B/copycanonical actual IPC scatter bit identity. Generated synthetic canonical B1/batch/IPC and pinned activation reference before actual tooling.',
    'development':'first8 fixed FIT records, same model-native tokenized immutable raw128 prefix across arms; source-token replay and <=32 EOG-stopped free run use independent fresh prefixes/KV; no benchmark success/failure.',
    'quality_interface':'explicit 256prefixIDs+256GT IDs; input position255 predicts GT0, input510 predicts GT255; all target NLL serial F64 stable logsumexp; requires separate quality protocol/jobs release. No PG acquisition/refit/bootstrap implemented.',
    'resources':'CPU-only eight distinct physical CPUs from launcher for all children; sequential model processes. Quality graph already computes original Down and IPC replacement: timing/RSS is harness overhead, no compressed B1 latency or8GB proof.',
    'review_schema':{'status':'approved','phase':'TOOLING|DEVELOPMENT|QUALITY','protocol_sha256':EXPECTED,'source_revision':'integer','source_freeze_sha256':'sha256(freeze_receipt.json)','registry_sha256':'sha256(explicit registry.json)','development_extra':'TOOLING_complete_sha256','quality_extra':'approved_jobs_sha256 and quality_protocol_sha256'},
    'compile_flags':FLAGS,'TEST_opened':False,'all_upstream_sources_unchanged':True}
def prepare():
    assert sha(PROTOCOL)==EXPECTED;OUT.mkdir(exist_ok=True);fr=OUT/'freezes';fr.mkdir(exist_ok=True);rev=1+max([int(p.name[3:]) for p in fr.glob('rev[0-9][0-9][0-9]')],default=0);rd=fr/f'rev{rev:03d}';rd.mkdir()
    for p in SOURCES:shutil.copy2(p,rd/p.name)
    save(rd/'algorithm_manifest.frozen.json',manifest());pr=read(PROTOCOL)
    assert sha(F0/'freeze_receipt.json')==pr['upstream_stage0_source_freeze_sha256']
    assert sha(F1/'freeze_receipt.json')==pr['upstream_stage1_source_freeze_sha256']
    registry={'protocol_sha256':EXPECTED,'operator_selection':'exact frozen Stage2 old8 registry, VALID-only; no SCORE reselect','operators':pr['operators'],'payload_manifests':[{'path':str(p),'sha256':sha(p)} for p in [S0/'FITVAL.complete.json',S1/'FITVAL.complete.json']]}
    for o in registry['operators']:o['preprocessing_path']=str(S0/'fitted'/o['model']/f"layer{o['layer']}")
    save(rd/'old8_registry.frozen.json',registry)
    includes=['-I'+str(UP/'include'),'-I'+str(UP/'ggml/include'),'-I'+str(UP/'vendor/nlohmann'),'-I'+str(rd)]
    commands=[['c++',*FLAGS,*includes,str(rd/(PREFIX+'_harness.cpp')),'-o',str(rd/'quality-harness'),*LIBS],['c++',*FLAGS,*includes,'-DLOCAL_AFFINE_LIBRARY',str(rd/(PREFIX+'_activation.cpp')),str(F0/'research_20261103_local_affine_stage0_numeric.cpp'),'-o',str(rd/'strict-activation'),*LIBS],['c++',*FLAGS,*includes,str(rd/(PREFIX+'_synthetic.cpp')),'-o',str(rd/'synthetic-cpp'),*LIBS]]
    save(rd/'build_commands.json',commands)
    for i,cmd in enumerate(commands):
        p=subprocess.run(cmd,capture_output=True,text=True);(rd/f'build{i}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,f'preserved failed build{i}; repair new revision before any data'
    tests=[['./synthetic-cpp',sys.executable,str(rd/(PREFIX+'_worker.py'))],[sys.executable,str(rd/(PREFIX+'_synthetic.py')),'--activation',str(rd/'strict-activation'),'--output',str(rd/'synthetic_python.json')]]
    for i,cmd in enumerate(tests):
        p=subprocess.run(cmd,cwd=rd,capture_output=True,text=True);(rd/f'synthetic{i}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,f'preserved failed synthetic{i}; no actual payload opened'
    save(rd/'synthetic_receipt.json',{'status':'passed','at_utc':now(),'argv':tests,'CPU_affinity':sorted(os.sched_getaffinity(0)),'generated_inputs_weights_only':True,'real_model_prompt_activation_target_fitted_payload_reads':False})
    dependencies={str(p):sha(p) for p in SOURCES};dependencies[str(PROTOCOL)]=EXPECTED;spec=Path(pr['source_specification']['path']);assert sha(spec)==pr['source_specification']['sha256'];dependencies[str(spec)]=sha(spec)
    for folder in [F0,F1]:
        fp=folder/'freeze_receipt.json';receipt=read(fp);dependencies[str(fp)]=sha(fp)
        for name,want in receipt['files'].items():assert sha(name)==want,name;dependencies[name]=want
    for name in LIBS:
        p=Path(name)
        if p.is_file():dependencies[str(p)]=sha(p)
    for p in [UP/'include/llama.h',UP/'ggml/include/ggml.h',UP/'ggml/include/ggml-cpu.h',UP/'ggml/include/ggml-backend.h',UP/'ggml/include/gguf.h',UP/'vendor/nlohmann/json.hpp']:dependencies[str(p)]=sha(p)
    linked=subprocess.run(['ldd',str(rd/'quality-harness')],capture_output=True,text=True,check=True).stdout;(rd/'linked_libraries.txt').write_text(linked)
    for name in re.findall(r'(/[\w/+.\-]+)',linked):
        p=Path(name)
        if p.is_file():dependencies[str(p.resolve())]=sha(p)
    dependencies[str(Path(sys.executable).resolve())]=sha(sys.executable)
    for p in rd.iterdir():
        if p.is_file():dependencies[str(p)]=sha(p)
    receipt={'status':'READY_implementation_build_synthetic_only_pending_Root_review','at_utc':now(),'revision':rev,'protocol_sha256':EXPECTED,'files':dependencies,'registry_path':str(rd/'old8_registry.frozen.json'),'registry_sha256':sha(rd/'old8_registry.frozen.json'),'manifest':manifest(),'payload_reads':False,'CPU_affinity':sorted(os.sched_getaffinity(0))}
    save(rd/'freeze_receipt.json',receipt);pointer={'revision':rev,'freeze_receipt':str(rd/'freeze_receipt.json'),'sha256':sha(rd/'freeze_receipt.json')};save(OUT/'CURRENT_FREEZE.json',pointer);print(json.dumps({'READY':pointer,'files':len(dependencies),'registry_sha256':receipt['registry_sha256']}),flush=True)
def verify(review,phase,registry=None,jobs=None,quality_protocol=None):
    assert sha(PROTOCOL)==EXPECTED;ptr=read(OUT/'CURRENT_FREEZE.json');fp=Path(ptr['freeze_receipt']);assert sha(fp)==ptr['sha256'];r=read(fp);a=read(review);reg=Path(registry or r['registry_path'])
    assert a['status']=='approved' and a['phase']==phase and a['protocol_sha256']==EXPECTED and a['source_revision']==r['revision'] and a['source_freeze_sha256']==ptr['sha256'] and a['registry_sha256']==sha(reg)
    for p,h in r['files'].items():assert sha(p)==h,p
    if phase=='DEVELOPMENT':assert a['TOOLING_complete_sha256']==sha(OUT/'TOOLING.complete.json') and read(OUT/'TOOLING.complete.json')['status']=='complete'
    if phase=='QUALITY':assert jobs and quality_protocol and a['approved_jobs_sha256']==sha(jobs) and a['quality_protocol_sha256']==sha(quality_protocol)
    return fp.parent,fp,r,reg
def actual(a,phase):
    rd,fp,r,reg=verify(a.source_review,phase,a.registry,a.jobs,a.quality_protocol);started=OUT/(phase+'.started.json');assert not started.exists(),'phase already launched, preserve and review any repair'
    # Only after authorization do prompt bodies/fit/model payload identities get
    # read. Existing immutable full model hashes are reused with all4 stat keys.
    identities=read(S0/'content_identities.frozen.json');pr=read(PROTOCOL);records=pr['development_records'][:1 if phase=='TOOLING' else pr['development_gate_count']]
    if phase=='QUALITY':records=None
    chosen=[]
    if records is not None:
        byid={v['id']:v for v in identities['records']}
        for row in records:
            want=byid[row['id']];assert want['split']=='FIT';assert all(stat(row['path'])[k]==want[k]==row[k] for k in ['size','mtime_ns','dev','inode']);assert sha(row['path'])==want['sha256'];chosen.append({'id':row['id'],'path':row['path']})
    launch={'at_utc':now(),'PID':os.getpid(),'argv':sys.argv,'phase':phase,'source_freeze_sha256':sha(fp),'registry_sha256':sha(reg),'CPU_affinity':sorted(os.sched_getaffinity(0)),'models':[],'TEST_opened':False};save(started,launch)
    models=read(ROUND/'stage0_protocol.frozen.json')['models'];reports=[]
    for mid,model in enumerate(models,1):
        identity=next(v for v in identities['models'] if v['path']==model['gguf']);assert stat(identity['path'])=={k:identity[k] for k in ['size','mtime_ns','dev','inode']}
        jobs=chosen if phase!='QUALITY' else read(a.jobs)['models'][model['name']]
        folder=OUT/(phase+'.'+model['name']);folder.mkdir();plan={'phase':phase,'model_id':mid,'gguf':model['gguf'],'run_id':time.time_ns()&((1<<63)-1),'jobs':jobs,'output':str(folder/'result.json'),'worker_argv':[sys.executable,str(rd/(PREFIX+'_worker.py')),'--model',str(mid),'--registry',str(reg),'--freeze',str(fp),'--review',a.source_review,'--phase',phase,'--activation',str(rd/'strict-activation')]};save(folder/'plan.json',plan)
        cmd=[str(rd/'quality-harness'),'run',str(folder/'plan.json')];utc=now();t=time.monotonic()
        with open(folder/'stdout.log','w') as stdout,open(folder/'stderr.log','w') as stderr:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);info={'at_utc':utc,'PID':proc.pid,'argv':cmd,'log':str(folder/'stderr.log'),'CPU_affinity':sorted(os.sched_getaffinity(proc.pid)),'model_identity_reused':identity};launch['models'].append(info);save(started,launch);print(json.dumps({'launched':info}),flush=True);rc=proc.wait()
        closure={**info,'started_at_utc':utc,'completed_at_utc':now(),'returncode':rc,'wall_seconds':time.monotonic()-t};save(folder/'process_exit.json',closure);assert rc==0,'whole prompt tooling-invalid; preserve all outputs, no source fallback'
        assert stat(identity['path'])=={k:identity[k] for k in ['size','mtime_ns','dev','inode']};assert read(folder/'result.json')['status']=='complete';reports.append(folder/'result.json');print(f'{phase} completed {model["name"]} OS0',flush=True)
    save(OUT/(phase+'.complete.json'),{'status':'complete','at_utc':now(),'phase':phase,'source_revision':r['revision'],'source_freeze_sha256':sha(fp),'registry_sha256':sha(reg),'files':{str(p):sha(p) for p in reports},'TEST_opened':False,'scientific_classification':'none; bounded quality diagnostics only'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','tooling','development','quality']);p.add_argument('--source-review');p.add_argument('--registry');p.add_argument('--jobs');p.add_argument('--quality-protocol');p.add_argument('--cpus');a=p.parse_args();affinity=bind(a.cpus)
    if a.command=='prepare':prepare()
    else:assert a.source_review;actual(a,a.command.upper())
