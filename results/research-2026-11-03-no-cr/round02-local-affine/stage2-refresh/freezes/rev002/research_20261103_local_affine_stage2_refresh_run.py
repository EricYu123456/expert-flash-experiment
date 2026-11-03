#!/usr/bin/env python3
"""Refresh phase gates. prepare performs source/build/generated-synthetic only."""
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
import urllib.parse
import urllib.request
ROOT=Path('/home/ericyu/expert_flash_experiment');ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine';OUT=ROUND/'stage2-refresh';F2=ROUND/'stage2-quality/freezes/rev004';PROTOCOL=ROUND/'stage2_refresh.protocol.frozen.json';EXPECTED='78045b99f1e4647475bc7a35ef27110856ed081dc8b2faace7ddd332b5b5d167';PREFIX='research_20261103_local_affine_stage2_refresh';OLD_FREEZE='00a08bfc5eee76d1d6b97d2826457853932058e72c416754f4390f37bbc811e1';OLD_REGISTRY='7ed8766ed70ba1acb48e864003b1119f5b9e875fc03cd2832f963ee915a8bfc9'
SOURCES=[ROOT/'scripts'/(PREFIX+s) for s in ['_run.py','_core.py','_fit.py','_worker.py','_harness.cpp','_synthetic.cpp','_synthetic.py']]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def topology(c):
    p=Path(f'/sys/devices/system/cpu/cpu{c}/topology');return tuple((p/k).read_text().strip() for k in ['physical_package_id','core_id'])
def bind(cpus):
    allowed=sorted(os.sched_getaffinity(0))
    if cpus:chosen=list(map(int,cpus.split(',')))
    else:
        def sample():
            out={}
            for l in Path('/proc/stat').read_text().splitlines():
                k,*v=l.split()
                if re.fullmatch('cpu[0-9]+',k):a=list(map(int,v));out[int(k[3:])]=(sum(a),a[3]+a[4])
            return out
        a=sample();time.sleep(.25);b=sample();chosen=[];seen=set()
        for c in sorted(allowed,key=lambda c:(1-(b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]),c)):
            key=topology(c)
            if key not in seen:chosen.append(c);seen.add(key)
            if len(chosen)==8:break
    assert len(chosen)==len(set(chosen))==8 and set(chosen)<=set(allowed) and len({topology(c) for c in chosen})==8;os.sched_setaffinity(0,chosen)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT']:os.environ[k]='8'
    os.environ.update(OMP_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='');return sorted(chosen)
def manifest():return {'scope':'exactlyone same-Round2 refresh, noVALID/test payload or selection; source-only until explicit phase gates','protocol_sha256':EXPECTED,'immutable_old_helpers':OLD_FREEZE,'old_registry':OLD_REGISTRY,'collection':'newharness includes frozenrev004; fresh B3/C5/D6 each768 native GTprefix, max32 natural EOGstop, live routes/alpha; Bprefix and all3decode only, no sampling/dedup','provenance':'separate ownedX/IDs worker binary and actual finalalpha C++ binary joined by full116-byte typed epoch incl request/run/context/arm/position/phase; book/tokenIDs resolved from immutable prefix+actual generated metadata; unit-weight ordered multiset oldFIT then eachbook Bprefix/Bdecode/Cdecode/Ddecode','acquisition':'GCS deepmind-gutenberg exactgeneration query +responsegeneration/size/base64MD5; preserve raw bytes,SHA,URL,UTC. strictUTF8 CRLFthenCR toLF Unicode midpoint→after firstLF elsemidpoint; pinned native add_specialtrue parse_specialfalse first768 no chat; short means incomplete no replacement','existing_gap':'originalStage0 bound target SHA but notcaptureinput SHA; currentcapture hashes are newly established, never historical','audit':'BIND_OLD_CAPTURE separate readonly release: every current oldFIT row/operator canonical f bitcompared to historical authoritativeY, all8pass atomic; ordered row→target mapping, before/after capture/target/weights SHA+stat, numeric identities, queries/MAC/wall. Proves currentX f=frozenY only, not historicalX equality(noninjective). No mismatched labels/tolerance/row removal/refit','fit':'FIT_REFRESH separate release binds auditcomplete/currentcapture SHA andcollectioncomplete; alltrue boolmask allold/newrows. Rebuild sharedC32 tree/fullHleafmeans/P128normal once completeunion using exact old solvers/seeds/ties; commonA8design/stats, exactone newtarget query/operator/row; oldlabels onlyverified. Original eta/lambda/G_eta dimensionless fixed, trace-scaled beta recomputed; fixedPE6/G2q512W8hostidentity; actualFITresidualbias recalibration; no alternativegrid/guard/rank/format/gauge/geometry','unsupported':'real selector occupancy/group/rank/conditioning; incompletegeometry inconclusive, unsupported canonicalfallback explicit no fabricated rows','registry':'fresh preprocessing/model paths+SHA via separate payloadmanifest; oldsourceweights untouched; no staleoldpreproc fallback; model/geometry/dependencies frozen before PGvalidation','numerics':'immutable Stage0canonical and Stage1FMA; strictrev004activation, exactDGEMM BLAS1checkedcaller; FMA/OMP/GGML8strictRN/x87+MXCSRround0 FTZ0 DAZ0;8distinctphysicalCPUaffinity allchildren, CPUonly','costs':'extraaudit queries separate commonprovenance cost; collectionliveforward separate; maxnewtarget110592rows905969664F32bytes826244333568MAC; offlinecost/RSS notdeployed speed/8GBproof','review_schema':{'status':'approved','phase':'BIND_OLD_CAPTURE|COLLECT_REFRESH|FIT_REFRESH','protocol_sha256':EXPECTED,'source_revision':'integer','source_freeze_sha256':'SHA ofthisreceipt','old_harness_source_freeze_sha256':OLD_FREEZE,'old_registry_sha256':OLD_REGISTRY,'COLLECT_extra':'BIND_OLD_CAPTURE_complete_sha256','FIT_extra':'BIND_OLD_CAPTURE_complete_sha256,COLLECT_REFRESH_complete_sha256,old_capture_files_sha256 exactlyauditmapping'},'TEST_opened':False,'PGvalidation_opened':False}
def prepare():
    assert sha(PROTOCOL)==EXPECTED and sha(F2/'freeze_receipt.json')==OLD_FREEZE and sha(F2/'old8_registry.frozen.json')==OLD_REGISTRY;OUT.mkdir(exist_ok=True);fr=OUT/'freezes';fr.mkdir(exist_ok=True);rev=1+max([int(p.name[3:]) for p in fr.glob('rev[0-9][0-9][0-9]')],default=0);rd=fr/f'rev{rev:03d}';rd.mkdir()
    for p in SOURCES:shutil.copy2(p,rd/p.name)
    save(rd/'algorithm_manifest.frozen.json',manifest());oldcommands=read(F2/'build_commands.json');base=oldcommands[0];sourceindex=next(i for i,v in enumerate(base) if v.endswith('_harness.cpp'));linkindex=base.index('-o')+2;flags=base[1:sourceindex];libs=base[linkindex:];commands=[['c++',*flags,'-I'+str(rd),str(rd/(PREFIX+'_harness.cpp')),'-o',str(rd/'refresh-harness'),*libs],['c++',*flags,'-I'+str(rd),str(rd/(PREFIX+'_synthetic.cpp')),'-o',str(rd/'synthetic-cpp'),*libs]];save(rd/'build_commands.json',commands)
    for i,cmd in enumerate(commands):
        p=subprocess.run(cmd,capture_output=True,text=True);(rd/f'build{i}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'preserved failed build; noactualpayload'
    tests=[[str(rd/'synthetic-cpp'),sys.executable,str(F2/'research_20261103_local_affine_stage2_worker.py'),str(rd)],[sys.executable,str(rd/(PREFIX+'_synthetic.py')),'--output',str(rd/'synthetic_python.json')]]
    for i,cmd in enumerate(tests):
        p=subprocess.run(cmd,cwd=rd,capture_output=True,text=True);(rd/f'synthetic{i}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'preserved failed synthetic; noactualpayload'
    save(rd/'synthetic_receipt.json',{'status':'passed','at_utc':now(),'argv':tests,'generated_only':True,'actual_payload_reads':False,'CPU_affinity':sorted(os.sched_getaffinity(0))});deps={str(p):sha(p) for p in SOURCES};oldreceipt=read(F2/'freeze_receipt.json')
    for p,h in oldreceipt['files'].items():assert sha(p)==h,p;deps[p]=h
    deps[str(F2/'freeze_receipt.json')]=OLD_FREEZE;pr=read(PROTOCOL)
    for p,h in [(PROTOCOL,EXPECTED),(Path(pr['train_metadata']),pr['train_metadata_sha256']),(Path(pr['source_theory_specification']),pr['source_theory_specification_sha256']),(ROUND/'stage0/FITVAL.complete.json',None),(ROUND/'stage0/content_identities.frozen.json',None),(ROUND/'stage1_controls.protocol.frozen.json',None),(ROUND/'stage2-quality/critic_old_capture_binding_concept.20261103T041741Z.json','58c285c4d26f23306a063c2fdd3f53ff8892a493ffc07e8cff17e0e3076105be')]:
        actual=sha(p);assert h is None or h==actual;deps[str(p)]=actual
    for p in rd.iterdir():
        if p.is_file():deps[str(p)]=sha(p)
    receipt={'status':'READY_source_build_generatedsynthetic_only_pending_Root_critic','at_utc':now(),'revision':rev,'protocol_sha256':EXPECTED,'old_harness_source_freeze_sha256':OLD_FREEZE,'old_registry_sha256':OLD_REGISTRY,'files':deps,'manifest':manifest(),'actual_new_model_context_book_activation_target_fitted_payload_reads':False,'CPU_affinity':sorted(os.sched_getaffinity(0))};save(rd/'freeze_receipt.json',receipt);ptr={'revision':rev,'freeze_receipt':str(rd/'freeze_receipt.json'),'sha256':sha(rd/'freeze_receipt.json')};save(OUT/'CURRENT_FREEZE.json',ptr);print(json.dumps({'READY':ptr,'files':len(deps)}),flush=True)
def verify(review,phase):
    ptr=read(OUT/'CURRENT_FREEZE.json');fp=Path(ptr['freeze_receipt']);assert sha(fp)==ptr['sha256'];sys.path.insert(0,str(fp.parent));import research_20261103_local_affine_stage2_refresh_core as r
    assert Path(r.__file__).resolve()==fp.parent/(PREFIX+'_core.py');receipt,approval=r.authorize(fp,review,phase);return fp,receipt,approval,r
def acquire(r):
    pr=read(PROTOCOL);meta=read(pr['train_metadata']);assert sha(pr['train_metadata'])==pr['train_metadata_sha256'] and meta['items']==pr['train16'];folder=OUT/'acquisition';folder.mkdir();receipts=[]
    for book in pr['train16']:
        url='https://storage.googleapis.com/download/storage/v1/b/deepmind-gutenberg/o/'+urllib.parse.quote(book['name'],safe='')+'?alt=media&generation='+book['generation'];start=now();req=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=60) as response:raw=response.read(int(book['size_bytes'])+1);headers=dict(response.headers);generation=response.headers.get('x-goog-generation');assert generation==book['generation'],'generation response mismatch'
        rawpath=folder/(str(book['book_id'])+'.raw');rawpath.write_bytes(raw);suffix,info=r.preprocess(raw,book);suffixpath=folder/(str(book['book_id'])+'.suffix.utf8');suffixpath.write_bytes(suffix);receipts.append({'book_id':book['book_id'],'pinned_metadata':book,'URL':url,'at_utc':start,'response_headers':headers,'raw_path':str(rawpath),'suffix_path':str(suffixpath),**info});save(folder/'acquisition.partial.json',{'status':'partial','books':receipts})
    save(folder/'acquisition.complete.json',{'status':'complete','books':receipts,'files':{str(p):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='acquisition.complete.json'}});return receipts
def collect(fp,review,r):
    books=acquire(r);models=read(ROUND/'stage0_protocol.frozen.json')['models'];identities=read(ROUND/'stage0/content_identities.frozen.json')['models'];reports=[];collected=OUT/'collected';collected.mkdir()
    for mid,m in enumerate(models,1):
        ident=next(v for v in identities if v['path']==m['gguf']);st=Path(m['gguf']).stat();assert all(v==getattr(st,{'size':'st_size','mtime_ns':'st_mtime_ns','dev':'st_dev','inode':'st_ino'}[k]) for k,v in ident.items() if k in ['size','mtime_ns','dev','inode']);folder=OUT/('collection.'+m['name']);folder.mkdir();run=time.time_ns()&((1<<63)-1);contexts={str(bi*3+ai+1):{'book_id':b['book_id'],'arm':arm} for bi,b in enumerate(books) for ai,arm in enumerate([3,5,6])};cp=folder/'contexts.json';save(cp,{'protocol_sha256':EXPECTED,'model':mid,'run':run,'contexts':contexts});plan={'phase':'COLLECT_REFRESH','model_id':mid,'gguf':m['gguf'],'run_id':run,'jobs':[{'book_id':b['book_id'],'suffix_path':b['suffix_path']} for b in books],'alpha_ledger':str(folder/'alpha.bin'),'output':str(folder/'result.json'),'worker_argv':[sys.executable,str(fp.parent/(PREFIX+'_worker.py')),'--model',str(mid),'--freeze',str(fp),'--review',review,'--registry',str(F2/'old8_registry.frozen.json'),'--activation',str(F2/'strict-activation'),'--records',str(folder/'owned_inputs.bin'),'--context-map',str(cp)]};save(folder/'plan.json',plan);cmd=[str(fp.parent/'refresh-harness'),'collect',str(folder/'plan.json')];t=time.monotonic();start=now()
        with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
            p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);launch={'started_at_utc':start,'PID':p.pid,'argv':cmd,'log':str(folder/'stderr.log'),'CPU_affinity':sorted(os.sched_getaffinity(p.pid)),'model_identity_reused':ident};save(folder/'launch.json',launch);print(json.dumps({'launched':launch}),flush=True);rc=p.wait()
        save(folder/'exit.json',{**launch,'completed_at_utc':now(),'returncode':rc,'wall_seconds':time.monotonic()-t});assert rc==0,'collection incomplete/tooling-invalid, no substitution or fallback';after=Path(m['gguf']).stat();assert all(getattr(after,k)==getattr(st,k) for k in ['st_size','st_mtime_ns','st_dev','st_ino']),'source modelstat changed';report=r.close_collection(m['name'],folder/'owned_inputs.bin',folder/'alpha.bin',folder/'result.json',collected/m['name'],[b['book_id'] for b in books]);save(folder/'closure.json',report);reports.append(report);print(f'COLLECT_REFRESH complete {m["name"]}',flush=True)
    files=[p for folder in [OUT/'acquisition',OUT/'collection.deepseek',OUT/'collection.olmoe',collected] for p in folder.rglob('*') if p.is_file()];save(OUT/'COLLECT_REFRESH.complete.json',{'status':'complete','at_utc':now(),'source_freeze_sha256':sha(fp),'BIND_OLD_CAPTURE_complete_sha256':sha(OUT/'BIND_OLD_CAPTURE.complete.json'),'models':reports,'files':{str(p):sha(p) for p in files},'PGvalidation_opened':False,'TEST_opened':False,'offroute_teacher_targets_queried':False})
def actual(a,phase):
    fp,receipt,approval,r=verify(a.source_review,phase);started=OUT/(phase+'.started.json');assert not started.exists(),'preserve prior phase, reviewed repair required';save(started,{'at_utc':now(),'PID':os.getpid(),'argv':sys.argv,'phase':phase,'source_freeze_sha256':sha(fp),'CPU_affinity':sorted(os.sched_getaffinity(0)),'TEST_opened':False})
    if phase=='COLLECT_REFRESH':collect(fp,a.source_review,r);return
    sys.path.insert(0,str(fp.parent));import research_20261103_local_affine_stage2_refresh_fit as fit
    fma=r.core.FMA(r.w.FROZEN1/'fma.so');num=r.w.GuardedNumeric(fma,F2/'strict-activation')
    try:
        if phase=='BIND_OLD_CAPTURE':fit.bind_old_capture(num,sha(fp))
        else:fit.fit_refresh(num,fma,sha(fp))
    finally:num.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','bind-old-capture','collect','fit']);p.add_argument('--source-review');p.add_argument('--cpus');a=p.parse_args();bind(a.cpus)
    if a.command=='prepare':prepare()
    else:assert a.source_review;actual(a,{'bind-old-capture':'BIND_OLD_CAPTURE','collect':'COLLECT_REFRESH','fit':'FIT_REFRESH'}[a.command])
