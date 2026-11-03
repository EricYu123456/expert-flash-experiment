#!/usr/bin/env python3
"""PG19 validation source/build gate and separately approved bounded screen."""
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
ROOT=Path('/home/ericyu/expert_flash_experiment');ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine';OUT=ROUND/'stage2-pg19-quality';F2=ROUND/'stage2-quality/freezes/rev004';FREF=ROUND/'stage2-refresh/freezes/rev002';PROTOCOL=ROUND/'stage2_pg19_quality.protocol.frozen.json';EXPECTED='3f95838e8af920ad6fe477ee0d11153621680c10488452cd7585fc1b4273ac01';REFRESH_FREEZE='8f89fc18a3583e28e22ff0129c5ac09eb3d23a06fdcf09ec6aaddcf333e9206b';PREFIX='research_20261103_local_affine_stage2_pg19_quality';SOURCES=[ROOT/'scripts'/(PREFIX+s) for s in ['_run.py','_core.py','_worker.py','_harness.cpp','_synthetic.cpp','_synthetic.py']]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def stat(p):
    s=Path(p).stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'dev':s.st_dev,'inode':s.st_ino}
def physical(c):
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
            if physical(c) not in seen:chosen.append(c);seen.add(physical(c))
            if len(chosen)==8:break
    assert len(chosen)==len(set(chosen))==8 and set(chosen)<=set(allowed) and len({physical(c) for c in chosen})==8;os.sched_setaffinity(0,chosen)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT']:os.environ[k]='8'
    os.environ.update(OMP_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='');return chosen
def manifest():return {'quality_protocol_sha256':EXPECTED,'refresh_source_freeze_sha256':REFRESH_FREEZE,'scope':'fixed16 officialPG19 validation books only after separate PG19_QUALITY Root+critic release; noTRAINfit/reselection/TEST/automatic science classification','acquisition':'exact deepmind-gutenberg validation object generation, responsegeneration+advertisedsize+base64MD5; originalraw/SHA/URL/UTC preserved, missing/changed no substitution','preprocess':'immutable refresh preprocess strictUTF8 CRLFthenCR→LF, Unicode midpoint thenafterfirstLF elsemidpoint; joint fullsuffix native add_specialtrue parse_specialfalse no chat; first512 once/model/book, same IDs allarms; short retained and entireprimaryfamily inconclusive','inference':'same immutable rev004 Hook/Worker/numeric libraries; fresh context/KV eacharm, ctx1024/batch=ubatch128/threads8; four batches128 feed all512 actualGT IDs. Score target indices256..511 frominput255..510 stable serialF64logsumexp actualfiniteF32fullvocab. Lastinput511 is extra UNUSED diagnostic and never scored, same allarms. No source sums/router/alpha/shared change','source_reference':'Primary source reportedarm0 is internalOBSERVER1 with no overrides, potentiallivepilothits only. Fixedfirstbook actualnohook(SOURCE0) vsobserver fullvocab512 memcmp; observer vsCOPYORIGINAL2 logits+ownedX/IDs/livealpha trace memcmp. Never callobserver nohook. B3 vsCOPYCANONICAL4 fullvocab+trace memcmp, freshKV each. Controls areprospectivebook0; no move toanotherbook ifshort','contexts':'context_plan puremetadata creates67unique contexts/model: firstbook nohook1/sourceobserver2/copyoriginal3/B4/copyB5/C6/D7; laterbooks source/B/C/D uniqueincrements. Worker only receivesactual modes3/4/5/6; source internalOBSERVER1 andcopy2 neverIPC. Fresh immutablejointIDs sameeveryarm','registry':'explicitRoot registrySHA and FIT_REFRESH.completeSHA; Registry loadsrefreshed tree/P/model paths fromnewpayloadmanifest, canonicalB weights unchanged. No silent oldbank/defaultpreprocessing','exposure':'C++ counts only after actualsuccessfulDown/scatter; sourcepotentialhits vsBCDactualrows separate. Worker unchangedpredictreturnedflags keyedmodel/book/context/arm/layer/expert/zone, strictnonnegativeinteger0<=fallback<=returned==Cppscatter. C/D flag0 approximation,flag1 canonicalfallback. Zones0..254/255..510/511. Potentialeffects acrossallcausal0..510 inclprefixKV;511unused cannotexplain256targetloss. Actual B/C zero totalcausalreplacements with differingpriorfullvocabmemcmp abortinstrumentation','summary':'source/B/C/D perbook mean256NLL and customsubset tokenPPL, withinmodeltokenizer only. C-Bprimary; B-source/C-source/D-source/DB/CD decomposition; noC-Bfail impliesC-sourceharm claim','bootstrap':'bothmodelprimaryC-B endpoints retained; same16pairedbook indexdraws10000 seed2026110302, centeredmaxabs t fixedbootstrapSD ddof1, empirical95linearquantile; hats±cSD. Missing/nonfinite/anyzeroSD entirefamilyinconclusive no drop/redraw/literalshortcut.16bookapproximation notpopulationguarantee. Secondarydescriptivepairedpercentiles neverdecision','overflow':'finite logNLL/Delta/bands preserved; derived exp/expm1 overflow null+explicitoverflowflag, never reclassified nonfiniteNLL','quality_budget':'2% relativecustomsubsetPPL marginlog1.02 fixed. Root+critic adjudicate boundedrefreshed8oponly, no automatic failure/success oruniversalimpossibility','numerics_resources':'CPUonly8distinctphysicalCPUs allchildren fromlaunch, BLAS1checkedcaller/frozenFMA/OMP/GGML8 strictx87RN+MXCSRround0 FTZDAZ0. Sourcegraph/originalDown/IPC duplicatework,RSS/timeharnessonly notcompressedruntimeor8GBproof. Fullvocabcontrolbuffers inRAMonly, no longlogitdump','review_schema':{'status':'approved','phase':'PG19_QUALITY','protocol_sha256':EXPECTED,'source_revision':'integer','source_freeze_sha256':'sha256newfreeze','refresh_source_freeze_sha256':REFRESH_FREEZE,'FIT_REFRESH_complete_sha256':'sha256completedTRAINrefresh','registry_sha256':'sha256explicitrefreshedregistry'},'PGvalidation_payload_read_during_prepare':False,'TEST_opened':False}
def prepare():
    assert sha(PROTOCOL)==EXPECTED and sha(FREF/'freeze_receipt.json')==REFRESH_FREEZE;OUT.mkdir(exist_ok=True);fr=OUT/'freezes';fr.mkdir(exist_ok=True);rev=1+max([int(p.name[3:]) for p in fr.glob('rev[0-9][0-9][0-9]')],default=0);rd=fr/f'rev{rev:03d}';rd.mkdir()
    for p in SOURCES:shutil.copy2(p,rd/p.name)
    save(rd/'algorithm_manifest.frozen.json',manifest());base=read(F2/'build_commands.json')[0];i=next(j for j,v in enumerate(base) if v.endswith('_harness.cpp'));flags=base[1:i];libs=base[base.index('-o')+2:];commands=[['c++',*flags,'-I'+str(rd),str(rd/(PREFIX+'_harness.cpp')),'-o',str(rd/'quality-harness'),*libs],['c++',*flags,'-I'+str(rd),str(rd/(PREFIX+'_synthetic.cpp')),'-o',str(rd/'synthetic-cpp'),*libs]];save(rd/'build_commands.json',commands)
    for j,cmd in enumerate(commands):
        p=subprocess.run(cmd,capture_output=True,text=True);(rd/f'build{j}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'preservedfailedbuild, repairbeforepayload'
    tests=[[str(rd/'synthetic-cpp'),sys.executable,str(F2/'research_20261103_local_affine_stage2_worker.py')],[sys.executable,str(rd/(PREFIX+'_synthetic.py')),'--output',str(rd/'synthetic_python.json')]]
    for j,cmd in enumerate(tests):
        p=subprocess.run(cmd,cwd=rd,capture_output=True,text=True);(rd/f'synthetic{j}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'preservedfailedgeneratedsynthetic, nopayload'
    save(rd/'synthetic_receipt.json',{'status':'passed','at_utc':now(),'argv':tests,'CPU_affinity':sorted(os.sched_getaffinity(0)),'generated_only':True,'PGvalidation_payload_read':False});deps={str(p):sha(p) for p in SOURCES};previous=read(FREF/'freeze_receipt.json')
    for p,h in previous['files'].items():assert sha(p)==h,p;deps[p]=h
    deps[str(FREF/'freeze_receipt.json')]=REFRESH_FREEZE;pr=read(PROTOCOL);deps[str(PROTOCOL)]=EXPECTED;mp=Path(pr['data']['metadata_proposal']);assert sha(mp)==pr['data']['metadata_proposal_sha256'];deps[str(mp)]=sha(mp)
    for p in rd.iterdir():
        if p.is_file():deps[str(p)]=sha(p)
    receipt={'status':'READY_source_build_generatedsynthetic_only_pending_Root_critic','at_utc':now(),'revision':rev,'protocol_sha256':EXPECTED,'refresh_source_freeze_sha256':REFRESH_FREEZE,'files':deps,'manifest':manifest(),'PGvalidation_model_prompt_activation_target_fitted_payload_read_during_prepare':False,'CPU_affinity':sorted(os.sched_getaffinity(0))};save(rd/'freeze_receipt.json',receipt);ptr={'revision':rev,'freeze_receipt':str(rd/'freeze_receipt.json'),'sha256':sha(rd/'freeze_receipt.json')};save(OUT/'CURRENT_FREEZE.json',ptr);print(json.dumps({'READY':ptr,'files':len(deps)}),flush=True)
def actual(a):
    ptr=read(OUT/'CURRENT_FREEZE.json');fp=Path(ptr['freeze_receipt']);assert sha(fp)==ptr['sha256'];sys.path.insert(0,str(fp.parent));import research_20261103_local_affine_stage2_pg19_quality_core as q
    assert Path(q.__file__).resolve()==fp.parent/(PREFIX+'_core.py');freeze,fit,registry=q.authorize(fp,a.source_review);startfile=OUT/'QUALITY.started.json';assert not startfile.exists();review_sha=sha(a.source_review);save(startfile,{'at_utc':now(),'PID':os.getpid(),'argv':sys.argv,'CPU_affinity':sorted(os.sched_getaffinity(0)),'source_freeze_sha256':sha(fp),'FIT_REFRESH_complete_sha256':sha(q.REFRESH/'FIT_REFRESH.complete.json'),'registry_sha256':sha(registry),'phase':'PG19_QUALITY','TEST_opened':False});pre_utc=now();fit_before=q.fit_dependency_identities(fit);save(OUT/'FIT_dependencies.before.json',{'at_utc':pre_utc,'review_sha256':review_sha,'files':fit_before});pr=read(PROTOCOL);books=pr['data']['validation16'];assert len(books)==16;acq=OUT/'acquisition';acq.mkdir();acquired=[]
    for b in books:
        url='https://storage.googleapis.com/download/storage/v1/b/deepmind-gutenberg/o/'+urllib.parse.quote(b['name'],safe='')+'?alt=media&generation='+b['generation'];utc=now();request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=60) as response:raw=response.read(b['size_bytes']+1);headers=dict(response.headers);assert response.headers.get('x-goog-generation')==b['generation']
        rp=acq/(str(b['book_id'])+'.raw');rp.write_bytes(raw);suffix,metadata=q.r.preprocess(raw,b);sp=acq/(str(b['book_id'])+'.suffix.utf8');sp.write_bytes(suffix);acquired.append({'book_id':b['book_id'],'pinned_metadata':b,'URL':url,'at_utc':utc,'response_headers':headers,'raw_path':str(rp),'suffix_path':str(sp),**metadata});save(acq/'partial.json',{'status':'partial','books':acquired})
    save(acq/'complete.json',{'status':'complete','books':acquired,'files':{str(p):sha(p) for p in acq.iterdir() if p.is_file() and p.name!='complete.json'}});models=read(ROUND/'stage0_protocol.frozen.json')['models'];identities=read(ROUND/'stage0/content_identities.frozen.json')['models'];outputs=[];exposures=[];jobs,contexts=q.context_plan([b['book_id'] for b in books])
    for mid,m in enumerate(models,1):
        ident=next(v for v in identities if v['path']==m['gguf']);before=stat(m['gguf']);assert before=={k:ident[k] for k in before};folder=OUT/m['name'];folder.mkdir();run=time.time_ns()&((1<<63)-1);cp=folder/'contexts.json';save(cp,{'protocol_sha256':EXPECTED,'model':mid,'run':run,'contexts':contexts});plan={'phase':'PG19_QUALITY','model_id':mid,'gguf':m['gguf'],'run_id':run,'jobs':[{**j,'suffix_path':acquired[i]['suffix_path']} for i,j in enumerate(jobs)],'output':str(folder/'result.json'),'worker_argv':[sys.executable,str(fp.parent/(PREFIX+'_worker.py')),'--model',str(mid),'--freeze',str(fp),'--review',a.source_review,'--activation',str(F2/'strict-activation'),'--context-map',str(cp),'--exposure',str(folder/'exposure.jsonl')]};save(folder/'plan.json',plan);cmd=[str(fp.parent/'quality-harness'),'quality',str(folder/'plan.json')];utc=now();t=time.monotonic()
        with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
            p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);launch={'started_at_utc':utc,'PID':p.pid,'argv':cmd,'log':str(folder/'stderr.log'),'CPU_affinity':sorted(os.sched_getaffinity(p.pid)),'model_identity_reused':ident};save(folder/'launch.json',launch);print(json.dumps({'launched':launch}),flush=True);rc=p.wait()
        save(folder/'exit.json',{**launch,'completed_at_utc':now(),'returncode':rc,'wall_seconds':time.monotonic()-t});assert rc==0,'quality tooling-invalid, preserveall outputs no fallback/reselection';assert stat(m['gguf'])==before;outputs.append(folder/'result.json');exposures.append(folder/'exposure.jsonl');print(f'PG19_QUALITY completed {m["name"]}',flush=True)
    fma=q.core.FMA(q.w.FROZEN1/'fma.so');assert fma.lib.lac_environment()==8;summary=q.summarize_outputs(outputs,exposures,[b['book_id'] for b in books]);assert fma.lib.lac_environment()==8;save(OUT/'summary.json',summary);assert sha(a.source_review)==review_sha;post_freeze,post_fit,post_registry=q.authorize(fp,a.source_review);fit_after=q.fit_dependency_identities(post_fit);assert fit_before==fit_after;identity_receipt=OUT/'FIT_dependency_identity.prepost.json';save(identity_receipt,{'status':'complete_all_FIT_dependencies_identity_equal','before_at_utc':pre_utc,'after_at_utc':now(),'same_review_sha256':review_sha,'source_freeze_sha256':sha(fp),'FIT_REFRESH_complete_sha256':sha(q.REFRESH/'FIT_REFRESH.complete.json'),'registry_sha256':sha(post_registry),'before':fit_before,'after':fit_after,'pre_acquire_full_FIT_dependency_hashes_checked':True,'post_summary_same_authorization_and_all_dependency_hashes_checked':True});paths=[p for folder in [acq,OUT/'deepseek',OUT/'olmoe'] for p in folder.rglob('*') if p.is_file()]+[OUT/'summary.json',OUT/'FIT_dependencies.before.json',identity_receipt];save(OUT/'QUALITY.complete.json',{'status':'complete_no_automatic_scientific_classification','at_utc':now(),'source_freeze_sha256':sha(fp),'FIT_REFRESH_complete_sha256':sha(q.REFRESH/'FIT_REFRESH.complete.json'),'registry_sha256':sha(registry),'primary_family_inconclusive':summary['primary_family_inconclusive'],'files':{str(p):sha(p) for p in paths},'TEST_opened':False,'PGvalidation_fit_or_selection':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--source-review');p.add_argument('--cpus');a=p.parse_args();bind(a.cpus)
    if a.command=='prepare':prepare()
    else:assert a.source_review;actual(a)
