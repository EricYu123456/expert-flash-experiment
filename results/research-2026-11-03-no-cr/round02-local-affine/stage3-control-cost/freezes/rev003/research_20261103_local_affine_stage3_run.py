#!/usr/bin/env python3
"""One explicit Stage3 phase per invocation; no implicit scientific release."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path('/home/ericyu/expert_flash_experiment');ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine';OUT=ROUND/'stage3-control-cost';PREFIX='research_20261103_local_affine_stage3';PROTOCOL=ROUND/'stage3_control_cost.protocol.frozen.json';EXPECTED='5f21869ae968057a09ed1f5f64cf06f83aa22fa1dc442303fa6ce5c25f9b58be';AMEND=ROUND/'stage3_phase_order.amendment.frozen.json';AMEND_SHA='cd8fb02247d0be93850c598097d95963c3538607118553d1b4941da1a64a835b';FQ=ROUND/'stage2-pg19-quality/freezes/rev001';FREF=ROUND/'stage2-refresh/freezes/rev002';SOURCES=[ROOT/'scripts'/(PREFIX+s) for s in ['_run.py','_core.py','_numeric.hpp','_numeric.cpp','_worker.py','_harness.cpp','_synthetic.py']]
PHASE={'bind':'BIND','bench-bcd':'BENCH_BCD','gu-fit':'GU_FIT','bench-nl':'BENCH_NL','dev-quality':'DEV_QUALITY'}
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def bind_cpu():
    cpus=list(range(8));assert set(cpus)<=os.sched_getaffinity(0);physical=[];nodes=[]
    for c in cpus:
        t=Path(f'/sys/devices/system/cpu/cpu{c}');physical.append(tuple((t/'topology'/k).read_text().strip() for k in ['physical_package_id','core_id']));nodes.append(sorted(p.name for p in t.glob('node[0-9]*')))
    assert len(set(physical))==8 and len({tuple(v) for v in nodes})==1,'eight distinct cores in one NUMA node required';os.sched_setaffinity(0,cpus)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT']:os.environ[k]='8'
    os.environ.update(OMP_DYNAMIC='FALSE',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');return {'CPU_affinity':cpus,'physical_package_core':physical,'NUMA_nodes':nodes}
def process_samples(pid):
    pending=[pid];seen=set();out=[]
    while pending:
        q=pending.pop()
        if q in seen:continue
        seen.add(q);d=Path(f'/proc/{q}')
        try:
            status={k:v.strip() for k,v in (line.split(':',1) for line in (d/'status').read_text().splitlines())};threads=list((d/'task').iterdir());aff={}
            for t in threads:
                try:
                    a=sorted(os.sched_getaffinity(int(t.name)));aff[t.name]=a
                    pending+=list(map(int,(t/'children').read_text().split()))
                except (FileNotFoundError,ProcessLookupError):pass
            out.append({'PID':q,'name':status['Name'],'RSS_KiB':int(status.get('VmRSS','0 kB').split()[0]),'VmHWM_KiB':int(status.get('VmHWM','0 kB').split()[0]),'thread_affinities':aff})
        except (FileNotFoundError,ProcessLookupError):pass
    return out
def hardware_snapshot():
    cores=[]
    for c in range(8):
        d=Path(f'/sys/devices/system/cpu/cpu{c}');v={'CPU':c,'package':(d/'topology/physical_package_id').read_text().strip(),'core':(d/'topology/core_id').read_text().strip(),'NUMA_nodes':[p.name for p in d.glob('node[0-9]*')]}
        for key in ['scaling_governor','scaling_cur_freq','scaling_min_freq','scaling_max_freq']:
            f=d/'cpufreq'/key;v[key]=f.read_text().strip() if f.exists() else None
        v['caches']=[{k:(q/k).read_text().strip() for k in ['level','type','size','shared_cpu_list']} for q in (d/'cache').glob('index*')];cores.append(v)
    return {'at_utc':now(),'cores':cores,'CPU_affinity':sorted(os.sched_getaffinity(0)),'cpuinfo':Path('/proc/cpuinfo').read_text().split('\n\n')[0],'loadavg':Path('/proc/loadavg').read_text().strip(),'other_jobs_not_modified':True,'disk_free_bytes':shutil.disk_usage(OUT).free,'actual_EPYC_not_target_i9':True,'compiler':subprocess.run(['c++','--version'],capture_output=True,text=True).stdout.splitlines()[0]}
def monitored(cmd,folder,cap,checkpoint=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);assert not (folder/'launch.json').exists();t=time.monotonic();utc=now();samples=[];violations=[];peak=0
    with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
        p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True);save(folder/'launch.json',{'at_utc':utc,'PID':p.pid,'parent_PID':os.getpid(),'argv':cmd,'CPU_affinity':sorted(os.sched_getaffinity(p.pid)),'RSS_cap_bytes':cap,'checkpoint_seconds':checkpoint});print(json.dumps({'launched_PID':p.pid,'phase_folder':str(folder),'at_utc':utc}),flush=True)
        with (folder/'resource_samples.jsonl').open('w') as f:
            while p.poll() is None:
                pp=process_samples(os.getpid());rss=sum(v['RSS_KiB'] for v in pp)*1024;peak=max(peak,rss);sample={'at_utc':now(),'aggregate_RSS_bytes':rss,'processes':pp};f.write(json.dumps(sample)+'\n');f.flush();samples.append(sample)
                bad=[(v['PID'],tid,a) for v in pp for tid,a in v['thread_affinities'].items() if not a or not set(a)<=set(range(8))]
                if rss>cap or bad or (checkpoint is not None and time.monotonic()-t>checkpoint):
                    violations.append({'at_utc':now(),'RSS_bytes':rss,'affinity_violations':bad,'checkpoint_exhausted':checkpoint is not None and time.monotonic()-t>checkpoint});os.killpg(p.pid,signal.SIGTERM);break
                time.sleep(.2)
        rc=p.wait()
    receipt={'started_at_utc':utc,'completed_at_utc':now(),'PID':p.pid,'returncode':rc,'wall_seconds':time.monotonic()-t,'peak_observed_aggregate_RSS_bytes':peak,'sampling_interval_seconds':.2,'observed_peak_is_lower_bound':True,'violations':violations,'samples':len(samples),'CPU_affinity':list(range(8))};save(folder/'exit.json',receipt);assert rc==0 and not violations,'preserved tooling/resource/checkpoint failure; same stage needs review';return receipt
def manifest():
    return {'base_protocol_sha256':EXPECTED,'phase_order_amendment_sha256':AMEND_SHA,'new_arms':{'NL':7,'COPY_NL':8,'old_D_unchanged':6},'phase_order':['prepare','bind','bench-bcd','Root_critic_checkpoint','gu-fit','bench-nl','dev-quality'],'bench_arms':{'BENCH_BCD':[3,5,6],'BENCH_NL':[7]},'bench_arm_rotation':'trialmod3 forBCD; NLonly separatephase, never fused simultaneousrace','GU':'one productionpass, complete union236888 expertrows; rawtargets retained; one cached actual FMA GU prediction pass then one Down calibration pass; additional audits/B1calls separately counted','numeric':'exactAVX2 signextendedI16 nonsaturatingmaddI32, width1..2048 APIaccepts[-128,127], absbound33554432; canonical127; FMA original8lanes/pairtree/tail; sequentialFP64 tree; original8FP64 P lanes withtailtosums0; separateF32rescale/bias/alpha','activation':'timed onlypinnedfullI GGML SiLU×F32Up+finite usingpreallocatedscratch; fusedbitcomparison auditonly; no IPCintimedregion','team':'persistent8 stdthreads pinnedCPU0..7, crossselectedslots firststage once/completeoutputstage once, privatewholeI activation cohort once; Cnofallback oneoutputteam; scratchpreallocatedcountedRAM, actualx87/SSE RN FTZDAZ0 eachworker before/afterjob; identicalrowlaneorder andtruebarriers','bench_rows':'SHA256 domain bytes stage3-bench-v1\\0 + canonical JSON(sortkeys,compact,ensure_ascii) provenance;32 perlayer frozenbeforearrayread; first4 numericallrows/offroute','cohorts':'operator0/operator1 rawx-tooutput(alpha1); naturalpartial orderedlivepilotIDs/finalalpha; fullKresourceonly templatescycle2 physicallydistinctallocations. Sharedinput iffidenticalfunctions; zero naturalhit no P/tree','cache':'warm5warmups/31trials; cold all8 workers each128MiB of1GiB stride64 beforeeachtrial, barrieroutsideinvocation; observedscenario notuniversalcacheclaim','source_quality':'newmode loop uses immutableHook/ownedX/finalalpha/scatter. PG16 alreadyexposed devonly, nativeGT512 IDs frompreviousapprovedqualityresult, freshKVeacharm, fixed256targets; no booksdownload/re-tokenization/windowchange. Firstfixedbook actualsource/copy/B controls, BCDprior64bitchecksum+NLL consistency explicitlynot historicalmemcmp, NLcopy actualmemcmp; zero-causal needsactualB replay','dev_statistics':'NL-B2endpoint centeredmaxabsT10000shared16bookdraws seed2026110303 ddof1 linear95; anymissing/nonfinite/zeroSD familyinconclusive; no classification/reselection','review_schema':{'status':'approved','phase':'BIND|BENCH_BCD|GU_FIT|BENCH_NL|DEV_QUALITY','protocol_sha256':EXPECTED,'phase_order_amendment_sha256':AMEND_SHA,'source_revision':'int','source_freeze_sha256':'SHA','upstream_receipts':'exactfreeze mapping','BIND_complete_sha256':'required allafterBIND','BENCH_BCD_complete_sha256':'required GU_FIT andlater','GU_FIT_complete_sha256':'required BENCH_NL/DEV','NL_registry_sha256':'requiredDEV'},'no_actual_payload_during_prepare':True,'optional_structural_preflight_omitted':True,'cost_scope':'EPYC9354AVX2 actualservice only, targetISA/bandwidthunmeasured; conditionalordinaryAiFDAG no fixedspeedup; allhost/cache andflashGUhostDown andsamebankordinaryload alternatives retained'}
def prepare():
    assert sha(PROTOCOL)==EXPECTED and sha(AMEND)==AMEND_SHA;approved=ROUND/'Root_stage3_SOURCE_BUILD_GENERATED.approved.json';assert sha(approved)=='12279b2de05196968f6311dd805e5b8b268b34600fb0ea149238de1a5e755e2c';OUT.mkdir(exist_ok=True);fr=OUT/'freezes';fr.mkdir(exist_ok=True);rev=1+max([int(p.name[3:]) for p in fr.glob('rev[0-9][0-9][0-9]')],default=0);rd=fr/f'rev{rev:03d}';rd.mkdir()
    for p in SOURCES:shutil.copy2(p,rd/p.name)
    save(rd/'algorithm_manifest.frozen.json',manifest());save(rd/'hardware.json',hardware_snapshot());base=read(FQ/'build_commands.json')[0];i=next(k for k,v in enumerate(base) if v.endswith('_harness.cpp'));flags=base[1:i];libs=base[base.index('-o')+2:];numericflags=[v for v in flags if v!='-march=native']+['-mavx2','-mfma','-mno-avx512f'];cmds=[['c++',*numericflags,'-I'+str(rd),str(rd/(PREFIX+'_numeric.cpp')),'-o',str(rd/'host-numeric'),*libs],['c++',*numericflags,'-I'+str(rd),'-fPIC','-shared','-DLS3_LIBRARY',str(rd/(PREFIX+'_numeric.cpp')),'-o',str(rd/'numeric.so')],['c++',*flags,'-I'+str(rd),str(rd/(PREFIX+'_harness.cpp')),'-o',str(rd/'dev-harness'),*libs]];save(rd/'build_commands.json',cmds)
    for index,cmd in enumerate(cmds):monitored(cmd,rd/f'build{index}',8000000000,1800)
    tests=[[str(rd/'host-numeric'),'synthetic'],[sys.executable,str(rd/(PREFIX+'_synthetic.py')),'--folder',str(rd/'generated')]]
    testreceipts=[]
    for index,cmd in enumerate(tests):testreceipts.append(monitored(cmd,rd/f'test{index}',8000000000,1800))
    sys.path.insert(0,str(rd));import research_20261103_local_affine_stage3_core as s
    upstream={str(p):sha(p) for p in s.UPSTREAM};deps={str(p):sha(p) for p in SOURCES+[PROTOCOL,AMEND,approved]}
    for p,h in read(FREF/'freeze_receipt.json')['files'].items():
        # This predecessor source freeze contains source/build/manifest files,
        # not numerical payloads. Fail closed rather than hash a forbidden body.
        q=Path(p);assert q.suffix not in ['.npy','.npz','.i8','.f32','.gguf','.raw','.utf8'];assert sha(q)==h,p;deps[p]=h
    for p in [FREF/'freeze_receipt.json',FQ/'freeze_receipt.json',*map(Path,upstream)]:deps[str(p)]=sha(p)
    for p,h in read(FQ/'freeze_receipt.json')['files'].items():assert Path(p).suffix not in ['.npy','.npz','.i8','.f32','.gguf','.raw','.utf8'];assert sha(p)==h;deps[p]=h
    for p in rd.rglob('*'):
        if p.is_file():deps[str(p)]=sha(p)
    f={'status':'READY_source_build_generated_only_pending_Root_critic_payload_review','revision':rev,'at_utc':now(),'protocol_sha256':EXPECTED,'phase_order_amendment_sha256':AMEND_SHA,'upstream_receipts':upstream,'files':deps,'manifest':manifest(),'generated_tests':testreceipts,'actual_payload_read':False,'TEST_opened':False};save(rd/'freeze_receipt.json',f);pointer={'revision':rev,'freeze_receipt':str(rd/'freeze_receipt.json'),'sha256':sha(rd/'freeze_receipt.json')};save(OUT/'CURRENT_FREEZE.json',pointer);print(json.dumps({'READY':pointer,'files':len(deps)}),flush=True)
def context_plan(book_ids):
    jobs=[];contexts={};n=0
    for bi,b in enumerate(book_ids):
        modes=[('nohook',0),('source',1),('copyoriginal',2),('B',3),('copyB',4),('C',5),('D',6),('NL',7),('copyNL',8)] if bi==0 else [('NL',7),('zeroB',3)];job={'book_id':b,'contexts':{}}
        for key,mode in modes:n+=1;job['contexts'][key]=n;contexts[str(n)]={'book_id':b,'mode':mode,'zeroB_reserved_integrity_only':key=='zeroB'}
        jobs.append(job)
    assert n==39;return jobs,contexts
def bench(s,phase,freeze,review):
    cp=s.read(OUT/'BIND.complete.json');deps={**s.fit_files(),**cp['files']}
    if phase=='BENCH_NL':deps.update(s.read(OUT/'GU_FIT.complete.json')['files'])
    before=s.identities(deps);dest=OUT/phase.lower();dest.mkdir();save(dest/'machine.before.json',hardware_snapshot());outputs=[]
    for name,layers in [('deepseek',[1,13]),('olmoe',[0,8])]:
        for layer in layers:
            bf=OUT/'bound'/name/f'layer{layer}';lp=bf/'layer.json' if phase=='BENCH_BCD' else OUT/'NL'/name/f'layer{layer}'/'layer.json';folder=dest/f'{name}.L{layer}';folder.mkdir();plan={'phase':phase,'review':str(review),'protocol_sha256':EXPECTED,'phase_order_amendment_sha256':AMEND_SHA,'source_freeze_sha256':sha(freeze),'layer_manifest':str(lp),'arms':[3,5,6] if phase=='BENCH_BCD' else [7],**read(bf/'sample.json'),'output':str(folder/'results.json')};save(folder/'plan.json',plan);monitored([str(Path(freeze).parent/'host-numeric'),'bench',str(folder/'plan.json')],folder/'process',8000000000);outputs.append(str(folder/'results.json'));print(f'{phase} completed {name} L{layer}',flush=True)
    assert s.identities(deps)==before;save(dest/'machine.after.json',hardware_snapshot());return s.finish(OUT,phase,freeze,[p for p in dest.rglob('*') if p.is_file()],{'results':outputs,'all_dependencies_prepost_equal':True,'no_fullmodel_latency_claim':True,'phase_separate_not_simultaneous_race':True})
def dev(s,freeze,review):
    deps={**s.fit_files(),**s.read(OUT/'GU_FIT.complete.json')['files'],**s.read(QOUT/'QUALITY.complete.json')['files']};before=s.identities(deps);dest=OUT/'dev-quality';dest.mkdir();outputs=[];oldvalues=[];nlvalues=[];models=s.read(ROUND/'stage0_protocol.frozen.json')['models'];ids=s.read(ROUND/'stage0/content_identities.frozen.json')['models'];books=None
    for mid,m in enumerate(models,1):
        name=m['name'];previous=s.read(QOUT/name/'result.json');assert previous['status']=='complete' and len(previous['jobs'])==16;bb=[v['book_id'] for v in previous['jobs']];books=bb if books is None else books;assert books==bb;jobs,contexts=context_plan(books);folder=dest/name;folder.mkdir();run=time.time_ns()&((1<<63)-1);cp=folder/'contexts.json';save(cp,{'protocol_sha256':EXPECTED,'model':mid,'run':run,'contexts':contexts})
        for a,b in zip(jobs,previous['jobs']):a.update(immutable_joint_ids=b['immutable_joint_ids'],historical_runs=b['runs'])
        identity=next(v for v in ids if v['path']==m['gguf']);st=Path(m['gguf']).stat();stat={k:getattr(st,v) for k,v in [('size','st_size'),('mtime_ns','st_mtime_ns'),('dev','st_dev'),('inode','st_ino')]};assert stat=={k:identity[k] for k in stat}
        plan={'phase':'DEV_QUALITY','model_id':mid,'gguf':m['gguf'],'run_id':run,'jobs':jobs,'output':str(folder/'result.json'),'worker_argv':[sys.executable,str(Path(freeze).parent/(PREFIX+'_worker.py')),'--model',str(mid),'--freeze',str(freeze),'--review',str(review),'--context-map',str(cp),'--exposure',str(folder/'exposure.jsonl')]};save(folder/'plan.json',plan);monitored([str(Path(freeze).parent/'dev-harness'),'dev',str(folder/'plan.json')],folder/'process',32000000000);st2=Path(m['gguf']).stat();assert (st.st_size,st.st_mtime_ns,st.st_dev,st.st_ino)==(st2.st_size,st2.st_mtime_ns,st2.st_dev,st2.st_ino);d=s.read(folder/'result.json');assert d['status']=='complete' and [v['book_id'] for v in d['jobs']]==books
        events=[json.loads(v) for v in (folder/'exposure.jsonl').read_text().splitlines()];lookup={}
        for e in events:
            assert e['model']==mid and e['arm'] in [3,4,5,6,7,8] and type(e['returned_rows']) is type(e['canonical_fallback_rows']) is int and 0<=e['canonical_fallback_rows']<=e['returned_rows'];key=e['context'],e['layer'],e['expert'],e['zone'];assert key not in lookup;lookup[key]=e
        oldm=[];nlm=[];used=set()
        for job,prev in zip(d['jobs'],previous['jobs']):
            assert job['status']=='complete' and job['immutable_joint_ids']==prev['immutable_joint_ids'];r=job['NL'];assert r['model']==mid and r['reported_arm']==r['internal_hook_mode']==7 and r['fresh_KV'] and r['four_batches_128'] and r['unused511_fed'];score=r['nll_targets'];assert len(score)==256 and [v['target_position'] for v in score]==list(range(256,512)) and [v['input_position'] for v in score]==list(range(255,511)) and [v['target_id'] for v in score]==prev['immutable_joint_ids'][256:];vals=s.np.asarray([v['nll'] for v in score]);assert s.np.isfinite(vals).all();nlm.append(float(vals.mean()));oldm.append([float(s.np.mean([v['nll'] for v in rr['nll_targets']])) for rr in prev['runs']])
            for l,ee in r['pilot_exposure'].items():
                for e,zz in ee.items():
                    for z,count in zz.items():
                        key=r['context'],int(l),int(e),z;event=lookup.get(key);assert type(count) is int and count>=0 and (event['returned_rows'] if event else 0)==count
                        if event:assert event['book_id']==job['book_id'] and event['arm']==7;used.add(key)
        assert all(k in used or lookup[k]['arm']!=7 for k in lookup),'unmatched NL exposure';oldvalues.append(oldm);nlvalues.append(nlm);outputs.append(str(folder/'result.json'));print(f'DEV_QUALITY completed {name}',flush=True)
    summary=s.development_summary(oldvalues,nlvalues,books);save(dest/'summary.json',summary);s.authorize(freeze,review,'DEV_QUALITY');assert s.identities(deps)==before;save(dest/'dependency_identity.prepost.json',{'before':before,'after':before,'same_review_reauthorized':True});return s.finish(OUT,'DEV_QUALITY',freeze,[p for p in dest.rglob('*') if p.is_file()],{'results':outputs,'primary_family_inconclusive':summary['primary_family_inconclusive'],'no_outcome_classification':True})
QOUT=ROUND/'stage2-pg19-quality'
def actual(phase,freeze,review):
    sys.path.insert(0,str(Path(freeze).parent));import research_20261103_local_affine_stage3_core as s
    assert Path(s.__file__).resolve()==Path(freeze).parent/(PREFIX+'_core.py');s.authorize(freeze,review,phase);assert not (OUT/(phase+'.complete.json')).exists()
    if phase in ['BENCH_BCD','BENCH_NL']:return bench(s,phase,freeze,review)
    if phase=='DEV_QUALITY':return dev(s,freeze,review)
    fma=s.core.FMA(s.w.FROZEN1/'fma.so');num=s.w.GuardedNumeric(fma,s.F2/'strict-activation')
    try:return s.bind(freeze,review,num,fma) if phase=='BIND' else s.gu_fit(freeze,review,num,fma)
    finally:num.close()
def report():
    # Pure artifact ledger; no new model queries, simulation or classification.
    p=OUT/'cost_window.structural_report.json';assert not p.exists();known={k:read(OUT/(k+'.complete.json')) for k in ['BIND','BENCH_BCD','GU_FIT','BENCH_NL','DEV_QUALITY'] if (OUT/(k+'.complete.json')).exists()};save(p,{'status':'structural_conditional_ledger_only','completed_phases':{k:sha(OUT/(k+'.complete.json')) for k in known},'RAM_budgets_bytes':[8000000000,8589934592],'native_parameters':['R_PE usefulproducts/sec','R_E usefulECCbytes/sec','setup/inputcontext','FIFO/drain/backpressure','actualcachehit/CPUoverlap'],'physics':{'chips':16,'planes_per_chip':4,'page_bytes':16384,'ordinary_LSB_seconds':28e-6,'ordinary_CSB_MSB_seconds':46e-6,'CR_allowed':False},'isolated_matchedP_break_even':'D_map < V_control - V_local_host only if services/overlap legal; not wholeDAG proof','mandatory_alternatives':['allhostoriginalOLM pendinginclusivecapacity','samebankordinaryload+CPUGEMV+samecache','originalflashGU+hostDown','NLhostDown/cache orprivateflashDown','fixedD anduntestedglobal/sharedoutputcontrols'],'whole_model_capacity':'UNRESOLVED: scales/bias/P/tree/nonFFN/shared/router/KV/cache/workspace/allocator required; microRSS proves none','whole_DAG_services':'UNRESOLVED: attention/KV/head/shared andresourcecontention/overlap; do not sum standalone walltimes as lowerbound','nativePE_FIFO_service':'UNMEASURED; no stockAiF bitexact/speed claim','EPYC_AVX2_not_best_target_i9':'target VEX AVXVNNI/OS support andsustained bandwidth unmeasured; paper86.4 andARK89.6GB/s neitherapplicationthroughput','retained_original':'G/U/D fallback retained charged; NLDown aliases existingDown once; loadedtyped/NPZ duplicates counted separately','strict_quality_Pareto_claim':False,'automatic_scientific_classification':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','report','_execute',*PHASE]);p.add_argument('--source-review');p.add_argument('--phase',choices=list(PHASE.values()));p.add_argument('--freeze');a=p.parse_args();hardware=bind_cpu()
    if a.command=='prepare':prepare()
    elif a.command=='report':report()
    elif a.command=='_execute':assert a.source_review and a.freeze and a.phase;actual(a.phase,Path(a.freeze),Path(a.source_review))
    else:
        assert a.source_review;ptr=read(OUT/'CURRENT_FREEZE.json');fp=Path(ptr['freeze_receipt']);assert sha(fp)==ptr['sha256'];sys.path.insert(0,str(fp.parent));import research_20261103_local_affine_stage3_core as s
        phase=PHASE[a.command];s.authorize(fp,a.source_review,phase);runs=OUT/'launches';runs.mkdir(exist_ok=True);attempt=1+len(list(runs.glob(phase+'.attempt*')));folder=runs/(phase+f'.attempt{attempt:03d}');cap=32000000000 if phase=='DEV_QUALITY' else 8000000000;budget={'BENCH_BCD':600,'BENCH_NL':600,'GU_FIT':5400,'DEV_QUALITY':3600}.get(phase);monitored([sys.executable,str(fp.parent/(PREFIX+'_run.py')),'_execute','--phase',phase,'--source-review',a.source_review,'--freeze',str(fp)],folder,cap,budget)
