#!/usr/bin/env python3
"""Exactly one reviewed Stage4 phase; no implicit data release or successor."""
import argparse
import ctypes
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
import urllib.parse
import urllib.request
ROOT=Path('/home/ericyu/expert_flash_experiment')
ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine'
OUT=ROUND/'stage4-earliest64'
PREFIX='research_20261103_local_affine_stage4'
PROTOCOL=ROUND/'stage4_earliest64.protocol.frozen.json'
EXPECTED='32213b7b76cba5c18254f8bb13c7cfb8091c1db6c800468a541c4020e66c6b33'
F2=ROUND/'stage2-quality/freezes/rev004'
F3=ROUND/'stage3-control-cost/freezes/rev006'
PHASE={'bind':'BIND_EXISTING64','preflight':'AUDIT_PREFLIGHT','fit':'FIT_C64','acquire':'ACQUIRE_FRESH16_TOKENIZE','quality':'QUALITY_C64'}
SOURCES=[ROOT/'scripts'/(PREFIX+s) for s in ['_run.py','_core.py','_fit.py','_weights.cpp','_worker.py','_harness.cpp','_synthetic.py']]
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
def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();v=text[text.rfind(')')+2:].split();return {'PID':int(pid),'starttime_ticks':int(v[19]),'PPID':int(v[1]),'process_group':int(v[2]),'session':int(v[3]),'state':v[0]}
    except (FileNotFoundError,ProcessLookupError):return None
def direct_children(pid):
    children=set()
    try:
        for t in Path(f'/proc/{pid}/task').iterdir():
            try:children.update(map(int,(t/'children').read_text().split()))
            except (FileNotFoundError,ProcessLookupError):pass
    except (FileNotFoundError,ProcessLookupError):pass
    return children
def open_pidfd(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid,0)
    # This pinned EPYC host's Anaconda build omits os.pidfd_open and its libc
    # predates the wrapper. Linux x86_64 syscall434 supplies the same pidfd.
    assert os.uname().sysname=='Linux' and os.uname().machine=='x86_64','pidfd syscall fallback is Linux x86_64 only'
    # Primary local UAPI: /usr/include/x86_64-linux-gnu/asm/unistd_64.h,
    # __NR_pidfd_open = 434. Unsupported kernel -> errno -> fail closed.
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long;fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if fd<0:
        code=ctypes.get_errno();raise OSError(code,os.strerror(code))
    return int(fd)
class OwnedProcesses:
    """Linux subreaper + pidfds: new sessions cannot escape owned teardown.

    Capture existing children before spawning; never signal/reap those. New
    direct children adopted after double-fork are ours because this supervisor
    starts one monitored tree at a time. pidfds prevent PID-reuse signaling.
    """
    def __init__(self):
        assert hasattr(signal,'pidfd_send_signal'),'Linux pidfd signal support required, fail before spawn';probe=open_pidfd(os.getpid());os.close(probe)
        libc=ctypes.CDLL(None,use_errno=True);assert libc.prctl(36,1,0,0,0)==0,'PR_SET_CHILD_SUBREAPER required';enabled=ctypes.c_int();assert libc.prctl(37,ctypes.byref(enabled),0,0,0)==0 and enabled.value==1
        self.supervisor=os.getpid();self.baseline={pid:process_identity(pid) for pid in direct_children(self.supervisor)};self.owned={};self.signals=[];self.reaped=[];self.root=None
    def add(self,pid,reason):
        if pid in self.owned or pid==self.supervisor:return
        before=process_identity(pid)
        if before is None:return
        base=self.baseline.get(pid)
        if base is not None and base['starttime_ticks']==before['starttime_ticks']:return
        try:fd=open_pidfd(pid)
        except ProcessLookupError:return
        after=process_identity(pid)
        if after is None or after['starttime_ticks']!=before['starttime_ticks']:os.close(fd);return
        self.owned[pid]={'identity':after,'pidfd':fd,'ownership_reason':reason}
    def discover(self):
        # Discover/re-discover fixed points before any kill. Descendants may
        # create their own process groups or sessions without leaving this tree.
        for _ in range(8):
            count=len(self.owned)
            for pid,item in list(self.owned.items()):
                current=process_identity(pid)
                if current and current['starttime_ticks']==item['identity']['starttime_ticks']:
                    for child in direct_children(pid):self.add(child,'descendant_of_owned_identity')
            for child in direct_children(self.supervisor):self.add(child,'new_direct_child_or_subreaper_adoption')
            if len(self.owned)==count:break
    def current(self,pid):
        v=process_identity(pid);return v if v and v['starttime_ticks']==self.owned[pid]['identity']['starttime_ticks'] else None
    def reap(self):
        for pid in list(self.owned):
            if pid==self.root:continue # Popen retains its actual OS returncode.
            v=self.current(pid)
            if v and v['state']=='Z' and v['PPID']==self.supervisor:
                try:
                    reaped,status=os.waitpid(pid,os.WNOHANG)
                    if reaped:self.reaped.append({'PID':pid,'starttime_ticks':v['starttime_ticks'],'wait_status':status})
                except ChildProcessError:pass
    def alive(self):return [pid for pid in self.owned if (v:=self.current(pid)) and v['state']!='Z']
    def send(self,pid,sig):
        if not self.current(pid):return
        try:signal.pidfd_send_signal(self.owned[pid]['pidfd'],sig,None,0);self.signals.append({'at_utc':now(),'PID':pid,'starttime_ticks':self.owned[pid]['identity']['starttime_ticks'],'signal':int(sig),'pidfd_identity_safe':True})
        except ProcessLookupError:pass
    def teardown(self,p,forced):
        t=time.monotonic();self.discover();self.reap();unexpected=False
        if not forced:
            # Successful numerical roots must quiesce their children; briefly
            # allow normal EOF/shutdown races, then reject surviving workload.
            until=time.monotonic()+1
            while self.alive() and time.monotonic()<until:self.discover();p.poll();self.reap();time.sleep(.02)
            unexpected=bool(self.alive());forced=unexpected
        if forced:
            until=time.monotonic()+.6;termed=set()
            while True:
                self.discover();p.poll();self.reap()
                for pid in self.alive():
                    if pid not in termed:self.send(pid,signal.SIGTERM);termed.add(pid)
                if not self.alive() or time.monotonic()>=until:break
                time.sleep(.02)
            until=time.monotonic()+5
            while True:
                self.discover();p.poll();self.reap()
                for pid in self.alive():self.send(pid,signal.SIGKILL)
                if not self.alive() or time.monotonic()>=until:break
                time.sleep(.02)
        p.wait(timeout=5)
        # After root exit, escaped-session orphans are synchronously adopted.
        # Require two stable empty scans, reap every adopted zombie as well.
        stable=0;until=time.monotonic()+5
        while stable<2 and time.monotonic()<until:
            self.discover();self.reap();remaining=[pid for pid in self.owned if self.current(pid)]
            if remaining:
                stable=0
                for pid in self.alive():self.send(pid,signal.SIGKILL)
            else:stable+=1
            time.sleep(.02)
        remaining=[pid for pid in self.owned if self.current(pid)];receipt={'supervisor_PID':self.supervisor,'subreaper_enabled':True,'root_PID':self.root,'forced_cleanup':bool(forced),'unexpected_descendants_after_normal_root_exit':unexpected,'owned_processes':{str(pid):{k:v for k,v in item.items() if k!='pidfd'} for pid,item in self.owned.items()},'preexisting_children_excluded':self.baseline,'pidfd_signals':self.signals,'adopted_children_reaped':self.reaped,'remaining_owned_identities':remaining,'all_owned_processes_gone':not remaining,'wall_seconds':time.monotonic()-t}
        return receipt
    def close(self):
        for v in self.owned.values():os.close(v['pidfd'])
def monitored(cmd,folder,cap,checkpoint=None,_generated_exception_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);assert not (folder/'launch.json').exists();assert _generated_exception_after is None or 'generated' in folder.parts
    t=time.monotonic();utc=now();violations=[];peak=0;sample_count=0;failure=None;life=None;p=None;owned=OwnedProcesses()
    try:
        with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
            p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True);owned.root=p.pid;owned.add(p.pid,'explicit_Popen_root');assert p.pid in owned.owned,'root pidfd identity capture';save(folder/'launch.json',{'at_utc':utc,'PID':p.pid,'parent_PID':os.getpid(),'root_identity':owned.owned[p.pid]['identity'],'argv':cmd,'CPU_affinity':sorted(os.sched_getaffinity(p.pid)),'RSS_cap_bytes':cap,'checkpoint_seconds':checkpoint,'owned_subreaper_pidfd_supervision':True});print(json.dumps({'launched_PID':p.pid,'phase_folder':str(folder),'at_utc':utc}),flush=True)
            try:
                with (folder/'resource_samples.jsonl').open('w') as f:
                    while p.poll() is None:
                        owned.discover();pp=process_samples(os.getpid());rss=sum(v['RSS_KiB'] for v in pp)*1024;peak=max(peak,rss);sample={'at_utc':now(),'aggregate_RSS_bytes':rss,'processes':pp};f.write(json.dumps(sample)+'\n');f.flush();sample_count+=1
                        bad=[(v['PID'],tid,a) for v in pp for tid,a in v['thread_affinities'].items() if not a or not set(a)<=set(range(8))];expired=checkpoint is not None and time.monotonic()-t>checkpoint
                        if rss>cap or bad or expired:violations.append({'at_utc':now(),'RSS_bytes':rss,'affinity_violations':bad,'checkpoint_exhausted':expired});break
                        if _generated_exception_after is not None and time.monotonic()-t>=_generated_exception_after:raise RuntimeError('GENERATED supervisor exception fixture')
                        time.sleep(.2)
            except BaseException as e:failure=e
            finally:life=owned.teardown(p,bool(violations) or failure is not None)
    except BaseException as e:
        if failure is None:failure=e
        if p is not None and life is None:
            try:life=owned.teardown(p,True)
            except BaseException as cleanup_error:life={'all_owned_processes_gone':False,'cleanup_exception':repr(cleanup_error)}
    finally:owned.close()
    rc=p.returncode if p is not None else None;receipt={'started_at_utc':utc,'completed_at_utc':now(),'PID':None if p is None else p.pid,'returncode':rc,'wall_seconds':time.monotonic()-t,'peak_observed_aggregate_RSS_bytes':peak,'sampling_interval_seconds':.2,'observed_peak_is_lower_bound':True,'violations':violations,'exception':None if failure is None else repr(failure),'samples':sample_count,'CPU_affinity':list(range(8)),'lifecycle':life};save(folder/'exit.json',receipt)
    if failure is not None:raise failure
    assert rc==0 and not violations and life['all_owned_processes_gone'] and not life['unexpected_descendants_after_normal_root_exit'],'preserved tooling/resource/checkpoint/lifecycle failure; same stage needs review';return receipt

def manifest():
    return {'protocol_sha256':EXPECTED,'version':'first64-v1','new_modes':{'B64':9,'COPY_B64':10,'C64':11,'COPY_C64':12},'mounted_layers':{'deepseek':[1],'olmoe':[0]},'experts_per_model':64,'old4_reused_byte_identical':True,'later_D_NL_not_loaded':True,'numeric':'Pinned Stage0rev003 numeric.so, Stage2rev004 strictactivation, Stage1rev003 FMA environment; exactFP64 integerDGEMM; stdround tiesaway; ownwholeI A8; F32scale andbias order unchanged','fit':'Frozen union_statistics/fixed_fit(make_factor=False), eta0lambda.01 new124, originalold4 untouched inclOLMe17lambda.001; withinleafdemean pooled prior, same1e-10 cutoff/normalizedtrace and actualdeployedbias','teacher':'One completeY pass eachnewexpert over fullorderedunion; rawY solepersistenttarget, no GU/predcache. Meter invocationcategories includeinternalI64 andfullrowprivateI64/archive/B1/IPC audits; no extra work calledfree','atomic':'Y.complete fsync+SHA+numeric/source/input/weights; operator.complete after archive/B1/IPC; noautomaticretry. Root continuation exactcompletedreceipt bindings required; completeY canfinishincompletefit withoutquery, completeops skipped','preflight':'Exact protocol serialization/domain/strata/digesttie, firstselectedold/prefix/Bdecode/Ddecode four; metadata frozen beforevalues','contexts':'52fresh/model:16sourceobserver/B64/C64 plusfirstfixedbook nohook/copyoriginal/copyB64/copyC64. nativejoint512IDs once/model/book, four128chunks, target256..511 predecessor255..510;511extraunused','invariance':'Literal all512 earliestownedX/orderedIDs/finalactualalpha bytes, realu32comparison viamemcpy (no aliasing UB), no laterlayer equality or statepatch','copy':'source0-nohook vsobserver1 logits only; sourceobserver1-copy2 logits+ownedtrace;9-10and11-12 logits+ownedtrace','summary':'two C64-B64 sharedpaired16book10000draws seed2026110304 ddof1 centeredmaxabsT linear95; bootstrapdefined andinterval-marginrelations distinct. Noautomatic scientificclassification, no missing/outlierdrop','supervisor':'Textreuse Stage3rev006 bind_cpu/hardware/processidentity/subreaper+pidfd/same-session-resistant cleanup; outercheckpoint coversentirephase,8CPUs0..7;8e9 actualfit vs32e9 referencequality','review_schema':{'status':'approved','phase':'one exact PHASE','protocol_sha256':EXPECTED,'source_revision':'int','source_freeze_sha256':'SHA','parent_receipts':'exactfreeze mapping','PRIORPHASE_complete_sha256':'all preceding phases required','registry_sha256':'ACQUIRE/QUALITY required','continuation':'FIT existing atomicunits only, explicitapproved completed_units mapping andadditional/cumulative budgets'},'actual_payload_in_prepare':False,'no_automatic_push':True}

def import_core(folder):
    sys.path.insert(0,str(folder));import research_20261103_local_affine_stage4_core as s
    assert Path(s.__file__).resolve()==Path(folder)/(PREFIX+'_core.py');return s

def prepare():
    started=time.monotonic();deadline=started+1800;s=import_core(ROOT/'scripts');assert sha(PROTOCOL)==EXPECTED and sha(s.APPROVAL)==s.APPROVAL_SHA;s.old_checks();OUT.mkdir(exist_ok=True);fr=OUT/'freezes';fr.mkdir(exist_ok=True);rev=1+max([int(p.name[3:]) for p in fr.glob('rev[0-9][0-9][0-9]')],default=0);rd=fr/f'rev{rev:03d}';rd.mkdir()
    for p in SOURCES:shutil.copy2(p,rd/p.name)
    save(rd/'algorithm_manifest.frozen.json',manifest());save(rd/'hardware.json',hardware_snapshot());parents={v['path']:v['sha256'] for v in s.spec()['parents'].values()}
    for path,h in parents.items():assert sha(path)==h,path
    # Only immutable JSONL provenance metadata are read here, never arrays.
    fitmeta=read(s.REF/'FIT_REFRESH.complete.json')
    for name in ['deepseek','olmoe']:
        m=s.config(name);p=s.REF/'fitted'/name/f'layer{m["layer"]}'/'union_provenance.jsonl';assert sha(p)==fitmeta['files'][str(p)]
        save(rd/(name+'.selection.frozen.json'),s.preflight_selection(name,s.provenance(name)));parents[str(p)]=sha(p)
    for p in [s.REF/'FIT_REFRESH.complete.json',s.REF/'BIND_OLD_CAPTURE.complete.json',s.REF/'COLLECT_REFRESH.complete.json',s.REF/'refreshed_registry.frozen.json',ROUND/'stage0/content_identities.frozen.json',ROUND/'stage0_protocol.frozen.json']:
        parents[str(p)]=sha(p)
    base=read(F2/'build_commands.json')[0];i=next(k for k,v in enumerate(base) if v.endswith('_harness.cpp'));flags=base[1:i];libs=base[base.index('-o')+2:];numeric=s.core.FROZEN0/'numeric.so'
    commands=[['c++',*flags,'-I'+str(rd),str(rd/(PREFIX+'_weights.cpp')),'-o',str(rd/'weights-export'),*libs,str(numeric),'-Wl,-rpath,'+str(numeric.parent)],['c++',*flags,'-I'+str(rd),str(rd/(PREFIX+'_harness.cpp')),'-o',str(rd/'quality-harness'),*libs]];save(rd/'build_commands.json',commands);receipts=[]
    for i,cmd in enumerate(commands):receipts.append(monitored(cmd,rd/f'build{i}',8000000000,max(.01,deadline-time.monotonic())))
    tests=[[str(rd/'weights-export'),'synthetic'],[str(rd/'quality-harness'),'synthetic',sys.executable,str(rd/(PREFIX+'_worker.py'))],[sys.executable,str(rd/(PREFIX+'_synthetic.py')),'--folder',str(rd/'generated')]];save(rd/'test_commands.json',tests)
    for i,cmd in enumerate(tests):receipts.append(monitored(cmd,rd/f'test{i}',8000000000,max(.01,deadline-time.monotonic())))
    assert time.monotonic()<deadline;s.old_checks();deps={str(p):sha(p) for p in SOURCES};deps.update(s.old_checks());deps[str(PROTOCOL)]=EXPECTED;deps[str(s.APPROVAL)]=s.APPROVAL_SHA
    # Transitive pinned module/library dependencies only. Do not traverse a
    # historical freeze's generated/payload array entries during preparation.
    frozen_dirs=[F3,s.upstream.FREF,F2,s.w.FROZEN1,s.core.FROZEN0,s.upstream.FQ]
    for folder in frozen_dirs:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.py','.cpp','.hpp','.h','.so']:deps[str(p)]=sha(p)
        p=folder/'freeze_receipt.json';deps[str(p)]=sha(p)
    for p in [F2/'strict-activation',*map(Path,[a for a in libs if a.startswith('/')])]:deps[str(p)]=sha(p)
    for p in rd.rglob('*'):
        if p.is_file():deps[str(p)]=sha(p)
    f={'status':'READY_SOURCE_BUILD_GENERATED_ONLY_PENDING_REAL_PHASE_REVIEW','at_utc':now(),'revision':rev,'protocol_sha256':EXPECTED,'parent_receipts':parents,'files':deps,'manifest':manifest(),'build_and_test_receipts':receipts,'prepare_wall_seconds':time.monotonic()-started,'actual_model_weight_activation_target_fittedarray_book_eval_payload_read':False,'metadata_only_actual_provenance_selection':True,'old37_sources_unchanged':True,'TEST_opened':False};save(rd/'freeze_receipt.json',f);ptr={'revision':rev,'freeze_receipt':str(rd/'freeze_receipt.json'),'sha256':sha(rd/'freeze_receipt.json')};save(OUT/'CURRENT_FREEZE.json',ptr);print(json.dumps({'READY':ptr,'files':len(deps)}),flush=True)

def bind_existing(s,freeze,review):
    parent=s.read(s.REF/'FIT_REFRESH.complete.json');before=s.identities(parent['files']);seed=s.read(s.REF/'refreshed_registry.frozen.json');dest=OUT/'bound';dest.mkdir();models={};model_bridges=[]
    for name in ['deepseek','olmoe']:
        m=s.config(name);folder=dest/name;folder.mkdir();selection=read(Path(freeze).parent/(name+'.selection.frozen.json'));assert selection==s.preflight_selection(name,s.provenance(name));s.atomic_json(folder/'selection.before_values.json',selection)
        x,ids,alpha,prov,identity=s.load_original_union(name)
        for field,a in [('inputs',x),('ids',ids),('alpha',alpha)]:s.np.save(folder/(field+'.npy'),a)
        pp=folder/'union_provenance.jsonl';pp.write_text(''.join(json.dumps(v,separators=(',',':'),allow_nan=False)+'\n' for v in prov));s.atomic_json(folder/'union.json',{'model':name,'layer':m['layer'],'union_identity':identity,'selection':selection,'source_union_provenance_sha256':s.sha(s.REF/'fitted'/name/f'layer{m["layer"]}'/'union_provenance.jsonl'),'input_and_rows_current_binding_only_not_historicalXclaim':True})
        mm,ident=s.model_source(name);model_before=s.stat(mm['gguf']);wf=folder/'weights';wf.mkdir()
        for e in range(64):
            s.disk_check();target=wf/f'expert{e}';monitored([str(Path(freeze).parent/'weights-export'),'weights',mm['gguf'],name,str(m['layer']),str(e),str(target)],folder/'export-processes'/f'expert{e}',8000000000)
        assert s.stat(mm['gguf'])==model_before
        own=[o for o in seed['operators'] if o['model']==name and o['layer']==m['layer']];assert {o['expert'] for o in own}==set(m['reused_experts']);oldops=[];geometries=set()
        for o in own:
            e=o['expert'];weights=Path(o['canonical_weights_path'])
            for n in ['weights.json','gate.i8','gate.scales.f32','up.i8','up.scales.f32','down.i8','down.scales.f32']:assert s.sha(wf/f'expert{e}'/n)==s.sha(weights/n),(name,e,n)
            lm=s.old.load_model(o['local_model_path']);s.validate_local(lm);assert s.local_hash(lm)==o['local_model_hash'];geometries.add(o['preprocessing_path'])
            # Shared old+new labels retain their historical whole-union binding;
            # no BIND teacher query or old-C refit is performed.
            targetid=s.read(s.REF/'fitted'/name/f'layer{m["layer"]}'/f'expert{e}'/'shared_target_identity.json')
            assert s.sha(targetid['old_path'])==targetid['old_sha256'] and s.sha(targetid['new_path'])==targetid['new_sha256']
            oldops.append({'expert':e,'local_model_path':o['local_model_path'],'local_model_hash':o['local_model_hash'],'weights_path':str(wf/f'expert{e}'),'reused_old_C':True,'eta':0,'lambda':.001 if name=='olmoe' and e==17 else .01,'old_union_target_binding':targetid})
        assert len(geometries)==1
        models[name]={'geometry':geometries.pop(),'old_operators':oldops};model_bridges.append({'model':name,'original_identity_receipt_sha256':s.sha(ROUND/'stage0/content_identities.frozen.json'),'historical_full_model_identity':ident,'current_stat_before':model_before,'current_stat_after':s.stat(mm['gguf']),'all_size_mtime_dev_inode_equal':True,'full_GGUF_content_hash_recomputed_this_phase':False,'scope':'Reuse historically hashed samefile identity with complete stat bridge; actualexport reads only required early-layer packed rows.'});del x,ids,alpha,lm
    s.atomic_json(dest/'models.json',models);s.atomic_json(dest/'model_identity.reuse_bridge.json',model_bridges);assert s.identities(parent['files'])==before;s.old_checks()
    return s.finish('BIND_EXISTING64',freeze,[p for p in dest.rglob('*') if p.is_file()],{'canonical_teacher_queries':0,'old4_C_refits':0,'all_parent_FIT_dependencies_prepost_equal':True,'model_identity_reuse_bridges':model_bridges,'source_late_maps_not_mounted':True})

def acquire(s,freeze,review):
    fit=s.read(OUT/'FIT_C64.complete.json');before=s.identities({**fit['files'],**fit['runtime_payload_files']});dest=OUT/'acquisition';dest.mkdir();books=s.spec()['evaluation']['selected16'];acquired=[]
    for b in books:
        s.disk_check(b['size_bytes']*3);url='https://storage.googleapis.com/download/storage/v1/b/deepmind-gutenberg/o/'+urllib.parse.quote(b['name'],safe='')+'?alt=media&generation='+b['generation'];utc=now();request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=60) as response:raw=response.read(b['size_bytes']+1);headers=dict(response.headers);assert response.headers.get('x-goog-generation')==b['generation']
        rp=dest/(str(b['book_id'])+'.raw');assert not rp.exists();rp.write_bytes(raw);suffix,meta=s.r.preprocess(raw,b);sp=dest/(str(b['book_id'])+'.suffix.utf8');sp.write_bytes(suffix);acquired.append({'book_id':b['book_id'],'metadata':b,'URL':url,'at_utc':utc,'response_headers':headers,'raw_path':str(rp),'suffix_path':str(sp),**meta});save(dest/'partial.json',{'status':'partial','books':acquired})
    s.atomic_json(dest/'objects.complete.json',{'status':'complete','books':acquired})
    for mid,name in enumerate(['deepseek','olmoe'],1):
        m,identity=s.model_source(name);pre=s.stat(m['gguf']);plan={'phase':'ACQUIRE_FRESH16_TOKENIZE','model_id':mid,'gguf':m['gguf'],'jobs':[{'book_id':b['book_id'],'suffix_path':b['suffix_path']} for b in acquired],'output':str(dest/(name+'.joint_ids.json'))};s.atomic_json(dest/(name+'.tokenize.plan.json'),plan);monitored([str(Path(freeze).parent/'quality-harness'),'tokenize',str(dest/(name+'.tokenize.plan.json'))],dest/(name+'.tokenize.process'),8000000000);assert s.stat(m['gguf'])==pre
    s.authorize(freeze,review,'ACQUIRE_FRESH16_TOKENIZE');assert s.identities({**fit['files'],**fit['runtime_payload_files']})==before
    return s.finish('ACQUIRE_FRESH16_TOKENIZE',freeze,[p for p in dest.rglob('*') if p.is_file()],{'FIT_payload_prepost_equal':True,'no_book_substitution':True,'native_tokenize_once_per_model_book':True,'vocab_only_no_model_forward':True})

def quality(s,freeze,review):
    fit=s.read(OUT/'FIT_C64.complete.json');acq=s.read(OUT/'ACQUIRE_FRESH16_TOKENIZE.complete.json');deps={**fit['files'],**fit['runtime_payload_files'],**acq['files']};before=s.identities(deps);dest=OUT/'quality';dest.mkdir();books=[v['book_id'] for v in s.spec()['evaluation']['selected16']];jobs,contexts=s.context_plan(books);outputs=[]
    for mid,name in enumerate(['deepseek','olmoe'],1):
        folder=dest/name;folder.mkdir();tokens=s.read(OUT/'acquisition'/(name+'.joint_ids.json'));assert tokens['status']=='complete' and [j['book_id'] for j in tokens['jobs']]==books;m,identity=s.model_source(name);pre=s.stat(m['gguf']);run=time.time_ns()&((1<<63)-1);contextfile=folder/'contexts.json';s.atomic_json(contextfile,{'protocol_sha256':EXPECTED,'model':mid,'run':run,'contexts':contexts});plan={'phase':'QUALITY_C64','model_id':mid,'gguf':m['gguf'],'run_id':run,'jobs':[{**j,'joint_ids':tokens['jobs'][i]['joint_ids']} for i,j in enumerate(jobs)],'output':str(folder/'result.json'),'worker_argv':[sys.executable,str(Path(freeze).parent/(PREFIX+'_worker.py')),'--model',str(mid),'--freeze',str(freeze),'--review',str(review),'--context-map',str(contextfile),'--exposure',str(folder/'exposure.jsonl')]};s.atomic_json(folder/'plan.json',plan);monitored([str(Path(freeze).parent/'quality-harness'),'quality',str(folder/'plan.json')],folder/'process',32000000000);assert s.stat(m['gguf'])==pre;outputs.append(folder);print(f'QUALITY_C64 completed {name}',flush=True)
    summary=summarize_quality(s,outputs,books);s.atomic_json(dest/'summary.json',summary);s.authorize(freeze,review,'QUALITY_C64');after=s.identities(deps);assert before==after;s.atomic_json(dest/'dependency_identity.prepost.json',{'before':before,'after':after,'same_review_reauthorized':True});s.old_checks()
    return s.finish('QUALITY_C64',freeze,[p for p in dest.rglob('*') if p.is_file()],{'primary_family_inconclusive':summary['primary_family_inconclusive'],'no_automatic_classification':True,'all_dependencies_prepost_equal':True})

def summarize_quality(s,folders,books):
    assert len(folders)==2;values=s.np.full((2,16,3),s.np.nan);exposures=[];counts={}
    for mi,folder in enumerate(folders):
        name=['deepseek','olmoe'][mi];m=s.config(name);d=s.read(folder/'result.json');contexts=s.read(folder/'contexts.json');assert d['status']=='complete' and [v['book_id'] for v in d['jobs']]==books
        events=[json.loads(v) for v in (folder/'exposure.jsonl').read_text().splitlines()];lookup={}
        for e in events:
            key=(e['context'],e['layer'],e['expert'],e['zone']);assert key not in lookup and e['model']==mi+1 and e['run']==contexts['run'] and e['arm'] in [9,10,11,12];lookup[key]=e;c=contexts['contexts'][str(e['context'])];assert c['mode']==e['arm'] and c['book_id']==e['book_id'];assert type(e['returned_rows']) is type(e['canonical_fallback_rows']) is int and 0<=e['canonical_fallback_rows']<=e['returned_rows']
        used=set();executed=[]
        for bi,j in enumerate(d['jobs']):
            if j['status']=='insufficient_window':continue
            assert j['status']=='complete' and len(j['immutable_joint_ids'])==512 and len(j['runs'])==3
            if bi==0:assert all(v is True for v in j['firstbook_controls'].values())
            assert j['source_vs_B64_invariant']['passed'] and j['source_vs_C64_invariant']['passed'] and j['source_vs_B64_invariant']['positions_compared']==j['source_vs_C64_invariant']['positions_compared']==512
            for ai,(mode,r) in enumerate(zip([1,9,11],j['runs'])):
                assert r['model']==mi+1 and r['internal_hook_mode']==mode and r['reported_arm']==(0 if mode==1 else mode) and r['run']==contexts['run'] and r['subrun']==9 and r['fresh_KV'] and r['four_batches_128'] and r['unused511_fed'];ctx=r['context'];assert ctx not in executed;executed.append(ctx);c=contexts['contexts'][str(ctx)];assert c['mode']==mode and c['book_id']==j['book_id']
                scores=r['nll_targets'];assert len(scores)==256 and [v['input_position'] for v in scores]==list(range(255,511)) and [v['target_position'] for v in scores]==list(range(256,512)) and [v['target_id'] for v in scores]==j['immutable_joint_ids'][256:];a=s.np.asarray([v['nll'] for v in scores]);assert s.np.isfinite(a).all();values[mi,bi,ai]=a.mean()
                totals={z:0 for z in s.ZONES}
                assert set(r['expert_exposure'])=={str(m['layer'])} and set(r['expert_exposure'][str(m['layer'])])=={str(e) for e in range(64)}
                for e in range(64):
                    for z in s.ZONES:
                        count=r['expert_exposure'][str(m['layer'])][str(e)][z];assert type(count) is int and count>=0;totals[z]+=count;key=(ctx,m['layer'],e,z);event=lookup.get(key)
                        if mode==1:assert event is None;fallback=0
                        else:
                            assert (event['returned_rows'] if event else 0)==count;fallback=event['canonical_fallback_rows'] if event else 0
                            if event:used.add(key)
                        exposures.append({'model':name,'book_id':j['book_id'],'arm':0 if mode==1 else mode,'layer':m['layer'],'expert':e,'zone':z,'source_potential_live_hits':count if mode==1 else None,'actual_scattered_rows':0 if mode==1 else count,'canonical_fallback_rows':fallback,'approximation_rows':count-fallback if mode==11 else 0,'reused_old_expert':e in m['reused_experts']})
                assert totals==dict(zip(s.ZONES,[255*m['K'],256*m['K'],m['K']]))
        assert all(k in used or e['arm'] in [10,12] for k,e in lookup.items())
        expected_actual=3*sum(j['status']=='complete' for j in d['jobs'])+(4 if d['jobs'][0]['status']=='complete' else 0);assert d['actual_contexts']==expected_actual
        if not d['insufficient_books']:assert expected_actual==52
        counts[name]={'actual_contexts':d['actual_contexts'],'complete_books':sum(j['status']=='complete' for j in d['jobs']),'all64_selectedslot_join':True,'all_512pos_earliest_invariance':True,'insufficient_books':d['insufficient_books']}
    result=s.paired_summary(values,books);result.update(model_integrity=counts,exposure=exposures,all64_early_only=True,TEST_opened=False,no_recipe_selection=True);return result

def execute(phase,freeze,review):
    s=import_core(Path(freeze).parent);s.authorize(freeze,review,phase);assert not (OUT/(phase+'.complete.json')).exists();s.disk_check()
    if phase=='BIND_EXISTING64':return bind_existing(s,freeze,review)
    if phase=='ACQUIRE_FRESH16_TOKENIZE':return acquire(s,freeze,review)
    if phase=='QUALITY_C64':return quality(s,freeze,review)
    import research_20261103_local_affine_stage4_fit as fit
    fma=s.core.FMA(s.w.FROZEN1/'fma.so');base=s.w.GuardedNumeric(fma,F2/'strict-activation');num=s.Meter(base)
    try:return fit.preflight(freeze,num) if phase=='AUDIT_PREFLIGHT' else fit.fit(freeze,review,num,fma)
    finally:base.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','_execute',*PHASE]);p.add_argument('--source-review');p.add_argument('--phase',choices=list(PHASE.values()));p.add_argument('--freeze');a=p.parse_args();bind_cpu()
    if a.command=='prepare':prepare()
    elif a.command=='_execute':assert a.phase and a.freeze and a.source_review;execute(a.phase,Path(a.freeze),Path(a.source_review))
    else:
        assert a.source_review;ptr=read(OUT/'CURRENT_FREEZE.json');fp=Path(ptr['freeze_receipt']);assert sha(fp)==ptr['sha256'];s=import_core(fp.parent);phase=PHASE[a.command];f,review=s.authorize(fp,a.source_review,phase,verify_prior_payload=False);assert not (OUT/(phase+'.complete.json')).exists();s.disk_check();attempt=1+len(list((OUT/'launches').glob(phase+'.attempt*')));budget=s.spec()['resources']['phase_checkpoints_seconds'][phase]
        if phase=='FIT_C64' and review.get('continuation'):budget=review['continuation']['additional_budget_seconds']
        monitored([sys.executable,str(fp.parent/(PREFIX+'_run.py')),'_execute','--phase',phase,'--freeze',str(fp),'--source-review',str(a.source_review)],OUT/'launches'/(phase+f'.attempt{attempt:03d}'),32000000000 if phase=='QUALITY_C64' else 8000000000,budget)
