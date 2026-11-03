#!/usr/bin/env python3
"""Generated arrays only; no original model, capture, target or book payload."""
import argparse
import ctypes as ct
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage4_core as s
import research_20261103_local_affine_stage4_worker as worker
import research_20261103_local_affine_stage4_fit as fit
import research_20261103_local_affine_stage4_run as run
np,old,core,w=s.np,s.old,s.core,s.w
def lifecycle_fixtures(folder,activation_pid):
    folder=Path(folder)/'lifecycle';folder.mkdir();dummy=folder/'dummy.py'
    dummy.write_text('''import json,os,pathlib,signal,subprocess,sys,time
role,mode,log=sys.argv[1:];signal.signal(signal.SIGTERM,signal.SIG_IGN)
if role=='root':subprocess.Popen([sys.executable,__file__,'middle',mode,log],start_new_session=True)
elif role=='middle':subprocess.Popen([sys.executable,__file__,'leaf',mode,log],start_new_session=True)
blob=bytearray(256<<20) if role=='leaf' and mode=='rss' else bytearray(1<<20)
for i in range(0,len(blob),4096):blob[i]=1
text=pathlib.Path('/proc/self/stat').read_text();v=text[text.rfind(')')+2:].split();entry={'PID':os.getpid(),'starttime_ticks':int(v[19]),'role':role,'session':os.getsid(0)}
fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600);os.write(fd,(json.dumps(entry)+'\\n').encode());os.close(fd)
if role=='leaf':pathlib.Path(log+'.ready').write_text('ready')
if role=='root' and mode=='orphan':
 while not pathlib.Path(log+'.ready').exists():time.sleep(.01)
 sys.exit(0)
while True:time.sleep(.05)
''')
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True);receipts=[]
    try:
        protected={sentinel.pid,activation_pid}
        for mode in ['timeout','rss','exception','orphan']:
            case=folder/mode;trace=folder/(mode+'.identities.jsonl');base=sum(v['RSS_KiB'] for v in run.process_samples(os.getpid()))*1024;cap=base+(160<<20) if mode=='rss' else 8000000000
            try:
                run.monitored([sys.executable,str(dummy),'root',mode,str(trace)],case,cap,.45 if mode=='timeout' else None,_generated_exception_after=.45 if mode=='exception' else None);raise RuntimeError('lifecycle expected failure not raised')
            except (AssertionError,RuntimeError) as ex:assert str(ex)!='lifecycle expected failure not raised'
            r=json.loads((case/'exit.json').read_text());life=r['lifecycle'];assert life['all_owned_processes_gone'] and not life['remaining_owned_identities'];assert any(v['signal']==9 for v in life['pidfd_signals']),'TERM-resistant workload requires recorded escalation';known=[json.loads(v) for v in trace.read_text().splitlines()];assert {v['role'] for v in known}=={'root','middle','leaf'} and len(known)==3
            assert all(str(v['PID']) in life['owned_processes'] and life['owned_processes'][str(v['PID'])]['identity']['starttime_ticks']==v['starttime_ticks'] for v in known);assert not protected.intersection(map(int,life['owned_processes'])) and protected<=set(map(int,life['preexisting_children_excluded']))
            for v in known:
                current=run.process_identity(v['PID']);assert current is None or current['starttime_ticks']!=v['starttime_ticks']
            assert sentinel.poll() is None and run.process_identity(activation_pid) is not None
            if mode=='timeout':assert any(v['checkpoint_exhausted'] for v in r['violations'])
            if mode=='rss':assert r['peak_observed_aggregate_RSS_bytes']>cap
            if mode=='exception':assert r['exception']=='RuntimeError(\'GENERATED supervisor exception fixture\')'
            if mode=='orphan':assert r['returncode']==0 and life['unexpected_descendants_after_normal_root_exit']
            receipts.append({'mode':mode,'exit_receipt':str(case/'exit.json'),'root_returncode':r['returncode'],'all3_identity_bound_processes_gone':True,'preexisting_sentinel_and_activation_survived':True,'TERM_KILL_escalation_recorded':True})
        result={'status':'passed','generated_only':True,'subreaper_pidfd_owned_identity_teardown':True,'cases':receipts,'actual_model_BENCH_not_run':True};s.atomic_json(folder/'receipt.json',result);return result
    finally:
        if sentinel.poll() is None:sentinel.terminate()
        sentinel.wait(timeout=5)

def generated_provenance(name):
    m=s.config(name);out=[]
    for origin,phase,arm,count in [('oldFIT','prefill',None,140),('pg19train','prefix',3,70),('pg19train','decode',3,30),('pg19train','decode',5,30),('pg19train','decode',6,30)]:
        for _ in range(count):
            i=len(out);p={'row':i,'model':name,'layer':m['layer'],'origin':origin,'phase':phase,'group':'generated:'+str(i%16),'input_sha256':str(i)}
            if arm is not None:p['arm']=arm
            out.append(p)
    return out

def weights(rng,I):
    def make(rows,cols):return rng.integers(-3,4,(rows,cols),dtype=np.int8),np.full(rows,.0003,np.float32)
    return {'gate':make(I,2048),'up':make(I,2048),'down':make(2048,I)}

def save_weights(folder,weights,I):
    folder.mkdir();info={'projections':{}}
    for role,(q,scale) in weights.items():
        q.tofile(folder/(role+'.i8'));scale.tofile(folder/(role+'.scales.f32'));info['projections'][role]={'R':q.shape[0],'C':q.shape[1],'source_type':'generated'}
    s.atomic_json(folder/'weights.json',info)

def registry_fixture(folder,tree,P,sigma,num,rng):
    folder.mkdir();geometry=folder/'geometry';xx=np.zeros((64,2048),np.float32);s.r.archive_geometry(geometry,tree,P,sigma,num,xx,np.arange(64),np.full(64,'generated'))
    bank=folder/'models';bank.mkdir();ops=[]
    # Zero maps are legitimate and let every expert have distinct biases while
    # using one stored map per archive. No deployment-RAM inference is made.
    zero=np.zeros((32,2048,128),np.int8);scale=np.ones((32,2048),np.float32)
    for e in range(64):
        model={'qmaps':zero,'scales':scale,'bias':np.full((32,2048),e/1000,np.float32),'map_hashes':['zero']*32,'unsupported':[31] if e==7 else []};path,key=old.archive_model(bank,model);ops.append({'expert':e,'local_model_path':path,'local_model_hash':key})
    models={};payload={}
    for name,I,layer in [('deepseek',1408,1),('olmoe',1024,0)]:
        wf=folder/name;wf.mkdir();paths=[]
        # Three different canonical functions expose wrong-expert dispatch;
        # paths repeat deliberately as generated aliases, never resource proof.
        for group in range(3):
            ww=weights(rng,I);ww['down']=(ww['down'][0],ww['down'][1]*np.float32(group+1));p=wf/f'weights{group}';save_weights(p,ww,I);paths.append(str(p))
            payload.update({str(f):s.sha(f) for f in p.iterdir()})
        modelops=[{**o,'weights_path':paths[o['expert']%3]} for o in ops];models[name]={'I':I,'layer':layer,'geometry':str(geometry),'operators':modelops}
    payload.update({str(p):s.sha(p) for p in bank.iterdir()});payload.update({str(p):s.sha(p) for p in geometry.iterdir()})
    pm=folder/'payload.json';s.atomic_json(pm,{'files':payload});rp=folder/'registry.json';s.atomic_json(rp,{'protocol_sha256':s.EXPECTED,'version':'first64-v1','models':models,'payload_manifest':{'path':str(pm),'sha256':s.sha(pm)},'generated_only':True});return rp

def binary_roundtrip(reg,x,ids,mode):
    rows=reg.evaluate(mode,reg.layer,x,ids);h=w.header(kind=1,mode=mode,arm=mode,model=reg.model,layer=reg.layer,T=len(x),K=reg.K,request=1,run=1,context=mode,subrun=8,decode=1,slots=len(rows),bytes=4*len(x)*(2048+reg.K));frame=w.pack(h)+x.tobytes()+ids.tobytes();stream=io.BytesIO();worker.serve(reg,io.BytesIO(frame),stream);raw=stream.getvalue();reply=w.unpack(raw[w.WIRE.size:2*w.WIRE.size]);assert reply['slots']==len(rows)==len(x)*reg.K and reply['mode']==mode;off=2*w.WIRE.size
    for t,k,e,flag,y in rows:
        assert w.struct.unpack('<4I',raw[off:off+16])==(t,k,e,flag);off+=16;assert raw[off:off+8192]==y.tobytes();off+=8192
    assert off==len(raw)
    return rows,frame

def main(folder):
    folder=Path(folder);folder.mkdir();rng=np.random.default_rng(2026110304);fma=core.FMA(w.FROZEN1/'fma.so');base=w.GuardedNumeric(fma,s.upstream.F2/'strict-activation');num=s.Meter(base)
    try:
        with num.category('generated_numeric_controls'):
            for width in [1,7,8,15,16,17,127,128,129,1024,1408,2048]:
                q=rng.integers(-128,128,(4,width),np.int8);ww=rng.integers(-128,128,(7,width),np.int8);q[0]=-128;ww[0]=-128
                assert width*128**2<2**53;assert np.array_equal(num.dots(q,ww),q.astype(np.int64)@ww.astype(np.int64).T)
            near=np.asarray([0.,-0.,.5,-.5,np.nextafter(np.float32(.5),np.float32(0)),np.nextafter(np.float32(-.5),np.float32(0)),127.,-127.],np.float32)[None,:];fit.own_quantization(num,near)
            for I in [1024,1408]:
                g=(rng.standard_normal((2,I))*.1).astype(np.float32);u=(rng.standard_normal((2,I))*.1).astype(np.float32);a=num.swiglu(g,u);fit.own_quantization(num,a)
        N=64;H=2048;x=(rng.standard_normal((N,H))*.1).astype(np.float32)
        for row in range(N):
            for bit in range(5):x[row,bit]=1 if (row//2>>(4-bit))&1 else -1
        tree={}
        for node in range(63):
            normal=np.zeros(H,np.float32) if node<31 else None
            if normal is not None:normal[(node+1).bit_length()-1]=1
            tree[node]={'mean':np.zeros(H,np.float32),'normal':normal,'threshold':0.,'depth':(node+1).bit_length()-1}
        P=(rng.standard_normal((128,H))*.01).astype(np.float32);sigma=np.ones(128);ww=weights(rng,1408)
        with num.category('generated_complete_fit_archive'):
            d=old.design(num,x,tree,32,P,sigma,128,'normal');assert np.bincount(d['leaf'],minlength=32).tolist()==[2]*32
            inputs=s.r.union_statistics(d,np.ones(N,np.bool_),np.arange(N),True);y=old.teacher(num,x,ww);model,unused,info=s.r.fixed_fit(num,fma,inputs,y,0.,.01,'PE_RRR',make_factor=False);assert unused is None;s.validate_local(model)
            bank=folder/'fit-models';bank.mkdir();path,key=old.archive_model(bank,model);loaded=old.load_model(path);assert s.local_hash(loaded)==key
            audit=fit.archive_audit(num,'deepseek',0,model,path,d,x,ww,tree,P,sigma,y,[0,16,32,63]);assert audit['batch4_archive4_B1four_all_H_bits']
            altered={**model,'unsupported':[0]};p2,k2=old.archive_model(bank,altered);assert k2!=key and old.load_model(p2)['unsupported']==[0]
        for name in ['deepseek','olmoe']:
            pp=generated_provenance(name);a=s.preflight_selection(name,pp);b=s.preflight_selection(name,[{k:p[k] for k in reversed(p)} for p in pp]);assert a==b and len(a['rows'])==256 and len(set(a['four_audit_rows']))==4
            bad=pp.copy();bad[0]={**bad[0],'row':999}
            try:s.preflight_selection(name,bad);raise RuntimeError('expected invalid provenance')
            except AssertionError:pass
        unit=folder/'atomic-unit';unit.mkdir();binding={'protocol_sha256':s.EXPECTED,'source_freeze_sha256':'generated','union_json_sha256':'generated','canonical_weights_files':{}}
        with num.category('generated_atomic_unit'):
            yp,yr,reused=fit.atomic_Y(unit,'deepseek',0,x[:4],ww,num,binding);assert not reused;before=num.receipt();yp2,yr2,reused=fit.atomic_Y(unit,'deepseek',0,x[:4],ww,num,binding);assert reused and yp2==yp and num.receipt()==before
        partial=folder/'partial-unit';partial.mkdir();(partial/'Y.npy').write_bytes(b'partial')
        try:fit.atomic_Y(partial,'deepseek',0,x[:4],ww,num,binding);raise RuntimeError('expected partial preservation')
        except AssertionError:assert (partial/'Y.npy').read_bytes()==b'partial'
        fixture=folder/'mutation.bin';fixture.write_bytes(b'generated');done=folder/'mutation.complete.json';s.atomic_json(done,{'status':'complete','files':{str(fixture):s.sha(fixture)}});s.verify_unit(done);fixture.write_bytes(b'mutated')
        try:s.verify_unit(done);raise RuntimeError('expected identity mismatch')
        except AssertionError:pass
        rp=registry_fixture(folder/'all64-registry',tree,P,sigma,num,rng);all64={}
        for mid in [1,2]:
            reg=worker.Registry(mid,rp,num);ids=np.asarray([[(t+k)%64 for k in range(reg.K)] for t in range(64)],np.int32)
            canonical=None;local=None
            for mode in [9,10,11,12]:
                with num.category('generated_all64_registry_IPC'):
                    rows,frame=binary_roundtrip(reg,x,ids,mode)
                assert {e for t,k,e,flag,y in rows}==set(range(64))
                if mode in [9,11]:
                    if mode==9:canonical=rows
                    else:local=rows
                else:
                    reference=canonical if mode==10 else local
                    assert all(a[:4]==b[:4] and a[4].tobytes()==b[4].tobytes() for a,b in zip(rows,reference))
            assert any(flag==1 for t,k,e,flag,y in local) and any(flag==0 for t,k,e,flag,y in local)
            for mutation in ['short','duplicate','oldmode','wrongmodel','duplicateIDs','wrongslots','wronglayer']:
                h=w.unpack(frame[:w.WIRE.size]);body=frame[w.WIRE.size:]
                if mutation=='short':bad=frame[:-1]
                elif mutation=='duplicate':bad=frame+frame
                elif mutation=='oldmode':h['mode']=h['arm']=7;bad=w.pack(h)+body
                elif mutation=='wrongmodel':h['model']=3-mid;bad=w.pack(h)+body
                elif mutation=='wronglayer':h['layer']+=1;bad=w.pack(h)+body
                elif mutation=='wrongslots':h['slots']-=1;bad=w.pack(h)+body
                else:
                    ii=ids.copy();ii[0,1]=ii[0,0];bad=w.pack(h)+x.tobytes()+ii.tobytes()
                try:worker.serve(reg,io.BytesIO(bad),io.BytesIO());raise RuntimeError('expected malformed IPC')
                except (AssertionError,RuntimeError) as e:assert str(e)!='expected malformed IPC'
            all64[str(mid)]={'loaded64runtime_archives':True,'all64_IDs_selected_K6orK8':True,'modes9_10_11_12_bits_and_flags':True,'malformed_oldmode_duplicate_short_wrongmodel_layer_slots_rejected':True};del reg,rows,canonical,local
        jobs,ctx=s.context_plan(list(range(16)));assert len(ctx)==52 and len({c for j in jobs for c in j['contexts'].values()})==52
        values=rng.standard_normal((2,16,3))*.03+2;values[:,:,2]=values[:,:,1]+rng.standard_normal((2,16))*.01;summary=s.paired_summary(values,list(range(16)));assert summary['bootstrap_defined']
        draws=np.random.default_rng(2026110304).integers(0,16,(10000,16));delta=values[:,:,2]-values[:,:,1];boot=np.stack([np.asarray([sum(delta[m,j] for j in row)/16 for row in draws]) for m in range(2)]);sd=boot.std(axis=1,ddof=1);cut=float(np.quantile(np.max(np.abs((boot-delta.mean(axis=1)[:,None])/sd[:,None]),axis=0),.95,method='linear'));assert abs(cut-summary['cutoff'])<1e-12
        assert s.paired_summary(np.zeros((2,16,3)),list(range(16)))['primary_family_inconclusive'];missing=values.copy();missing[1,3,2]=np.nan;assert s.paired_summary(missing,list(range(16)))['primary_family_inconclusive'];cross=values.copy();cross[:,:,2]=cross[:,:,1]+np.log(1.02)+rng.standard_normal((2,16))*.03;crossing=s.paired_summary(cross,list(range(16)));assert crossing['bootstrap_defined'] and crossing['interval_relation']=='crossing_boundary_or_undefined' and crossing['primary_family_inconclusive']
        large=values.copy();large[:,:,2]+=1000;json.dumps(s.paired_summary(large,list(range(16))),allow_nan=False)
        lifecycle=lifecycle_fixtures(folder,base.worker.pid)
        result={'status':'passed','generated_only':True,'actual_model_weights_capture_targets_fittedpayload_books_eval_read':False,'old37_sources_unchanged':s.old_checks(),'FP64_integer_DGEMM_worst_signed_I64_all_channels':True,'ties_away_nearhalf_F32ratio_reference':True,'private_fullI1024_1408_actual_quantization':True,'fixed_local_only_complete_fit_archive_and_allrow_I64_B1_IPC':audit,'provenance_selection_keyorder_and_badrow':True,'atomic_completeY_reuse_zero_requery_and_partial_failclosed':True,'unsupported_in_archive_identity':True,'runtime_registry64_models':all64,'actual52_context_plan':True,'bootstrap_independent_loop_and_crossing_SD_missing_overflow':True,'numeric_ledger_generated_only':num.receipt(),'lifecycle':lifecycle};s.atomic_json(folder/'receipt.json',result);print(json.dumps({'status':'passed','generated_only':True,'receipt':str(folder/'receipt.json')}),flush=True)
    finally:base.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();main(a.folder)
