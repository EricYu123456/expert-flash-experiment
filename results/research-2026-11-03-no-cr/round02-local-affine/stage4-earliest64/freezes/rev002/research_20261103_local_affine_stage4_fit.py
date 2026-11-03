"""One canonical Y production pass per new expert; complete atomic units only."""
from collections import Counter
import io
import json
import os
from pathlib import Path
import time
import research_20261103_local_affine_stage4_core as s
import research_20261103_local_affine_stage4_worker as worker
np,old,core,r,pipe,w=s.np,s.old,s.core,s.r,s.pipe,s.w

def own_quantization(num,x):
    q,scale=num.quantize(x);amax=np.max(np.abs(x),axis=1)
    want=np.where(amax==0,np.float32(1),np.divide(amax,np.float32(127),dtype=np.float32)).astype(np.float32)
    s.bits(scale,want)
    ratio=np.divide(x,scale[:,None],dtype=np.float32).astype(np.float64)
    qr=np.clip(np.copysign(np.floor(np.abs(ratio)+.5),ratio),-127,127).astype(np.int8)
    assert np.array_equal(q,qr)
    return q,scale,{'amax_bits':amax.view(np.uint32).tolist(),'scale_bits':scale.view(np.uint32).tolist(),'q_array_hash':old.array_hash(q),'ties_away_ratio_F32_then_F64_reference':True}

def private_audit(num,x,weights,target=None):
    q,sx,xrec=own_quantization(num,x);g=num.linear(q,sx,*weights['gate']);u=num.linear(q,sx,*weights['up'])
    for role,y in [('gate',g),('up',u)]:
        ww,sw=weights[role];ref=q.astype(np.int64)@ww.astype(np.int64).T
        num.add('independent_fullrow_I64_MAC',len(q)*len(ww)*q.shape[1]);s.bits(y,num.rescale(ref.astype(np.float64),sw,sx))
    a=num.swiglu(g,u);qa,sa,arec=own_quantization(num,a);ww,sw=weights['down'];y=num.linear(qa,sa,ww,sw)
    ref=qa.astype(np.int64)@ww.astype(np.int64).T;num.add('independent_fullrow_I64_MAC',len(qa)*len(ww)*qa.shape[1]);s.bits(y,num.rescale(ref.astype(np.float64),sw,sa))
    if target is not None:s.bits(y,np.asarray(target,np.float32))
    return {'input_quantization':xrec,'private_full_I_product_quantization':arec,'all_GU_Down_output_channels_vs_I64_bits':True,'Y_array_hash':old.array_hash(y),'rows':len(x)}

def one_registry(mid,layer,e,weights,model,tree,P,sigma,num):
    reg=worker.Registry.__new__(worker.Registry);reg.model=mid;reg.name='deepseek' if mid==1 else 'olmoe';reg.K=6 if mid==1 else 8;reg.num=num;reg.layer=layer;reg.weights={e:weights};reg.locals={e:model};reg.ops={e:{}};reg.tree=tree;reg.P=P;reg.sigma=sigma;reg.contextmap=None;reg.events={};reg.current=None
    return reg

def ipc_audit(reg,x,mode):
    assert len(x)<=128
    e=next(iter(reg.weights));ids=np.asarray([[(e+k)%64 for k in range(reg.K)] for _ in x],np.int32)
    before=reg.evaluate(mode,reg.layer,x,ids);h=w.header(kind=1,mode=mode,arm=mode,request=1,run=1,context=mode,subrun=8,decode=1,model=reg.model,layer=reg.layer,T=len(x),K=reg.K,slots=len(before),bytes=4*len(x)*(2048+reg.K))
    stream=io.BytesIO();worker.serve(reg,io.BytesIO(w.pack(h)+x.tobytes()+ids.tobytes()),stream);raw=stream.getvalue();reply=w.unpack(raw[w.WIRE.size:2*w.WIRE.size]);assert reply['kind']==2 and reply['slots']==len(before) and reply['mode']==mode
    offset=2*w.WIRE.size
    for t,k,ee,flag,y in before:
        assert w.struct.unpack('<4I',raw[offset:offset+16])==(t,k,ee,flag);offset+=16
        assert raw[offset:offset+8192]==y.tobytes();offset+=8192
    assert offset==len(raw)
    return {'mode':mode,'rows':len(before),'binary_roundtrip_all_output_elements_bits':True}

def archive_audit(num,name,e,model,path,d,x,weights,tree,P,sigma,target,four):
    loaded=old.load_model(path);s.validate_local(loaded);assert s.local_hash(loaded)==s.local_hash(model)
    sd=core.subset(d,np.asarray(four));xx=np.ascontiguousarray(x[four]);expected=old.predict(num,model,sd);actual=old.predict(num,loaded,sd)
    s.bits(actual,expected)
    singles=np.concatenate([old.predict(num,loaded,core.subset(sd,np.asarray([i]))) for i in range(4)])
    s.bits(actual,singles)
    # Independent full-row I64 map check for every selected audit channel.
    for i,c in enumerate(sd['leaf']):
        q=sd['q'][i:i+1];ww=loaded['qmaps'][c];ref=q.astype(np.int64)@ww.astype(np.int64).T
        num.add('independent_fullrow_C_I64_MAC',q.shape[1]*len(ww))
        pred=num.add_bias(num.rescale(ref.astype(np.float64),loaded['scales'][c],sd['sx'][i:i+1]),loaded['bias'][c]);s.bits(pred,actual[i:i+1])
    private=private_audit(num,xx,weights,np.asarray(target[four]))
    reg=one_registry(1 if name=='deepseek' else 2,s.config(name)['layer'],e,weights,loaded,tree,P,sigma,num)
    ipc=[ipc_audit(reg,xx,mode) for mode in [9,10,11,12]]
    return {'archive_model_hash':s.local_hash(loaded),'four_ordered_union_rows':four,'finite_all_coefficients_scales_bias':True,'batch4_archive4_B1four_all_H_bits':True,'map_independent_I64_all_output_channels_bits':True,'private_canonical':private,'single_operator_IPC_audits':ipc,'allrow_not_spot_channel_checks':True}

def preflight(freeze,num):
    bind=s.read(s.OUT/'BIND_EXISTING64.complete.json');before=s.identities(bind['files']);dest=s.OUT/'preflight';dest.mkdir();reports=[]
    for name in ['deepseek','olmoe']:
        m=s.config(name);x,ids,alpha,prov,receipt=s.load_bound_union(name);selection=receipt['selection'];rows=[p['row'] for p in selection['rows']];e=selection['expert'];folder=dest/name;folder.mkdir();weights=pipe.load_weights(s.OUT/'bound'/name/'weights'/f'expert{e}',m['I']);xx=np.ascontiguousarray(x[rows]);yp=folder/'Y.npy';t=time.monotonic()
        with num.category('preflight_extra_teacher256'):
            y=old.teacher(num,xx,weights)
        np.save(yp,y);s.identities({str(yp):s.sha(yp)});loaded=np.load(yp,mmap_mode='r',allow_pickle=False);s.bits(y,loaded);assert np.isfinite(loaded).all()
        with num.category('preflight_deliberate_all256_B1_teacher'):
            for i in range(256):s.bits(old.teacher(num,xx[i:i+1],weights),loaded[i:i+1])
        with num.category('preflight_private_I64_four'):
            four=[rows.index(i) for i in selection['four_audit_rows']];audit=private_audit(num,xx[four],weights,np.asarray(loaded[four]))
        report={'model':name,'layer':m['layer'],'expert':e,'rows':256,'selection':selection,'Y_sha256':s.sha(yp),'Y_bytes':y.nbytes,'extra_teacher_integer_MAC':256*3*2048*m['I'],'teacher256_and_256individual_B1_bits':True,'private_audit':audit,'seconds':time.monotonic()-t,'counts_cumulative':num.receipt(),'rate_scope':'teacher+serialization+B1/privateaudit wall; not productionB1latency or quality'}
        p=folder/'receipt.json';s.atomic_json(p,report);reports.append(str(p));print(f'PREFLIGHT completed {name} E{e}',flush=True);s.disk_check();del x,ids,alpha,xx,y,loaded,weights
    assert s.identities(bind['files'])==before
    return s.finish('AUDIT_PREFLIGHT',freeze,[p for p in dest.rglob('*') if p.is_file()],{'operator_reports':reports,'primitive_ledger':num.receipt(),'minimum_extra_teacher_integer_MAC':3825205248,'no_selection_or_fit':True,'BIND_dependencies_prepost_equal':True})

def atomic_Y(folder,name,e,x,weights,num,unit_binding):
    receipt=folder/'Y.complete.json';yp=folder/'Y.npy'
    if receipt.exists():return yp,s.verify_unit(receipt,unit_binding),True
    assert not yp.exists(),'partial Y preserved; no implicit requery or recovery'
    s.disk_check(len(x)*2048*4+4096);y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.float32,shape=(len(x),2048));start=time.monotonic();before=num.receipt()
    with num.category('production_Y'):
        for a in range(0,len(x),256):
            s.disk_check();y[a:a+256]=old.teacher(num,x[a:a+256],weights)
    y.flush();del y
    fd=os.open(yp,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    info={'status':'complete',**unit_binding,'model':name,'expert':e,'shape':[len(x),2048],'dtype':'<f4','production_passes':1,'expert_rows':len(x),'F32_data_bytes':len(x)*2048*4,'integer_MAC':len(x)*3*2048*weights['gate'][0].shape[0],'wall_seconds':time.monotonic()-start,'counts_before':before,'counts_after':num.receipt(),'files':{str(yp):s.sha(yp)}}
    s.atomic_json(receipt,info);return yp,info,False

def fit(freeze,review,num,fma):
    bind=s.read(s.OUT/'BIND_EXISTING64.complete.json');before=s.identities(bind['files']);dest=s.OUT/'fit';dest.mkdir(exist_ok=True);allops={};newreports=[];runtime={};production=Counter();skipped=[]
    bound=s.read(s.OUT/'bound/models.json')
    for name in ['deepseek','olmoe']:
        m=s.config(name);x,ids,alpha,prov,urec=s.load_bound_union(name);tree,P,sigma,_=pipe.tree_load(bound[name]['geometry'])
        with num.category('common_frozen_geometry_design'):
            d=old.design(num,x,tree,32,P,sigma,128,'normal')
        include=np.ones(len(x),np.bool_);groups=np.asarray([p['group'] for p in prov]);inputs=r.union_statistics(d,include,groups,True);assert np.bincount(d['leaf'],minlength=32).tolist()==urec['union_identity']['runtime_leaf_counts']
        allops[name]=list(bound[name]['old_operators']);mf=dest/name;mf.mkdir(exist_ok=True)
        for e in m['new_experts']:
            s.disk_check();folder=mf/f'expert{e}';folder.mkdir(exist_ok=True);opdone=folder/'operator.complete.json'
            binding={'protocol_sha256':s.EXPECTED,'source_freeze_sha256':s.sha(freeze),'union_json_sha256':s.sha(s.OUT/'bound'/name/'union.json'),'canonical_weights_files':{str(p):s.sha(p) for p in (s.OUT/'bound'/name/'weights'/f'expert{e}').iterdir() if p.is_file()}}
            if opdone.exists():
                done=s.verify_unit(opdone,binding);allops[name].append(done['operator']);production.update(done['production_totals']);skipped.append(str(opdone));newreports.append(str(opdone));continue
            weights=pipe.load_weights(s.OUT/'bound'/name/'weights'/f'expert{e}',m['I']);yp,yi,reused_Y=atomic_Y(folder,name,e,x,weights,num,binding);y=np.load(yp,mmap_mode='r',allow_pickle=False);assert y.shape==(len(x),2048) and y.dtype==np.float32 and np.isfinite(y).all()
            attempts=list(folder.glob('fit.attempt*'));fitdir=folder/f'fit.attempt{len(attempts)+1:03d}';fitdir.mkdir();started=time.monotonic()
            with num.category('fixed_C_fit_and_actual_bias'):
                model,factor,info=r.fixed_fit(num,fma,inputs,y,0.,.01,'PE_RRR',make_factor=False)
            assert factor is None;s.validate_local(model)
            bank=fitdir/'models';bank.mkdir();path,mh=old.archive_model(bank,model)
            with num.category('archive_B1_private_I64_IPC_audits'):
                audit=archive_audit(num,name,e,model,path,d,x,weights,tree,P,sigma,y,urec['selection']['four_audit_rows'])
            local_solve=info['penalties_recomputed_from_union']['leaf_solve'];pooled=info['penalties_recomputed_from_union']['pooled_solve']
            # fixed_fit currently invokes33actual+33diagnostic solve calls; each
            # nonzero-trace solve executes cov@prior,U.T@residual,U@correction.
            solve_calls=2*(sum(float(np.trace(st['cov']))>0 for st in inputs['leaves'])+int(float(np.trace(inputs['pcov']))>0))
            fitcount={'response_cross_F64_MAC':len(x)*128*2048,'known_F64_solve_dense_MAC':solve_calls*3*128*128*2048,'solve_calls_nonzero_trace':solve_calls,'eig_common_reused':True,'BLAS_library_work_and_scalar_statistics_paid_not_a_zero_cost_claim':True}
            report={'model':name,'layer':m['layer'],'expert':e,'eta':0,'lambda':.01,'complete_union_rows':len(x),'explicit_allrows':True,'local_model_hash':mh,'local_unsupported':model['unsupported'],'local_solve':local_solve,'pooled_solve':pooled,'fit_info':info,'fit_primitive_source_counts':fitcount,'actual_numeric_primitive_ledger_cumulative':num.receipt(),'archive_audit':audit,'reused_completed_Y_without_query':reused_Y,'fit_seconds':time.monotonic()-started,'natural_union_hits_diagnostic_only':int(np.any(ids==e,axis=1).sum())}
            s.atomic_json(fitdir/'report.json',report)
            op={'expert':e,'local_model_path':path,'local_model_hash':mh,'weights_path':str(s.OUT/'bound'/name/'weights'/f'expert{e}'),'reused_old_C':False,'eta':0,'lambda':.01}
            totals={'expert_rows':len(x),'F32_data_bytes':len(x)*2048*4,'integer_MAC':len(x)*3*2048*m['I']}
            paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='operator.complete.json']
            done={'status':'complete',**binding,'operator':op,'production_totals':totals,'Y_complete_sha256':s.sha(folder/'Y.complete.json'),'files':{str(p):s.sha(p) for p in paths},'all_fit_archive_B1_IPC_audits_passed':True,'no_later_layer_or_factor_fit':True}
            s.atomic_json(opdone,done);allops[name].append(op);production.update(totals);newreports.append(str(opdone));print(f'FIT_C64 completed {name} E{e} union{len(x)}',flush=True)
            del y,model,weights,factor
        assert {o['expert'] for o in allops[name]}==set(range(64))
        runtime[name]={'layer':m['layer'],'I':m['I'],'geometry':bound[name]['geometry'],'operators':sorted(allops[name],key=lambda o:o['expert'])}
        del inputs,d,x,ids,alpha
    target=s.spec()['training']['new_Y'];assert production==Counter(expert_rows=target['expert_rows'],F32_data_bytes=target['float32_data_bytes'],integer_MAC=target['integer_MAC'])
    payload={}
    for name,model in runtime.items():
        for p in [Path(model['geometry'])/'tree.json',Path(model['geometry'])/'tree_projection.npz']:payload[str(p)]=s.sha(p)
        for op in model['operators']:
            payload[op['local_model_path']]=s.sha(op['local_model_path'])
            for p in Path(op['weights_path']).iterdir():
                if p.is_file():payload[str(p)]=s.sha(p)
    manifest=s.OUT/'C64_payload_manifest.json';s.atomic_json(manifest,{'files':payload,'runtime_only_targets_not_loaded_by_worker':True})
    registry=s.OUT/'C64_registry.frozen.json';s.atomic_json(registry,{'protocol_sha256':s.EXPECTED,'version':'first64-v1','models':runtime,'payload_manifest':{'path':str(manifest),'sha256':s.sha(manifest)},'old4_maps_unchanged':True,'late_layers_not_mounted':True,'new_experts_fixed_eta0_lambda01':True})
    assert s.identities(bind['files'])==before
    s.old_checks();s.disk_check()
    return s.finish('FIT_C64',freeze,[p for p in dest.rglob('*') if p.is_file()]+[manifest,registry],{'production_totals':dict(production),'operator_complete_receipts':newreports,'skipped_previously_complete_operators':skipped,'primitive_ledger_current_attempt':num.receipt(),'runtime_payload_files':payload,'registry_sha256':s.sha(registry),'BIND_dependencies_prepost_equal':True,'quality_payload_opened':False})
