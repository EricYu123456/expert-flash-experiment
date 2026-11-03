"""Stage3 isolated helpers. Import reads source/libraries, never payloads."""
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/home/ericyu/expert_flash_experiment');ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine';OUT=ROUND/'stage3-control-cost';REF=ROUND/'stage2-refresh';FREF=REF/'freezes/rev002';F2=ROUND/'stage2-quality/freezes/rev004';FQ=ROUND/'stage2-pg19-quality/freezes/rev001';QOUT=ROUND/'stage2-pg19-quality'
PROTOCOL=ROUND/'stage3_control_cost.protocol.frozen.json';EXPECTED='5f21869ae968057a09ed1f5f64cf06f83aa22fa1dc442303fa6ce5c25f9b58be';NL=7;COPY_NL=8
AMENDMENT=ROUND/'stage3_phase_order.amendment.frozen.json';AMEND_SHA='cd8fb02247d0be93850c598097d95963c3538607118553d1b4941da1a64a835b'
sys.path.insert(0,str(FREF))
import research_20261103_local_affine_stage2_refresh_core as r
np,w,old,core,pipe=r.np,r.w,r.old,r.core,r.pipe
read,save,sha,now=r.read,r.save,r.sha,r.now
assert Path(r.__file__).resolve()==FREF/'research_20261103_local_affine_stage2_refresh_core.py'
UPSTREAM=[REF/'BIND_OLD_CAPTURE.complete.json',REF/'COLLECT_REFRESH.complete.json',REF/'FIT_REFRESH.complete.json',REF/'refreshed_registry.frozen.json',REF/'refreshed_payload_manifest.json',QOUT/'QUALITY.complete.json',FREF/'freeze_receipt.json',FQ/'freeze_receipt.json']
PHASES=['BIND','GU_FIT','BENCH_BCD','BENCH_NL','DEV_QUALITY']
def identities(files):
    out={}
    for p,h in files.items():
        p=Path(p);a=p.stat();actual=sha(p);b=p.stat();assert actual==h and (a.st_size,a.st_mtime_ns,a.st_dev,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_dev,b.st_ino),str(p)
        out[str(p)]={'sha256':actual,'size':b.st_size,'mtime_ns':b.st_mtime_ns,'dev':b.st_dev,'inode':b.st_ino}
    return out
def authorize(freeze,review,phase):
    f=read(freeze);a=read(review);assert phase in PHASES and a['status']=='approved' and a['phase']==phase
    assert a['protocol_sha256']==f['protocol_sha256']==EXPECTED==sha(PROTOCOL)
    assert a['phase_order_amendment_sha256']==f['phase_order_amendment_sha256']==AMEND_SHA==sha(AMENDMENT)
    assert a['source_revision']==f['revision'] and a['source_freeze_sha256']==sha(freeze)
    assert a['upstream_receipts']==f['upstream_receipts']
    for p,h in f['files'].items():assert sha(p)==h,p
    for p,h in f['upstream_receipts'].items():assert sha(p)==h,p
    for key in (['BIND'] if phase!='BIND' else [])+(['BENCH_BCD'] if phase in ['GU_FIT','BENCH_NL','DEV_QUALITY'] else [])+(['GU_FIT'] if phase in ['BENCH_NL','DEV_QUALITY'] else []):
        cp=OUT/(key+'.complete.json');assert a[key+'_complete_sha256']==sha(cp) and read(cp)['status']=='complete'
    if phase=='DEV_QUALITY':
        assert a['NL_registry_sha256']==sha(OUT/'NL_registry.frozen.json')
    return f,a
def fit_files():return read(REF/'FIT_REFRESH.complete.json')['files']
def finish(folder,phase,freeze,files,extra=None):
    result={'status':'complete','phase':phase,'at_utc':now(),'protocol_sha256':EXPECTED,'phase_order_amendment_sha256':AMEND_SHA,'source_freeze_sha256':sha(freeze),'files':{str(p):sha(p) for p in files},'TEST_opened':False,'automatic_scientific_classification':False,**(extra or {})}
    pending=Path(folder)/(phase+'.pending.json');save(pending,result);os.replace(pending,Path(folder)/(phase+'.complete.json'));return result
def dump(folder,name,array):
    a=np.ascontiguousarray(array);p=Path(folder)/(name+'.bin');assert not p.exists();p.write_bytes(a.tobytes());return {'path':str(p),'dtype':a.dtype.str,'itemsize':a.itemsize,'shape':list(a.shape),'bytes':a.nbytes,'sha256':sha(p)}
def weight_export(folder,name,pair):
    q,s=pair;return {'rows':len(q),'cols':q.shape[1],'q':dump(folder,name+'.q',q),'s':dump(folder,name+'.s',s)}
def export_layer(folder,reg,layer):
    folder=Path(folder);folder.mkdir(parents=True);tree,P,sigma,_=reg.trees[layer];H=2048
    assert P.shape==(128,H) and all(tree[j]['normal'] is not None for j in range(31))
    j={'H':H,'K':reg.K,'layer':layer,'normals':dump(folder,'normals',np.stack([tree[k]['normal'] for k in range(31)])),'thresholds':dump(folder,'thresholds',np.asarray([tree[k]['threshold'] for k in range(31)],np.float64)),'means':dump(folder,'means',np.stack([tree[31+k]['mean'] for k in range(32)])),'P':dump(folder,'P',P),'operators':[]}
    for e in sorted(reg.pilots(layer)):
        o=reg.ops[layer,e];m=reg.locals[layer,e];f=reg.factors[layer,e];assert f['meta']['format']=='W8' and f['meta']['path']=='host_floatlatent' and f['meta']['gauge']=='identity' and o['factor_point']['guard']==100
        weights=reg.weights[layer,e];d={'expert':e,'I':weights['gate'][0].shape[0],'global':f['meta']['family']=='G_RRR','Q':512,'guard':100,'unsupported':list(map(int,m['unsupported'])),'dunsupported':list(map(int,f['unsupported']))}
        for key in ['gate','up','down']:d[key]=weight_export(folder,f'e{e}.{key}',weights[key])
        for key,a in {'cq':m['qmaps'],'cs':m['scales'],'cb':m['bias'],'bq':f['B'],'bs':f['Bs'],'uq':f['U'],'us':f['Us'],'db':f['bias']}.items():d[key]=dump(folder,f'e{e}.{key}',a)
        j['operators'].append(d)
    p=folder/'layer.json';save(p,j);return p
def bench_selection(provenance):
    ranked=[]
    for index,p in enumerate(provenance):
        b=json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode();h=hashlib.sha256(b'stage3-bench-v1\0'+b).hexdigest();ranked.append((h,index,p))
    ranked.sort(key=lambda v:(v[0],v[1]));assert len(ranked)>=32
    return [{'collection_row':i,'selection_sha256':h,'provenance':p} for h,i,p in ranked[:32]]
def load_union(name,layer):
    fitids=[v['id'] for v in read(core.STAGE0/'content_identities.frozen.json')['records'] if v['split']=='FIT'];assert len(fitids)==120
    ds=pipe.dataset(core.STAGE0/'capture'/name/'FIT',layer,fitids);nf=REF/'collected'/name/f'layer{layer}';x=np.concatenate([ds['x'],np.load(nf/'inputs.npy',allow_pickle=False)])
    expected=read(REF/'fitted'/name/f'layer{layer}'/'union_input_identity.json');assert len(x)==(29629 if name=='deepseek' else 29593) and old.array_hash(x)==expected['input_array_hash'];return x
def load_y(name,layer,e):
    info=read(REF/'fitted'/name/f'layer{layer}'/f'expert{e}'/'shared_target_identity.json')
    for k in ['old','new']:assert sha(info[k+'_path'])==info[k+'_sha256']
    y=np.concatenate([np.load(info['old_path'],mmap_mode='r',allow_pickle=False),np.load(info['new_path'],mmap_mode='r',allow_pickle=False)]);assert old.array_hash(y)==info['ordered_union_target_array_hash'];return y
def ffi_dots(lib,q,weights):
    q=np.ascontiguousarray(q,np.int8);weights=np.ascontiguousarray(weights,np.int8);out=np.empty((len(q),len(weights)),np.int32);assert lib.ls3_dots(old.Numeric.ptr(q),old.Numeric.ptr(weights),len(q),q.shape[1],len(weights),old.Numeric.ptr(out))==0;return out
def numeric_audit(folder,layer_manifest,sample,reg,layer,freeze,review,nl_models=None):
    folder=Path(folder);folder.mkdir();arms=[3,5,6,7] if nl_models else [3,5,6];phase='GU_FIT' if nl_models else 'BIND';plan={'phase':phase,'review':str(review),'protocol_sha256':EXPECTED,'phase_order_amendment_sha256':AMEND_SHA,'source_freeze_sha256':sha(freeze),'layer_manifest':str(layer_manifest),'arms':arms,**sample,'output_prefix':str(folder/'raw'),'output':str(folder/'direct.json')};save(folder/'plan.json',plan)
    started=time.monotonic();p=subprocess.run([str(Path(freeze).parent/'host-numeric'),'audit',str(folder/'plan.json')],capture_output=True,text=True);(folder/'stderr.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'direct numeric audit failed; preserve diagnostics'
    def raw(a):return np.fromfile(a['path'],np.dtype(a['dtype'])).reshape(a['shape'])
    x=raw(sample['x'])[:4];tree,P,sigma,_=reg.trees[layer];d=old.design(reg.num,x,tree,32,P,sigma,128,'normal');dg=old.design(reg.num,x,tree,32,P,sigma,2048,'identity');qx,sx=reg.num.quantize(x);lib=ct.CDLL(str(Path(freeze).parent/'numeric.so'));checks=[];extra_mac=0
    # Every output row, all4 fixed vectors, exact signed integer/I64 audit.
    for e in sorted(reg.pilots(layer)):
        weights=reg.weights[layer,e]
        for head in ['gate','up']:
            a=ffi_dots(lib,qx,weights[head][0]);ref=qx.astype(np.int64)@weights[head][0].astype(np.int64).T;assert np.array_equal(a,ref)
        for mode in arms:
            if mode==3:
                if nl_models:
                    previous=OUT/'bound'/reg.name/f'layer{layer}'/'audit';expected=np.stack([np.fromfile(previous/f'raw.r{row}.e{e}.a3.f32',np.float32) for row in range(4)])
                else:expected=old.teacher(reg.num,x,weights);extra_mac+=4*3*2048*weights['gate'][0].shape[0]
            elif mode==5:expected=old.predict(reg.num,reg.locals[layer,e],d)
            elif mode==6:expected=core.predict(reg.num,reg.fma,reg.factors[layer,e],dg if reg.factors[layer,e]['meta']['family']=='G_RRR' else d)
            else:expected=predict_nl(reg.num,reg.fma,nl_models[e],d,weights);extra_mac+=4*2048*weights['gate'][0].shape[0]
            for row in range(4):
                actual=np.fromfile(folder/f'raw.r{row}.e{e}.a{mode}.f32',np.float32);assert actual.shape==expected[row].shape and np.array_equal(actual.view(np.uint32),expected[row].view(np.uint32)),f'allrawbits {reg.name} L{layer} E{e} mode{mode} row{row}'
            checks.append({'expert':e,'arm':mode,'rows':4,'output_elements':int(expected.size),'all_bits_equal':True})
    report={'status':'passed','seconds':time.monotonic()-started,'checks':checks,'full_row_I64_dot_checks':True,'additional_reference_integer_MACs':extra_mac,'not_GU_production_query':True};save(folder/'closure.json',report);return report
def bind(freeze,review,num,fma):
    before=identities(fit_files());dest=OUT/'bound';dest.mkdir();selections={};registry=REF/'refreshed_registry.frozen.json'
    # Freeze row IDs/hash/order for all4 layers before opening selected arrays.
    for name,layers in [('deepseek',[1,13]),('olmoe',[0,8])]:
        for layer in layers:
            nf=REF/'collected'/name/f'layer{layer}';prov=[json.loads(v) for v in (nf/'provenance.jsonl').read_text().splitlines()];assert len(prov)==13824;selections[f'{name}.L{layer}']=bench_selection(prov)
    save(dest/'bench_rows.frozen.json',{'domain':'stage3-bench-v1\\0 + UTF8 canonicalJSON(sort_keys,compact,ensure_ascii=true)','selection_before_array_values':True,'layers':selections})
    exported=[];audits=[]
    for mid,name in [(1,'deepseek'),(2,'olmoe')]:
        reg=w.Registry(mid,registry,num,fma)
        for layer in sorted(reg.trees):
            nf=REF/'collected'/name/f'layer{layer}';chosen=selections[f'{name}.L{layer}'];ix=np.asarray([v['collection_row'] for v in chosen]);x=np.load(nf/'inputs.npy',mmap_mode='r',allow_pickle=False)[ix];ids=np.load(nf/'ids.npy',mmap_mode='r',allow_pickle=False)[ix];alpha=np.load(nf/'alpha.npy',mmap_mode='r',allow_pickle=False)[ix]
            for row,p in zip(x,chosen):assert hashlib.sha256(row.tobytes()).hexdigest()==p['provenance']['input_sha256']
            folder=dest/name/f'layer{layer}';lp=export_layer(folder,reg,layer);sample={'x':dump(folder,'sample.x',x),'ids':dump(folder,'sample.ids',ids),'alpha':dump(folder,'sample.alpha',alpha)};save(folder/'sample.json',sample);audits.append(numeric_audit(folder/'audit',lp,sample,reg,layer,freeze,review));union=load_union(name,layer);save(folder/'union.binding.json',{'rows':len(union),'input_array_hash':old.array_hash(union),'no_new_states':True});del union;exported.append(str(lp))
        del reg
    assert identities(fit_files())==before;save(dest/'FIT_dependencies.prepost.json',{'before':before,'after':before,'all_actual_hashes_checked_twice':True});files=[p for p in dest.rglob('*') if p.is_file()];return finish(OUT,'BIND',freeze,files,{'exported_layers':exported,'numeric_audits':audits,'FIT_dependency_identities':before,'benchmark_rows_manifest_sha256':sha(dest/'bench_rows.frozen.json'),'historical_X_equality_claim':False,'semantic_old_X_to_frozenY_binding_receipt':str(REF/'BIND_OLD_CAPTURE.complete.json')})
def predict_nl(num,fma,m,d,weights,bias=True):
    n=len(d['q']);I=len(m['g']);g=np.empty((n,I),np.float32);u=np.empty_like(g)
    for a in range(0,n,256):
        sl=slice(a,a+256);q=d['q'][sl].astype(np.float32);sx=d['sx'][sl,None]
        g[sl]=np.multiply(fma.dot(q,m['g']),sx,dtype=np.float32);u[sl]=np.multiply(fma.dot(q,m['u']),sx,dtype=np.float32)
        for c in np.unique(d['leaf'][sl]):
            ix=np.flatnonzero(d['leaf'][sl]==c);g[a+ix]=num.add_bias(g[a+ix],m['gb'][c]);u[a+ix]=num.add_bias(u[a+ix],m['ub'][c])
    a=num.swiglu(g,u);qa,sa=num.quantize(a);out=num.linear(qa,sa,*weights['down'])
    if bias:
        for c in np.unique(d['leaf']):
            ix=np.flatnonzero(d['leaf']==c);out[ix]=num.add_bias(out[ix],m['yb'][c])
    return out
def model_hash(m):return old.array_hash(*(m[k] for k in ['g','u','gb','ub','yb','unsupported']))
def save_model(path,m):np.savez(path,**m);return model_hash(m)
def load_model(path):
    with np.load(path,allow_pickle=False) as z:m={k:z[k].copy() for k in z.files}
    assert set(m)=={'g','u','gb','ub','yb','unsupported'} and m['g'].shape==m['u'].shape and m['g'].shape[1]==128 and m['gb'].shape==m['ub'].shape==(32,len(m['g'])) and m['yb'].shape==(32,2048)
    assert m['unsupported'].dtype==np.int32 and np.all((m['unsupported']>=0)&(m['unsupported']<32))
    for k in ['g','u','gb','ub','yb']:assert m[k].dtype==np.float32 and np.isfinite(m[k]).all()
    return m
def fit_one(num,fma,inputs,targets,y,weights,unsupported,folder):
    """One eta0 solve; cache one actual GU projection pass; one Down pass."""
    folder=Path(folder);N=len(y);I=targets.shape[1]//2;assert targets.shape==(N,2*I);As,prior,cross,stats,yn=core.fitted_slopes(inputs,targets,0.,'infinity');_,solve_info=old.solve(inputs['pcov'],cross,0.,factor=inputs['pfactor']);slopes=prior.T.astype(np.float32);m={'g':slopes[:I].copy(),'u':slopes[I:].copy(),'gb':np.zeros((32,I),np.float32),'ub':np.zeros((32,I),np.float32),'yb':np.zeros((32,2048),np.float32),'unsupported':np.asarray(unsupported,np.int32)};d=inputs['d']
    raw=np.lib.format.open_memmap(folder/'GU.actual_prediction_before_bias.npy',mode='w+',dtype=np.float32,shape=(N,2*I))
    for a in range(0,N,256):
        sl=slice(a,a+256);q=d['q'][sl].astype(np.float32);sx=d['sx'][sl,None];raw[sl,:I]=np.multiply(fma.dot(q,m['g']),sx,dtype=np.float32);raw[sl,I:]=np.multiply(fma.dot(q,m['u']),sx,dtype=np.float32)
    raw.flush()
    for c in range(32):
        ix=np.flatnonzero(d['leaf']==c)
        if len(ix):b=num.bias(targets[ix],raw[ix],np.ones(len(ix)));m['gb'][c]=b[:I];m['ub'][c]=b[I:]
    pred=np.lib.format.open_memmap(folder/'Down.actual_prediction_before_output_bias.npy',mode='w+',dtype=np.float32,shape=y.shape)
    for a in range(0,N,256):
        sl=slice(a,a+256);g=np.asarray(raw[sl,:I]).copy();u=np.asarray(raw[sl,I:]).copy()
        for c in np.unique(d['leaf'][sl]):
            ix=np.flatnonzero(d['leaf'][sl]==c);g[ix]=num.add_bias(g[ix],m['gb'][c]);u[ix]=num.add_bias(u[ix],m['ub'][c])
        act=num.swiglu(g,u);qa,sa=num.quantize(act);pred[sl]=num.linear(qa,sa,*weights['down'])
    pred.flush()
    for c in range(32):
        ix=np.flatnonzero(d['leaf']==c)
        if len(ix):m['yb'][c]=num.bias(y[ix],pred[ix],np.ones(len(ix)))
    return m,{'eta':0,'pooled_solve':solve_info,'trace':float(np.trace(inputs['pcov'])),'positive_relative_cutoff':1e-10,'GU_actual_FMA_prediction_passes':1,'GU_actual_FMA':int(N*2*I*128),'GU_prediction_cache_bytes':int(raw.nbytes),'Down_prediction_passes':1,'Down_integer_MACs':int(N*2048*I),'Down_prediction_cache_bytes':int(pred.nbytes),'bias_rule':'sequentialF64 residualmean via frozen la_calibrate; F32 intercept; no post-output-add recalibration','leaf_rows':[int(np.sum(d['leaf']==c)) for c in range(32)]}
def gu_fit(freeze,review,num,fma):
    bind_receipt=read(OUT/'BIND.complete.json');before=identities({**fit_files(),**bind_receipt['files']});dest=OUT/'NL';dest.mkdir();newops=[];reports=[];totals={'production_expert_rows':0,'production_GU_integer_MACs':0,'production_GU_target_bytes':0,'GU_prediction_FMA':0,'Down_calibration_integer_MACs':0};registry=REF/'refreshed_registry.frozen.json'
    for mid,name in [(1,'deepseek'),(2,'olmoe')]:
        reg=w.Registry(mid,registry,num,fma)
        for layer in sorted(reg.trees):
            x=load_union(name,layer);tree,P,sigma,_=reg.trees[layer];d=old.design(num,x,tree,32,P,sigma,128,'normal');inputs=r.union_statistics(d,np.ones(len(x),np.bool_),np.zeros(len(x),np.int32),False);folder=dest/name/f'layer{layer}';folder.mkdir(parents=True);lm=read(OUT/'bound'/name/f'layer{layer}'/'layer.json');models={}
            for e in sorted(reg.pilots(layer)):
                t=time.monotonic();ef=folder/f'expert{e}';ef.mkdir();weights=reg.weights[layer,e];I=weights['gate'][0].shape[0];target=np.lib.format.open_memmap(ef/'GU.canonical_targets.npy',mode='w+',dtype=np.float32,shape=(len(x),2*I))
                for a in range(0,len(x),256):
                    sl=slice(a,a+256);q,sx=num.quantize(x[sl]);target[sl,:I]=num.linear(q,sx,*weights['gate']);target[sl,I:]=num.linear(q,sx,*weights['up'])
                target.flush();y=load_y(name,layer,e);m,report=fit_one(num,fma,inputs,target,y,weights,reg.locals[layer,e]['unsupported'],ef);mp=ef/'NL.npz';mh=save_model(mp,m);loaded=load_model(mp);assert model_hash(loaded)==mh;models[e]=loaded;sample=np.arange(min(4,len(x)));a=predict_nl(num,fma,m,core.subset(d,sample),weights);b=np.concatenate([predict_nl(num,fma,loaded,core.subset(d,np.asarray([i])),weights) for i in sample]);assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
                report.update(model=name,layer=layer,expert=e,rows=len(x),GU_production_passes=1,GU_integer_MACs=int(len(x)*2*2048*I),GU_target_bytes=int(target.nbytes),GU_target_sha256=sha(ef/'GU.canonical_targets.npy'),model_hash=mh,seconds=time.monotonic()-t,archive_B1_batch_bits=True,support_inherited_from_C=True);save(ef/'fit_report.json',report);reports.append(report)
                totals['production_expert_rows']+=len(x);totals['production_GU_integer_MACs']+=report['GU_integer_MACs'];totals['production_GU_target_bytes']+=target.nbytes;totals['GU_prediction_FMA']+=report['GU_actual_FMA'];totals['Down_calibration_integer_MACs']+=report['Down_integer_MACs'];newops.append({'model':name,'layer':layer,'expert':e,'NL_model_path':str(mp),'NL_model_hash':mh,'unchanged_C_D_operator':reg.ops[layer,e]})
                o=next(o for o in lm['operators'] if o['expert']==e)
                for key,a in [('ng',m['g']),('nu',m['u']),('gb',m['gb']),('ub',m['ub']),('yb',m['yb'])]:o[key]=dump(ef,'export.'+key,a)
                print(f'GU_FIT completed {name} L{layer} E{e} rows{len(x)}',flush=True);del target,y,m,loaded
            lp=folder/'layer.json';save(lp,lm);numeric_audit(folder/'audit',lp,read(OUT/'bound'/name/f'layer{layer}'/'sample.json'),reg,layer,freeze,review,models);del x,d,inputs,models
        del reg
    assert totals['production_expert_rows']==236888 and totals['production_GU_integer_MACs']==1179989835776 and totals['production_GU_target_bytes']==2304667648 and totals['Down_calibration_integer_MACs']==589994917888
    payload=OUT/'NL_payload_manifest.json';files=[p for p in dest.rglob('*') if p.is_file()];save(payload,{'files':{str(p):sha(p) for p in files}});registry_new=OUT/'NL_registry.frozen.json';save(registry_new,{'protocol_sha256':EXPECTED,'arm':NL,'copy_arm':COPY_NL,'original_registry_path':str(registry),'original_registry_sha256':sha(registry),'operators':newops,'payload_manifest':str(payload),'payload_manifest_sha256':sha(payload),'same_C_D_P_tree':True,'fit_data':'complete fixed oldFIT120+PGTRAIN16 multiset','eta':0,'PG_development_not_opened':True});assert identities({**fit_files(),**bind_receipt['files']})==before;return finish(OUT,'GU_FIT',freeze,files+[payload,registry_new],{'totals':totals,'reports':reports,'NL_registry_sha256':sha(registry_new),'all_FIT_and_BCD_exports_prepost_equal':True,'additional_audit_costs':'separately in each layer/audit/closure.json; archive B1 checks additionally paid, not production GU pass'})
def development_summary(old_values,nl_values,book_ids):
    b=np.asarray(old_values,np.float64);nl=np.asarray(nl_values,np.float64);assert b.shape==(2,16,4) and nl.shape==(2,16) and len(book_ids)==len(set(book_ids))==16
    result={'development_only_exposed_validation16':True,'book_ids':list(book_ids),'primary_family_inconclusive':True,'seed':2026110303,'repeats':10000,'bootstrap_SD_ddof':1,'quantile_method':'linear','margin_NLL':math.log(1.02),'classification':'Root+critic only, no automatic scientific decision','no_PG_selection':True,'models':{},'TEST_opened':False}
    if not np.isfinite(b).all() or not np.isfinite(nl).all():result['reason']='missing/nonfinite endpoint, all family retained';return result
    draws=np.random.default_rng(2026110303).integers(0,16,size=(10000,16));delta=nl-b[:,:,1];hat=delta.mean(axis=1);boot=delta[:,draws].mean(axis=2);sd=boot.std(axis=1,ddof=1);valid=np.isfinite(sd).all() and (sd>0).all();cut=float(np.quantile(np.max(np.abs((boot-hat[:,None])/sd[:,None]),axis=0),.95,method='linear')) if valid else None;result.update(primary_family_inconclusive=not bool(valid),cutoff=cut,shared_draws_array_hash=old.array_hash(draws),reason=None if valid else 'zero/nonfinite bootstrapSD, no drop/redraw/literal shortcut')
    for i,name in enumerate(['deepseek','olmoe']):
        band=[float(hat[i]-cut*sd[i]),float(hat[i]+cut*sd[i])] if valid else None
        def expm(v):
            try:return {'relative_PPL_change':math.expm1(float(v)),'overflow':False}
            except OverflowError:return {'relative_PPL_change':None,'overflow':True}
        result['models'][name]={'meanNLL_source_B_C_D':b[i].mean(axis=0).tolist(),'meanNLL_NL':float(nl[i].mean()),'per_book_NL_meanNLL':nl[i].tolist(),'primary_NL_minus_B':{'Delta_hat':float(hat[i]),'bootstrapSD':float(sd[i]),'simultaneous95_NLL_band':band,'simultaneous95_relative_PPL_band':None if band is None else [expm(v) for v in band],'per_book_delta':delta[i].tolist()},'descriptive_NL_minus_C':{'mean_delta_NLL':float((nl[i]-b[i,:,2]).mean()),'per_book_delta':(nl[i]-b[i,:,2]).tolist()},'unchanged_C_minus_B':{'mean_delta_NLL':float((b[i,:,2]-b[i,:,1]).mean()),'per_book_delta':(b[i,:,2]-b[i,:,1]).tolist()}}
    return result
