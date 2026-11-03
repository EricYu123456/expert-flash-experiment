"""One-refresh helpers. Imports read source/libraries only, never payloads."""
import base64
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
ROOT=Path('/home/ericyu/expert_flash_experiment');ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine';S2=ROUND/'stage2-quality';F2=S2/'freezes/rev004';OUT=ROUND/'stage2-refresh'
PROTOCOL=ROUND/'stage2_refresh.protocol.frozen.json';EXPECTED='78045b99f1e4647475bc7a35ef27110856ed081dc8b2faace7ddd332b5b5d167';OLD_FREEZE='00a08bfc5eee76d1d6b97d2826457853932058e72c416754f4390f37bbc811e1';OLD_REGISTRY='7ed8766ed70ba1acb48e864003b1119f5b9e875fc03cd2832f963ee915a8bfc9'
sys.path.insert(0,str(F2))
import research_20261103_local_affine_stage2_worker as w
np,old,core,pipe=w.np,w.old,w.core,w.pipe
assert Path(w.__file__).resolve()==F2/'research_20261103_local_affine_stage2_worker.py'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):old.save(p,v)
def sha(p):return old.sha(p)
def authorize(freeze,review,phase):
    fp=Path(freeze);r=read(fp);a=read(review)
    assert phase in ['BIND_OLD_CAPTURE','COLLECT_REFRESH','FIT_REFRESH'] and a['status']=='approved' and a['phase']==phase
    assert a['protocol_sha256']==r['protocol_sha256']==EXPECTED and sha(PROTOCOL)==EXPECTED
    assert a['source_revision']==r['revision'] and a['source_freeze_sha256']==sha(fp)
    assert a['old_harness_source_freeze_sha256']==r['old_harness_source_freeze_sha256']==OLD_FREEZE
    assert a['old_registry_sha256']==r['old_registry_sha256']==OLD_REGISTRY
    for p,h in r['files'].items():assert sha(p)==h,p
    if phase in ['COLLECT_REFRESH','FIT_REFRESH']:
        assert a['BIND_OLD_CAPTURE_complete_sha256']==sha(OUT/'BIND_OLD_CAPTURE.complete.json')
        assert read(OUT/'BIND_OLD_CAPTURE.complete.json')['status']=='complete_all_old_targets_bitwise_verified'
    if phase=='FIT_REFRESH':
        assert a['COLLECT_REFRESH_complete_sha256']==sha(OUT/'COLLECT_REFRESH.complete.json')
        assert read(OUT/'COLLECT_REFRESH.complete.json')['status']=='complete'
        assert a['old_capture_files_sha256']==read(OUT/'BIND_OLD_CAPTURE.complete.json')['old_capture_files_sha256']
    return r,a
def preprocess(raw,metadata):
    assert len(raw)==int(metadata['size_bytes']),'pinned object size mismatch'
    assert base64.b64encode(hashlib.md5(raw).digest()).decode()==metadata['md5_base64'],'pinned object MD5 mismatch'
    text=raw.decode('utf-8',errors='strict').replace('\r\n','\n').replace('\r','\n');mid=len(text)//2;lf=text.find('\n',mid);start=mid if lf<0 else lf+1;suffix=text[start:]
    return suffix.encode('utf-8'),{'normalization':'strictUTF8 CRLF thenCR toLF','unicode_codepoints':len(text),'unicode_midpoint':mid,'suffix_start_codepoint':start,'first_LF_at_or_after_midpoint':None if lf<0 else lf,'raw_sha256':hashlib.sha256(raw).hexdigest(),'suffix_sha256':hashlib.sha256(suffix.encode()).hexdigest()}
def state_key(h):return tuple(h[k] for k in ['model','run','context','subrun','arm','decode','micro','layer','T','H','K','request','position','token_phase'])
def admitted(h):return (h['mode']==3 and h['token_phase']==0) or (h['mode'] in [3,5,6] and h['token_phase']==1)
def frames(path,magic):
    with open(path,'rb') as f:
        assert w.exact(f,8)==magic
        while True:
            b=w.exact(f,w.WIRE.size,eof=True)
            if b is None:return
            h=w.unpack(b);assert 1<=h['T']<=128 and h['H']==2048 and h['K'] in [6,8] and admitted(h)
            yield h,w.exact(f,h['bytes'])
def close_collection(model,input_path,alpha_path,result_path,destination,books):
    """Join actual alpha and owned inputs by exact typed epoch; no row dedup."""
    result=read(result_path);assert result['status']=='complete' and [j['book_id'] for j in result['jobs']]==books
    contexts={};mid=1 if model=='deepseek' else 2;layers=[1,13] if mid==1 else [0,8];K=6 if mid==1 else 8
    for bi,job in enumerate(result['jobs']):
        assert [r['mode'] for r in job['runs']]==[3,5,6];assert len(job['immutable_prefix_ids'])==768
        for ai,r in enumerate(job['runs']):
            assert r['model']==mid and r['context']==bi*3+ai+1 and r['fresh_KV'] and r['prefix_ids']==job['immutable_prefix_ids'] and r['subrun']==4 and r['prefix_input_rows']==768
            assert 0<=len(r['generated_ids'])<=32 and r['decode_input_rows']==len(r['generated_ids']) and r['stop'] in ['EOG','max32']
            assert (r['stop']=='EOG' and r['EOG_not_fed']>=0) or (r['stop']=='max32' and r['EOG_not_fed']==-1 and len(r['generated_ids'])==32)
            contexts[r['context']]={'book_id':job['book_id'],'book_index':bi,'arm_index':ai,**r}
    alpha={}
    for h,b in frames(alpha_path,b'LARFALP1'):
        assert h['kind']==4 and h['model']==mid and h['K']==K and h['bytes']==h['T']*K*8;key=state_key(h);assert key not in alpha
        count=h['T']*K;ids=np.frombuffer(b[:count*4],'<i4').reshape(h['T'],K).copy();a=np.frombuffer(b[count*4:],'<f4').reshape(h['T'],K).copy();assert np.isfinite(a).all() and np.all(a>=0);alpha[key]=(ids,a,h)
    rows={l:[] for l in layers};seen=set();framecount=0
    for h,b in frames(input_path,b'LARFXI01'):
        key=state_key(h);assert h['kind']==1 and h['model']==mid and h['K']==K and h['bytes']==h['T']*(2048+K)*4 and key not in seen and key in alpha;seen.add(key);framecount+=1
        ids,a,ah=alpha[key];assert h['slots']==ah['slots'];cut=h['T']*2048*4;x=np.frombuffer(b[:cut],'<f4').reshape(h['T'],2048).copy();ownids=np.frombuffer(b[cut:],'<i4').reshape(h['T'],K);assert np.array_equal(ownids,ids) and np.isfinite(x).all() and np.all((ids>=0)&(ids<64));assert all(len(np.unique(v))==K for v in ids)
        c=contexts[h['context']];assert all(h[k]==c[k] for k in ['run','subrun','arm','mode','model']) and h['layer'] in layers and h['micro']==0
        assert h['decode']==(h['position']//128+1 if h['token_phase']==0 else 7+h['position']-768)
        assert (h['token_phase']==0 and h['mode']==3 and h['T']==128 and h['position'] in range(0,768,128)) or (h['token_phase']==1 and h['T']==1 and 768<=h['position']<768+len(c['generated_ids']))
        for t in range(h['T']):
            pos=h['position']+t;token=c['prefix_ids'][pos] if h['token_phase']==0 else c['generated_ids'][pos-768]
            provenance={'model':model,'book_id':c['book_id'],'group':'pg19train:'+str(c['book_id']),'layer':h['layer'],'arm':h['arm'],'run':h['run'],'context':h['context'],'subrun':4,'decode_epoch':h['decode'],'request':h['request'],'micro':h['micro'],'absolute_position':pos,'token_id':int(token),'phase':'prefix' if h['token_phase']==0 else 'decode','token_in_batch':t,'input_sha256':hashlib.sha256(x[t].tobytes()).hexdigest(),'origin':'pg19train','unit_weight':1}
            order=(c['book_index'],0 if h['token_phase']==0 else c['arm_index']+1,pos);rows[h['layer']].append((order,x[t].copy(),ids[t].copy(),a[t].copy(),provenance))
    assert seen==set(alpha) and framecount==result['alpha_frames'] and sum(len(v) for v in rows.values())==result['alpha_rows_both_layers']
    receipts={};destination=Path(destination);destination.mkdir()
    for l,bank in rows.items():
        bank.sort(key=lambda z:z[0]);assert len(bank)<=13824 and len({(v[4]['context'],v[4]['absolute_position']) for v in bank})==len(bank)
        for bi,job in enumerate(result['jobs']):
            for ai,r in enumerate(job['runs']):
                actual=[z[4]['absolute_position'] for z in bank if z[4]['book_id']==job['book_id'] and z[4]['arm']==r['arm']];expected=(list(range(768)) if ai==0 else [])+list(range(768,768+len(r['generated_ids'])));assert actual==expected,'missing/duplicate/out-of-order true states'
        folder=destination/f'layer{l}';folder.mkdir();np.save(folder/'inputs.npy',np.stack([v[1] for v in bank]));np.save(folder/'ids.npy',np.stack([v[2] for v in bank]));np.save(folder/'alpha.npy',np.stack([v[3] for v in bank]));(folder/'provenance.jsonl').write_text(''.join(json.dumps(v[4],separators=(',',':'))+'\n' for v in bank))
        paths=list(folder.iterdir());receipts[l]={'rows':len(bank),'groups':len(books),'multiplicity_retained':True,'contributions':dict(Counter(str(z[4]['arm'])+':'+z[4]['phase'] for z in bank)),'files':{str(p):sha(p) for p in paths}}
    return {'status':'complete','model':model,'frames':framecount,'layers':receipts,'all_alpha_ownedinput_epochs_joined':True,'offroute_teacher_targets_queried':False}
def union_statistics(d,include,groups,local):
    """Explicit bool mask; no numeric prompt<nfit filter exists here."""
    include=np.asarray(include);assert include.dtype==np.bool_ and include.shape==(len(d['q']),) and include.all()
    groups=np.asarray(groups);assert len(groups)==len(include);ix=np.flatnonzero(include);f=core.subset(d,ix);r=f['v'].shape[1];pcov=np.zeros((r,r),np.float64);leaves=[]
    for c in range(32):
        rows=np.flatnonzero(f['leaf']==c);a=f['v'][rows];n=len(rows);mean=a.mean(axis=0) if n else np.zeros(r);dv=a-mean;cov=(dv.T*np.ones(n))@dv/n if n else np.zeros((r,r));pcov+=cov*(n/len(ix));leaves.append({'ix':rows,'mean_v':mean,'dv':dv,'sum_w':n,'cov':cov if local else None,'factor':old.eigen(cov) if local else None,'distinct_groups':int(len(np.unique(groups[ix[rows]])))})
    return {'d':f,'ix':ix,'leaves':leaves,'pcov':pcov,'pfactor':old.eigen(pcov),'local':local,'explicit_union_allrows':True}
def archive_geometry(folder,tree,P,sigma,num,x,groups,origins):
    # Reuse exact tree archive layout without its obsolete N120-only labels.
    folder=Path(folder);folder.mkdir();arrays={'basis':P,'sigma':sigma};meta={}
    for k,node in tree.items():
        arrays[f'mean{k}']=node['mean'];meta[str(k)]={a:b for a,b in node.items() if a not in ['mean','normal']}
        if node['normal'] is not None:arrays[f'normal{k}']=node['normal']
    leaf=old.select_tree(num,tree,x,32);assert np.all(leaf>=0),'incomplete geometry; do not reduceC or fabricate rows'
    np.savez(folder/'tree_projection.npz',**arrays);occ=[]
    for c in range(32):
        use=leaf==c;occ.append({'leaf':c,'rows':int(use.sum()),'distinct_groups':int(len(np.unique(groups[use]))),'old_synthetic_rows':int(np.sum(use&(origins=='oldFIT'))),'PG_TRAIN_rows':int(np.sum(use&(origins=='pg19train'))),'balanced_count_guarantee':False})
    save(folder/'tree.json',{'nodes':meta,'FIT_recipe':'FIT120_plus_pg19train16_one_refresh','runtime_FIT_occupancy':{'32':occ},'training_inputs_only':True,'iterations':1});return leaf
def fixed_fit(num,fma,inputs,y,eta,lam,family,factor_eta=None,q=512,make_factor=True):
    """Frozen fixed penalties/role; caller supplies common full union targets."""
    As,prior,cross,stats,yn=core.fitted_slopes(inputs,y,eta if family=='PE_RRR' else factor_eta,lam)
    model=old.lower(num,As,inputs['d'],yn,np.ones(len(yn)),32) if family=='PE_RRR' else None
    factor=None;info=None
    if make_factor:
        M=core.target_matrix(As,inputs,cross,stats);U,info=core.output_basis(M,maxq=q);B=np.stack([(A@U).T for A in As]);factor=core.calibrate(num,fma,core.lower_factors(num,U,B,inputs['d'],'W8','host_floatlatent','identity'),inputs['d'],yn)
    penalty={'pooled_dimensionless_eta':eta if family=='PE_RRR' else factor_eta,'local_dimensionless_lambda':lam if family=='PE_RRR' else 'infinity','pooled_trace':float(np.trace(inputs['pcov'])),'pooled_beta':None if (eta if family=='PE_RRR' else factor_eta)=='infinity' else float((eta if family=='PE_RRR' else factor_eta)*np.trace(inputs['pcov'])/inputs['pcov'].shape[0]),'leaf_beta':[None if lam=='infinity' else float(lam*np.trace(s['cov'])/inputs['pcov'].shape[0]) for s in stats] if family=='PE_RRR' else None,'leaf_solve':[old.solve(s['cov'],s['cross'],lam,prior,factor=s['factor'])[1] for s in stats] if family=='PE_RRR' else None}
    penalty['pooled_solve']=old.solve(inputs['pcov'],cross,eta if family=='PE_RRR' else factor_eta,factor=inputs['pfactor'])[1]
    return model,factor,{'output_basis':info,'penalties_recomputed_from_union':penalty,'explicit_allrows':inputs['explicit_union_allrows'],'target_rows':len(yn)}
