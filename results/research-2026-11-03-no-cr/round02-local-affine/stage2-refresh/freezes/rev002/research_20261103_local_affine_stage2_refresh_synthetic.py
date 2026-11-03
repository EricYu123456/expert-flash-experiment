#!/usr/bin/env python3
"""Generated book bytes/states/weights only; no actual corpus or fit payload."""
import argparse
import base64
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage2_refresh_core as r
import research_20261103_local_affine_stage2_refresh_fit as fit
from research_20261103_local_affine_stage2_refresh_worker import RecorderRegistry
np,old,core=r.np,r.old,r.core
def rejects(fn):
    try:fn()
    except (AssertionError,RuntimeError,UnicodeError):return
    raise AssertionError('invalid input accepted')
def bitsequal(a,b):assert a.dtype==b.dtype==np.float32 and a.shape==b.shape and np.array_equal(a.view(np.uint32),b.view(np.uint32))
def collection_test(folder):
    folder.mkdir();inputs=folder/'owned.bin';alpha=folder/'alpha.bin';rec=RecorderRegistry.__new__(RecorderRegistry);rec.model=1;rec.K=6;rec.num=SimpleNamespace(activation_calls=0,activation_ipc_bytes=0,activation_seconds=0);contexts={str(i+1):{'book_id':42,'arm':arm} for i,arm in enumerate([3,5,6])};rec.start_recording(inputs,contexts,11);reports=[];serial=0;alpha_frames=alpha_rows=0;last_input=None
    with alpha.open('wb') as af:
        af.write(b'LARFALP1')
        for context,arm in enumerate([3,5,6],1):
            prefix=list(range(1000,1768));gen={3:[901],5:[801,802],6:[]}[arm];reports.append({'model':1,'mode':arm,'arm':arm,'run':11,'context':context,'subrun':4,'prefix_ids':prefix,'generated_ids':gen,'stop':'EOG','EOG_not_fed':2,'decode_input_rows':len(gen),'prefix_input_rows':768,'fresh_KV':True})
            for pos,T,phase in [(p,128,0) for p in range(0,768,128)]+[(768+j,1,1) for j in range(len(gen))]:
                for layer in [1,13]:
                    serial+=1;x=np.broadcast_to((np.arange(pos,pos+T,dtype=np.float32)/1000)[:,None],(T,2048)).copy();ids=np.tile(np.array([14,40,0,1,2,3],np.int32),(T,1));a=np.tile(np.array([.1,.2,.3,.4,.5,.6],np.float32),(T,1));h=r.w.header(kind=1,mode=arm,arm=arm,request=serial,run=11,context=context,subrun=4,decode=pos//128+1 if phase==0 else 7+pos-768,micro=0,model=1,layer=layer,T=T,K=6,slots=T*2,bytes=x.nbytes+ids.nbytes,position=pos,token_phase=phase);rec.observe(h,x,ids)
                    if r.admitted(h):
                        ah={**h,'kind':4,'bytes':ids.nbytes+a.nbytes};af.write(r.w.pack(ah)+ids.tobytes()+a.tobytes());alpha_frames+=1;alpha_rows+=T;last_input=r.w.pack(h)+x.tobytes()+ids.tobytes()
    rec.finish_recording();result=folder/'result.json';r.save(result,{'status':'complete','jobs':[{'book_id':42,'immutable_prefix_ids':prefix,'runs':reports}],'alpha_frames':alpha_frames,'alpha_rows_both_layers':alpha_rows});closure=r.close_collection('deepseek',inputs,alpha,result,folder/'closed',[42]);assert all(v['rows']==771 for v in closure['layers'].values());prov=[json.loads(v) for v in (folder/'closed/layer1/provenance.jsonl').read_text().splitlines()];assert [p['arm'] for p in prov[-3:]]==[3,5,5] and [p['token_id'] for p in prov[-3:]]==[901,801,802] and prov[-3]['input_sha256']==prov[-2]['input_sha256'];assert len({p['group'] for p in prov})==1
    b=inputs.read_bytes();bad=folder/'duplicate.bin';bad.write_bytes(b+last_input);rejects(lambda:r.close_collection('deepseek',bad,alpha,result,folder/'bad_duplicate',[42]));bad=folder/'short.bin';bad.write_bytes(b[:-1]);rejects(lambda:r.close_collection('deepseek',bad,alpha,result,folder/'bad_short',[42]));bad=folder/'missing_alpha.bin';bad.write_bytes(alpha.read_bytes()[:-116-6*8]);rejects(lambda:r.close_collection('deepseek',inputs,bad,result,folder/'bad_missingalpha',[42]));wrong=bytearray(alpha.read_bytes());wrong[8+116]^=1;bad=folder/'wrong_id_alpha.bin';bad.write_bytes(wrong);rejects(lambda:r.close_collection('deepseek',inputs,bad,result,folder/'bad_alphaIDs',[42]));return {'frames':alpha_frames,'rows_per_layer':771,'identical_rows_retained':True,'fault_join_tests':True,'typed_token_book_epoch_order':True}
def main(out):
    out=Path(out);folder=out.parent/'generated';folder.mkdir();raw='αβ\r\nγδε\rζηθ\nijklm\n尾'.encode();meta={'size_bytes':len(raw),'md5_base64':base64.b64encode(r.hashlib.md5(raw).digest()).decode()};suffix,info=r.preprocess(raw,meta);text=raw.decode().replace('\r\n','\n').replace('\r','\n');mid=len(text)//2;lf=text.find('\n',mid);assert suffix==text[mid if lf<0 else lf+1:].encode();assert info['unicode_midpoint']==mid;rejects(lambda:r.preprocess(raw,{**meta,'md5_base64':'bad'}));bad=b'\xff';rejects(lambda:r.preprocess(bad,{'size_bytes':1,'md5_base64':base64.b64encode(r.hashlib.md5(bad).digest()).decode()}));collection=collection_test(folder/'collection')
    fma=core.FMA(r.w.FROZEN1/'fma.so');num=r.w.GuardedNumeric(fma,r.F2/'strict-activation');rng=np.random.default_rng(20261103)
    try:
        N,H,rank=96,16,8;x=rng.normal(size=(N,H)).astype(np.float32);tree=old.fit_tree(num,x,1);basis,sigma=old.fit_projection(x);groups=np.asarray(['oldFIT:'+str(i//3) if i<72 else 'pg19train:'+str(i//6) for i in range(N)]);origins=np.asarray(['oldFIT']*72+['pg19train']*24);leaf=r.archive_geometry(folder/'new_geometry',tree,basis[:rank],sigma[:rank],num,x,groups,origins);loaded,P,S,metadata=r.pipe.tree_load(folder/'new_geometry');d=old.design(num,x,loaded,32,P,S,rank,'normal');assert 'unsupervised_FIT_prompts' not in metadata;include=np.ones(N,np.bool_);inputs=r.union_statistics(d,include,groups,True);assert len(inputs['ix'])==N and sum(v['sum_w'] for v in inputs['leaves'])==N;rejects(lambda:r.union_statistics(d,np.arange(N)<72,groups,True));legacy=core.input_statistics(d,np.r_[np.arange(72),np.full(24,1000)],120,True);assert len(legacy['ix'])==72
        y=rng.normal(size=(N,H)).astype(np.float32);local,factor,diag=r.fixed_fit(num,fma,inputs,y,.001,.01,'PE_RRR',q=8);assert diag['penalties_recomputed_from_union']['pooled_beta']==.001*np.trace(inputs['pcov'])/rank and diag['penalties_recomputed_from_union']['pooled_solve']['rank']>0;bank=folder/'models';bank.mkdir();lp,lh=old.archive_model(bank,local);fp,fh=core.archive(bank,factor);lm,fm=old.load_model(lp),core.load(fp);assert core.model_hash(fm)==fh;bitsequal(old.predict(num,local,d),old.predict(num,lm,d));bitsequal(core.predict(num,fma,factor,d),core.predict(num,fma,fm,d));bitsequal(core.predict(num,fma,fm,d),np.concatenate([core.predict(num,fma,fm,core.subset(d,np.array([i]))) for i in range(N)]));unbiased=core.predict(num,fma,fm,d,bias=False)
        for c in range(32):
            ix=np.flatnonzero(d['leaf']==c);assert len(ix)>0;bitsequal(fm['bias'][c],num.bias(y[ix],unbiased[ix],np.ones(len(ix))))
        dg=old.design(num,x,loaded,32,P,S,H,'identity');ig=r.union_statistics(dg,include,groups,False);_,gf,gd=r.fixed_fit(num,fma,ig,y,.001,.01,'G_RRR',factor_eta=.1,q=8);assert gd['penalties_recomputed_from_union']['pooled_solve']['rank']>0 and gd['penalties_recomputed_from_union']['pooled_beta']==.1*np.trace(ig['pcov'])/H
        weights={role:num.quantize(rng.normal(0,.1,shape).astype(np.float32)) for role,shape in [('gate',(32,H)),('up',(32,H)),('down',(H,32))]};target=old.teacher(num,x[:4],weights);audit=fit.compare_targets(num,x[:4],weights,target,batch=3);assert audit['mismatch_elements']==0 and audit['audit_expert_row_queries']==4;wrong=target.copy();wrong[0,0]=np.nextafter(wrong[0,0],np.float32(np.inf));bad=fit.compare_targets(num,x[:4],weights,wrong,batch=2);assert bad['mismatch_elements']==1 and bad['mismatch_rows']==1
        result={'status':'passed','generated_only':True,'no_actual_payload_or_download':True,'Unicode_generation_MD5_preprocessing':True,'collection':collection,'explicit_union_mask_includes_new_groups':True,'legacy_filter_negative_control':True,'new_geometry_archive_roundtrip':True,'fixed_PE_G_trace_penalties_rank_conditioning':True,'archive_B1_batch_actual_bias':True,'semantic_oldlabel_bit_audit_and_mismatch_detection':True,'old_helpers_unchanged':True};r.save(out,result);print(json.dumps(result),flush=True)
    finally:num.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
