#!/usr/bin/env python3
"""Generated-only complete archive/worker/C++ wiring and arithmetic checks."""
import argparse
import ctypes as ct
import io
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage3_core as s
import research_20261103_local_affine_stage3_worker as sw
import research_20261103_local_affine_stage3_run as run
np,old,core,w=s.np,s.old,s.core,s.w
def bits(a,b):assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint32),b.view(np.uint32))
def main(folder):
    folder=Path(folder);folder.mkdir();rng=np.random.default_rng(20261103);fma=core.FMA(w.FROZEN1/'fma.so');num=w.GuardedNumeric(fma,s.F2/'strict-activation');lib=ct.CDLL(str(Path(__file__).parent/'numeric.so'));H=2048;I=17;Q=512;N=64
    try:
        for width in [1,7,8,15,16,17,127,128,129,511,512,1024,1408,2048]:
            q=rng.integers(-128,128,(4,width),dtype=np.int8);ww=rng.integers(-128,128,(7,width),dtype=np.int8);q[0]=-128;ww[0]=-128;actual=s.ffi_dots(lib,q,ww);assert np.array_equal(actual,q.astype(np.int64)@ww.astype(np.int64).T)
            x=(rng.standard_normal((4,width))*.1).astype(np.float32);coef=(rng.standard_normal((7,width))*.1).astype(np.float32);a=np.empty((4,7),np.float32);assert lib.ls3_fma(old.Numeric.ptr(x),old.Numeric.ptr(coef),4,width,7,4,old.Numeric.ptr(a))==0;bits(a,fma.dot(x,coef));proj=np.empty((4,7),np.float32);assert lib.ls3_projection(old.Numeric.ptr(x),old.Numeric.ptr(coef),4,width,7,old.Numeric.ptr(proj))==0;bits(proj,num.project(x,coef))
            qout=np.empty(x.shape,np.int8);sc=np.empty(4,np.float32);assert lib.ls3_quant(old.Numeric.ptr(x),4,width,old.Numeric.ptr(qout),old.Numeric.ptr(sc))==0;qr,sr=num.quantize(x);assert np.array_equal(qout,qr);bits(sc,sr)
        zero=np.zeros((2,17),np.float32);qz,sz=num.quantize(zero);assert np.all(qz==0) and np.all(sz==1)
        x=(rng.standard_normal((N,H))*.1).astype(np.float32);tree={}
        for node in range(63):
            normal=np.zeros(H,np.float32) if node<31 else None
            if normal is not None:normal[(node+1).bit_length()-1]=1
            tree[node]={'normal':normal,'mean':np.zeros(H,np.float32),'threshold':0.,'depth':(node+1).bit_length()-1}
        P=(rng.standard_normal((128,H))*.01).astype(np.float32);sigma=np.ones(128);x[0,:5]=-1;x[1,:5]=[-1,-1,-1,-1,1];x[2,:5]=[-1,-1,-1,1,-1];x[3,:5]=1
        d=old.design(num,x,tree,32,P,sigma,128,'normal');dg=old.design(num,x,tree,32,P,sigma,H,'identity');assert d['leaf'][:4].tolist()==[0,1,2,31]
        def weights():
            def m(rows,cols):return rng.integers(-127,128,(rows,cols),dtype=np.int8),np.full(rows,.0003,np.float32)
            return {'gate':m(I,H),'up':m(I,H),'down':m(H,I)}
        ww=weights();qx,sx=num.quantize(x);g=num.linear(qx,sx,*ww['gate']);u=num.linear(qx,sx,*ww['up']);target=np.concatenate([g,u],axis=1);y=old.teacher(num,x,ww)
        inp=s.r.union_statistics(d,np.ones(N,np.bool_),np.arange(N),False);fitfolder=folder/'fit';fitfolder.mkdir();m,report=s.fit_one(num,fma,inp,target,y,ww,[1],fitfolder);assert report['GU_actual_FMA_prediction_passes']==report['Down_prediction_passes']==1;path=folder/'NL.npz';mh=s.save_model(path,m);loaded=s.load_model(path);assert s.model_hash(loaded)==mh;changed={**m,'unsupported':np.asarray([2],np.int32)};assert s.model_hash(changed)!=mh
        allpred=s.predict_nl(num,fma,m,core.subset(d,np.arange(4)),ww);one=np.concatenate([s.predict_nl(num,fma,loaded,core.subset(d,np.asarray([i])),ww) for i in range(4)]);bits(allpred,one)
        class Dummy(sw.Registry):
            def __init__(self):
                self.model=1;self.name='deepseek';self.K=6;self.num=num;self.fma=fma;self.trees={1:(tree,P,sigma,None)};self.ops={};self.weights={};self.locals={};self.factors={};self.nl={}
                for index,e in enumerate([14,40]):
                    key=1,e;self.ops[key]={'factor_point':{'guard':100}};self.weights[key]=ww if index==0 else weights();self.nl[key]=loaded;self.locals[key]={'qmaps':rng.integers(-3,4,(32,H,128),dtype=np.int8),'scales':np.full((32,H),.001,np.float32),'bias':(rng.standard_normal((32,H))*.001).astype(np.float32),'unsupported':[1]};R=H if index else 128;C=1 if index else 32
                    self.factors[key]={'B':rng.integers(-3,4,(C,Q,R),dtype=np.int8),'Bs':np.full((C,Q),.001,np.float32),'U':rng.integers(-3,4,(H,Q),dtype=np.int8),'Us':np.full(H,.001,np.float32),'bias':(rng.standard_normal((32,H))*.001).astype(np.float32),'unsupported':np.asarray([2],np.int32),'meta':{'family':'G_RRR' if index else 'PE_RRR','format':'W8','path':'host_floatlatent','gauge':'identity','q':Q}}
        reg=Dummy();exp=folder/'export';lp=s.export_layer(exp,reg,1);lm=s.read(lp)
        for o in lm['operators']:
            for key,a in [('ng',m['g']),('nu',m['u']),('gb',m['gb']),('ub',m['ub']),('yb',m['yb'])]:o[key]=s.dump(exp,f'e{o["expert"]}.nl.{key}',a)
        s.save(lp,lm);ids=np.tile(np.asarray([14,1,40,2,3,4],np.int32),(32,1));alpha=np.tile(np.asarray([.2,.1,.3,.15,.15,.1],np.float32),(32,1));sample={'x':s.dump(exp,'sample.x',x[:32]),'ids':s.dump(exp,'sample.ids',ids),'alpha':s.dump(exp,'sample.alpha',alpha)};plan={'phase':'GENERATED_AUDIT','generated_only':True,'layer_manifest':str(lp),'arms':[3,5,6,7],**sample,'output_prefix':str(exp/'raw'),'output':str(exp/'audit.json')};s.save(exp/'plan.json',plan);p=subprocess.run([str(Path(__file__).parent/'numeric'),'audit',str(exp/'plan.json')],capture_output=True,text=True);(exp/'direct.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
        for mode in [3,5,6,7]:
            for e in [14,40]:
                key=1,e;weight=reg.weights[key]
                if mode==3:expected=old.teacher(num,x[:4],weight)
                elif mode==5:expected=old.predict(num,reg.locals[key],core.subset(d,np.arange(4)))
                elif mode==6:expected=core.predict(num,fma,reg.factors[key],core.subset(dg if e==40 else d,np.arange(4)))
                else:expected=s.predict_nl(num,fma,m,core.subset(d,np.arange(4)),weight)
                unsupported=[] if mode==3 else reg.factors[key]['unsupported'] if mode==6 else reg.locals[key]['unsupported'];bad=np.isin(d['leaf'][:4],unsupported)
                if bad.any():expected[bad]=old.teacher(num,x[:4][bad],weight)
                for row in range(4):bits(np.fromfile(exp/f'raw.r{row}.e{e}.a{mode}.f32',np.float32),expected[row])
        # Actual NL archive→prediction→new mode8 binary server, zero-mask too.
        for mode in [7,8]:
            for nohits in [False,True]:
                ii=ids[:4].copy()
                if nohits:ii=np.tile(np.arange(1,7,dtype=np.int32),(4,1))
                rows=reg.evaluate(mode,1,x[:4],ii);h=w.header(kind=1,mode=mode,arm=mode,request=1,run=1,context=mode,subrun=7,decode=1,model=1,layer=1,T=4,K=6,slots=len(rows),bytes=4*4*(H+6));stream=io.BytesIO();sw.serve(reg,io.BytesIO(w.pack(h)+x[:4].tobytes()+ii.tobytes()),stream);blob=stream.getvalue();reply=w.unpack(blob[w.WIRE.size:2*w.WIRE.size]);assert reply['mode']==mode and reply['slots']==len(rows);off=2*w.WIRE.size
                for t,k,e,flag,yy in rows:assert w.struct.unpack('<4I',blob[off:off+16])==(t,k,e,flag);off+=16;assert blob[off:off+H*4]==yy.tobytes();off+=H*4
                assert off==len(blob)
        for mutate in ['duplicate','short','unsupportedmode']:
            h=w.header(kind=1,mode=7,arm=7,request=1,run=1,context=7,subrun=7,decode=1,model=1,layer=1,T=4,K=6,slots=8,bytes=4*4*(H+6));frame=w.pack(h)+x[:4].tobytes()+ids[:4].tobytes()
            if mutate=='duplicate':frame+=frame
            elif mutate=='short':frame=frame[:-3]
            else:h['mode']=9;h['arm']=9;frame=w.pack(h)+frame[w.WIRE.size:]
            try:sw.serve(reg,io.BytesIO(frame),io.BytesIO());raise RuntimeError('expected failclosed')
            except (AssertionError,RuntimeError) as ex:assert str(ex)!='expected failclosed'
        oldvalues=rng.standard_normal((2,16,4))*.05+2;nlvalues=oldvalues[:,:,1]+rng.standard_normal((2,16))*.01;summary=s.development_summary(oldvalues,nlvalues,list(range(16)));draws=np.random.default_rng(2026110303).integers(0,16,(10000,16));delta=nlvalues-oldvalues[:,:,1];boot=np.stack([np.asarray([sum(delta[m,j] for j in row)/16 for row in draws]) for m in range(2)]);hat=delta.mean(axis=1);sd=boot.std(axis=1,ddof=1);cut=float(np.quantile(np.max(np.abs((boot-hat[:,None])/sd[:,None]),axis=0),.95,method='linear'));assert abs(cut-summary['cutoff'])<1e-12
        constant=s.development_summary(np.zeros((2,16,4)),np.zeros((2,16)),list(range(16)));assert constant['primary_family_inconclusive'];missing=nlvalues.copy();missing[0,0]=np.nan;assert s.development_summary(oldvalues,missing,list(range(16)))['primary_family_inconclusive'];large=s.development_summary(oldvalues,nlvalues+1000,list(range(16)));json.dumps(large,allow_nan=False)
        jobs,ctx=run.context_plan(list(range(16)));assert len(ctx)==39 and jobs[0]['contexts']['NL']!=jobs[0]['contexts']['D'] and ctx[str(jobs[-1]['contexts']['zeroB'])]['zeroB_reserved_integrity_only']
        # Metadata selection is independent of values/targets and stable under
        # JSON key insertion order, with identities frozen before array reads.
        pp=[{'row':i,'origin':'generated','phase':'prefix','context':1,'input_sha256':str(i)} for i in range(40)];assert s.bench_selection(pp)==s.bench_selection([{k:p[k] for k in reversed(p)} for p in pp])
        fixture=folder/'identity.bin';fixture.write_bytes(b'generated');identity=s.identities({str(fixture):s.sha(fixture)});fixture.write_bytes(b'changed')
        try:s.identities({str(fixture):identity[str(fixture)]['sha256']});raise RuntimeError('expected identity failure')
        except AssertionError:pass
        result={'status':'passed','generated_only':True,'real_weights_arrays_targets_books_read':False,'I64_all_signed_extremes_and_tails':True,'frozen_FMA_and_FP64_P_bits':True,'A8_frozen_bits':True,'eta0_rank_deficient_fit_archive_B1_batch':True,'actual_raw_GU_prediction_passes':report['GU_actual_FMA_prediction_passes'],'direct_CXX_B_C_D_NL_allrows_and_Gcentered_integerfirst_floatlift':True,'inherited_support_fallback':True,'NL7_COPY_NL8_distinct_binary_roundtrip_and_zero_mask':True,'duplicate_short_badmode_failclosed':True,'paired_twoendpoint_bootstrap_independent_loop':True,'zero_missing_family_preserved':True,'largefinite_report_serializes':True,'stable_provenance_selection_and_identitymutation_rejected':True};s.save(folder/'receipt.json',result);print(json.dumps(result),flush=True)
    finally:num.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();main(a.folder)
