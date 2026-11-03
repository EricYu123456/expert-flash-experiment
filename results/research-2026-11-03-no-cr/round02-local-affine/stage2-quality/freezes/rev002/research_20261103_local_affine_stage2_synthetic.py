#!/usr/bin/env python3
"""Generated inputs/weights only. Never opens real model/fit/corpus payload."""
import argparse
import io
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage2_worker as w
np,old,core=w.np,w.old,w.core
def bits(a,b):assert a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.array_equal(a.view(np.uint32),b.view(np.uint32))
def main(activation,out):
    rng=np.random.default_rng(20261103);fma=core.FMA(w.FROZEN1/'fma.so');num=w.GuardedNumeric(fma,activation);reference=old.Numeric(core.FROZEN0/'numeric.so');checks=[]
    try:
        for mid,I,l,es in [(1,1408,1,[14,40]),(2,1024,0,[6,17])]:
            H=2048;N=3;x=rng.normal(0,.1,(N,H)).astype(np.float32)
            weights={role:num.quantize(rng.normal(0,.015,shape).astype(np.float32)) for role,shape in [('gate',(I,H)),('up',(I,H)),('down',(H,I))]}
            q,sx=num.quantize(x);actual=num.dots(q,weights['gate'][0]);assert np.array_equal(actual,q.astype(np.int64)@weights['gate'][0].astype(np.int64).T)
            y=old.teacher(num,x,weights);bits(y,old.teacher(num,x,weights,batch=1));bits(y,old.teacher(reference,x,weights))
            tree={j:{'mean':np.zeros(H,np.float32),'normal':np.eye(1,H,0,dtype=np.float32)[0],'threshold':0.} for j in range(63)}
            P=np.eye(H,dtype=np.float32)[:128];sigma=np.ones(128)
            d=old.design(num,x,tree,32,P,sigma,128,'normal')
            local={'qmaps':rng.integers(-127,128,(32,H,128),dtype=np.int8),'scales':np.full((32,H),.0001,np.float32),'bias':rng.normal(0,.01,(32,H)).astype(np.float32),'unsupported':[]}
            U=rng.normal(0,.01,(H,512));B=rng.normal(0,.01,(32,512,128));factor=core.lower_factors(num,U,B,d,'W8','host_floatlatent','identity');factor['unsupported']=np.empty(0,np.int32);factor['bias']=local['bias'].copy()
            reg=w.Registry.__new__(w.Registry);reg.model=mid;reg.K=6 if mid==1 else 8;reg.num=num;reg.fma=fma;reg.ops={(l,e):{} for e in es};reg.trees={l:(tree,P,sigma,{})};reg.weights={(l,e):weights for e in es};reg.locals={(l,e):local for e in es};reg.factors={(l,e):factor for e in es}
            route=es+[e for e in range(64) if e not in es][:reg.K-2];ids=np.tile(np.asarray(route,np.int32),(N,1))
            for mode in [3,4,5,6]:
                got=reg.evaluate(mode,l,x,ids)
                expected=y if mode in [3,4] else old.predict(num,local,d) if mode==5 else core.predict(num,fma,factor,d)
                for t,k,e,flag,row in got:assert flag==0;bits(row,expected[t])
                # Full binary framing through the exact worker server, with
                # real reused numerical helpers and generated coefficients.
                h=w.header(kind=1,mode=mode,arm=mode,model=mid,layer=l,T=N,K=reg.K,request=1,run=9,context=10+mode,decode=1,position=128,token_phase=2,slots=2*N,bytes=4*N*(H+reg.K))
                stream=io.BytesIO();w.serve(reg,io.BytesIO(w.pack(h)+x.tobytes()+ids.tobytes()),stream);blob=stream.getvalue();ready=w.unpack(blob[:w.WIRE.size]);reply=w.unpack(blob[w.WIRE.size:2*w.WIRE.size]);assert ready['kind']==3 and reply=={**h,'kind':2,'bytes':2*N*(16+4*H)}
                pos=2*w.WIRE.size
                for t,k,e,flag,row in got:
                    assert struct.unpack('<4I',blob[pos:pos+16])==(t,k,e,flag);pos+=16;bits(np.frombuffer(blob[pos:pos+4*H],np.float32),row);pos+=4*H
                assert pos==len(blob)
                for t in range(N):
                    single=reg.evaluate(mode,l,x[t:t+1],ids[t:t+1])
                    for _,k,e,flag,row in single:bits(row,expected[t])
            checks.append({'model_adapter':mid,'I':I,'full_H':H,'teacher_all_I64_dot_reference':True,'pinned_full_I_activation_bits':True,'B1_batch_IPC_all_arms_bits':True})
        # FMA full-H control uses identity input, never an implicit PCA.
        df=old.design(num,x,tree,32,P,sigma,H,'identity');fg=core.lower_factors(num,U,rng.normal(0,.005,(1,512,H)),df,'W8','host_floatlatent','identity');fg['unsupported']=np.empty(0,np.int32);reg.factors={(l,e):fg for e in es}
        expected=core.predict(num,fma,fg,df)
        for t,k,e,flag,row in reg.evaluate(6,l,x,ids):bits(row,expected[t])
        # Unsupported leaf is a declared canonical fallback, distinct from
        # framing/environment errors, which must throw rather than fallback.
        local['unsupported']=list(np.unique(d['leaf']));reg.locals={(l,e):local for e in es}
        for t,k,e,flag,row in reg.evaluate(5,l,x,ids):assert flag==1;bits(row,y[t])
        bad=w.header();bad['reserved2']=1
        try:w.unpack(w.pack(bad));raise RuntimeError('bad header accepted')
        except AssertionError:pass
        result={'status':'passed','generated_synthetic_only':True,'checks':checks,'full_H_identity_factor':True,'declared_unsupported_fallback':True,'actual_numeric_guard_workers':8,'dot_audits':num.dot_audits,'wire_bytes':w.WIRE.size}
        Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:num.close();reference.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activation',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.activation,a.output)
