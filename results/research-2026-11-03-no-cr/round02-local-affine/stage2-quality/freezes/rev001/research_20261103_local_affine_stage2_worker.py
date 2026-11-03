#!/usr/bin/env python3
"""Persistent numerical worker. No driver imports; no payload reads at import."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
ROOT=Path('/home/ericyu/expert_flash_experiment')
ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine'
FROZEN1=ROUND/'stage1-controls/freezes/rev003'
sys.path.insert(0,str(FROZEN1))
import research_20261103_local_affine_stage1_controls_core as core
import numpy as np
old,pipe=core.old,core.pipe
PROTOCOL_SHA='0e59b5cc4d3fad695d500a1b50897864fa048ba682d51702914d66b9075e57c6'
WIRE=struct.Struct('<4I3Q2IQ7IQ2IQ2I');assert WIRE.size==116
FIELDS='magic version kind mode request run context subrun arm decode micro model layer T H K slots bytes status reserved position token_phase reserved2'.split()
MAX_BYTES=16<<20

def header(**kw):
    h=dict.fromkeys(FIELDS,0);h.update(magic=0x3251414c,version=1,H=2048);h.update(kw);return h
def pack(h):return WIRE.pack(*(h[k] for k in FIELDS))
def unpack(b):
    assert len(b)==WIRE.size;h=dict(zip(FIELDS,WIRE.unpack(b)))
    assert h['magic']==0x3251414c and h['version']==1 and h['reserved']==h['reserved2']==0 and h['token_phase'] in [0,1,2] and 0<=h['bytes']<=MAX_BYTES
    return h
def exact(stream,n,eof=False):
    out=bytearray()
    while len(out)<n:
        part=stream.read(n-len(out))
        if not part:
            if eof and not out:return None
            raise RuntimeError('binary worker truncated frame')
        out.extend(part)
    return bytes(out)

class GuardedNumeric(old.Numeric):
    def __init__(self,fma,activation):
        self.strict=fma.lib.lac_environment
        self.guard();super().__init__(core.FROZEN0/'numeric.so');self.activation_binary=Path(activation);self.guard()
    def guard(self):assert self.strict()==8,'all actual numeric OMP workers require x87/SSE RN FTZ0 DAZ0'
    def checked(self,name,*args,**kwargs):
        self.guard();out=getattr(super(),name)(*args,**kwargs);self.guard();return out
    def quantize(self,*a,**k):return self.checked('quantize',*a,**k)
    def dots(self,*a,**k):return self.checked('dots',*a,**k)
    def rescale(self,*a,**k):return self.checked('rescale',*a,**k)
    def swiglu(self,*a,**k):return self.checked('swiglu',*a,**k)
    def add_bias(self,*a,**k):return self.checked('add_bias',*a,**k)
    def project(self,*a,**k):return self.checked('project',*a,**k)
    def tree_project(self,*a,**k):return self.checked('tree_project',*a,**k)

def authorize(freeze,review,phase,registry):
    freeze=Path(freeze);r=json.loads(freeze.read_text());a=json.loads(Path(review).read_text())
    assert a['status']=='approved' and a['phase']==phase and phase in ['TOOLING','DEVELOPMENT','QUALITY']
    assert a['protocol_sha256']==r['protocol_sha256']==PROTOCOL_SHA
    assert a['source_revision']==r['revision'] and a['source_freeze_sha256']==old.sha(freeze)
    assert a['registry_sha256']==old.sha(registry),'explicit approved registry identity'
    for p,want in r['files'].items():assert old.sha(p)==want,p
    return r

class Registry:
    H=2048
    def __init__(self,model,registry,num,fma):
        self.model=model;self.K=6 if model==1 else 8;self.name='deepseek' if model==1 else 'olmoe';self.num=num;self.fma=fma
        r=json.loads(Path(registry).read_text());assert r['protocol_sha256']==PROTOCOL_SHA
        self.ops={(o['layer'],o['expert']):o for o in r['operators'] if o['model']==self.name}
        assert len(self.ops)==4
        deps={}
        for receipt in r['payload_manifests']:
            assert old.sha(receipt['path'])==receipt['sha256'],'approved payload manifest identity'
            for path,digest in json.loads(Path(receipt['path']).read_text())['files'].items():
                assert path not in deps or deps[path]==digest,'conflicting payload identities'
                deps[path]=digest
        def verify_file(p):
            p=Path(p);assert str(p) in deps and old.sha(p)==deps[str(p)],str(p)
        self.trees={};self.weights={};self.locals={};self.factors={}
        for (layer,e),o in self.ops.items():
            tf=Path(o['preprocessing_path'])
            if layer not in self.trees:
                for name in ['tree.json','tree_projection.npz']:verify_file(tf/name)
                self.trees[layer]=pipe.tree_load(tf);self.tree_paths=getattr(self,'tree_paths',{});self.tree_paths[layer]=tf
            else:assert self.tree_paths[layer]==tf,'same layer requires one shared tree/P'
            wf=Path(o['canonical_weights_path'])
            for name in ['weights.json','gate.i8','gate.scales.f32','up.i8','up.scales.f32','down.i8','down.scales.f32']:verify_file(wf/name)
            self.weights[layer,e]=pipe.load_weights(wf,1408 if model==1 else 1024)
            verify_file(o['local_model_path']);m=old.load_model(o['local_model_path'])
            assert old.array_hash(m['qmaps'],m['scales'],m['bias'],np.asarray(m['unsupported'],np.int32))==o['local_model_hash']
            self.locals[layer,e]=m
            fp=o['factor_point'];verify_file(fp['model_path']);f=core.load(fp['model_path']);assert core.model_hash(f)==fp['model_hash']
            assert f['meta']['format']=='W8' and f['meta']['path']=='host_floatlatent' and f['meta']['q']==512
            assert f['meta']['family']==fp['family'];self.factors[layer,e]=f
        # Immutable arrays; only request-owned inputs/designs/results are mutable.
        def readonly(obj):
            if isinstance(obj,np.ndarray):obj.setflags(write=False)
            elif isinstance(obj,dict):
                for v in obj.values():readonly(v)
            elif isinstance(obj,(list,tuple)):
                for v in obj:readonly(v)
        for obj in [self.trees,self.weights,self.locals,self.factors]:readonly(obj)
    def observe(self,epoch,x,ids):
        # Explicit owned-input/provenance extension point. No collector is
        # enabled by this source: future refresh requires a separate release.
        # x/ids here are detached copies and epoch carries position/phase.
        pass
    def pilots(self,layer):return {e for l,e in self.ops if l==layer}
    def canonical(self,layer,e,rows,q,sx):
        # Frozen teacher arithmetic, with its common complete-H input A8
        # factored out once per layer request and reused over selected experts.
        w=self.weights[layer,e];g=self.num.linear(q[rows],sx[rows],*w['gate']);u=self.num.linear(q[rows],sx[rows],*w['up'])
        a=self.num.swiglu(g,u);qa,sa=self.num.quantize(a)
        return self.num.linear(qa,sa,*w['down'])
    def evaluate(self,mode,layer,x,ids):
        self.num.guard();pairs=[(t,k,int(e)) for t in range(len(x)) for k,e in enumerate(ids[t]) if int(e) in self.pilots(layer)]
        result={};designs={};qx=sx=None
        if mode in [3,4] or pairs:qx,sx=self.num.quantize(x)
        tree,P,sigma,_=self.trees[layer]
        for e in sorted({p[2] for p in pairs}):
            own=[p for p in pairs if p[2]==e];rows=np.array([p[0] for p in own],np.int64);flags=np.zeros(len(rows),np.uint32)
            if mode in [3,4]:y=self.canonical(layer,e,rows,qx,sx)
            else:
                model=self.locals[layer,e] if mode==5 else self.factors[layer,e]
                family='PE_RRR' if mode==5 else model['meta']['family']
                if family not in designs:designs[family]=old.design(self.num,x,tree,32,P,sigma,128 if family=='PE_RRR' else 2048,'normal' if family=='PE_RRR' else 'identity')
                d=core.subset(designs[family],rows)
                y=old.predict(self.num,model,d) if mode==5 else core.predict(self.num,self.fma,model,d)
                unsupported=np.isin(d['leaf'],model['unsupported']);flags[unsupported]=1
                if unsupported.any():y[unsupported]=self.canonical(layer,e,rows[unsupported],qx,sx)
            assert y.shape==(len(rows),self.H) and y.dtype==np.float32 and np.isfinite(y).all()
            for slot,row,flag in zip(own,y,flags):result[slot]=(int(flag),row.copy())
        self.num.guard();return [(t,k,e,*result[(t,k,e)]) for t,k,e in pairs]

class SyntheticRegistry:
    H=2048
    def __init__(self,model):self.model=model;self.K=6 if model==1 else 8
    def pilots(self,layer):
        return ({1:{40,14},13:{56,10}} if self.model==1 else {0:{6,17},8:{3,9}}).get(layer,set())
    def evaluate(self,mode,layer,x,ids):
        return [(t,k,int(e),0,np.add(x[t],np.float32(int(e)/64),dtype=np.float32)) for t in range(len(x)) for k,e in enumerate(ids[t]) if int(e) in self.pilots(layer)]
    def observe(self,epoch,x,ids):pass

def serve(registry,instream,outstream,fault=None):
    outstream.write(pack(header(kind=3,model=registry.model)));outstream.flush();serial=0;seen=set()
    while True:
        raw=exact(instream,WIRE.size,eof=True)
        if raw is None:return
        h=unpack(raw);assert h['kind']==1 and h['status']==0 and h['mode'] in [3,4,5,6] and h['arm']==h['mode']
        assert h['request']==serial+1 and h['model']==registry.model and h['H']==registry.H and h['K']==registry.K
        assert 1<=h['T']<=128 and h['micro']==0 and h['run']>0 and h['context']>0 and h['decode']>0
        assert registry.pilots(h['layer']) and h['bytes']==4*h['T']*(h['H']+h['K'])
        epoch=tuple(h[k] for k in ['run','context','arm','subrun','decode','micro','layer','position','token_phase']);assert epoch not in seen
        # Keep only epoch identities, never graph aliases or activations.
        seen.add(epoch);serial=h['request'];data=exact(instream,h['bytes']);cut=4*h['T']*h['H']
        x=np.frombuffer(data[:cut],'<f4').reshape(h['T'],h['H']).copy();ids=np.frombuffer(data[cut:],'<i4').reshape(h['T'],h['K']).copy()
        assert np.isfinite(x).all() and np.all((ids>=0)&(ids<64))
        assert all(len(np.unique(row))==h['K'] for row in ids)
        registry.observe(dict(h),x.copy(),ids.copy())
        expected=[(t,k,int(e)) for t in range(h['T']) for k,e in enumerate(ids[t]) if int(e) in registry.pilots(h['layer'])]
        assert h['slots']==len(expected)
        rows=registry.evaluate(h['mode'],h['layer'],x,ids)
        assert [(t,k,e) for t,k,e,_,_ in rows]==expected
        payload=b''.join(struct.pack('<4I',t,k,e,flag)+np.ascontiguousarray(y,'<f4').tobytes() for t,k,e,flag,y in rows)
        reply={**h,'kind':2,'bytes':len(payload),'slots':len(rows)}
        if fault=='die':os._exit(42)
        if fault=='misepoch':reply['decode']+=1
        if fault=='duplicate' and len(rows)>1:
            stride=16+4*h['H'];payload=payload[:stride]+payload[:stride]+payload[2*stride:]
        if fault=='nonfinite' and rows:
            payload=payload[:16]+struct.pack('<f',float('nan'))+payload[20:]
        if fault=='short':outstream.write(pack(reply)[:37]);outstream.flush();return
        outstream.write(pack(reply));outstream.write(payload);outstream.flush()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=int,required=True,choices=[1,2]);ap.add_argument('--synthetic',action='store_true');ap.add_argument('--fault',choices=['die','misepoch','duplicate','nonfinite','short']);ap.add_argument('--registry');ap.add_argument('--freeze');ap.add_argument('--review');ap.add_argument('--phase');ap.add_argument('--activation');a=ap.parse_args()
    num=None
    try:
        if a.synthetic:reg=SyntheticRegistry(a.model)
        else:
            assert not a.fault;authorize(a.freeze,a.review,a.phase,a.registry)
            fma=core.FMA(FROZEN1/'fma.so');num=GuardedNumeric(fma,a.activation)
            reg=Registry(a.model,a.registry,num,fma)
        serve(reg,sys.stdin.buffer,sys.stdout.buffer,a.fault)
        if num:num.close()
    except BaseException as error:
        print('Stage2 worker invalid/tooling: '+repr(error),file=sys.stderr,flush=True)
        if num:
            try:num.close()
            except BaseException:pass
        raise
