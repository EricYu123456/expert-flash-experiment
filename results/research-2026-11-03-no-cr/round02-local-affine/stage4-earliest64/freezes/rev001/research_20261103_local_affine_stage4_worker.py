"""Explicit earliest-only all64 registry. No old four-operator constructor."""
import argparse
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage4_core as s
np,w,old,core,pipe=s.np,s.w,s.old,s.core,s.pipe

class Registry:
    H=2048
    def __init__(self,model,path,num):
        self.model=model;self.name='deepseek' if model==1 else 'olmoe';self.K=6 if model==1 else 8;self.num=num
        j=s.read(path);assert j['protocol_sha256']==s.EXPECTED and j['version']=='first64-v1'
        manifest=j['payload_manifest'];assert s.sha(manifest['path'])==manifest['sha256']
        deps=s.read(manifest['path'])['files'];s.identities(deps)
        m=j['models'][self.name];self.layer=1 if model==1 else 0;assert m['layer']==self.layer and m['I']==(1408 if model==1 else 1024)
        tf=Path(m['geometry']);assert all(str(tf/n) in deps for n in ['tree.json','tree_projection.npz'])
        self.tree,self.P,self.sigma,_=pipe.tree_load(tf);assert self.P.shape==(128,2048)
        self.weights={};self.locals={};self.ops={}
        for o in m['operators']:
            e=o['expert'];assert e not in self.ops and 0<=e<64;self.ops[e]=o
            wf=Path(o['weights_path'])
            assert all(str(wf/n) in deps for n in ['weights.json','gate.i8','gate.scales.f32','up.i8','up.scales.f32','down.i8','down.scales.f32'])
            self.weights[e]=pipe.load_weights(wf,m['I'])
            assert str(Path(o['local_model_path'])) in deps
            lm=old.load_model(o['local_model_path']);s.validate_local(lm);assert s.local_hash(lm)==o['local_model_hash'];self.locals[e]=lm
        assert set(self.ops)==set(range(64))
        def readonly(x):
            if isinstance(x,np.ndarray):x.setflags(write=False)
            elif isinstance(x,dict):
                for a in x.values():readonly(a)
            elif isinstance(x,(list,tuple)):
                for a in x:readonly(a)
        readonly(self.weights);readonly(self.locals);readonly(self.tree);readonly(self.P);readonly(self.sigma)
        self.contextmap=None;self.events={};self.current=None

    def pilots(self,layer):return set(self.weights) if layer==self.layer else set()

    def evaluate(self,mode,layer,x,ids):
        assert mode in [9,10,11,12] and layer==self.layer
        self.num.guard();pairs=[(t,k,int(e)) for t in range(len(x)) for k,e in enumerate(ids[t]) if int(e) in self.pilots(layer)]
        if not pairs:return []
        design=None;qx=sx=None;results={}
        if mode in [11,12]:design=old.design(self.num,x,self.tree,32,self.P,self.sigma,128,'normal')
        for e in sorted({p[2] for p in pairs}):
            own=[p for p in pairs if p[2]==e];ix=np.asarray([v[0] for v in own],np.int64);weights=self.weights[e];flags=np.zeros(len(ix),np.uint32)
            if mode in [9,10]:
                if qx is None:qx,sx=self.num.quantize(x)
                g=self.num.linear(qx[ix],sx[ix],*weights['gate']);u=self.num.linear(qx[ix],sx[ix],*weights['up']);a=self.num.swiglu(g,u);qa,sa=self.num.quantize(a);y=self.num.linear(qa,sa,*weights['down'])
            else:
                d=core.subset(design,ix);m=self.locals[e];flags=np.isin(d['leaf'],m['unsupported']).astype(np.uint32);y=old.predict(self.num,m,d)
                if flags.any():y[flags.astype(bool)]=old.teacher(self.num,x[ix[flags.astype(bool)]],weights)
            assert y.shape==(len(ix),2048) and y.dtype==np.float32 and np.isfinite(y).all()
            if mode in [10,12]:y=y.copy() # Copy the declared function, no alias of original GGUF output.
            for slot,row,flag in zip(own,y,flags):results[slot]=(int(flag),row.copy())
        self.num.guard();return [(t,k,e,*results[t,k,e]) for t,k,e in pairs]

    def observe(self,h,x,ids):
        if self.contextmap is not None:
            c=self.contextmap['contexts'][str(h['context'])]
            assert h['run']==self.contextmap['run'] and c['mode']==h['mode']==h['arm'] and h['subrun']==9
            assert h['T']==128 and h['position'] in [0,128,256,384] and h['decode']==h['position']//128+1
            self.current={**h,'book_id':c['book_id']}

    def record(self,rows):
        if self.contextmap is None:return
        h=self.current
        for t,k,e,flag,y in rows:
            pos=h['position']+t;z=s.ZONES[0 if pos<255 else 1 if pos<511 else 2]
            key=(h['context'],h['layer'],e,z)
            if key not in self.events:self.events[key]={'model':self.model,'run':h['run'],'book_id':h['book_id'],'context':h['context'],'arm':h['arm'],'layer':h['layer'],'expert':e,'zone':z,'returned_rows':0,'canonical_fallback_rows':0}
            self.events[key]['returned_rows']+=1;self.events[key]['canonical_fallback_rows']+=flag

    def finish(self,path):
        p=Path(path);assert not p.exists();p.write_text(''.join(json.dumps(self.events[k],separators=(',',':'))+'\n' for k in sorted(self.events)))

def serve(registry,instream,outstream):
    outstream.write(w.pack(w.header(kind=3,model=registry.model)));outstream.flush();serial=0;seen=set()
    while True:
        raw=w.exact(instream,w.WIRE.size,eof=True)
        if raw is None:return
        h=w.unpack(raw)
        assert h['kind']==1 and h['status']==0 and h['mode'] in [9,10,11,12] and h['arm']==h['mode'] and h['request']==serial+1
        assert h['model']==registry.model and h['H']==2048 and h['K']==registry.K and h['layer']==registry.layer
        assert 1<=h['T']<=128 and h['micro']==0 and h['run']>0 and h['context']>0 and h['decode']>0 and h['subrun'] in [8,9]
        assert h['position']+h['T']<=512 and h['bytes']==4*h['T']*(h['H']+h['K'])
        assert h['token_phase']==(0 if h['position']<256 else 2)
        key=tuple(h[k] for k in ['run','context','arm','subrun','decode','micro','layer','position','token_phase']);assert key not in seen;seen.add(key);serial=h['request']
        blob=w.exact(instream,h['bytes']);cut=4*h['T']*2048;x=np.frombuffer(blob[:cut],'<f4').reshape(h['T'],2048).copy();ids=np.frombuffer(blob[cut:],'<i4').reshape(h['T'],h['K']).copy()
        assert np.isfinite(x).all() and np.all((ids>=0)&(ids<64)) and all(len(np.unique(v))==h['K'] for v in ids)
        expected=[(t,k,int(e)) for t in range(h['T']) for k,e in enumerate(ids[t]) if int(e) in registry.pilots(h['layer'])]
        assert h['slots']==len(expected)
        registry.observe(dict(h),x.copy(),ids.copy());rows=registry.evaluate(h['mode'],h['layer'],x,ids)
        assert [(t,k,e) for t,k,e,_,_ in rows]==expected
        for _,_,_,flag,y in rows:assert flag in [0,1] and y.shape==(2048,) and y.dtype==np.float32 and np.isfinite(y).all()
        registry.record(rows)
        payload=b''.join(struct.pack('<4I',t,k,e,flag)+y.tobytes() for t,k,e,flag,y in rows)
        outstream.write(w.pack({**h,'kind':2,'bytes':len(payload),'slots':len(rows)}));outstream.write(payload);outstream.flush()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=int,choices=[1,2],required=True);p.add_argument('--freeze');p.add_argument('--review');p.add_argument('--context-map');p.add_argument('--exposure');p.add_argument('--generated',action='store_true');a=p.parse_args();num=None
    try:
        if a.generated:
            assert not any([a.freeze,a.review,a.context_map,a.exposure])
            fma=core.FMA(w.FROZEN1/'fma.so');assert fma.lib.lac_environment()==8
            class Generated(Registry):
                def __init__(self):self.model=a.model;self.K=6 if a.model==1 else 8;self.layer=1 if a.model==1 else 0;self.weights=dict.fromkeys(range(64));self.contextmap=None
                def evaluate(self,mode,layer,x,ids):
                    assert fma.lib.lac_environment()==8
                    return [(t,k,int(e),0,np.add(x[t],np.float32(e/64),dtype=np.float32)) for t in range(len(x)) for k,e in enumerate(ids[t])]
            serve(Generated(),sys.stdin.buffer,sys.stdout.buffer);raise SystemExit(0)
        assert all([a.freeze,a.review,a.context_map,a.exposure])
        # Launcher verifies every complete prior payload before/after the phase.
        # This child verifies the same receipt identities, then every actual
        # runtime dependency in Registry; it never loads teacher target arrays.
        s.authorize(a.freeze,a.review,'QUALITY_C64',verify_prior_payload=False);fma=core.FMA(w.FROZEN1/'fma.so');num=w.GuardedNumeric(fma,s.upstream.F2/'strict-activation')
        reg=Registry(a.model,s.OUT/'C64_registry.frozen.json',num);reg.contextmap=s.read(a.context_map)
        assert reg.contextmap['model']==a.model and reg.contextmap['protocol_sha256']==s.EXPECTED
        serve(reg,sys.stdin.buffer,sys.stdout.buffer);num.close();num=None;reg.finish(a.exposure)
    except Exception as e:
        print('Stage4 worker technical-invalid: '+repr(e),file=sys.stderr,flush=True)
        if num:
            try:num.close()
            except BaseException:pass
        raise
