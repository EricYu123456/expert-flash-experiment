#!/usr/bin/env python3
"""Separately identified NL7/COPY_NL8; immutable B/C/D dispatch unchanged."""
import argparse
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage3_core as s
np,w,core,old=s.np,s.w,s.core,s.old
class Registry(w.Registry):
    def __init__(self,model,registry,num,fma):
        j=s.read(registry);assert j['protocol_sha256']==s.EXPECTED and j['arm']==7 and j['copy_arm']==8 and j['original_registry_sha256']==s.sha(j['original_registry_path'])
        super().__init__(model,j['original_registry_path'],num,fma);assert s.sha(j['payload_manifest'])==j['payload_manifest_sha256'];deps=s.read(j['payload_manifest'])['files'];self.nl={}
        for o in j['operators']:
            if o['model']!=self.name:continue
            key=o['layer'],o['expert'];assert o['unchanged_C_D_operator']==self.ops[key] and s.sha(o['NL_model_path'])==deps[o['NL_model_path']]
            m=s.load_model(o['NL_model_path']);assert s.model_hash(m)==o['NL_model_hash'] and m['unsupported'].tolist()==self.locals[key]['unsupported'];self.nl[key]=m
        assert set(self.nl)==set(self.ops)
    def evaluate(self,mode,layer,x,ids):
        if mode not in [7,8]:return super().evaluate(mode,layer,x,ids)
        self.num.guard();pairs=[(t,k,int(e)) for t in range(len(x)) for k,e in enumerate(ids[t]) if int(e) in self.pilots(layer)];result={}
        if not pairs:return []
        tree,P,sigma,_=self.trees[layer];d=old.design(self.num,x,tree,32,P,sigma,128,'normal');qx=sx=None
        for e in sorted({p[2] for p in pairs}):
            own=[p for p in pairs if p[2]==e];rows=np.asarray([v[0] for v in own],np.int64);dd=core.subset(d,rows);m=self.nl[layer,e];flags=np.isin(dd['leaf'],m['unsupported']).astype(np.uint32);y=s.predict_nl(self.num,self.fma,m,dd,self.weights[layer,e])
            if flags.any():
                if qx is None:qx,sx=self.num.quantize(x)
                bad=flags.astype(bool);y[bad]=self.canonical(layer,e,rows[bad],qx,sx)
            assert y.dtype==np.float32 and y.shape==(len(rows),2048) and np.isfinite(y).all()
            for slot,row,flag in zip(own,y,flags):result[slot]=(int(flag),row.copy())
        self.num.guard();return [(t,k,e,*result[t,k,e]) for t,k,e in pairs]
class Exposure:
    def start(self,path,contexts):self.exposure_path=Path(path);assert not self.exposure_path.exists();self.contextmap=contexts;self.events={};self.current=None
    def observe(self,h,x,ids):
        c=self.contextmap['contexts'][str(h['context'])];assert h['run']==self.contextmap['run'] and h['mode']==h['arm']==c['mode'] and h['subrun']==7 and h['micro']==0 and h['T']==128 and h['position'] in [0,128,256,384] and h['token_phase']==(0 if h['position']<256 else 2);self.current={**h,'book_id':c['book_id']}
    def record(self,rows):
        h=self.current
        for t,k,e,flag,y in rows:
            assert flag in [0,1];pos=h['position']+t;zone='unscored_inputs_0_254' if pos<255 else 'scored_predecessors_255_510' if pos<511 else 'unused_input_511';key=(h['model'],h['book_id'],h['context'],h['arm'],h['layer'],e,zone)
            if key not in self.events:self.events[key]={'model':h['model'],'book_id':h['book_id'],'context':h['context'],'arm':h['arm'],'layer':h['layer'],'expert':e,'zone':zone,'returned_rows':0,'canonical_fallback_rows':0}
            self.events[key]['returned_rows']+=1;self.events[key]['canonical_fallback_rows']+=flag
    def finish(self):self.exposure_path.write_text(''.join(json.dumps(self.events[k],separators=(',',':'))+'\n' for k in sorted(self.events)))
class QualityRegistry(Exposure,Registry):
    def evaluate(self,*args):
        rows=super().evaluate(*args);self.record(rows);return rows
def serve(registry,instream,outstream):
    # The old framing is reused with a new, explicit mode whitelist. No
    # reinterpretation of D6, no monkeypatch of the immutable old server.
    outstream.write(w.pack(w.header(kind=3,model=registry.model)));outstream.flush();serial=0;seen=set()
    while True:
        raw=w.exact(instream,w.WIRE.size,eof=True)
        if raw is None:return
        h=w.unpack(raw);assert h['kind']==1 and h['status']==0 and h['mode'] in [3,4,5,6,7,8] and h['arm']==h['mode'] and h['request']==serial+1 and h['model']==registry.model and h['H']==registry.H and h['K']==registry.K
        assert 1<=h['T']<=128 and h['micro']==0 and h['run']>0 and h['context']>0 and h['decode']>0 and registry.pilots(h['layer']) and h['bytes']==4*h['T']*(h['H']+h['K'])
        epoch=tuple(h[k] for k in ['run','context','arm','subrun','decode','micro','layer','position','token_phase']);assert epoch not in seen;seen.add(epoch);serial=h['request'];data=w.exact(instream,h['bytes']);cut=4*h['T']*h['H'];x=np.frombuffer(data[:cut],'<f4').reshape(h['T'],h['H']).copy();ids=np.frombuffer(data[cut:],'<i4').reshape(h['T'],h['K']).copy();assert np.isfinite(x).all() and np.all((ids>=0)&(ids<64)) and all(len(np.unique(row))==h['K'] for row in ids)
        registry.observe(dict(h),x.copy(),ids.copy());expected=[(t,k,int(e)) for t in range(h['T']) for k,e in enumerate(ids[t]) if int(e) in registry.pilots(h['layer'])];assert h['slots']==len(expected);rows=registry.evaluate(h['mode'],h['layer'],x,ids);assert [(t,k,e) for t,k,e,_,_ in rows]==expected
        for _,_,_,flag,y in rows:assert flag in [0,1] and y.shape==(h['H'],) and y.dtype==np.float32 and np.isfinite(y).all()
        payload=b''.join(struct.pack('<4I',t,k,e,flag)+np.ascontiguousarray(y,'<f4').tobytes() for t,k,e,flag,y in rows);reply={**h,'kind':2,'bytes':len(payload),'slots':len(rows)};outstream.write(w.pack(reply));outstream.write(payload);outstream.flush()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=int,choices=[1,2],required=True);p.add_argument('--freeze',required=True);p.add_argument('--review',required=True);p.add_argument('--context-map',required=True);p.add_argument('--exposure',required=True);a=p.parse_args();num=None
    try:
        s.authorize(a.freeze,a.review,'DEV_QUALITY');ctx=s.read(a.context_map);assert ctx['protocol_sha256']==s.EXPECTED and ctx['model']==a.model and ctx['run']>0;fma=core.FMA(w.FROZEN1/'fma.so');num=w.GuardedNumeric(fma,s.F2/'strict-activation');reg=QualityRegistry(a.model,s.OUT/'NL_registry.frozen.json',num,fma);reg.start(a.exposure,ctx);serve(reg,sys.stdin.buffer,sys.stdout.buffer);num.close();num=None;reg.finish()
    except BaseException as e:
        print('Stage3 worker tooling-invalid: '+repr(e),file=sys.stderr,flush=True)
        if num:
            try:num.close()
            except BaseException:pass
        raise
