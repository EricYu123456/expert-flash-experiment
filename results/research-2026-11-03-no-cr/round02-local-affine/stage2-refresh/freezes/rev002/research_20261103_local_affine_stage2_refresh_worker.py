#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage2_refresh_core as r
w,np=r.w,r.np
class RecorderRegistry(w.Registry):
    def start_recording(self,path,contexts,run):
        self.contexts=contexts;self.run=run;self.seen=set();self.record_counts={};self.live_canonical_rows={};self.actual_mode=None;self.record_path=Path(path);assert not self.record_path.exists();self.record_file=self.record_path.open('wb');self.record_file.write(b'LARFXI01')
    def evaluate(self,mode,*args):
        self.actual_mode=mode;return super().evaluate(mode,*args)
    def canonical(self,layer,e,rows,q,sx):
        key=str(self.actual_mode)+':'+str(layer)+':'+str(e);self.live_canonical_rows[key]=self.live_canonical_rows.get(key,0)+len(rows);return super().canonical(layer,e,rows,q,sx)
    def observe(self,h,x,ids):
        c=self.contexts[str(h['context'])];assert h['run']==self.run and h['arm']==h['mode']==c['arm'] and h['subrun']==4 and h['micro']==0
        assert (h['token_phase']==0 and h['T']==128 and h['position'] in range(0,768,128)) or (h['token_phase']==1 and h['T']==1 and h['position'] in range(768,800))
        if not r.admitted(h):return
        key=r.state_key(h);assert key not in self.seen;self.seen.add(key);assert h['bytes']==x.nbytes+ids.nbytes
        self.record_file.write(w.pack(h));self.record_file.write(np.ascontiguousarray(x,'<f4').tobytes());self.record_file.write(np.ascontiguousarray(ids,'<i4').tobytes());self.record_file.flush();name=str(h['layer'])+':'+str(h['arm'])+':'+str(h['token_phase']);self.record_counts[name]=self.record_counts.get(name,0)+h['T']
    def finish_recording(self):
        self.record_file.close();r.save(self.record_path.with_suffix('.receipt.json'),{'status':'complete','counts':self.record_counts,'frames':len(self.seen),'sha256':r.sha(self.record_path),'owned_input_only':True,'canonical_offroute_targets_queried':False,'live_forward_canonical_expert_rows_by_arm_layer_expert':self.live_canonical_rows,'live_forward_activation_calls':self.num.activation_calls,'live_forward_activation_IPC_bytes':self.num.activation_ipc_bytes,'live_forward_activation_seconds':self.num.activation_seconds})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=int,required=True,choices=[1,2]);p.add_argument('--freeze',required=True);p.add_argument('--review',required=True);p.add_argument('--registry',required=True);p.add_argument('--activation',required=True);p.add_argument('--records',required=True);p.add_argument('--context-map',required=True);a=p.parse_args();num=None
    try:
        r.authorize(a.freeze,a.review,'COLLECT_REFRESH');assert r.sha(a.registry)==r.OLD_REGISTRY
        fma=r.core.FMA(w.FROZEN1/'fma.so');num=w.GuardedNumeric(fma,a.activation);reg=RecorderRegistry(a.model,a.registry,num,fma);m=r.read(a.context_map);books=r.read(r.PROTOCOL)['train16'];expected={str(bi*3+ai+1):{'book_id':b['book_id'],'arm':arm} for bi,b in enumerate(books) for ai,arm in enumerate([3,5,6])};assert m['protocol_sha256']==r.EXPECTED and m['model']==a.model and m['run']>0 and m['contexts']==expected;reg.start_recording(a.records,m['contexts'],m['run']);w.serve(reg,sys.stdin.buffer,sys.stdout.buffer);num.close();num=None;reg.finish_recording()
    except BaseException as e:
        print('Refresh worker tooling-invalid: '+repr(e),file=sys.stderr,flush=True)
        if num:
            try:num.close()
            except BaseException:pass
        raise
