#!/usr/bin/env python3
"""Quality worker: unchanged frozen predictions plus typed returned-row counts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage2_pg19_quality_core as q
w,r=q.w,q.r
class QualityRegistry(w.Registry):
    def start_exposure(self,path,contextmap):
        self.exposure_path=Path(path);assert not self.exposure_path.exists();self.contextmap=contextmap;self.events={};self.current=None
    def observe(self,h,x,ids):
        c=self.contextmap['contexts'][str(h['context'])];assert h['run']==self.contextmap['run'] and h['mode']==h['arm']==c['mode'] and h['subrun']==5 and h['micro']==0 and h['T']==128 and h['position'] in [0,128,256,384]
        assert h['token_phase']==(0 if h['position']<256 else 2);self.current={**h,'book_id':c['book_id']}
    def evaluate(self,mode,layer,x,ids):
        rows=super().evaluate(mode,layer,x,ids);self.record_returned(rows,mode,layer);return rows
    def record_returned(self,rows,mode,layer):
        h=self.current;assert h['mode']==mode and h['layer']==layer
        for t,k,e,flag,y in rows:
            assert type(flag) is int and flag in [0,1]
            z=q.zone(h['position']+t);key=(h['model'],h['book_id'],h['context'],mode,layer,e,z)
            if key not in self.events:self.events[key]={'model':h['model'],'book_id':h['book_id'],'context':h['context'],'arm':mode,'layer':layer,'expert':e,'zone':z,'returned_rows':0,'canonical_fallback_rows':0,'approximation_rows':0,'canonical_B_rows':0}
            event=self.events[key];event['returned_rows']+=1;event['canonical_fallback_rows']+=int(flag)
            if mode in [3,4]:assert flag==0;event['canonical_B_rows']+=1
            else:event['approximation_rows']+=int(flag==0)
    def finish_exposure(self):
        with self.exposure_path.open('w') as f:
            for key in sorted(self.events):f.write(json.dumps(self.events[key],separators=(',',':'))+'\n')
        r.save(self.exposure_path.with_suffix('.receipt.json'),{'status':'complete','sha256':r.sha(self.exposure_path),'events':len(self.events),'scope':'returnedrows; paired against actual successful C++scatter before reporting','raw_X_retained':False,'targets_queried_for_fitting':False,'PGvalidation_never_fit_or_select':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=int,choices=[1,2],required=True);p.add_argument('--freeze',required=True);p.add_argument('--review',required=True);p.add_argument('--activation',required=True);p.add_argument('--context-map',required=True);p.add_argument('--exposure',required=True);a=p.parse_args();num=None
    try:
        freeze,fit,registry=q.authorize(a.freeze,a.review);contextmap=r.read(a.context_map);assert contextmap['protocol_sha256']==q.EXPECTED and contextmap['model']==a.model and contextmap['run']>0
        books=[b['book_id'] for b in r.read(q.PROTOCOL)['data']['validation16']];assert contextmap['contexts']==q.context_plan(books)[1]
        fma=q.core.FMA(w.FROZEN1/'fma.so');num=w.GuardedNumeric(fma,r.F2/'strict-activation');reg=QualityRegistry(a.model,registry,num,fma);reg.start_exposure(a.exposure,contextmap);w.serve(reg,sys.stdin.buffer,sys.stdout.buffer);num.close();num=None;reg.finish_exposure()
    except BaseException as e:
        print('PGquality worker tooling-invalid: '+repr(e),file=sys.stderr,flush=True)
        if num:
            try:num.close()
            except BaseException:pass
        raise
