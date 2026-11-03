#!/usr/bin/env python3
"""Diagnostic relaxation: arbitrary A8 locations, ordinary full-row job costs.

Not a stored layout: no finite-copy or native-speed claim. For each exact A/B
expert subset, identical A8 jobs are assigned to arbitrary chips. Each expert
uses eight distinct chips. The complete bipartite degree constraints reduce to
sum min(nA, floor((T-base_c)/cost_A_job)) >= 8*nA.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'g4-elastic-prefix-screen/free-span-bound'
INC=BASE/'compact-coupled-solver/continuation/native/0107.npz'
CFG=BASE/'compact-coupled-solver/protocol.frozen.json'
SELF=Path(__file__).resolve()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def freeze():
 OUT.mkdir(exist_ok=False,parents=True)
 paths=[SELF,INC,CFG,ROOT/'scripts/research_20261103_ordinary_read_backend.py']
 dump(OUT/'protocol.frozen.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),inputs=[dict(path=str(p),sha256=sha(p)) for p in paths],scope='G4 L0 TRAIN512/currentCPB; free arbitrary A8 chip locations foreachquery, unlimitedcopies; impossible-as-given placement oracle usedonlyforupperheadroom, notproposedhardware.',objective='Exactminimax chipownedwork under64 A/B expertmasks; eachA contributes8identical full128row/3wave GUjobs on8distinctchips, B GU/Down/shared unchanged. Fullinputreload/commands/outputfringe perjob paid. No native/readiness/selector/fixedaddress restriction.',capacity_condition='For nA jobsources, bipartite source-rows eachdegree8 and16 chipcolumns capacity<=nA. Sufficient andnecessary: sumcaps>=8nA. Foranyk<=nA, sum min(cap,k)>=(k/nA)sumcap>=8k, satisfying allcompletebipartite mincuts.',gate='If optimisticmean<2% and P99<5%, stopbroaderApositionsearch. Otherwise report, not claimattainment orlaunchalllayers/heldout.'))
 print(sha(OUT/'protocol.frozen.json'))

def run():
 p=json.loads((OUT/'protocol.frozen.json').read_text());assert not (OUT/'report.json').exists()
 for x in p['inputs']:assert sha(x['path'])==x['sha256']
 import research_20261103_ordinary_read_backend as b
 s=b.Service(**json.loads(CFG.read_text())['service']);assert s.command_ns==250 and s.input_channel_copies==1
 with np.load(INC) as z:
  rows=z['rows'].copy();routes=z['routes'].copy();cost=z['cost'].copy();plan={k:z[k] for k in ['offsets','jobs','input_columns','total_rows']}
 t=b.occupancy_table(plan,s)['chip_ns'];fixed=t[routes,1].sum(1)+t[64].sum(0);gu=t[routes,0]
 # A128 has12pages,3waves,128complete outputrows; reload full1536bytes.
 job=s.command_ns+s.transfer(1536,s.channel_bytes_per_second)+3*s.ordinary_read_ns+s.transfer(128*4,s.channel_bytes_per_second)
 assert job==85274
 all_values=[];mask_bases=[];allocations=[]
 for mask in range(64):
  active=[k for k in range(6) if mask>>k&1];n=len(active);keep=[k for k in range(6) if not(mask>>k&1)]
  base=fixed+gu[:,keep].sum(1);lo=base.max(1).copy();hi=lo+n*job
  if n:
   while np.any(lo<hi):
    mid=(lo+hi)//2;capacity=np.minimum(n,np.maximum(0,(mid[:,None]-base)//job));ok=(mid>=base.max(1))&(capacity.sum(1)>=8*n)
    hi=np.where(ok,mid,hi);lo=np.where(ok,lo,mid+1)
   cap=np.minimum(n,np.maximum(0,(lo[:,None]-base)//job));assert np.all(cap.sum(1)>=8*n)
   prev=np.minimum(n,np.maximum(0,((lo-1)[:,None]-base)//job));assert np.all(((lo-1)<base.max(1))|(prev.sum(1)<8*n))
  else:cap=np.zeros_like(base)
  all_values.append(lo);mask_bases.append(base);allocations.append(cap)
 values=np.stack(all_values);winner=values.argmin(0);best=values[winner,np.arange(512)]+s.host_submit_ns
 assert np.all(best<=cost)
 np.savez_compressed(OUT/'arrays.npz',rows=rows,routes=routes,baseline_native=cost,allmask_work=values,allmask_base=np.stack(mask_bases),allmask_chip_capacities=np.stack(allocations),winner=winner,best_LB=best)
 report=dict(status='exact relaxed necessary-work oracle only',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(SELF),A_job_ns=int(job),allmask_queries=512*64,baseline_native_mean_ns=float(cost.mean()),baseline_native_p99_ns=float(np.percentile(cost,99)),relaxed_native_mean_LB_ns=float(best.mean()),relaxed_native_p99_LB_ns=float(np.percentile(best,99)),mean_improvement_upper_pct=float(100*(1-best.mean()/cost.mean())),p99_improvement_upper_pct=float(100*(1-np.percentile(best,99)/np.percentile(cost,99))),selected_A_histogram={str(n):int(sum(int(m).bit_count()==n for m in winner)) for n in range(7)},limitations='No finitecopy/staticaddress/readiness/hostscatter/selectiontime. No feasibleplacement/nativeperformance/newhardwareclaim. AppliesonlyA128x8 vsfrozenB GU; Down/shared fixed.')
 dump(OUT/'report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args();freeze() if args.mode=='freeze' else run()
