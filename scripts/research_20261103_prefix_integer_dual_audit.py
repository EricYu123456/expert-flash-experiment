#!/usr/bin/env python3
"""Independent exact-integer box-dual certificates for the prefix LP screen.

No solver import or producer implementation import. Duals are rounded toward
zero to rational numbers; all subsequent certificate arithmetic is integer.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'results/research-2026-11-02/round2/g4-elastic-prefix-screen/work-v3-independent-prefix'
OUT=IN/'integer-dual-audit'
SELF=Path(__file__).resolve();DEN=10**9
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def freeze():
 OUT.mkdir(exist_ok=False)
 ps=[SELF,IN/'LP_arrays.npz',IN/'contract_arrays.npz',IN/'protocol.frozen.json']
 dump(OUT/'protocol.frozen.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),denominator=DEN,inputs=[dict(path=str(p),sha256=sha(p)) for p in ps],method='Convert scaled-LP duals to actual ns constraints, round multipliers towardzero to rational denominator1e9. Requirey<=0 andresidualT>=0. Useexactinteger y*b+sum min(-A^T*y,0) overbox[0,1]^54. Floor rational result; nofloating primal value used as bound.',scope='2048 fixed-layout LP necessary bounds, not integer-feasible schedules or native latency. No newsolver/weights/heldout.'))

def run():
 p=json.loads((OUT/'protocol.frozen.json').read_text());assert not (OUT/'report.json').exists()
 for x in p['inputs']:assert sha(x['path'])==x['sha256']
 with np.load(IN/'contract_arrays.npz') as z:D,Z,base,routes=[z[k].copy() for k in ['D','Z','base','routes']]
 with np.load(IN/'LP_arrays.npz') as z:y=z['dual_multipliers'].copy();old=z['necessary_work_LB'].copy();primal=z['primal_work'].copy()
 assert y.shape==(4,512,70) and np.all(np.isfinite(y)) and np.all(y<=0)
 scaled=y.copy();scaled[:,:,16:]*=28000
 nums=np.trunc(scaled*DEN).astype(np.int64);assert np.all(nums<=0)
 bound=np.zeros((4,512),np.int64);resT=DEN+nums[:,:,:16].sum(2);assert np.all(resT>=0)
 A=np.zeros((70,54),np.int64);rhs=np.zeros(70,np.int64)
 for e in range(6):
  for i in range(8):A[16+e*8+i,48+e]=1;A[16+e*8+i,e*8+i]=-1
  A[64+e,e*8:e*8+8]=1;A[64+e,48+e]=-1;rhs[64+e]=7
 for lay in range(4):
  for q,sel in enumerate(routes):
   A[:16,:48]=D[lay,sel].reshape(48,16).T;A[:16,48:54]=Z[sel].T;rhs[:16]=-base[q]
   # Python integer objects avoid any signed-machine-integer overflow.
   n=nums[lay,q].astype(object);r=-(A.astype(object).T@n)
   numerator=sum(int(x)*int(b) for x,b in zip(n,rhs))+sum(min(int(v),0) for v in r)
   bound[lay,q]=numerator//DEN
 assert np.all(bound<=primal+1e-7) and np.all(abs(bound-old)<=1)
 np.savez_compressed(OUT/'certificates.npz',dual_numerators=nums,work_lower_bound=bound,T_residual_numerator=resT)
 r=dict(status='PASS exact rational dual bounds',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(SELF),certificates=2048,denominator=DEN,all_multipliers_nonpositive=True,all_T_residuals_nonnegative=True,arithmetic='Python arbitraryprecision integer; floor rational',min_delta_vs_previous_floor_ns=int((bound-old).min()),max_delta_vs_previous_floor_ns=int((bound-old).max()),mean_work_LB_ns=bound.mean(1).tolist(),scope='Necessary continuous relaxation only. No feasible integerassignment/native/precisionABI proof.')
 dump(OUT/'report.json',r)
 files=[dict(path=str(x.relative_to(OUT)),sha256=sha(x)) for x in sorted(OUT.iterdir()) if x.is_file()]
 dump(OUT/'completion.json',dict(status='sealed_exact_dual_check',files=files));print(json.dumps(r,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args();freeze() if args.mode=='freeze' else run()
