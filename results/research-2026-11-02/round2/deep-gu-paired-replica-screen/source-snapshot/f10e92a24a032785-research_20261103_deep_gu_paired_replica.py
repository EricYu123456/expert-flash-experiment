#!/usr/bin/env python3
"""TRAIN-only paired boundary replicas: necessary-work screen before native cost.

This uses ordinary full-row GEMVs, no input reuse, partial dots or new contexts.
The graph-orientation construction is standard, not a new theorem.
"""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'deep-gu-paired-replica-screen'
INC=BASE/'native-joint-refinement/pilot-v1/deepseek.L0.input4.independent_paired.seed20261106'
TRAIN=BASE/'w8-format-training/deepseek/decode.routes.npz'
SELF=Path(__file__).resolve()
CPP=SELF.with_name('research_20261103_deep_gu_paired_replica_oracle.cpp')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def freeze():
 OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
 rec=json.loads((INC/'result.json').read_text());svc=rec['service']
 with np.load(INC/'arrays.npz') as z:
  r=z['rows'];assert r.shape==(65,2,16)
  assert np.all(np.sort(r[:64,0],axis=1)==np.array([160]*8+[192]*8))
  with np.load(TRAIN) as t:assert t['split'].item()=='train' and t['completed'].all() and np.array_equal(z['routes'],t['routes'][:,:,0].reshape(512,6))
 assert svc['input_channel_copies']==4 and svc['command_ns']==250 and svc['down_first']==0
 paths=[SELF,CPP,INC/'result.json',INC/'arrays.npz',TRAIN,ROOT/'scripts/research_20261103_ordinary_read_backend.py',ROOT/'scripts/research_20261103_ordinary_read_native.cpp',ROOT/'scripts/research_20261103_topology_placement_replay.py']
 p=dict(frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='Deep L0/input4 actual W8 TRAIN512 only. Finite6 routeblind pairings; no global placement optimum, inference, heldout, native-device or publication claim.',service=svc,inputs=[dict(path=str(x),sha256=sha(x)) for x in paths],pairing_menu=['sorted_high_low','sorted_high_reverse_low']+[f'random_{s}' for s in range(202611081,202611085)],replicas=dict(experts=64,per_expert=8,tile_rows=32,columns=2048,extra_coefficient_bytes=33554432,extra_pages=2048,extra_waves=512),contract=['All16 chips retain160 unique full rows per routedexpert. Each original192 source retains its final32 logicalrows; matching160 destination receives copy as contiguoussuffix. Exactlyone of thetwo prefixes executes those32rows.','Every query uses16 GU jobs/expert and88waves, with independent full2048byte input reload eachjob. SharedGU/Down and privateDown remain unchanged.','Only32row/4page waves used, nofringe, splitdot, inputreuse, earlycompletion, newMAC/decoder/context. Logicalrow scatter is required software, not a provednativeoutput ABI.','K6 selectedexperts give6regular union of perfectmatchings; balanced orientation givesindegree3. This standardEuler/flow fact is notanewbatchcode theorem.','Work-optimal orientation is a necessary chip-context bound only. Add currentfixedexecutor firstdispatch offsets c*250ns for a stronger bound; no native-speed inference from work alone.'],selector='Exact integral maxflow minimax of joined chip work for fixedpairing. Zero selectorcost only an optimistic initialscreen; beforeperformanceclaim charge measuredproductionC++ selection and strong samecapacity controls.',static_control='Same edge locations, zerocopies fixedone-endchoice perexpert. Two complete coordinatepasses overall256 masks/expert starting fromoriginalA. Workobjectiveonly.',gate='If every finitepairing has fixedexecutor optimistic native mean uppergain<2%, sealno-go. Otherwise report to root before native replay andsame32MiB whole/partial replicas. Noalllayer/heldoutautomaticexpansion.',whole_backup_control_requirement='32MiB funds5 complete5.5MiB GUbackups plus4.5MiB remainingusefulcopies; 6complete=33MiB is a stronger slightlyoverbudget control, not exactbudget. Do notclaimpartialcopies win against5copies whileignoringremaining4.5MiB.')
 for x in p['inputs']:
  if Path(x['path']).suffix in ['.py','.cpp']:
   dst=OUT/'source-snapshot'/(x['sha256'][:16]+'-'+Path(x['path']).name);shutil.copy2(x['path'],dst);x['snapshot']=str(dst)
 dump(OUT/'protocol.frozen.json',p);print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'))))

def run():
 p=json.loads((OUT/'protocol.frozen.json').read_text());assert not (OUT/'work_report.json').exists()
 for x in p['inputs']:assert sha(x['path'])==x['sha256'],x['path']
 import research_20261103_ordinary_read_backend as backend
 began=time.perf_counter();s=backend.Service(**p['service'])
 with np.load(INC/'arrays.npz') as z:
  rows=z['rows'].copy();routes=np.ascontiguousarray(z['routes'],np.int32);oldcost=z['cost'].copy();oldlb=z['lower_bound'].copy()
  plan={k:z[k].copy() for k in ['offsets','jobs','input_columns','total_rows']}
 table=backend.occupancy_table(plan,s)['chip_ns'];original=table[routes].sum((1,2))+table[64].sum(0)
 # Exact whole-row job occupancy difference: one ordinary wave and128B output.
 delta=s.ordinary_read_ns+s.transfer(128,s.channel_bytes_per_second);assert delta==28064
 high=(rows[:64,0]==192).astype(np.int64);base=original-(high[routes].sum(1)*delta)
 baseline_work=original.max(1);assert np.all(baseline_work+s.host_submit_ns<=oldcost)
 binary=OUT/'oracle.so';subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(CPP),'-o',str(binary)],check=True)
 lib=ctypes.CDLL(str(binary));fn=lib.paired_orientation
 i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
 fn.argtypes=[i64,i32,i32,ctypes.c_int,ctypes.c_int64,i64,i32];fn.restype=None
 def orient(work,edges):
  v=np.zeros(512,np.int64);a=np.zeros((512,48),np.int32);fn(np.ascontiguousarray(work,np.int64),routes,np.ascontiguousarray(edges,np.int32),512,delta,v,a)
  counts=np.asarray([np.bincount(t,minlength=16) for t in a],np.int64)
  assert np.array_equal((work+counts*delta).max(1),v)
  for k in range(6):
   ep=edges[routes[:,k]];take=a[:,k*8:(k+1)*8];assert np.all((take==ep[:,:,0])|(take==ep[:,:,1]))
  return v,a
 layouts=[]
 for name in p['pairing_menu']:
  rng=np.random.default_rng(int(name.split('_')[-1]) if name.startswith('random_') else 0);edges=[]
  for e in range(64):
   h=np.flatnonzero(rows[e,0]==192).astype(np.int32);l=np.flatnonzero(rows[e,0]==160).astype(np.int32)
   if name=='sorted_high_reverse_low':l=l[::-1]
   elif name.startswith('random_'):l=rng.permutation(l)
   edges.append(np.stack([h,l],axis=1))
  layouts.append(np.asarray(edges,np.int32))
 records=[];arrays=dict(rows=rows,routes=routes,original_native=oldcost,original_native_LB=oldlb,original_chip_ns=original,base_chip_ns=base,layouts=np.stack(layouts));members=[np.flatnonzero((routes==e).any(1)) for e in range(64)]
 order=sorted(range(64),key=lambda e:(-len(members[e]),e))
 for j,edges in enumerate(layouts):
  v,a=orient(base,edges);dispatch=base+np.arange(16,dtype=np.int64)*s.command_ns;fv,fa=orient(dispatch,edges)
  # Verify the routing-independent balanced-GU guarantee separately.
  zero=np.zeros_like(base);gu,ga=orient(zero,edges);assert np.all(gu==3*delta)
  assert np.all(np.asarray([np.bincount(t,minlength=16) for t in ga])==3)
  # Same physical prefix choices as a fixed one-copy map; no residency benefit.
  static=original.copy();bits=np.zeros((64,8),np.int8);scores=static.max(1);passes=[]
  effects=np.zeros((64,8,16),np.int64)
  for e in range(64):
   for k,(src,dst) in enumerate(edges[e]):effects[e,k,src]-=delta;effects[e,k,dst]+=delta
  for iteration in range(2):
   changed=0
   for e in order:
    qq=members[e];old=np.einsum('j,jc->c',bits[e],effects[e]);core=static[qq]-old;best=int(scores[qq].sum());bestbits=bits[e].copy();bestscore=scores[qq].copy();bestloads=static[qq].copy()
    for mask in range(256):
     b=np.asarray([(mask>>k)&1 for k in range(8)],np.int8);loads=core+np.einsum('j,jc->c',b,effects[e]);sc=loads.max(1);val=int(sc.sum())
     if val<best:best=val;bestbits=b;bestscore=sc;bestloads=loads
    changed+=int(np.any(bestbits!=bits[e]));bits[e]=bestbits;scores[qq]=bestscore;static[qq]=bestloads
   passes.append(dict(iteration=iteration+1,changed_experts=changed,mean_ns=float(scores.mean())))
  assert np.array_equal(static,original+np.einsum('qkj,qkjc->qc',bits[routes],effects[routes]))
  arrays.update({f'layout{j}_work':v,f'layout{j}_assignment':a,f'layout{j}_dispatch_work':fv,f'layout{j}_dispatch_assignment':fa,f'layout{j}_balanced_GU_assignment':ga,f'layout{j}_static_bits':bits,f'layout{j}_static_work':scores})
  rec=dict(layout=j,name=p['pairing_menu'][j],mean_work_ns=float(v.mean()),p99_work_ns=float(np.percentile(v,99)),work_improvement_pct=float(100*(1-v.mean()/baseline_work.mean())),fixed_executor_mean_LB_ns=float(fv.mean()+s.host_submit_ns),optimistic_native_gain_upper_pct=float(100*(1-(fv.mean()+s.host_submit_ns)/oldcost.mean())),same_edges_static_mean_work_ns=float(scores.mean()),dynamic_vs_static_work_pct=float(100*(1-v.mean()/scores.mean())),static_passes=passes)
  records.append(rec);print(json.dumps(rec),flush=True)
 best=min(records,key=lambda x:x['fixed_executor_mean_LB_ns']);np.savez_compressed(OUT/'work_arrays.npz',**arrays)
 for x in p['inputs']:assert sha(x['path'])==x['sha256']
 report=dict(status='PASS finite necessary-work screen only',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(SELF),cpp_sha256=sha(CPP),binary_sha256=sha(binary),baseline_native_mean_ns=float(oldcost.mean()),baseline_work_mean_ns=float(baseline_work.mean()),delta_ns=int(delta),replicas=p['replicas'],records=records,best_by_optimistic_fixed_executor_LB=best,queries_per_layout=512,balanced_GU_guarantee_checked_cases=512*6,gate_reached=bool(best['optimistic_native_gain_upper_pct']>=2),seconds=time.perf_counter()-began,limitations='Finite routeblind layouts and fixedexecutor LB. Minimax joinedwork doesnotensure bestnativecausal latency. Staticcontroliswork-only. Selector/hostrow scatter/wholeandpartial samebudgetstrongcontrols/notactualNANDABI remainbeforeclaim.')
 dump(OUT/'work_report.json',report);print(json.dumps(dict(best=best,seconds=report['seconds'],gate_reached=report['gate_reached']),indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args();freeze() if args.action=='freeze' else run()
