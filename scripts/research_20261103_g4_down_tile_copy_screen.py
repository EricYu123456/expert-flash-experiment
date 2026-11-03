#!/usr/bin/env python3
"""Frozen, bounded TRAIN-only Down full-row replica necessary-work screen."""
import argparse,collections,ctypes,hashlib,json,pathlib,subprocess,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'g4-down-tile-copy-screen'
BASELINE=BASE/'compact-coupled-solver/continuation/native/0107.npz';SERVICE=BASE/'compact-coupled-solver/protocol.frozen.json';TRAIN=BASE/'w8-format-training/granite4tiny/decode.routes.npz';SELF=pathlib.Path(__file__).resolve();CPP=SELF.with_name('research_20261103_g4_down_copy_oracle.cpp')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def freeze():
 OUT.mkdir(exist_ok=False);s=json.loads(SERVICE.read_text());z=np.load(BASELINE);rows=z['rows'];assert rows.shape==(65,2,16) and z['routes'].shape==(512,6);assert z['cost'].mean()==561099.01953125
 with np.load(TRAIN) as t:assert t['split'].item()=='train' and t['completed'].all() and np.array_equal(z['routes'],t['routes'][:,:,0].reshape(512,6))
 assert np.all(np.sort(rows[:64,1],axis=1)==np.asarray([0]*4+[128]*12));svc=s['service'];assert svc['command_ns']==250 and svc['input_channel_copies']==1 and svc['output_fringe']==1
 sources=[SELF,CPP,BASELINE,SERVICE,TRAIN,ROOT/'scripts/research_20261103_ordinary_read_backend.py',ROOT/'scripts/research_20261103_ordinary_read_native.cpp']
 protocol=dict(frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),model='granite4tiny',layer=0,service=svc,baseline_native_mean_ns=561099.01953125,input_sha256={str(p):sha(p) for p in sources},routeblind_layouts=32,routeblind_seed=202611071,coordinate_passes=2,edge_candidates_per_expert=180,expert_order='Descending actualTRAIN frequency, tie expertID',objective='Mean exact minimum maxchip ownership work across all64 perquery A/B masks. Necessary work only, no native speed claim.',copies=dict(experts=64,one_tile_per_expert=True,tile_rows=128,columns=512,bytes_per_tile=65536,total_resident_extra_bytes=4194304,total_resident_extra_waves=64),allowed='Each source is baseline Down128, destination any otherchip; A unchanged; B source128->0 and dest0->128 or128->256. GU/shared fixed; original logical rowIDs retained, original destcore then copiedsource suffix.',static_control='Same fixed copy edges but oneA/B choice perexpert acrossallTRAIN; zero coefficient replication. Two deterministic coordinate passes fromallA andallB. Fixedfullinput/onejob prefix grouping exactlysame.',gate='If selected finite-layout work-oracle potential>=2% mean, report before fullnative and same64wave strong controls. Otherwise seal no-go for tested finite screen; not a global replication impossibility.',scope='TRAIN512 only; no heldout, new weights, quantization, coactivation/router rescaling, inference or hardware additions. All64copies countresident evenif never used. Job coefficients/output exactwholewaves, no validR/fringe assumption.')
 dump(OUT/'protocol.frozen.json',protocol);(OUT/'source.py').write_bytes(SELF.read_bytes());(OUT/'oracle.cpp').write_bytes(CPP.read_bytes());print(json.dumps(dict(status='FROZEN',protocol_sha256=sha(OUT/'protocol.frozen.json')),indent=2))
def run():
 protocol=json.loads((OUT/'protocol.frozen.json').read_text());assert not (OUT/'work_report.json').exists()
 for p,h in protocol['input_sha256'].items():assert sha(p)==h,p
 import research_20261103_ordinary_read_backend as backend
 began=time.monotonic();svc=backend.Service(**protocol['service']);z=np.load(BASELINE);rows=z['rows'].copy();routes=np.ascontiguousarray(z['routes'],np.int32);plan={k:z[k] for k in ['offsets','jobs','input_columns','total_rows']};table=backend.occupancy_table(plan,svc)['chip_ns'];base=np.ascontiguousarray(table[routes].sum((1,2))+table[64].sum(0),np.int64);original=base.max(1);ii=np.arange(512,dtype=np.int32);members=[np.flatnonzero((routes==e).any(1)).astype(np.int32) for e in range(64)];order=sorted(range(64),key=lambda e:(-len(members[e]),e))
 def cost(n):
  if n==0:return 0
  pages=(n*512+16383)//16384;return 250+256+((pages+3)//4)*28000+2*(pages*16384//512)
 c128,c256=cost(128),cost(256);assert (c128,c256)==(28762,57018)
 edges=[[(int(s),t) for s in np.flatnonzero(rows[e,1]) for t in range(16) if t!=s] for e in range(64)];assert all(len(x)==180 for x in edges)
 def delta_for(e,edge):
  s,t=edge;d=np.zeros(16,np.int64);d[s]=-c128;d[t]=cost(int(rows[e,1,t])+128)-cost(int(rows[e,1,t]));return d
 binary=OUT/'oracle.so';assert not binary.exists();argv=['g++','-O3','-std=c++17','-shared','-fPIC',str(CPP),'-o',str(binary)];subprocess.run(argv,check=True);lib=ctypes.CDLL(str(binary));p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fn=lib.down_copy_oracle;fn.argtypes=[p64,p32,p64,p32,ctypes.c_int,p64,p32];fn.restype=None;calls=queries=0
 def evaluate(delta,selected=ii):
  nonlocal calls,queries
  v=np.zeros(len(selected),np.int64);m=np.zeros(len(selected),np.int32);fn(base,routes,np.ascontiguousarray(delta,np.int64),selected,len(selected),v,m);calls+=1;queries+=len(selected);return v,m
 rng=np.random.default_rng(protocol['routeblind_seed']);blind=[];layout_arrays=[];deltas=[];blind_values=[]
 for j in range(protocol['routeblind_layouts']):
  layout=np.asarray([ee[int(rng.integers(len(ee)))] for ee in edges],np.int32);d=np.stack([delta_for(e,layout[e]) for e in range(64)]);v,m=evaluate(d);blind.append(dict(index=j,mean_work_ns=float(v.mean()),p99_work_ns=float(np.percentile(v,99)),mean_work_saving_pct=float(100*(1-v.mean()/original.mean()))));layout_arrays.append(layout);deltas.append(d);blind_values.append(v)
 bj=min(range(len(blind)),key=lambda j:blind[j]['mean_work_ns']);layout=layout_arrays[bj].copy();delta=deltas[bj].copy();values=blind_values[bj].copy();coordinate=[];proposals=0
 for pass_index in range(protocol['coordinate_passes']):
  changed=0
  for e in order:
   selected=members[e];old=delta[e].copy();bestedge=tuple(layout[e]);bestvalue=values[selected].copy();besttotal=int(bestvalue.sum())
   for edge in edges[e]:
    delta[e]=delta_for(e,edge);v,_=evaluate(delta,selected);proposals+=1;total=int(v.sum())
    if total<besttotal:besttotal=total;bestedge=edge;bestvalue=v.copy()
   delta[e]=delta_for(e,bestedge)
   if tuple(layout[e])!=tuple(bestedge):changed+=1
   layout[e]=bestedge;values[selected]=bestvalue
  full,masks=evaluate(delta);assert np.array_equal(full,values);coordinate.append(dict(pass_index=pass_index+1,changed_experts=changed,mean_work_ns=float(values.mean()),p99_work_ns=float(np.percentile(values,99))));print('coordinate',coordinate[-1],flush=True)
 final,masks=evaluate(delta);assert np.array_equal(final,values)
 # Independent Python exhaustive mask checks on fixed representative TRAIN rows.
 exhaustive=[]
 for q in [0,255,511]:
  candidates=[]
  for mask in range(64):
   load=base[q].copy()
   for k,e in enumerate(routes[q]):
    if mask>>k&1:load+=delta[e]
   candidates.append(int(load.max()))
  assert min(candidates)==int(final[q]) and candidates[int(masks[q])]==int(final[q]);exhaustive.append(dict(query=q,minimum_ns=int(final[q]),all64_masks_checked=True))
 def static_control(d):
  results=[]
  for start in [0,1]:
   bits=np.full(64,start,np.int8);loads=base+(d[routes]*bits[routes,None]).sum(1);vv=loads.max(1);passes=[]
   for p in range(2):
    accepted=0
    for e in order:
     qq=members[e];change=d[e]*(1 if bits[e]==0 else -1);candidate=loads[qq]+change;nv=candidate.max(1)
     if nv.sum()<vv[qq].sum():loads[qq]=candidate;vv[qq]=nv;bits[e]^=1;accepted+=1
    passes.append(dict(pass_index=p+1,accepted=accepted,mean_work_ns=float(vv.mean())))
   direct=base+(d[routes]*bits[routes,None]).sum(1);assert np.array_equal(direct,loads);results.append(dict(start=start,mean_work_ns=float(vv.mean()),p99_work_ns=float(np.percentile(vv,99)),bits=bits.tolist(),passes=passes,values=vv))
  return min(results,key=lambda r:r['mean_work_ns'])
 static=static_control(delta);blindstatic=static_control(deltas[bj]);static_values=static.pop('values');blindstatic_values=blindstatic.pop('values')
 # Logical-row ownership ledger, for both copy and corresponding zero-copy B.
 ledger=[]
 for e,(s,t) in enumerate(layout):
  rowids={c:list(range(sum(rows[e,1,:c]),sum(rows[e,1,:c+1]))) for c in range(16) if rows[e,1,c]};packet=rowids[int(s)];assert len(packet)==128
  b={c:rr.copy() for c,rr in rowids.items() if c!=s};b[int(t)]=b.get(int(t),[])+packet;assert sorted(r for rr in b.values() for r in rr)==list(range(1536));assert all(len(rr) in [128,256] for rr in b.values());assert sum(map(len,rowids.values()))==1536
  ledger.append(dict(expert=e,source=int(s),destination=int(t),source_logical_rowids=packet,destination_original_rowids=rowids.get(int(t),[]),A_chip_logical_rows=rowids,B_chip_logical_rows=b,duplicate_bytes=65536,duplicate_pages=4,duplicate_waves=1,B_source_job_absent=True,B_destination_one_contiguous_prefix=True,output_fringe_rows=0))
 np.savez_compressed(OUT/'work_arrays.npz',rows=rows,routes=routes,baseline_chip_load_ns=base,baseline_work_ns=original,baseline_native_ns=z['cost'],baseline_native_LB_ns=z['lower_bound'],routeblind_layouts=np.stack(layout_arrays),routeblind_values=np.stack(blind_values),routeblind_best_layout=layout_arrays[bj],routeblind_best_delta=deltas[bj],selected_layout=layout,selected_delta=delta,selected_oracle_values=final,selected_oracle_masks=masks,selected_static_values=static_values,routeblind_best_static_values=blindstatic_values)
 saving=100*(1-final.mean()/original.mean());nativeoptimistic=100*(1-(final.mean()+svc.host_submit_ns)/z['cost'].mean())
 for p,h in protocol['input_sha256'].items():assert sha(p)==h,p
 report=dict(status='PASS bounded necessary-work screen',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(SELF),oracle_source_sha256=sha(CPP),oracle_binary_sha256=sha(binary),service=protocol['service'],TRAIN_queries=512,baseline_native_mean_ns=float(z['cost'].mean()),baseline_chip_work_mean_ns=float(original.mean()),baseline_native_LB_mean_ns=float(z['lower_bound'].mean()),routeblind=blind,routeblind_best_index=bj,coordinate_passes=coordinate,coordinate_edge_proposals=proposals,oracle_calls=calls,oracle_query_evaluations=queries,selected_oracle_mean_ns=float(final.mean()),selected_oracle_p99_ns=float(np.percentile(final,99)),selected_oracle_work_saving_pct=float(saving),optimistic_native_saving_ceiling_using_hostprefix_pct=float(nativeoptimistic),same_edges_zero_copy_static=static,routeblind_best_same_edges_zero_copy_static=blindstatic,selected_vs_same_edges_static_work_saving_pct=float(100*(1-final.mean()/static_values.mean())),replica_budget=protocol['copies'],independent_python64_checks=exhaustive,gate_reached_2pct=bool(saving>=2 or nativeoptimistic>=2),scope='Finite32routeblind layouts plus atmost2TRAIN marginal coordinatepasses; perquery exact64choice necessary-work oracle only. No fullnative speed, selector, global optimum, inference or heldout claim.',seconds=time.monotonic()-began)
 dump(OUT/'copy_rowid_ledger.json',ledger);dump(OUT/'work_report.json',report);print(json.dumps({k:report[k] for k in ['status','baseline_chip_work_mean_ns','selected_oracle_mean_ns','selected_oracle_work_saving_pct','optimistic_native_saving_ceiling_using_hostprefix_pct','selected_vs_same_edges_static_work_saving_pct','gate_reached_2pct','seconds']},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
if __name__=='__main__':main()
