#!/usr/bin/env python3
"""Train-only GU row-stream packing oracle; no full-FFN speed claim.

All four frozen full-input geometry families receive identical ordering and
coalescing controls. No external/test payload is opened. Per-chip dynamic
programs minimize aggregate native context work, relaxing inter-chip/channel
queues, expert readiness and host demultiplexing. This is a necessary-work
screen, not the production resource-DAG scheduler.
"""
from datetime import datetime,timezone
import ctypes,hashlib,json,math,subprocess
from pathlib import Path
import numpy as np
from research_20261102_qwen38_independent_audit import templates,coverage,SHAPES

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/qwen38-placement'
DEST=BASE.parent/'gu-coalescing'
TRAIN=BASE.parent/'qwen38-trace-audit/train_decode.npz'
FROZEN=BASE/'native-full48/plans.frozen.json'
FREEZE_SHA='9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505'
TRAIN_SHA='953e8db87d4bc4a5b5b9a77f5aebae1578b1e522df5bf3ac6a32a591f9e8b145'
CASES=['group_g16','group_best','overlap_best','whole_row']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')


def geometry(plan,inv):
 result=np.zeros((513,16),dtype=np.int32)
 for e in range(513):
  spec=plan['fixed_shared']['gate_up'] if e==512 else plan['plans']['gate_up']
  label=spec['color'] if e==512 else spec['labels'][e]
  jobs=inv['gate_up'][spec['label']]['variants'][label];coverage(jobs,SHAPES['gate_up'])
  for j in jobs:
   assert j['columns']==2560 and j['column_start']==0
   result[e,j['chip']]=j['rows']
 return result


def orderings(rows,routes):
 active=np.zeros((len(routes),513),dtype=np.int32)
 for i,r in enumerate(routes):active[i,r]=1
 active[:,512]=1;co=active.T@active;freq=active.sum(axis=0)
 result={key:[] for key in ['identity','frequency','coactivation_path']}
 for chip in range(16):
  ids=np.flatnonzero(rows[:,chip]);result['identity'].append(ids.tolist())
  result['frequency'].append(sorted(ids.tolist(),key=lambda e:(-int(freq[e]),e)))
  # Standard greedy degree<=2 maximum-weight path heuristic, no novelty.
  parent={int(e):int(e) for e in ids};degree={int(e):0 for e in ids};neighbors={int(e):[] for e in ids}
  def find(e):
   while parent[e]!=e:parent[e]=parent[parent[e]];e=parent[e]
   return e
  edges=sorted([(int(co[a,b]),int(a),int(b)) for ix,a in enumerate(ids) for b in ids[ix+1:]],key=lambda t:(-t[0],t[1],t[2]))
  for _,a,b in edges:
   if degree[a]==2 or degree[b]==2 or find(a)==find(b):continue
   parent[find(a)]=find(b);degree[a]+=1;degree[b]+=1;neighbors[a].append(b);neighbors[b].append(a)
  start=min((e for e in ids if degree[int(e)]<2),key=lambda e:(-int(freq[e]),int(e)))
  path=[];last=None;current=int(start)
  while True:
   path.append(current);n=[e for e in neighbors[current] if e!=last]
   if not n:break
   last,current=current,n[0]
  assert set(path)==set(ids.tolist());result['coactivation_path'].append(path)
 return result


def pack(rows,order,align,padded):
 capacity=(192*65536//(align*2560))*align
 layout=np.zeros((513,16,2,3),dtype=np.int32);inventory=[]
 for chip in range(16):
  block=0;used=0;filled=[];useful=0
  for e in order[chip]:
   r=int(rows[e,chip]);useful+=r*2560
   if padded:
    slot=(r+align-1)//align*align
    if used+slot>capacity:filled.append(used);block+=1;used=0
    layout[e,chip,0]=[block,used,used+r];used+=slot
   else:
    left=r;segment=0
    while left:
     if used==capacity:filled.append(used);block+=1;used=0
     take=min(left,capacity-used);layout[e,chip,segment]=[block,used,used+take]
     segment+=1;used+=take;left-=take
    assert segment<=2
  filled.append(used)
  inventory.append(dict(chip=chip,useful_bytes=useful,blocks=len(filled),allocated_LSB_block_bytes=len(filled)*192*65536,
       programmed_dense_bytes=sum((n*2560+16383)//16384*16384 for n in filled),
       reserved_wave_bytes=sum((n*2560+65535)//65536*65536 for n in filled),
       row_capacity_per_block=capacity,programmed_padding_included=padded))
 assert layout.shape==(513,16,2,3)
 for e in range(513):
  for c in range(16):assert sum(int(v[2]-v[1]) for v in layout[e,c])==int(rows[e,c])
 return np.ascontiguousarray(layout),inventory


def summarize(x):return dict(mean=float(np.mean(x)),p99=float(np.percentile(x,99)),minimum=int(np.min(x)),maximum=int(np.max(x)))


def main():
 assert sha(FROZEN)==FREEZE_SHA and sha(TRAIN)==TRAIN_SHA
 DEST.mkdir(parents=True,exist_ok=True)
 cpp=Path(__file__).with_suffix('.cpp');libpath=DEST/'gu_coalescing.so'
 assert not libpath.exists();subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(libpath)],check=True)
 inv=templates();freeze=json.loads(FROZEN.read_text());fr={(r['layer'],r['case']):r for r in freeze['records']}
 with np.load(TRAIN) as f:routes=f['routes'][:,::4,:3].copy()
 assert routes.shape==(16,200,3,10)
 source={str(p):sha(p) for p in [Path(__file__),cpp,FROZEN,TRAIN,libpath]}
 protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=source,
    train_only='All16originaltrainrequests, steps0,4,...796;3200queries/layer;first3layers;noexternal/test.',
    geometry='Original frozen GU ownership unchanged, fullH2560 only, one copy. Reorder complete rows; down unchanged and not merged.',
    contracts={'strict_WL':'128rows=5waves, blockcapacity4864rows/190waves; no plane-start extension.',
               'conditional_page':'32rows=5pages, blockcapacity4896rows/765pages; arbitraryplane/page start NOT verified.',
               'aligned_stride':'Each expert begins at joint row/address boundary; padded complete dummy rows count toward storage/read/output.',
               'dense_stream':'Tight row stream; selected intervals require preceding legal-start prefix. Inactive rows really read/compute/output.'},
    objective='Per-chip exact DP aggregate context ns=28us*waves+250ns/call+C/2ns input+2ns/emitted4B row; complete-wave outputs; sharedalwaysactive.',
    relaxation='All GU work initially ready; independent chips; channel/controller/PCIe queues and expert GU->activation->down DAG not scheduled. Mean/P99 are scalar work bounds, not attainable FFN times.',
    no_CR=True,no_new_hardware=True,no_replication=True)
 write(DEST/'protocol.frozen.json',protocol)
 lib=ctypes.CDLL(str(libpath));i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
 lib.gu_coalescing.argtypes=[i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64]
 records=[];originals=[];proofs=[]
 for layer in range(3):
  selected=np.ascontiguousarray(routes[:,:,layer].reshape(-1,10),dtype=np.int32)
  chosen=np.column_stack([selected,np.full(len(selected),512,dtype=np.int32)])
  for case in CASES:
   rec=fr[layer,case];p=ROOT/rec['plan'];assert sha(p)==rec['plan_sha256'];plan=json.loads(p.read_text());rr=geometry(plan,inv)
   weights=(rr.astype(np.int64)*2560+65535)//65536
   er=weights*65536//2560
   single=weights*28000+(rr>0)*1530+er*2
   old=single[chosen].sum(axis=1);oldwaves=weights[chosen].sum(axis=1)
   originals.append(dict(layer=layer,case=case,context_ns=summarize(old.max(axis=1)),waves=summarize(oldwaves.max(axis=1))))
   if case=='whole_row':
    sizes=rr[chosen].astype(np.int64)*2560
    pooled=(sizes.sum(axis=1)+65535)//65536
    assert np.array_equal(pooled,oldwaves)
    proofs.append(dict(layer=layer,all3200x16query_chip_cases_have_zero_NAND_coalescing_headroom=True,
        nonzero_rows=[int(v) for v in sorted(set(rr.ravel())-{0})],max_fractional_wave_slack_per_job=0.0703125,
        max11jobs_total_slack=0.7734375,proof='sum ceil(x_i)-ceil(sum x_i)=floor(sum fractional_slacks)=0. Ignoring adjacency/start alignment/blocks favors fusion.'))
   orders=orderings(rr,selected)
   for name,order in orders.items():
    for align in [128,32]:
     for padded in [False,True]:
      layout,inventory=pack(rr,order,align,padded);out=np.zeros((3200,16,4),dtype=np.int64)
      lib.gu_coalescing(layout,selected,len(selected),align,2560,out)
      tag=f'L{layer}.{case}.{name}.a{align}.{"stride" if padded else "dense"}'
      np.savez_compressed(DEST/(tag+'.npz'),cost=out,original_context_ns=old,original_waves=oldwaves,layout=layout)
      records.append(dict(layer=layer,case=case,ordering=name,alignment_rows=align,padded_stride=padded,label=tag,
          context_ns=summarize(out[:,:,0].max(axis=1)),waves=summarize(out[:,:,1].max(axis=1)),
          commands_per_query=summarize(out[:,:,2].sum(axis=1)),emitted_rows_per_query=summarize(out[:,:,3].sum(axis=1)),
          mean_context_reduction_vs_original_pct=100*(1-out[:,:,0].max(axis=1).mean()/old.max(axis=1).mean()),
          mean_wave_reduction_vs_original_pct=100*(1-out[:,:,1].max(axis=1).mean()/oldwaves.max(axis=1).mean()),
          inventory=inventory,npz_sha256=sha(DEST/(tag+'.npz'))))
   print('complete',layer,case,flush=True)
 for p,h in source.items():assert sha(p)==h
 # Every geometry gets the same three fixed-order controls and exact query DP.
 best=[]
 for align in [128,32]:
  for case in CASES:
   for layer in range(3):
    r=min([x for x in records if x['alignment_rows']==align and x['layer']==layer and x['case']==case],key=lambda x:x['context_ns']['mean'])
    best.append({k:r[k] for k in ['alignment_rows','layer','case','label','context_ns','waves','mean_context_reduction_vs_original_pct']})
 write(DEST/'screen.json',dict(status='complete train-only GU native-work screen; no fullFFN latency claim',
    source_sha256=source,originals=originals,whole_row_zero_headroom_proofs=proofs,records=records,best_train_only=best,
    external_or_heldout_opened=False,original_placement_refit=False,static_ordering_train_only=True,
    preserved_shared_and_routed_GU_same_input=True,no_new_dtype_or_context=True))
 print(json.dumps(best,indent=2))


if __name__=='__main__':main()
