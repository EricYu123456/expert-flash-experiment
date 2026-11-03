#!/usr/bin/env python3
"""Route-blind half-chip symmetry control, not a new placement method."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import gzip,hashlib,heapq,json,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
COMP=ROOT/'results/research-2026-11-02/round2/ordinary-read-compiler'
OUT=ROOT/'results/research-2026-11-02/round2/capacity-symmetry-control'
ANCHOR=COMP/'W8_selected_plans.frozen.json'
VALIDATED=COMP/'independent/input-copy-and-physical-guards/snapshots'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(p):return Path(p) if Path(p).is_absolute() else ROOT/p
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def cdiv(a,b):return (int(a)+int(b)-1)//int(b)

def inventory(jobs,L):
 h=np.zeros((16,L+1),dtype=np.int64)
 np.add.at(h,(jobs[:,0],jobs[:,4]),1);assert h[:,0].sum()==0
 return h

def bfd_hist(hist,L):
 """Exact BFD count for size multiset;bulk-fill same-size items per bin."""
 residual=[0]*(L+1);blocks=0
 for w in range(L,0,-1):
  n=int(hist[w])
  while n:
   fit=next((left for left in range(w,L+1) if residual[left]),None)
   if fit is None:
    per=L//w;full,part=divmod(n,per)
    if full:blocks+=full;residual[L-per*w]+=full
    if part:blocks+=1;residual[L-part*w]+=1
    n=0;continue
   per=fit//w;full=min(residual[fit],n//per)
   if full:
    residual[fit]-=full;residual[fit-per*w]+=full;n-=full*per
   else:
    assert n<per;residual[fit]-=1;residual[fit-n*w]+=1;n=0
 return blocks


def pack(jobs,L):
 """Same BFD size/index policy as root,with residual-capacity buckets."""
 addresses=[None]*len(jobs);counts=[];intervals=[]
 for chip in range(16):
  queues=[[] for _ in range(L+1)];available=0;blocks=0
  ids=np.flatnonzero(jobs[:,0]==chip).tolist();ids.sort(key=lambda j:(-int(jobs[j,4]),j))
  for j in ids:
   w=int(jobs[j,4]);possible=available & ~((1<<w)-1)
   if possible:
    left=(possible & -possible).bit_length()-1;block=heapq.heappop(queues[left])
    if not queues[left]:available &= ~(1<<left)
   else:block=blocks;blocks+=1;left=L
   begin=L-left;rest=left-w
   assert 0<=begin and begin+w<=L
   heapq.heappush(queues[rest],block);available |=1<<rest
   addresses[j]=dict(job=j,chip=chip,block=block,wordline=begin,reserved_wordlines=w,pages=int(jobs[j,3]),first_plane=0,page_order='wordline-major; plane=page_index%planes')
   intervals.append((chip,block,begin,begin+w,j))
  counts.append(blocks)
 # Independent sorted reserved interval check rather than allocator state.
 intervals.sort();assert all((a[0],a[1])!=(b[0],b[1]) or a[3]<=b[2] for a,b in zip(intervals,intervals[1:]))
 return addresses,counts


def choose_flip(H,L):
 """Exact count-class search if every layer has identical swap histogram delta."""
 layers=len(H);base=H.sum(0);delta=H[:,np.arange(16)^8,:]-H
 assert np.all(delta==delta[0]),'Stop:count-class reduction does not apply;do not silently approximate.'
 wave=np.arange(L+1,dtype=np.int64);records=[]
 for k in range(layers+1):
  hist=base+k*delta[0];assert np.all(hist>=0)
  loads=hist@wave;lb=[cdiv(x,L) for x in loads];blocks=[bfd_hist(h,L) for h in hist]
  objective=(sum(blocks),max(blocks),int(np.sum((loads*16-int(loads.sum()))**2)),k)
  records.append(dict(flipped_layers=k,blocks=blocks,waves=loads.tolist(),work_blocks_LB=lb,objective=list(objective)))
 chosen=min(records,key=lambda x:tuple(x['objective']));k=chosen['flipped_layers'];flags=[int(l<k) for l in range(layers)]
 # Relaxing each chip's binary decisions independently is another valid bound.
 per_layer_load=H@wave
 coordinate_min=np.minimum(per_layer_load,per_layer_load[:,np.arange(16)^8]).sum(0)
 coordinate_lb=[cdiv(x,L) for x in coordinate_min]
 best_count_lb=min(sum(r['work_blocks_LB']) for r in records)
 return flags,dict(method='Exact0..Lflip-count search:alllayers verified identical swap-delta histograms;prefix-layer tie rule;minimize BFDtotalblocks,maxblocks,waveimbalance,k. No routes/costs used.',chosen=chosen,candidates=records,coordinate_relaxation_min_waves=coordinate_min.tolist(),coordinate_relaxation_min_blocks=coordinate_lb,
  minimum_work_LB_over_all_flipcounts=best_count_lb,achieved_BFD_total_blocks=chosen['objective'][0],certified_minimum_total_blocks_within_swap8_family=chosen['objective'][0]==best_count_lb,
  stronger_coordinate_LB_tight=chosen['objective'][0]==sum(coordinate_lb),scope='Exact over thisbinarywhole-layerchiprenamingfamily anddeclared BFDpolicy. Only a matched work LB certifies arbitrary binpacking within thisfamily;not a general all-placementcapacity optimum.')


def trace_rename(t):
 x=t.copy();array=(x[:,0]>=11)&(x[:,0]<27);x[array,0]=11+((x[array,0]-11)^8)
 chip=x[:,6]>=0;x[chip,6]^=8;return x

def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.frozen.json').exists()
 assert sha(ANCHOR)=='00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00'
 anchor=json.loads(ANCHOR.read_text());snap=OUT/'snapshots';snap.mkdir()
 inputs={str(ANCHOR):sha(ANCHOR),str(Path(__file__)):sha(Path(__file__))};campaigns=[]
 for name in ('research_20261103_ordinary_read_backend.py','research_20261103_ordinary_read_native.cpp','research_20261102_whole_row_compiler.py'):
  src=VALIDATED/name;shutil.copy2(src,snap/name);inputs[str(src)]=sha(src)
 for ent in anchor['campaigns']:
  for key in ('summary','protocol','snapshot_manifest'):
   p=resolve(ent[key]);assert sha(p)==ent[key+'_sha256'];inputs[str(p)]=sha(p)
  base=resolve(ent['summary']).parent;summary=json.loads(resolve(ent['summary']).read_text());totals=json.loads((base/'model_totals.json').read_text())
  assert summary['status'].startswith('complete') and len(totals['records'])==12
  inputs[str(base/'model_totals.json')]=sha(base/'model_totals.json')
  for r in summary['records']:
   for a in r['artifacts']:
    for key in ('arrays','plan'):
     p=resolve(a[key]);assert sha(p)==a[key+'_sha256'];inputs[str(p)]=a[key+'_sha256']
  for rec in totals['records']:
   p=resolve(rec['physical_plan']);assert sha(p)==rec['physical_plan_sha256'];inputs[str(p)]=rec['physical_plan_sha256']
  campaigns.append((ent,base,summary,totals))
 dump(OUT/'protocol.frozen.json',dict(inputs_sha256=inputs,scope='Postfreeze route-blind strong symmetry/packing control. Every layer jointly swaps allrouted+shared GU+Down chips cxor8,with nativejob order/rows/fullC/padpages unchanged. No weightscopied ornewinstructions. All24maps treated identically.',selection='Static per-layer chip wave histograms only. No routecost,TRAINfrequency,heldout orresult-dependent swap choice. TRAIN2query traces are used only after staticchoice to verify graphisomorphism.',assumptions='16identicalchips;chip%8channel unchanged;homogeneousordinaryservice;no CR/addressdependentlatency. Commands globallyprioritized byphase/expert/nodeID ratherthanchipid. Controller/PCIe/host/channels unchanged. Same graph/nodeorder.',limits='Virtual coefficientmaps only,no physicaldevicebudget/nativePE numericalproof. Exact min BFD count inverified count-equivalence class;arbitrarypacking optimum onlywhen lowerbound matches. Standard symmetrycontrol,not a new method.'))
 sys.path.insert(0,str(snap))
 from research_20261103_ordinary_read_backend import Service,NativeReplay
 native=NativeReplay(OUT/'build');results=[];diagnostics=[]
 # Tiny count-packer checks against explicit address-packer,independent of realmaps.
 rng=np.random.default_rng(38611)
 for _ in range(60):
  L=int(rng.integers(2,30));sizes=rng.integers(1,L+1,size=int(rng.integers(1,80)));j=np.array([[0,1,1,1,int(w)] for w in sizes],dtype=np.int32)
  _,count=pack(j,L);assert count[0]==bfd_hist(np.bincount(sizes,minlength=L+1),L)
 for ent,base,summary,totals in campaigns:
  target=OUT/ent['campaign'];target.mkdir()
  for rec in totals['records']:
   model,cmd,family=rec['model'],rec['command_ns'],rec['family'];layers=sorted([r for r in summary['records'] if r['model']==model and r['command_ns']==cmd],key=lambda x:x['layer'])
   s=Service(**layers[0]['service']);assert s.chips==16 and s.channels==8;L=s.block_wordlines;P=s.planes;b=s.page_bytes;B=P*b
   local=[];H=[]
   for r in layers:
    a=next(a for a in r['artifacts'] if a['family']==family)
    with np.load(resolve(a['arrays'])) as f:plan={k:f[k].copy() for k in ('offsets','jobs','input_columns','total_rows')}
    local.append((r,a,plan));H.append(inventory(plan['jobs'],L))
   H=np.array(H);flags,choice=choose_flip(H,L)
   label=f'{model}.cmd{cmd}.{family}';dump(target/(label+'.choice.json'),dict(flags=flags,**choice))
   # Verify an actual full swap before relying on chosen identities;then chosen flips.
   for li in sorted(set([0,len(layers)//2,len(layers)-1])):
    r,a,old=local[li]
    with np.load(resolve(a['arrays'])) as f:routes=f['selection_routes'][[0,63]].copy();saved=f['cost'][[0,63]].copy()
    for flip in sorted(set([1,flags[li]])):
     new={k:v.copy() for k,v in old.items()}
     if flip:new['jobs'][:,0]^=8
     before=native.run(old,routes,r['shape'][2],1,s,True);after=native.run(new,routes,r['shape'][2],1,s,True)
     expected=trace_rename(before['trace']) if flip else before['trace']
     if not (np.array_equal(before['cost'],saved) and np.array_equal(before['cost'],after['cost']) and np.array_equal(before['lower_bound'],after['lower_bound']) and np.array_equal(expected,after['trace'])):
      dump(OUT/'invariance_counterexample.json',dict(campaign=ent['campaign'],label=label,layer=li,flip=flip,before=before['cost'].tolist(),after=after['cost'].tolist(),source='Do not continue packing after failed isomorphism.'));raise AssertionError('Native trace isomorphism failed.')
     diagnostics.append(dict(campaign=ent['campaign'],label=label,layer=li,flip=flip,queries=2,exact_cost=True,exact_LB=True,exact_resource_chip_renamed_trace=True))
   offsets=[];jobs=[];columns=[];totals_rows=[];logical=[];jb=eb=0
   for li,(r,a,old) in enumerate(local):
    new=old['jobs'].copy()
    if flags[li]:new[:,0]^=8
    assert np.array_equal(new[:,1:],old['jobs'][:,1:]) and np.array_equal(new[:,0]%8,old['jobs'][:,0]%8)
    offsets.append(old['offsets'].astype(np.int64)+jb);jobs.append(new);columns.append(old['input_columns']);totals_rows.append(old['total_rows'])
    doc=json.loads(resolve(a['plan']).read_text())
    for k,j in enumerate(doc['logical']):
     x=dict(j,expert=j['expert']+eb,chip=j['chip']^8 if flags[li] else j['chip'],moe_layer_index=li,physical_model_layer=r['training']['physical_model_layer']);assert x['chip']==int(new[k,0]);logical.append(x)
    jb+=len(new);eb+=len(old['offsets'])
   jobs=np.concatenate(jobs);offsets=np.concatenate(offsets).astype(np.int32);columns=np.concatenate(columns);totals_rows=np.concatenate(totals_rows)
   hist=inventory(jobs,L);assert np.array_equal(hist,H.sum(0)+sum(flags)*(H[:,np.arange(16)^8,:]-H)[0])
   addresses,counts=pack(jobs,L);assert counts==choice['chosen']['blocks']
   with gzip.open(resolve(rec['physical_plan']),'rt') as f:original=json.load(f)
   assert len(logical)==len(original['logical'])==len(jobs)
   for new,old in zip(logical,original['logical']):assert {k:v for k,v in new.items() if k!='chip'}=={k:v for k,v in old.items() if k!='chip'}
   useful=int((jobs[:,1].astype(np.int64)*jobs[:,2]).sum());pages=int(jobs[:,3].astype(np.int64).sum());waves=int(jobs[:,4].astype(np.int64).sum())
   assert useful==rec['useful_bytes'] and pages*b==rec['programmed_page_bytes'] and waves*B==rec['reserved_wave_bytes']
   assert sum(counts)*L>=waves
   # Full row coverage and physical page containment independently rechecked.
   for e,phases in enumerate(offsets):
    for ph,(lo,hi) in enumerate(phases):
     bands=sorted((x['row_start'],x['row_start']+x['rows']) for x in logical[lo:hi]);assert bands[0][0]==0 and bands[-1][1]==totals_rows[e,ph] and all(x[1]==y[0] for x,y in zip(bands,bands[1:]))
   for a,j,x in zip(addresses,jobs,logical):
    assert tuple(map(int,j))==(x['chip'],x['rows'],x['columns'],x['pages'],x['waves']) and x['useful_bytes']==int(j[1])*int(j[2])
    assert cdiv(int(j[1])*int(j[2]),b)==int(j[3]) and cdiv(int(j[3]),P)==int(j[4])
    assert a['wordline']*B+int(j[3])*b<=L*B
   # Check untransformed BFD reconstruction exactly against root's frozen allocator.
   untransformed=np.concatenate([x[2]['jobs'] for x in local]);oldaddr,oldcounts=pack(untransformed,L)
   assert oldcounts==rec['chip_blocks'] and oldaddr==original['addresses']
   arrays=target/(label+'.arrays.npz');np.savez_compressed(arrays,offsets=offsets,jobs=jobs,input_columns=columns,total_rows=totals_rows,swap8_flags=np.array(flags,dtype=np.int8))
   physical=target/(label+'.physical.json.gz')
   payload=dict(model=model,family=family,command_ns=cmd,service=asdict(s),layers=original['layers'],complete_all_MoE_layers=True,swap8_flags=flags,logical=logical,addresses=addresses,chip_blocks=counts,useful_bytes=useful,programmed_page_bytes=pages*b,reserved_wave_bytes=waves*B,allocated_lsb_block_bytes=sum(counts)*L*B,row_coverage_and_address_nonoverlap=True,scope='Postfreeze routeblind chiprename control,no weight duplication;coefficientonly virtual sameaddressplaneblockbundles.')
   with gzip.GzipFile(filename=str(physical),mode='wb',mtime=0) as f:f.write(json.dumps(payload,separators=(',',':')).encode())
   result=dict(campaign=ent['campaign'],model=model,command_ns=cmd,family=family,layers=len(layers),flipped_layers=sum(flags),jobs=len(jobs),old_blocks=rec['chip_blocks'],new_blocks=counts,old_allocated_bytes=rec['allocated_lsb_block_bytes'],new_allocated_bytes=sum(counts)*L*B,old_GiB=rec['allocated_lsb_block_bytes']/2**30,new_GiB=sum(counts)*L*B/2**30,saved_blocks=sum(rec['chip_blocks'])-sum(counts),useful_bytes=useful,reserved_wave_bytes=waves*B,minimum_blocks_coordinate_relaxation=choice['coordinate_relaxation_min_blocks'],minimum_blocks_countsearch_LB=choice['minimum_work_LB_over_all_flipcounts'],minimum_within_symmetry_family_certified=choice['certified_minimum_total_blocks_within_swap8_family'],arrays=str(arrays),arrays_sha256=sha(arrays),physical_plan=str(physical),physical_plan_sha256=sha(physical))
   results.append(result);dump(OUT/'partial.json',results);print(ent['campaign'],label,'blocks',sum(rec['chip_blocks']),'->',sum(counts),'flip',sum(flags),'cert',result['minimum_within_symmetry_family_certified'],flush=True)
 assert all(sha(p)==h for p,h in inputs.items())
 dump(OUT/'summary.json',dict(status='PASS all24maps routeblind symmetry control',records=results,diagnostics=diagnostics,trace_comparisons=len(diagnostics),native_costs_compared=sum(x['queries'] for x in diagnostics),native_provenance=native.provenance,scope='Graphisomorphism under the declared homogeneous service contract,not native hardware measurement or allgeometrycapacity optimality.'))
 dump(OUT/'completion.json',dict(status='complete',sha256={str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='completion.json'}))

if __name__=='__main__':main()
