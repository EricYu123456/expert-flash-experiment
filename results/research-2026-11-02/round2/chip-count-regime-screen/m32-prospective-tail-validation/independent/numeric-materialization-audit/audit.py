#!/usr/bin/env python3
"""Independent read-only bank and full-FFN CPU proof. No producer helper import."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gzip,hashlib,json,time
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]
ROOT=HERE.parents[6]
ROUND=ROOT/'results/research-2026-11-02/round2'
OLD=ROUND/'shared-materialize'
NEW=PARENT/'numeric-materialization'
PAGE=16384;BLOCK=192*4*PAGE
NAMES=('free','common_cyclic','phase_independent')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while chunk:=f.read(4<<20):h.update(chunk)
 return h.hexdigest()
def bh(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def up(n,d):return (int(n)+int(d)-1)//int(d)
def readjson(p):return json.loads(Path(p).read_text())

def freeze():
 assert not (HERE/'protocol.frozen.json').exists()
 producer=readjson(NEW/'protocol.frozen.json');report=readjson(NEW/'report.json')
 assert sha(NEW/'protocol.frozen.json')=='2ba2f0eae49ed14ae1c7ada5ff0e3ddd128ef1979368defe78d8c0d64339512e'
 assert sha(NEW/'completion.json')=='ad23a4242cfb174ee99e7413c3482a4a8d545d83d61318e9ec3fb25970919de1'
 manifest=readjson(OLD/'onecopy/manifest.json')
 paths=[OLD/'onecopy/manifest.json',OLD/'logical_scales.f32',OLD/'logical_scales.json',OLD/'source_ranges.json',OLD/'fixtures.npz',
   NEW/'protocol.frozen.json',NEW/'report.json',NEW/'completion.json',NEW/'train_route_fixtures.frozen.npz',NEW/'numeric_arrays.npz',
   ROUND/'w8-format-training/granite4tiny/decode.routes.npz',PARENT/'protocol.frozen.json']
 for name in NAMES:paths.extend([PARENT/'plans'/(name+'.frozen.npz'),NEW/'banks'/name/'manifest.json.gz'])
 refs=[{'path':str(p),'sha256':sha(p)} for p in paths]
 oldbanks=[{'path':str(OLD/c['path']),'sha256':c['sha256'],'size':BLOCK} for c in manifest['chips']]
 newbanks=[{'path':c['path'],'sha256':c['sha256'],'size':BLOCK} for name in NAMES for c in report['storage'][name]['chips']]
 assert len(oldbanks)==16 and len(newbanks)==96 and not any('data-capture' in r['path'] for r in refs+oldbanks+newbanks)
 protocol={'status':'frozen_before_independent_bank_or_numeric_execution','utc':datetime.now(timezone.utc).isoformat(),
  'source_sha256':sha(__file__),'references':refs,'original_banks':oldbanks,'three_layout_banks':newbanks,
  'source_reconstruction':'Use129 execution_views plus8 single-copy sharedGU pair views(choice1), not producer reconstruct()/storage-allocation placement. Reconstruct every row and host scale from original16banks. Crosscheck1312sourceallocations and195functional projection hashes.',
  'new_address_oracle':'Derive every logical row interval from frozen plan offsets/jobs; derive page addresses from block/WL/plane fields. Compare all7272jobs full pages to independently reconstructed W8. Crosscheck producer gzmetadata afterwards.',
  'integer_oracle':'Explicit I32 element products reduced with dtype=I64, no numpy matmul or producer GEMV helper. Every complete emitted padding row is also computed and dropped.',
  'float_oracle':'Independent round-half-away A8; explicit F32 ufunc stages for scale, sigmoid/SiLU, gating and ordered mix. Native output ABI is not inferred.',
  'cases':'All4 Cartesian cases x0/x1 plusTRAINq0/q511, all3layouts plus canonical. No old fixture reference outputs used.',
  'comparisons':'All stage arrays versus independent canonical and producer saved arrays, exact bytes; full sharedexpert included. Verify every old/newbank SHA before and after run.',
  'limits':'Only software L0 physical bytes/address and full-FFN scalar composition. No new capture/test payload, no GGUF/read/requantization/inference/device/timing/new pairedfullmodel claim.',
  'target_compute_seconds':90}
 dump(HERE/'protocol.frozen.json',protocol)
 print(json.dumps({'protocol_sha256':sha(HERE/'protocol.frozen.json'),'source_sha256':sha(__file__)},indent=2))

def q_a8(v):
 v=np.asarray(v,np.float32);assert np.all(np.isfinite(v))
 m=max(abs(float(x)) for x in v)
 scale=np.float32(m/127.) if m else np.float32(1.)
 normalized=np.divide(v,scale,dtype=np.float32)
 magnitude=np.abs(normalized);whole=np.floor(magnitude).astype(np.int32)
 rounded=whole+(np.subtract(magnitude,whole,dtype=np.float32)>=np.float32(.5))
 signed=np.where(np.signbit(normalized),-rounded,rounded)
 return np.clip(signed,-127,127).astype(np.int8),scale

def nonlinearity(y):
 assert y.dtype==np.float32 and len(y)%2==0
 g=y[:len(y)//2];u=y[len(y)//2:]
 ex=np.exp(np.negative(g,dtype=np.float32),dtype=np.float32)
 denominator=np.add(ex,np.float32(1),dtype=np.float32)
 sigmoid=np.divide(np.float32(1),denominator,dtype=np.float32)
 return np.multiply(np.multiply(g,sigmoid,dtype=np.float32),u,dtype=np.float32)

def integer_rows(w,x):
 assert w.dtype==x.dtype==np.int8 and w.shape[1]==len(x)
 products=np.multiply(w.astype(np.int32),x.astype(np.int32)[None,:],dtype=np.int32)
 return np.sum(products,axis=1,dtype=np.int64)

def f32_scale(z,sx,scales):
 factor=np.multiply(scales,sx,dtype=np.float32)
 return np.multiply(z.astype(np.float32),factor,dtype=np.float32)

def run():
 started=time.monotonic();p=readjson(HERE/'protocol.frozen.json');assert sha(__file__)==p['source_sha256']
 for r in p['references']:assert sha(r['path'])==r['sha256']
 bankchecks=[]
 for r in p['original_banks']+p['three_layout_banks']:
  assert Path(r['path']).stat().st_size==r['size'] and sha(r['path'])==r['sha256']
  bankchecks.append({'path':r['path'],'sha256':r['sha256']})
 io_hash_seconds=time.monotonic()-started
 manifest=readjson(OLD/'onecopy/manifest.json');sm=readjson(OLD/'logical_scales.json')
 allscales=np.fromfile(OLD/'logical_scales.f32',dtype='<f4')
 banks={c['chip']:np.memmap(OLD/c['path'],mode='r',dtype=np.uint8) for c in manifest['chips']}
 views={key:list(vals) for key,vals in manifest['execution_views'].items()}
 views['shared.gu']=[v for pair in range(8) for v in manifest['shared_gu_legal_views'][str(pair)]['1']]
 assert len(views)==130
 W={};S={};source_allocations=set();source_info={}
 for key,parts in views.items():
  nr=sm[key]['rows'];C=parts[0]['columns'];w=np.empty((nr,C),np.int8);seen=np.zeros(nr,np.uint8)
  so=sm[key]['offset']//4;s=allscales[so:so+nr].copy()
  assert len(s)==nr and bh(s)==sm[key]['sha256'] and np.all(np.isfinite(s)) and np.all(s>0)
  for v in parts:
   r=int(v['logical_row_start']);n=int(v['rows']);c=int(v['columns']);assert c==C and v['start_plane']==v['byte_in_plane_page']==0
   at=(int(v['block'])*192+int(v['start_wordline']))*4*PAGE
   assert at==v['linear_byte_offset'] and v['scale_file_byte_offset']==sm[key]['offset']+4*r
   raw=np.asarray(banks[v['chip']][at:at+n*C]);assert len(raw)==n*C
   w[r:r+n]=raw.view(np.int8).reshape(n,C);seen[r:r+n]+=1
   aid=v['storage_allocation'];assert aid not in source_allocations;source_allocations.add(aid)
   allocation=manifest['storage_allocations'][aid]
   assert allocation['logical_matrix']==key and allocation['row_start']==r and allocation['rows']==n
   assert hashlib.sha256(raw).hexdigest()==allocation['payload_sha256']
  assert np.all(seen==1);W[key]=w;S[key]=s
  source_info[key]={'shape':[nr,C],'weight_sha256':bh(w),'scales_sha256':bh(s)}
 assert source_allocations==set(range(1312)) and sum(x.nbytes for x in W.values())==155713536
 projection_checks=[]
 for r in readjson(OLD/'source_ranges.json'):
  prefix='shared' if r['expert'] is None else 'routed.'+str(r['expert'])
  if '_down_' in r['tensor']:w=W[prefix+'.down'];s=S[prefix+'.down']
  else:
   total=W[prefix+'.gu'];sc=S[prefix+'.gu'];half=len(total)//2;at=0 if '_gate_' in r['tensor'] else half
   w=total[at:at+half];s=sc[at:at+half]
  assert w.shape==(r['rows'],r['columns']) and bh(w)==r['functional_i8_sha256'] and bh(s)==r['functional_row_scales_sha256']
  projection_checks.append({'tensor':r['tensor'],'expert':r['expert'],'W8_and_scale_hash_exact':True})
 assert len(projection_checks)==195

 physical={};storage=[]
 for name in NAMES:
  with np.load(PARENT/'plans'/(name+'.frozen.npz')) as z:plan={k:z[k] for k in z.files}
  with gzip.open(NEW/'banks'/name/'manifest.json.gz','rt') as f:declared=json.load(f)
  byjob={a['job']:a for a in declared['allocations']};assert len(byjob)==2424
  newbanks=[np.memmap(NEW/'banks'/name/f'chip{c:02d}.bin',mode='r',dtype=np.uint8) for c in range(32)]
  cursor_by_chip=[[] for _ in range(32)];matrix_views={};totals={'jobs':0,'pages':0,'waves':0,'valid_rows':0,'emitted_rows':0,'partial_tail_bytes':0,'useful_bytes':0,'page_padding_bytes':0}
  for e in range(65):
   for phase in range(2):
    key=('shared' if e==64 else 'routed.'+str(e))+('.gu' if phase==0 else '.down')
    C=W[key].shape[1];start=0;coverage=np.zeros(len(W[key]),np.uint8);parts=[]
    lo,hi=map(int,plan['offsets'][e,phase])
    for j in range(lo,hi):
     chip,n,c,pages,waves=map(int,plan['jobs'][j]);ac,block,wl,span,ap,plane=map(int,plan['addresses'][j])
     assert (chip,pages,waves)==(ac,ap,span) and c==C and block==plane==0 and wl+waves<=192
     assert pages==up(n*C,PAGE) and waves==up(pages,4)
     at=(block*192*4+wl*4+plane)*PAGE
     raw=newbanks[chip][at:at+pages*PAGE];payload=W[key][start:start+n].view(np.uint8).ravel()
     assert len(raw)==pages*PAGE and np.array_equal(raw[:n*C],payload) and not np.any(raw[n*C:])
     emitted=len(raw)//C;tail=len(raw)%C
     assert emitted>=n and not np.any(raw[emitted*C:])
     d=byjob[j]
     expected={'logical_matrix':key,'chip':chip,'block':block,'wordline':wl,'plane':plane,'row_start':start,
      'rows':n,'columns':C,'pages':pages,'waves':waves,'physical_byte_offset':at,'scale_byte_offset':sm[key]['offset']+4*start,
      'weight_bytes':n*C,'emitted_rows':emitted,'partial_tail_bytes':tail}
     assert all(d[k]==v for k,v in expected.items())
     assert hashlib.sha256(raw[:n*C]).hexdigest()==d['payload_sha256'] and hashlib.sha256(raw).hexdigest()==d['programmed_sha256']
     parts.append({'chip':chip,'address':at,'pages':pages,'row_start':start,'rows':n,'C':C,'emitted':emitted,'partial':tail})
     coverage[start:start+n]+=1;start+=n;cursor_by_chip[chip].append((wl,wl+waves))
     for k,v in [('jobs',1),('pages',pages),('waves',waves),('valid_rows',n),('emitted_rows',emitted),('partial_tail_bytes',tail),('useful_bytes',n*C),('page_padding_bytes',len(raw)-n*C)]:totals[k]+=v
    assert start==len(W[key]) and np.all(coverage==1);matrix_views[key]=parts
  for spans in cursor_by_chip:
   spans.sort();assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
  assert totals['jobs']==2424 and totals['pages']==9764 and totals['waves']==2441 and totals['useful_bytes']==155713536
  assert totals['valid_rows']==167424
  physical[name]=(newbanks,matrix_views)
  storage.append({'layout':name,'full_page_payloads_and_zeros_exact':True,'all130_matrices_onecover':True,
   'all_job_row_addresses_and_scale_offsets_exact':True,'all32_oneblock_intervals_nonoverlap':True,'ledger':totals})
 bank_and_row_seconds=time.monotonic()-started-io_hash_seconds

 def mvm(key,x,sx,layout=None):
  if layout is None:z=integer_rows(W[key],x);return z,f32_scale(z,sx,S[key]),None
  nb,v=physical[layout];zall=np.empty(len(W[key]),np.int64);extra={'valid_rows':0,'computed_complete_rows':0,'dropped_complete_rows':0,'dropped_partial_bytes':0,'pages_read':0}
  for a in v[key]:
   raw=nb[a['chip']][a['address']:a['address']+a['pages']*PAGE]
   ww=raw[:a['emitted']*a['C']].view(np.int8).reshape(a['emitted'],a['C'])
   zi=integer_rows(ww,x);r=a['row_start'];n=a['rows'];assert not np.any(zi[n:])
   zall[r:r+n]=zi[:n]
   extra['valid_rows']+=n;extra['computed_complete_rows']+=len(zi);extra['dropped_complete_rows']+=len(zi)-n
   extra['dropped_partial_bytes']+=a['partial'];extra['pages_read']+=a['pages']
  return zall,f32_scale(zall,sx,S[key]),extra

 with np.load(OLD/'fixtures.npz') as f:fixtures={k:f[k].copy() for k in ('x0','x1','alpha0','alpha1')}
 with np.load(NEW/'train_route_fixtures.frozen.npz') as f:indices=f['query_indices'].copy();routes=f['routes'].copy()
 with np.load(ROUND/'w8-format-training/granite4tiny/decode.routes.npz') as f:train=f['routes'][:,:,0].reshape(512,6)
 assert np.array_equal(indices,[0,511]) and np.array_equal(routes,train[indices])
 producer_arrays=np.load(NEW/'numeric_arrays.npz');saved={};case_checks=[];compared={};numeric_started=time.monotonic()
 def evaluate(fi,route,layout=None):
  aq,as_=q_a8(fixtures[f'x{fi}']);mix=np.zeros(1536,np.float32);out={};fringe={}
  for ordinal,e in enumerate(list(map(int,route))+[64]):
   key='shared' if e==64 else 'routed.'+str(e)
   gi,gu,gl=mvm(key+'.gu',aq,as_,layout)
   act=nonlinearity(gu);dq,ds=q_a8(act);di,down,dl=mvm(key+'.down',dq,ds,layout)
   for stage,value in [('gu_i64',gi),('gu_f32',gu),('activation_f32',act),('down_a8',dq),('down_a8_scale_f32',np.asarray(ds)),('down_i64',di),('down_f32',down)]:out[key+'.'+stage]=value
   fringe[key]={'GU':gl,'Down':dl}
   term=np.multiply(down,fixtures[f'alpha{fi}'][ordinal],dtype=np.float32) if ordinal<6 else down
   mix=np.add(mix,term,dtype=np.float32)
  out['output']=mix;return out,fringe
 for fi in (0,1):
  for query,route in zip(indices,routes):
   label=f'fixture{fi}.trainq{query}';canonical,_=evaluate(fi,route)
   for stage,val in canonical.items():
    key=label+'.canonical.'+stage;assert val.tobytes()==producer_arrays[key].tobytes();saved[key]=val
   for name in NAMES:
    actual,fringe=evaluate(fi,route,name)
    for stage,value in actual.items():
     assert value.dtype==canonical[stage].dtype and value.shape==canonical[stage].shape and value.tobytes()==canonical[stage].tobytes()
     key=label+'.'+name+'.'+stage;assert value.tobytes()==producer_arrays[key].tobytes();saved[key]=value
     tag='ordered_mixed_output_f32' if stage=='output' else stage.rsplit('.',1)[1]
     compared[tag]=compared.get(tag,0)+value.size
    case_checks.append({'fixture':fi,'TRAIN_query':int(query),'ordered_experts':list(map(int,route)),
      'layout':name,'every_stage_matches_independent_canonical_and_producer_bytes':True,
      'output_sha256':bh(actual['output']),'physical_fringe_computed_then_dropped':fringe})
 assert len(case_checks)==12 and sum(compared.values())==571476
 numeric_seconds=time.monotonic()-numeric_started
 for r in p['references']:assert sha(r['path'])==r['sha256']
 for r in p['original_banks']+p['three_layout_banks']:assert sha(r['path'])==r['sha256']
 np.savez_compressed(HERE/'independent_numeric_arrays.npz',**saved)
 dump(HERE/'source_reconstruction.independent.json',{'matrix_count':len(W),'source_views_and_allocations_once':len(source_allocations),
  'matrices':source_info,'functional_projection_hash_checks':projection_checks})
 report={'status':'PASS_INDEPENDENT_PHYSICAL_AND_FULL_FFN_NUMERIC_RECOMPUTATION','utc':datetime.now(timezone.utc).isoformat(),
  'protocol_sha256':sha(HERE/'protocol.frozen.json'),'source_sha256':sha(__file__),
  'independence':['No producer source/helper imported or executed.','Canonical rows reconstructed from original execution views and source banks, not producer W dictionaries.',
    'New row/address oracle derived from frozen plan arrays, then compared with producer manifests.','Explicit I32 element products + I64 reduction instead of matmul.',
    'Independent F32 quantization/nonlinearity/rescale/mix; all stages compared bytewise with own canonical and producer outputs.'],
  'old_banks_full_sha_checked':16,'new_banks_full_sha_checked':96,'all_hashes_rechecked_after':True,
  'source_matrices':130,'source_allocations':1312,'functional_projection_hashes':195,'storage':storage,
  'cases':case_checks,'matrix_phases':168,'compared_scalars_by_stage':compared,'total_compared_scalars':sum(compared.values()),'bit_mismatches':0,
  'seconds':{'initial_hash_io':io_hash_seconds,'source_and_new_bank_checks':bank_and_row_seconds,'numeric':numeric_seconds,'total':time.monotonic()-started},
  'numeric_scope':'Fixed explicit x/alpha fixtures crossed with two actual TRAIN ordered top6 sets. Not actual full-model input activations for those TRAIN queries.',
  'limitations':['CPU physical row bytes/address and L0 complete-FFN proof only.','No new capture/test payload, GGUF read, requantization or model inference.',
    'No new three-map paired full-model proof; no native PE dtype/scales/reset/ECC/partial-flush/timing/device proof.','Only valid logical rows receive their original scales; zero fringe output is read/computed/dropped without invented fringe scales.']}
 dump(HERE/'audit.json',report)
 files=[f for f in HERE.rglob('*') if f.is_file() and f.name!='completion.json']
 dump(HERE/'completion.json',{'status':'SEALED_INDEPENDENT_PASS','files':{str(f.relative_to(HERE)):sha(f) for f in sorted(files)}})
 print(json.dumps({'status':report['status'],'total_compared_scalars':report['total_compared_scalars'],'seconds':report['seconds'],
  'audit_sha256':sha(HERE/'audit.json'),'completion_sha256':sha(HERE/'completion.json')},indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);a=ap.parse_args()
 {'freeze':freeze,'run':run}[a.mode]()
