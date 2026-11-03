#!/usr/bin/env python3
"""Independent array-derived whole-model address/ledger audit; no backend import."""
from collections import defaultdict
from pathlib import Path
import argparse,csv,gc,gzip,hashlib,json,time
import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment')
PARSER=argparse.ArgumentParser();PARSER.add_argument('--campaign',required=True,choices=['full-campaign-v5-W8-training','full-campaign-v6-W8-input4']);ARGS=PARSER.parse_args()
COMPILER=ROOT/'results/research-2026-11-02/round2/ordinary-read-compiler'
BASE=COMPILER/ARGS.campaign
OUT=Path(__file__).resolve().parent/ARGS.campaign
OUT.mkdir(parents=True,exist_ok=True)
SHAPES={'deepseek':(2048,1408,64,6,26,2816),'granite4tiny':(1536,512,64,6,40,1024),'qwen38':(2560,640,512,10,48,640)}
FIELDS=('expert','phase','chip','row_start','rows','columns','useful_bytes','pages','waves','moe_layer_index','physical_model_layer')
LEDGERS=('useful_bytes','programmed_page_bytes','reserved_wave_bytes','allocated_lsb_block_bytes')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,x):
 p=OUT/name;p.write_text(json.dumps(x,indent=2)+'\n');return p
def path(s):
 p=Path(s);return p if p.is_absolute() else ROOT/p
def cdiv(x,y):return (x+y-1)//y

def main():
 assert not (OUT/'protocol.frozen.json').exists(),'Refuse to replace audit freeze.'
 totals=json.loads((BASE/'model_totals.json').read_text())
 campaign=json.loads((BASE/'summary.json').read_text())
 records=campaign['records'];assert len(records)==132 and len(totals['records'])==12
 assert campaign['status'].startswith('complete') and totals['status'].startswith('complete')
 anchor_path=COMPILER/'W8_selected_plans.frozen.json';assert sha(anchor_path)=='00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00'
 anchor=json.loads(anchor_path.read_text());entry=next(x for x in anchor['campaigns'] if x['campaign']==ARGS.campaign)
 for k in ('summary','protocol','snapshot_manifest'):assert sha(path(entry[k]))==entry[k+'_sha256']
 frozen_inputs_path=path(anchor['training_manifest']);assert sha(frozen_inputs_path)==anchor['training_manifest_sha256']
 frozen_inputs=json.loads(frozen_inputs_path.read_text())
 sources=json.loads((BASE/'source-snapshot/manifest.json').read_text())
 inputs={str(p):sha(p) for p in [Path(__file__),BASE/'model_totals.json',BASE/'summary.json',BASE/'protocol.frozen.json',BASE/'source-snapshot/manifest.json']}
 inputs[str(anchor_path)]=sha(anchor_path);inputs[str(frozen_inputs_path)]=sha(frozen_inputs_path)
 for src in sources['sources_and_inputs']:
  p=path(src['snapshot']);assert sha(p)==src['sha256'];inputs[str(p)]=src['sha256']
 for r in records:
  for a in r['artifacts']:
   for key in ('arrays','plan'):
    p=path(a[key]);assert sha(p)==a[key+'_sha256'];inputs[str(p)]=a[key+'_sha256']
 for r in totals['records']:
  p=path(r['physical_plan']);assert sha(p)==r['physical_plan_sha256'];inputs[str(p)]=r['physical_plan_sha256']
  p=BASE/'models'/f"{r['model']}.cmd{r['command_ns']}.{r['family']}.selection-cost.npy";inputs[str(p)]=sha(p)
 # Verify existing source route arrays once, never fetch new data.
 route_cache={}
 for r in records:
  p=path(r['training']['path']);h=r['training']['sha256']
  if str(p) not in route_cache:
   assert sha(p)==h
   with np.load(p) as a:route_cache[str(p)]={k:a[k].copy() for k in a.files}
   data=route_cache[str(p)];assert np.all(data['completed']) and str(data['split'])=='train' and str(data['representation'])=='fixed_full_FFN_W8A8_canonical_only'
   assert data['prompt_ids'].tolist()==frozen_inputs['prompt_ids'] and data['routes'].shape[:2]==(32,16)
   declaration=frozen_inputs['models'][r['model']];assert declaration['sha256']==h and path(declaration['archive'])==p
   assert data['routes'].shape==tuple(declaration['shape']) and data['layers'].tolist()==declaration['layers']
   inputs[str(p)]=h
 write('protocol.frozen.json',dict(status='frozen before global address evaluation',inputs_sha256=inputs,
  scope='All12 sealed actual-W8 model/family/command global coefficient maps in this campaign. Reconstruct jobs and logical rows from native array offsets, rows, full input C and known shapes; compare global logical metadata only after reconstruction. Audit address intervals, independent page/wave and coefficient-capacity ledgers, route alignment and saved layer-cost sums. No event replay, compiler imports, new fitting or hardware calibration.',
  excluded='Numerical coefficient payload/programming, PE output ABI, actual hardware capacity, overprovisioning, row-scale metadata, router/attention/SSM/dense-first-layer weights, heldout performance and v3 family-search attribution.'))
 results=[]
 for rec in totals['records']:
  started=time.monotonic();model,family,cmd=rec['model'],rec['family'],rec['command_ns']
  d,i,e,k,layers,shared_i=SHAPES[model]
  rs=sorted([r for r in records if r['model']==model and r['command_ns']==cmd],key=lambda r:r['layer'])
  assert [r['layer'] for r in rs]==list(range(layers))
  expected=[];costs=[];expert_base=0;route_identity=None;physical_layers=[]
  for r in rs:
   assert tuple(r['shape'])==SHAPES[model]
   s=r['service'];assert s['command_ns']==cmd and s.get('input_channel_copies',1)==entry['input_channel_copies']
   assert s['chips']==16 and s['channels']==8 and s['planes']==4 and s['page_bytes']==16384 and s['block_wordlines']==192 and s['input_bytes']==1 and s['output_bytes']==4 and s['output_fringe']==1 and s['ordinary_read_ns']==28000 and s['host_submit_ns']==2000 and s['input_buffer_bytes']==32768
   assert s['pcie_bytes_per_second']==8_000_000_000 and s['channel_bytes_per_second']==2_000_000_000 and s['activation_ns']==0 and s['combine_ns']==0 and s['down_first']==0
   P,L,b=s['planes'],s['block_wordlines'],s['page_bytes'];B=P*b
   a=next(x for x in r['artifacts'] if x['family']==family)
   with np.load(path(a['arrays'])) as f:
    offsets=f['offsets'].copy();jobs=f['jobs'].copy();columns=f['input_columns'].copy();total_rows=f['total_rows'].copy();costs.append(f['cost'].copy());sel=f['selection_routes'].copy()
   assert offsets.shape==(e+1,2,2) and columns.shape==total_rows.shape==(e+1,2)
   assert np.all(columns[:e]==[d,i]) and np.array_equal(columns[e],[d,shared_i])
   assert np.all(total_rows[:e]==[2*i,d]) and np.array_equal(total_rows[e],[2*shared_i,d])
   assert int(columns.max())*s['input_bytes']<=s['input_buffer_bytes']
   tr=r['training'];fit_idx=np.array(tr['selected_indices']);selection_idx=np.array(r['selection_training_indices'])
   identity=(tr['path'],tr['sha256'],tr['original_events'],tuple(fit_idx.tolist()),tuple(selection_idx.tolist()))
   if route_identity is None:route_identity=identity
   else:assert route_identity==identity,'Cross-layer cost sums use different token identities.'
   original=route_cache[tr['path']]
   li=r['layer'];physical=li if model=='qwen38' else int(original['layers'][li]);assert physical==tr['physical_model_layer'];physical_layers.append(physical)
   raw=(original['routes'][:,::4,li] if model=='qwen38' else original['routes'][:,:,li]).reshape(-1,k)
   assert len(raw)==tr['original_events'] and np.array_equal(sel,raw[fit_idx[selection_idx]])
   assert len(sel)==64 and np.all((sel>=0)&(sel<e))
   assert np.all(np.diff(np.sort(sel,axis=1),axis=1)>0)
   local=json.loads(path(a['plan']).read_text());assert local['service']==s and local['layer']==li and local['model']==model
   assert len(local['logical'])==len(jobs)
   next_job=0
   for ex in range(e+1):
    for ph in range(2):
     lo,hi=map(int,offsets[ex,ph]);assert lo==next_job and hi>lo and hi<=len(jobs)
     cursor=0
     for j in range(lo,hi):
      chip,rows,C,pages,waves=map(int,jobs[j]);useful=rows*C
      assert 0<=chip<s['chips'] and rows>0 and C==int(columns[ex,ph])
      assert pages==cdiv(useful,b) and waves==cdiv(pages,P) and 1<=waves<=L
      core=(ex,ph,chip,cursor,rows,C,useful,pages,waves)
      assert tuple(local['logical'][j][f] for f in FIELDS[:9])==core
      expected.append((ex+expert_base,ph,chip,cursor,rows,C,useful,pages,waves,li,physical))
      cursor+=rows
     assert cursor==int(total_rows[ex,ph]);next_job=hi
   assert next_job==len(jobs)
   expert_base+=e+1
  expected=np.asarray(expected,dtype=np.int64)
  p=path(rec['physical_plan'])
  with gzip.open(p,'rt') as f:doc=json.load(f)
  service=rs[0]['service'];assert all(r['service']==service for r in rs)
  assert doc['service']==service and doc['model']==model and doc['family']==family and doc['command_ns']==cmd
  assert doc['layers']==list(range(layers)) and doc['complete_all_MoE_layers'] is True and rec['complete_all_MoE_layers'] is True
  assert len(doc['logical'])==len(doc['addresses'])==len(expected)
  for j,(x,want) in enumerate(zip(doc['logical'],expected)):
   assert tuple(x[f] for f in FIELDS)==tuple(want),(model,family,cmd,'global logical mismatch',j)
  intervals=np.empty((len(expected),4),dtype=np.int64)
  chip_blocks=[set() for _ in range(service['chips'])]
  for j,(a,want) in enumerate(zip(doc['addresses'],expected)):
   ex,ph,chip,rowstart,rows,C,useful,pages,waves,li,physical=map(int,want)
   assert a['job']==j and a['chip']==chip and a['reserved_wordlines']==waves and a['pages']==pages and a['first_plane']==0
   assert a['page_order']=='wordline-major; plane=page_index%planes'
   block,wl=int(a['block']),int(a['wordline']);assert block>=0 and 0<=wl and wl+waves<=L
   # Real useful bytes and all padded programmed pages lie in a single same-address plane-block bundle.
   assert useful<=pages*b<=waves*B
   assert wl*B+pages*b<=L*B and wl*B+useful<=L*B
   intervals[j]=chip,block,wl,wl+waves;chip_blocks[chip].add(block)
  order=np.lexsort((intervals[:,2],intervals[:,1],intervals[:,0]));iv=intervals[order]
  same=(iv[1:,0]==iv[:-1,0])&(iv[1:,1]==iv[:-1,1])
  assert np.all(iv[1:,2][same]>=iv[:-1,3][same]),'Overlapping reserved waves or programmed coefficient pages.'
  counts=[len(x) for x in chip_blocks]
  assert all(x==set(range(len(x))) for x in chip_blocks)
  assert counts==doc['chip_blocks']==rec['chip_blocks']
  derived={'useful_bytes':int(expected[:,6].sum()),'programmed_page_bytes':int(expected[:,7].sum())*b,'reserved_wave_bytes':int(expected[:,8].sum())*B,'allocated_lsb_block_bytes':sum(counts)*L*B}
  assert derived['useful_bytes']==layers*3*d*(e*i+shared_i)
  assert derived['useful_bytes']<=derived['programmed_page_bytes']<=derived['reserved_wave_bytes']<=derived['allocated_lsb_block_bytes']
  assert doc['service'].get('input_channel_copies',1)==entry['input_channel_copies']
  for key in LEDGERS:assert derived[key]==doc[key]==rec[key],(model,family,cmd,key,derived[key],doc[key],rec[key])
  values=np.stack(costs).sum(axis=0)
  saved=np.load(BASE/'models'/f'{model}.cmd{cmd}.{family}.selection-cost.npy')
  assert np.array_equal(values,saved)
  assert float(values.mean())==rec['mean_ns'] and float(np.percentile(values,99))==rec['p99_ns']
  certified=sum(bool(next(x for x in r['selection'] if x['family']==family)['finite_pool_certified_best']) for r in rs)
  assert certified==rec['finite_pool_mean_certified_layers']
  result=dict(model=model,family=family,command_ns=cmd,layers=layers,physical_model_layers=physical_layers,experts_including_shared=expert_base,jobs=len(expected),**derived,chip_blocks=counts,
   input_channel_copies=entry['input_channel_copies'],coefficient_payload_MiB=derived['useful_bytes']/2**20,allocated_lsb_bundle_MiB=derived['allocated_lsb_block_bytes']/2**20,
   route_alignment='64 identical source token identities across all layers; actual per-layer routes checked against source arrays',
   mean_training_selection_FFN_service_ns=float(values.mean()),finite_pool_mean_certified_layers=certified,
   elapsed_seconds=time.monotonic()-started,status='PASS')
  results.append(result);write('partial.json',results);print(model,cmd,family,'PASS',len(expected),'jobs',flush=True)
  del doc,expected,intervals,iv;gc.collect()
 assert all(sha(p)==h for p,h in inputs.items())
 write('results.json',dict(status='PASS all12 actual-W8 global maps',input_channel_copies=entry['input_channel_copies'],anchor_sha256=sha(anchor_path),maps=len(results),logical_and_address_jobs=sum(x['jobs'] for x in results),records=results,
  interpretation='Virtual coefficient address legality and simultaneous-resident LSB bundle ledgers only; not physical hardware capacity or model latency. Deduplicated fitting optimality/numeric capture quality remain outside address correctness. Deep dense-prefix coefficients and service are separately excluded.'))
 with (OUT/'ledger.csv').open('w') as f:
  keys=['model','command_ns','family','layers','jobs',*LEDGERS]
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in results)
 write('completion.json',dict(status='complete',outputs_sha256={str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}))

if __name__=='__main__':main()
