#!/usr/bin/env python3
"""Independent supplementary all-layer sums/statistics/addresses, 24 traces."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,json,sys
import numpy as np
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[1];P=BASE/'all-layer-fixed-map-transfer';E=P/'execution';NAMES=('free','common_cyclic','phase_independent','tail_cyclic');CONTROLS=NAMES[1:]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def mod(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def freeze():
 assert not (OUT/'protocol.frozen.json').exists()
 own=BASE/'independent/replay-statistics-audit-v2'
 paths=[Path(__file__),own/'audit.py',own/'source-snapshot/research_20261103_ordinary_read_backend.py',own/'source-snapshot/research_20261103_wave_drain.py',own/'source-snapshot/research_20261102_whole_row_compiler.py',P/'protocol.frozen.json',P/'execution_protocol.frozen.json',E/'completion.frozen.json',E/'analysis.json',E/'costs.npz',E/'bootstrap_clusters.npz',E/'physical/report.json',BASE/'data-capture/granite4tiny/decode.routes.npz',BASE/'protocol.frozen.json']
 paths+=list((E/'physical').glob('*.resident.npz'))+list(E.glob('*.csv'))+[P/f'plans/{n}.frozen.npz' for n in NAMES]
 selection=[{'plan':n,'copies':c,'layer':[0,13,26,39][(j+i)%4],'query':[0,255,511][i]} for j,n in enumerate(NAMES) for c in (1,4) for i in range(3)]
 paths += [E/f'traces/input{x["copies"]}.{x["plan"]}.L{x["layer"]}.q{x["query"]}.npz' for x in selection]
 dump(OUT/'protocol.frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'files':{str(f):sha(f) for f in paths},'trace_selection':selection,'rules':['Check producer full completion and source references; keep secondary distinct from original prospectiveL0.','All8costarrays32x16x40 and lowerbounds: metadata/L0 bridge; recompute sum40 then linearP99, everylayer/prompt/source CSV and ALL negative records. No costs independently replayed except24predeclared traces.','Recompute 2000 PCG64seed20261115 promptcluster draw and CI using independently implemented sorted quantiles/integerweightedmeans.','24 deterministic sampled traces cover4maps/2fanouts/4layers/3q positions; frozenPython plus independently maintained scalarheap graph, allnodes anddependency/owner checks.','All4resident ledgers: independently reconstruct complete row bands fromjobs; verify96,960jobs/map, full40onecopy/shared, pages/waves/oneblock/nooverlap, perchipcapacity and523/525/519/523blocks.','No infer/fit/mapchoice/newsource/outputABI/wholemodelperformance claim.']})
 print('protocol',sha(OUT/'protocol.frozen.json'))
def run():
 proto=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==proto['source_sha256']
 for f,h in proto['files'].items():assert sha(f)==h
 seal=json.loads((E/'completion.frozen.json').read_text());records=seal['files']
 if isinstance(records,dict):records=[{'path':k,'sha256':v} for k,v in records.items()]
 for x in records:
  f=Path(x['path']);assert sha(f)==x['sha256']
 assert sha(E/'completion.frozen.json')=='65609b122e1d74dc2cf212d8192906d9ac3efd69514a991aec62e35a0848f93e'
 own=BASE/'independent/replay-statistics-audit-v2';helpers=mod(own/'audit.py','own_prospective_helpers');q=helpers.quantile;compare=helpers.compare
 sys.path.insert(0,str(own/'source-snapshot'));b=mod(own/'source-snapshot/research_20261103_ordinary_read_backend.py','all40_independent_backend');w=mod(own/'source-snapshot/research_20261103_wave_drain.py','all40_independent_scalar')
 service=json.loads((BASE/'protocol.frozen.json').read_text())['hardware_primary'];plans={n:load(P/f'plans/{n}.frozen.npz') for n in NAMES};raw=load(BASE/'data-capture/granite4tiny/decode.routes.npz');a=load(E/'costs.npz');saved=json.loads((E/'analysis.json').read_text());physical=json.loads((E/'physical/report.json').read_text());old=load(BASE/'replay-adapter-v2/prospective/costs.npz')
 for n in NAMES:
  for c in (1,4):
   k=f'input{c}.{n}';assert a[k+'.cost'].shape==(32,16,40) and np.all(a[k+'.cost']>=a[k+'.bound'])
   if n!='tail_cyclic':
    for suffix in ('cost','bound'):assert np.array_equal(a[k+'.'+suffix][:,:,0],old[k+'.'+suffix])
 assert np.array_equal(a['prompt_ids'],raw['prompt_ids']) and np.array_equal(a['source_keys'],raw['source_keys']) and np.array_equal(a['token_positions'],raw['token_positions'])
 draw=np.random.Generator(np.random.PCG64(20261115)).integers(0,32,(2000,32));assert np.array_equal(draw,load(E/'bootstrap_clusters.npz')['prompt_indices']);mult=np.array([np.bincount(r,minlength=32) for r in draw],np.int64)
 def stats(v):return {'mean_ns':float(v.sum(dtype=np.int64)/v.size),'p99_ns':float(q(v,.99)),'max_ns':int(v.max())}
 def ci(v):return [float(q(v,.025)),float(q(v,.975))]
 sources=raw['source_keys'].astype(str);ids=raw['prompt_ids'].astype(str);tables=[];prompts=[];stables=[];neg=[];nlayer=[];profiles={};extra=[]
 for c in (1,4):
  values={n:a[f'input{c}.{n}.cost'] for n in NAMES};tot={n:x.sum(2,dtype=np.int64) for n,x in values.items()};profiles[f'input{c}']={'plans':{},'comparisons':{}}
  for n in NAMES:
   v=values[n];t=tot[n];profiles[f'input{c}']['plans'][n]=stats(t)
   for L in range(40):tables.append({'input_copies':c,'plan':n,'layer':L,**stats(v[:,:,L])})
   for p in range(32):prompts.append({'input_copies':c,'plan':n,'prompt_index':p,'prompt_id':ids[p],'source':sources[p],**stats(t[p])})
   for s in sorted(set(sources)):stables.append({'input_copies':c,'plan':n,'source':s,'prompts':int((sources==s).sum()),**stats(t[sources==s])})
  for control in CONTROLS:
   f=tot['free'];o=tot[control];d=f-o
   rec={'free_minus_control_mean_ns':float(d.sum()/512),'mean_gain_percent':100*(o.mean()-f.mean())/o.mean(),'free_minus_control_p99_ns':float(q(f,.99)-q(o,.99)),'p99_gain_percent':100*float(q(o,.99)-q(f,.99))/float(q(o,.99)),'regressed_queries':int((d>0).sum()),'tied_queries':int((d==0).sum()),'improved_queries':int((d<0).sum()),'max_regression_ns':max(0,int(d.max())),'max_regression_percent':max(0.,float(np.max(100*d/o)))}
   if control in ('phase_independent','tail_cyclic'):
    fm=mult@f.sum(1)/512;om=mult@o.sum(1)/512;fq=q(f[draw].reshape(2000,512),.99,axis=1);oq=q(o[draw].reshape(2000,512),.99,axis=1)
    rec['paired_prompt_cluster_95pct_CI']={'free_minus_control_mean_ns':ci(fm-om),'mean_gain_percent':ci(100*(om-fm)/om),'free_minus_control_p99_ns':ci(fq-oq),'p99_gain_percent':ci(100*(oq-fq)/oq)}
   profiles[f'input{c}']['comparisons'][control]=rec
   for p,t in np.argwhere(d>0):neg.append({'input_copies':c,'control':control,'prompt_index':int(p),'prompt_id':ids[p],'source':sources[p],'decode_index':int(t),'token_position':int(raw['token_positions'][p,t]),'free_ns':int(f[p,t]),'control_ns':int(o[p,t]),'regression_ns':int(d[p,t])})
   ld=values['free']-values[control]
   for p,t,L in np.argwhere(ld>0):nlayer.append({'input_copies':c,'control':control,'prompt_index':int(p),'prompt_id':ids[p],'source':sources[p],'decode_index':int(t),'layer':int(L),'regression_ns':int(ld[p,t,L])})
   pd=d.sum(1);extra.append({'copies':c,'control':control,'prompt_mean_improve':int((pd<0).sum()),'prompt_mean_tie':int((pd==0).sum()),'prompt_mean_regress':int((pd>0).sum())})
 errors=compare(profiles,saved['profiles']);assert len(neg)==saved['negative_sum_records'] and len(nlayer)==saved['negative_layer_records']
 csvchecks=[]
 for filename,expect in [('all_40_layer_stats.csv',tables),('all_32_prompt_stats.csv',prompts),('both_source_stats.csv',stables),('all_regressed_40layer_sum_queries.csv',neg),('all_regressed_layer_queries.csv',nlayer)]:
  with (E/filename).open() as f:actual=list(csv.DictReader(f))
  assert len(actual)==len(expect)
  for ar,er in zip(actual,expect):
   assert set(ar)==set(er)
   for k,v in er.items():
    if isinstance(v,(int,np.integer)):assert int(ar[k])==v
    elif isinstance(v,(float,np.floating)):assert abs(float(ar[k])-v)<=1e-8
    else:assert ar[k]==v
  csvchecks.append({'file':filename,'all_rows_exact':len(expect)})
 addresschecks=[]
 for n in NAMES:
  plan=plans[n];z=load(E/f'physical/{n}.resident.npz');ledger=z['ledger'];assert ledger.shape==(96960,13)
  one=[]
  for e in range(65):
   for ph in (0,1):
    row=0;lo,hi=map(int,plan['offsets'][e,ph])
    for j in range(lo,hi):
     chip,R,C,pg,wav=map(int,plan['jobs'][j]);assert pg==(R*C+16383)//16384 and wav==(pg+3)//4
     one.append([j,e,ph,chip,row,R,C,pg,wav]);row+=R
    assert row==int(plan['total_rows'][e,ph])
  expected=np.concatenate([np.column_stack([np.full(len(one),L,np.int64),np.array(one,np.int64)]) for L in range(40)])
  assert np.array_equal(ledger[:,:10],expected);assert np.all(ledger[:,12]==0) and np.all(ledger[:,10:12]>=0) and np.all(ledger[:,11]+ledger[:,9]<=192)
  blocks=[];waves=[]
  for chip in range(32):
   rows=ledger[ledger[:,4]==chip];nb=int(rows[:,10].max())+1;blocks.append(nb);waves.append(int(rows[:,9].sum()));assert nb==(waves[-1]+191)//192
   for block in range(nb):
    rr=rows[rows[:,10]==block];order=np.argsort(rr[:,11],kind='stable');rr=rr[order];assert len(rr)>0 and np.all(rr[:-1,11]+rr[:-1,9]<=rr[1:,11])
  assert np.array_equal(z['chip_blocks'],blocks) and np.array_equal(z['chip_waves'],waves)
  fact={'jobs':len(ledger),'logical_rows':int(ledger[:,6].sum()),'coefficient_useful_bytes':int((ledger[:,6]*ledger[:,7]).sum()),'coefficient_copies_bytes':0,'programmed_pages':int(ledger[:,8].sum()),'programmed_page_bytes':int(ledger[:,8].sum())*16384,'reserved_waves':int(ledger[:,9].sum()),'reserved_wave_bytes':int(ledger[:,9].sum())*65536,'per_chip_waves':waves,'per_chip_blocks':blocks,'total_blocks':sum(blocks),'allocated_bytes':sum(blocks)*192*65536,'whole_fringe_rows_drained_and_dropped':int((ledger[:,8]*16384//ledger[:,7]-ledger[:,6]).sum()),'partial_row_bytes_charged_and_dropped':int((ledger[:,8]*16384%ledger[:,7]).sum())}
  for k,v in fact.items():assert physical[n][k]==v,(n,k)
  assert physical[n]['shared_included_once_per_layer'];addresschecks.append({'plan':n,**fact,'all_rows_and_addresses_exact':True,'fixed_chip_assignment_block_count_lower_bound_attained':True})
 assert [x['total_blocks'] for x in addresschecks]==[523,525,519,523]
 tracechecks=[]
 for pick in proto['trace_selection']:
  n,c,L,t=pick['plan'],pick['copies'],pick['layer'],pick['query'];plan=plans[n];route=raw['routes'].reshape(512,40,6)[t,L];s=dict(service,input_channel_copies=c)
  st=load(E/f'traces/input{c}.{n}.L{L}.q{t}.npz');assert 'trace' in st
  g=w.ScalarGraph.build(plan,list(map(int,route))+[64],s,0,c);cost,tr=g.run();assert np.array_equal(tr,st['trace']) and cost==int(a[f'input{c}.{n}.cost'].reshape(512,40)[t,L]);check=g.audit(tr)
  ref=b.ReferenceGraph.build(plan,np.r_[route,64],b.Service(**s)).run();assert ref['cost']==cost and np.array_equal(ref['trace'],tr)
  tracechecks.append({**pick,'cost_ns':cost,'all_trace_fields_exact':True,**check})
 for f,h in proto['files'].items():assert sha(f)==h
 audit={'status':'PASS_SECONDARY_ALL40_SUM_STATISTICS_PHYSICAL_AND24TRACES','utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.frozen.json'),'arrays_validated_costs':163840,'independently_replayed_costs_and_full_traces':24,'trace_nodes':sum(x['nodes'] for x in tracechecks),'trace_checks':tracechecks,'all_source_statistics_max_abs_error':max(errors,default=0),'csv_checks':csvchecks,'physical':addresschecks,'profiles':profiles,'prompt_negative_counts':extra,'scope':'Supplementary rules fixed after capture before costs; not original prospective primary. Every sum/statistic/negative/capacity ledger independently checked; native costs independently replayed only at24declared traces, not all163840. No capture/infer/refit/placedfullmodel/device proof.'}
 dump(OUT/'audit.json',audit);dump(OUT/'completion.json',{'status':'SEALED_PASS','files':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completion.json' and '__pycache__' not in str(f)}})
 print(json.dumps({'status':audit['status'],'traces':24,'negative_sum':len(neg),'negative_layer':len(nlayer),'blocks':[x['total_blocks'] for x in addresschecks],'audit_sha256':sha(OUT/'audit.json'),'completion_sha256':sha(OUT/'completion.json')}))
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
