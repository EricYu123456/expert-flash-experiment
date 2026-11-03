#!/usr/bin/env python3
"""Independent full-route replay and physical/contract audit of frozen pilot."""
import collections,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';PILOT=BASE/'native-joint-refinement/pilot-v1';OUT=BASE/'native-joint-refinement/independent-pilot-v1'
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261103_ordinary_read_backend as backend
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 began=time.monotonic();OUT.mkdir(exist_ok=False);protocol_path=PILOT/'protocol.frozen.json';protocol=json.loads(protocol_path.read_text());protocol_sha=sha(protocol_path);assert protocol_sha=='97f0905057941a6dd5fc3d6e90abcf160935bd35bd4d2e64523d93e012493761'
 launch_path=PILOT/'launch_results.json';launch=json.loads(launch_path.read_text());assert len(launch)==8 and all(r['exit_code']==0 for r in launch)
 source=ROOT/'scripts/research_20261103_native_joint_refine.py';old=BASE/'native-joint-refinement/independent-preflight/reviewed_target.py';core=lambda text:text[text.index('class ExactFit:'):text.index('\ndef main():')];assert core(source.read_text())==core(old.read_text());assert sha(source)==protocol['source_sha256']
 inputs={str(protocol_path):protocol_sha,str(launch_path):sha(launch_path),str(old):sha(old)}
 for entry in protocol['inputs']:assert sha(entry['path'])==entry['sha256'];inputs[entry['path']]=entry['sha256']
 native=backend.NativeReplay(OUT/'build');checks=[];traces={};ref_families=set();recomputed={};initials={}
 for launched in launch:
  dest=pathlib.Path(launched['path']);result_path=dest/'result.json';r=json.loads(result_path.read_text());inputs[str(result_path)]=sha(result_path);array_path=pathlib.Path(r['arrays']);assert sha(array_path)==r['arrays_sha256'];inputs[str(array_path)]=sha(array_path)
  assert r['script_sha256']==protocol['source_sha256'] and r['protocol_sha256']==protocol_sha and pathlib.Path(r['protocol'])==protocol_path
  assert r['family']==launched['family'] and r['seed']==launched['seed'];assert r['seed'] in protocol['seeds'] and r['family'] in protocol['families'];assert r['fit']['proposals']==protocol['iterations'] and r['fit']['initial_temperature_ns']==protocol['initial_temperature_ns']
  case_path=pathlib.Path(r['case']);assert str(case_path) in protocol['cases'] and sha(case_path)==r['case_sha256'];case=json.loads(case_path.read_text());assert case['service']==r['service'] and case['shape']==r['shape'];D,I,E,K,L,SI=r['shape'];assert E==64
  with np.load(array_path) as f:a={k:f[k] for k in f.files}
  rows,initial,routes=a['rows'],a['initial_rows'],a['routes'];assert rows.shape==initial.shape==(65,2,16);assert routes.shape==(512,K)
  trainpath=pathlib.Path(r['training']['path']);assert sha(trainpath)==r['training']['sha256'];assert r['training']==case['training'];inputs[str(trainpath)]=sha(trainpath)
  with np.load(trainpath) as f:
   assert f['split'].item()=='train' and f['completed'].all();assert np.array_equal(routes,f['routes'][:,:,case['layer']].reshape(-1,K));assert all(str(x).startswith('train_') for x in f['prompt_ids'])
  assert np.array_equal(np.sort(rows,axis=2),np.sort(initial,axis=2)) and np.array_equal(rows[-1],initial[-1]);assert np.all(rows[:,0].sum(1)==np.r_[[2*I]*E,2*SI]) and np.all(rows[:,1].sum(1)==D)
  inherited=case['winners'][r['family']];assert inherited==r['inherited'];upstream=inherited[inherited['selected']];upstream_path=pathlib.Path(upstream['arrays']);assert sha(upstream_path)==upstream['sha256'];inputs[str(upstream_path)]=sha(upstream_path)
  with np.load(upstream_path) as f:
   assert np.array_equal(initial,f['rows']) and np.array_equal(routes,f['routes']) and np.array_equal(a['initial_cost'],f['cost']);upstream_initial_lb=f['lower_bound'].copy() if 'lower_bound' in f.files else None
  service=backend.Service(**r['service']);final_plan=backend.lower_rows(rows[:,0],rows[:,1],D,[I]*E+[SI],service);initial_plan=backend.lower_rows(initial[:,0],initial[:,1],D,[I]*E+[SI],service)
  for key in ['offsets','jobs','input_columns','total_rows']:assert np.array_equal(a[key],final_plan[key]),key
  final=native.run(final_plan,routes,E,1,service);initial_replay=native.run(initial_plan,routes,E,1,service)
  assert np.array_equal(final['cost'],a['cost']) and np.array_equal(final['lower_bound'],a['lower_bound']);assert np.array_equal(initial_replay['cost'],a['initial_cost']);assert upstream_initial_lb is not None and np.array_equal(initial_replay['lower_bound'],upstream_initial_lb)
  assert float(final['cost'].mean())==r['mean_ns']==launched['mean_ns']==r['fit']['final_mean_ns'];assert float(initial_replay['cost'].mean())==r['fit']['initial_mean_ns']==launched['initial_mean_ns'];assert np.percentile(final['cost'],99)==r['p99_ns']==launched['p99_ns'];assert int(final['cost'].sum())<=int(initial_replay['cost'].sum())
  expectedlogical=np.asarray([[v[k] for k in ['expert','phase','chip','row_start','rows','columns']] for v in final_plan['logical']],np.int32);assert np.array_equal(expectedlogical,a['logical'])
  # Independent interval/chip/page checks, using the saved arrays directly.
  physical_ranges=collections.defaultdict(list);logical_rows=jobs=extra=partialbytes=0
  assert len(a['addresses'])==len(a['logical'])==len(a['jobs'])
  for e in range(65):
   for phase in [0,1]:
    lo,hi=map(int,a['offsets'][e,phase]);cursor=0;seen=set();C=D if phase==0 else SI if e==64 else I
    for ji in range(lo,hi):
     chip,n,c,pages,waves=map(int,a['jobs'][ji]);le,lp,lc,start,ln,lcol=map(int,a['logical'][ji]);achip,block,wl,reserved,apages,plane=map(int,a['addresses'][ji]);assert chip not in seen;seen.add(chip)
     assert (le,lp,lc,start,ln,lcol)==(e,phase,chip,cursor,n,C);assert c==C and n==rows[e,phase,chip];cursor+=n
     assert pages==(n*C+16383)//16384 and waves==(pages+3)//4;assert (achip,reserved,apages,plane)==(chip,waves,pages,0);assert block>=0 and wl>=0 and wl+waves<=192
     physical_ranges[chip,block].append((wl,wl+waves,ji));logical_rows+=n;jobs+=1;extra+=pages*16384//C-n;partialbytes+=pages*16384%C
    assert cursor==rows[e,phase].sum() and seen==set(np.flatnonzero(rows[e,phase]))
  for intervals in physical_ranges.values():
   intervals.sort();assert all(b<=c for (_,b,_),(c,_,_) in zip(intervals,intervals[1:]))
  independently_physical=backend.physical_addresses(final_plan,service);expectedaddresses=np.asarray([[v[k] for k in ['chip','block','wordline','reserved_wordlines','pages','first_plane']] for v in independently_physical.pop('addresses')],np.int32);assert np.array_equal(expectedaddresses,a['addresses']);assert independently_physical==r['physical']
  orbit=None
  if r['kind']=='cyclic':
   selected_record,stage=inherited,inherited['selected']
   if stage=='inherited':
    oldcase=pathlib.Path(case['v1_case_path']);inputs[str(oldcase)]=sha(oldcase);oldrec=json.loads(oldcase.read_text());matches=[(rec,st) for rec in oldrec['records'] for st in ['initial','final'] if rec[st]['arrays']==upstream['arrays']];assert matches;selected_record,stage=matches[0]
   initialperm=np.asarray(selected_record['fit']['final_global_permutations'] if stage=='final' else [np.arange(16),np.arange(16)]);finalperm=np.asarray(r['fit']['final_global_permutations']);assert initialperm.shape==finalperm.shape==(2,16)
   for p in [0,1]:
    assert sorted(initialperm[p])==sorted(finalperm[p])==list(range(16));base=initial[0,p,np.argsort(initialperm[p])]
    for rr,perm in [(initial,initialperm),(rows,finalperm)]:
     variants={tuple(np.roll(base,k)[perm[p]]) for k in range(16)};assert all(tuple(x) in variants for x in rr[:64,p])
   orbit=True
  graphkey=(case['model'],r['family']);graphcheck=None
  if graphkey not in ref_families:
   ref_families.add(graphkey);q=127;graph=backend.ReferenceGraph.build(final_plan,np.r_[routes[q],E],service);g=graph.run();assert g['cost']==int(a['cost'][q]);label=dest.name;traces[label]=g['trace'];graphcheck=dict(query=q,native_ns=int(a['cost'][q]),ReferenceGraph_ns=int(g['cost']),nodes=len(g['trace']))
  label=dest.name;recomputed[label+'.initial_cost']=initial_replay['cost'];recomputed[label+'.initial_LB']=initial_replay['lower_bound'];recomputed[label+'.final_cost']=final['cost'];recomputed[label+'.final_LB']=final['lower_bound']
  checks.append(dict(fit=label,model=case['model'],physical_model_layer=r['training']['physical_model_layer'],family=r['family'],seed=r['seed'],full_routes=512,initial_final_costs_and_LBs_exact=True,rows_offsets_jobs_logical_addresses_exact=True,logical_rows=logical_rows,jobs=jobs,one_job_per_phase_chip=True,physical_ranges=len(physical_ranges),same_projection_row_multisets=True,shared_unchanged=True,cyclic_orbits_legal=orbit,requested_pages=int(a['jobs'][:,3].sum()),reserved_waves=int(a['jobs'][:,4].sum()),extra_page_emitted_complete_rows=extra,page_partial_row_bytes=partialbytes,initial_mean_ns=float(initial_replay['cost'].mean()),final_mean_ns=float(final['cost'].mean()),refinement_mean_saving_ns=float(initial_replay['cost'].mean()-final['cost'].mean()),ReferenceGraph_check=graphcheck,native_query_evaluations=r['fit']['native_query_evaluations']))
 assert inputs=={p:sha(p) for p in inputs};assert sha(source)==protocol['source_sha256'];np.savez_compressed(OUT/'full512_recomputed_costs_and_bounds.npz',**recomputed);np.savez_compressed(OUT/'reference_graph_traces.npz',**traces)
 report=dict(status='PASS',protocol_sha256=protocol_sha,ExactFit_core_identical_to_reviewed_version=True,ExactFit_core_sha256=hashlib.sha256(core(source.read_text()).encode()).hexdigest(),all_protocol_source_hashes_unchanged=True,TRAIN_only_routes_exact=True,fits=8,replayed_cost_values=8192,replayed_LB_values=8192,ReferenceGraph_checks=len(traces),checks=checks,source_sha256=sha(__file__),inputs_sha256=inputs,native_provenance=native.provenance,seconds=time.monotonic()-began,scope='Independent full512 initial/final replay and row/address audit only; no fitting, heldout, model inference, source mutation or native hardware claim.')
 dump(OUT/'report.json',report);(OUT/'audit_source.py').write_bytes(pathlib.Path(__file__).read_bytes());dump(OUT/'completion.json',dict(status='PASS',artifact_sha256={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}));print(json.dumps({k:report[k] for k in ['status','protocol_sha256','ExactFit_core_identical_to_reviewed_version','fits','replayed_cost_values','replayed_LB_values','ReferenceGraph_checks','seconds']},indent=2))
if __name__=='__main__':main()
