#!/usr/bin/env python3
"""Narrow TRAIN-only packing-vs-native contract audit; no fitting or inference."""
import collections,hashlib,itertools,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'coupled-work-contract-audit'
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261103_ordinary_read_backend as backend
import research_20261103_shared_counterbalanced as candidate
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def wave_rows(rows,C):return (np.asarray(rows)*C+65535)//65536
def packing_ledger():
 models={
 'granite4tiny':([83]*6+[82]*3+[40]*7,[128]*12+[0]*4,1536,512),
 'deepseek':([192]*8+[160]*8,[137]*13+[89]*3,2048,1408),
 'qwen38':([101]*3+[76]*2+[75]*11,[199]*4+[198]*6+[96]*6,2560,640)}
 result={}
 for name,(gu,down,C,D) in models.items():
  gw=wave_rows(gu,C);dw=wave_rows(down,D);a,b=int(gw.min()),int(dw.min());h=int((gw==a+1).sum());k=int((dw==b+1).sum());assert set(gw)=={a,a+1} and set(dw)=={b,b+1}
  # Enumerate the complete feasible 2x2 contingency table, independent of masks.
  legal=[]
  for overlap in range(17):
   counts=[16-h-k+overlap,h-overlap,k-overlap,overlap]
   if min(counts)>=0:legal.append((overlap,counts))
  overlap,counts=legal[0];assert overlap==max(0,h+k-16)
  joined=collections.Counter({a+b:counts[0],a+b+1:counts[1]+counts[2],a+b+2:counts[3]});joined={str(w):int(n) for w,n in joined.items() if n};assert len(joined)<=2
  phases=[]
  for rows,col,waves in [(gu,C,gw),(down,D,dw)]:
   pages=(np.asarray(rows)*col+16383)//16384;assert np.all((pages+3)//4==waves)
   phases.append(dict(rows=rows,columns=col,total_rows=sum(rows),row_histogram=dict(collections.Counter(rows)),wave_histogram=dict(collections.Counter(map(int,waves))),useful_bytes=sum(rows)*col,pages=int(pages.sum()),waves=int(waves.sum()),page_padding_bytes=int(pages.sum())*16384-sum(rows)*col,reserved_wave_bytes=int(waves.sum())*65536))
  result[name]=dict(chips=16,GU_low_wave=a,GU_high_count=h,Down_low_wave=b,Down_high_count=k,min_overlap=overlap,categories=dict(neither=counts[0],GU_only=counts[1],Down_only=counts[2],both=counts[3]),joined_wave_histogram=joined,joined_total_waves=int(gw.sum()+dw.sum()),phases=phases,scope='One-copy full-row coefficient geometry only; basic 2x2 packing identity, no new theorem or latency guarantee.')
 return result
def main():
 started=time.monotonic();dest=OUT/'sealed';dest.mkdir(exist_ok=False)
 summary=BASE/'ordinary-read-compiler/full-campaign-v5-W8-training/summary.json';rec=next(r for r in json.loads(summary.read_text())['records'] if r['model']=='granite4tiny' and r['layer']==0 and r['command_ns']==250)
 s=backend.Service(**rec['service']);nr=backend.NativeReplay(dest/'build');route=np.ascontiguousarray(np.load(rec['training']['path'])['routes'][0,0,0][None],dtype=np.int32);assert route.tolist()==[[1,49,45,33,10,20]]
 discovery_inputs=[OUT/p for p in ['early_witness.json','early_witness.npz','enumeration.scratch.json','enumeration.scratch.npz','random_decompositions.scratch.json','random_decompositions.scratch.npz']]
 early=np.load(OUT/'early_witness.npz');single=np.load(OUT/'enumeration.scratch.npz');random=np.load(OUT/'random_decompositions.scratch.npz');shared=early['shared'];samework=json.loads((OUT/'enumeration.scratch.json').read_text())['samework'];reverse=json.loads((OUT/'random_decompositions.scratch.json').read_text())['reverse'];assert [r['index'] for r in samework]==[0,1] and [r['index'] for r in reverse]==[965,0]
 maps={'base':early['rowsA'],'wave_equal_other_decomposition':early['rowsB'],'ns_equal_other_decomposition':single['rows'][1],'lower_joined_ns':random['rows'][965],'higher_joined_ns':random['rows'][0]}
 records={};trace_summary={};arrays={};sourcefiles=[pathlib.Path(backend.__file__),pathlib.Path(candidate.__file__),ROOT/'scripts/research_20261103_ordinary_read_native.cpp',summary,pathlib.Path(rec['training']['path']),*discovery_inputs];source_hashes={str(p):sha(p) for p in sourcefiles}
 for label,rows in maps.items():
  assert rows.shape==(64,2,16) and np.all(rows[:,0].sum(1)==1024) and np.all(rows[:,1].sum(1)==1536)
  gw=wave_rows(rows[:,0],1536);dw=wave_rows(rows[:,1],512);assert np.all((gw==2).sum(1)==9) and np.all((dw==1).sum(1)==12);assert np.all(((gw==2)&(dw==1)).sum(1)==5)
  allrows=np.r_[rows,shared[None]];plan=backend.lower_rows(allrows[:,0],allrows[:,1],1536,[512]*64+[1024],s);counts=plan['offsets'][:,:,1]-plan['offsets'][:,:,0];assert np.all(counts[:,0]==16) and np.all(counts[:64,1]==12)
  native=nr.run(plan,route,64,1,s,trace=True);selected=np.r_[route[0],64];g=backend.ReferenceGraph.build(plan,selected,s);reference=g.run();T=int(native['cost'][0]);assert reference['cost']==T
  table=backend.occupancy_table(plan,s);chip_work=table['chip_ns'][selected].sum((0,1));W=int(chip_work.max());sharedwork=sum(candidate.cost(shared[p],C,rec['service']) for p,C in enumerate([1536,1024]));sourceW=int(candidate.work(rows,route,sharedwork,[1536,512],rec['service'])[0]);assert W==sourceW
  physical=backend.physical_addresses(plan,s);assert physical['row_coverage_and_address_nonoverlap'];assert all(j['waves']<=192 for j in plan['logical'])
  np.savez_compressed(dest/(label+'.npz'),rows=allrows,jobs=plan['jobs'],offsets=plan['offsets'],input_columns=plan['input_columns'],total_rows=plan['total_rows'],route=route,native_trace=native['trace'],reference_trace=reference['trace'],joined_wave_rows=gw+dw,chip_work_ns=chip_work)
  dump(dest/(label+'.physical.json'),dict(logical=plan['logical'],**physical))
  perexpert=[]
  for ei,e in enumerate(selected):
   ready=[i for i,n in enumerate(g.nodes) if n['expert']==ei and n['phase']==0 and n['resource']==s.resources-1 and n['job']==-1]
   assert len(ready)==1;act=[i for i,n in enumerate(g.nodes) if n['expert']==ei and n['phase']==1 and n['resource']==0]
   perexpert.append(dict(logical_expert=int(e),selected_index=ei,GU_ready_ns=int(reference['trace'][ready[0],3]),host_phase_nodes=[dict(node=i,start_ns=int(reference['trace'][i,2]),end_ns=int(reference['trace'][i,3]),duration_ns=n['duration']) for i,n in enumerate(g.nodes) if i in act]))
  jobs=[]
  for i,node in enumerate(g.nodes):jobs.append(dict(node=i,**node,start_ns=int(reference['trace'][i,2]),end_ns=int(reference['trace'][i,3])))
  dump(dest/(label+'.reference_graph.json'),dict(cost_ns=T,service=rec['service'],selected=selected.tolist(),perexpert=perexpert,nodes=jobs))
  records[label]=dict(native_ns=T,ReferenceGraph_ns=int(reference['cost']),native_lower_bound_ns=int(native['lower_bound'][0]),joined_ns=W,residual_ns=T-W,joined_chip_ns=chip_work.tolist(),joined_wave_histogram=dict(collections.Counter(map(int,(gw+dw)[route[0]].ravel()))),rows_artifact=label+'.npz',physical_artifact=label+'.physical.json',ReferenceGraph_artifact=label+'.reference_graph.json',native_jobs_for_selected=int(counts[selected].sum()),all_full_rows_onejob_per_phase_chip=True)
  trace_summary[label]=perexpert
 base=maps['base'];basewaves=wave_rows(base[:,0],1536)+wave_rows(base[:,1],512)
 assert all(np.array_equal(basewaves,wave_rows(r[:,0],1536)+wave_rows(r[:,1],512)) for r in maps.values())
 assert records['base']['joined_ns']==records['ns_equal_other_decomposition']['joined_ns'] and records['base']['native_ns']!=records['ns_equal_other_decomposition']['native_ns']
 assert records['lower_joined_ns']['joined_ns']<records['higher_joined_ns']['joined_ns'] and records['lower_joined_ns']['native_ns']>records['higher_joined_ns']['native_ns']
 ledger=packing_ledger();dump(dest/'packing_rowbyte_ledger.json',ledger)
 assert source_hashes=={str(p):sha(p) for p in sourcefiles}
 snapshot=dest/'source-snapshot';snapshot.mkdir()
 for p in sourcefiles[:3]:
  q=snapshot/(sha(p)[:16]+'-'+p.name);q.write_bytes(p.read_bytes())
 report=dict(status='PASS',script_sha256=sha(__file__),scope='TRAIN-only existing Service and one-copy full-row legal G4 geometry; no inference, new row types, arbitrary delays, controller/backend change, fitting or heldout access.',model='granite4tiny',model_layer=0,service=rec['service'],TRAIN_archive=rec['training'],route=route.tolist(),shared_rows=shared.tolist(),all_maps_same_every_expert_chip_joined_wave_vector=True,all_maps_minimal_overlap=True,source_work_ns_equals_independent_occupancy_sum=True,all_native_costs_match_independent_ReferenceGraph=True,source_sha256=source_hashes,native_provenance=nr.provenance,packing_ledger=ledger,witnesses=records,witness_pairs=dict(joined_waves_equal=['base','wave_equal_other_decomposition'],joined_ns_equal=['base','ns_equal_other_decomposition'],lower_joined_ns_but_slower_native=['lower_joined_ns','higher_joined_ns']),discovery_enumeration=dict(single_swap_candidates=169,random_phase_decompositions=1024,random_seed=202611033,base_seed=20261103,no_additional_candidates_evaluated_by_audit=True),seconds=time.monotonic()-started)
 dump(dest/'result.json',report);seal={str(p):sha(p) for p in dest.rglob('*') if p.is_file()};dump(dest/'completion.json',dict(status='PASS',artifact_sha256=seal));print(json.dumps({k:report[k] for k in ['status','all_maps_same_every_expert_chip_joined_wave_vector','all_native_costs_match_independent_ReferenceGraph','witnesses','seconds']},indent=2))
if __name__=='__main__':main()
