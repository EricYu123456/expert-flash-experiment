#!/usr/bin/env python3
"""Crosscheck the new generic backend against sealed independently checked vectors."""
from dataclasses import asdict
import datetime,hashlib,json,pathlib,sys,time
import numpy as np
OUT=pathlib.Path(__file__).resolve().parent
REPLAY=OUT.parent
FRESH=REPLAY.parent
BASE=FRESH.parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from research_20261103_ordinary_read_backend import NativeReplay,Service,lower_rows
PLAN=BASE/'geometry-study/granite4tiny.frozen.json'
CASES={'frozen_phase_group':'group_rows','frozen_whole_rows':'rows'}
NAMES=['host','controller','pcie']+[f'channel{i}' for i in range(8)]+[f'chip{i}' for i in range(16)]+['sync']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 start=time.monotonic();protocol_path=OUT/'generic_backend_crosscheck.protocol.frozen.json';report_path=OUT/'generic_backend_crosscheck.report.json'
 if protocol_path.exists() or report_path.exists():raise RuntimeError('Never overwrite sealed crosscheck')
 source_paths=[pathlib.Path(__file__),ROOT/'scripts/research_20261103_ordinary_read_backend.py',ROOT/'scripts/research_20261103_ordinary_read_native.cpp',ROOT/'scripts/research_20261102_whole_row_compiler.py']
 input_paths=[PLAN,REPLAY/'protocol.frozen.json',REPLAY/'routes.npz',REPLAY/'route_extraction.json',REPLAY/'summary.json',OUT/'audit.json',OUT/'frozen_phase_group.independent_layers.npy',OUT/'frozen_whole_rows.independent_layers.npy']
 for layer in range(40):
  for case in CASES:input_paths.extend([REPLAY/f'L{layer}.{case}.npz',REPLAY/f'L{layer}.{case}.trace.json'])
 hashes={str(p):sha(p) for p in input_paths};sources={str(p):sha(p) for p in source_paths}
 assert sha(PLAN)=='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
 service=Service();assert service.output_fringe==0 and service.down_first==0
 engine=NativeReplay(OUT/'generic_backend_build')
 backend_validation=json.loads((BASE/'ordinary-read-compiler/backend-validation/summary.json').read_text())
 assert engine.provenance['source_sha256']==backend_validation['build']['source_sha256'] and engine.provenance['binary_sha256']==backend_validation['build']['binary_sha256']
 write(protocol_path,{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs_sha256':hashes,'source_sha256':sources,'native_build':engine.provenance,'service':asdict(service),'experts':64,'shared':1,'fixed_shared':{'GU_rows_per_chip':128,'GU_columns':1536,'down_rows_per_chip':96,'down_columns':1024},'comparison':'All16prompts*16decode*40layers*2layouts, against old Graph NPZ and independently set-queue-recompiled vectors. No refit. First-query node traces also checked exactly.','backend_mutation':False})
 with np.load(REPLAY/'routes.npz') as f:routes=f['routes']
 assert routes.shape==(16,16,40,6)
 doc=json.loads(PLAN.read_text());phases={(p['layer'],p['name']):p for p in doc['phases']};independent={case:np.load(OUT/f'{case}.independent_layers.npy') for case in CASES}
 records=[];all_costs={c:np.zeros((40,16,16),dtype=np.int64) for c in CASES};total_node_checks=0
 for layer in range(40):
  for case,key in CASES.items():
   gu=np.vstack([np.asarray(phases[layer,'gate_up'][key],dtype=np.int64),np.full((1,16),128,dtype=np.int64)])
   down=np.vstack([np.asarray(phases[layer,'down'][key],dtype=np.int64),np.full((1,16),96,dtype=np.int64)])
   assert gu.shape==down.shape==(65,16)
   lowered=lower_rows(gu,down,np.full(65,1536,dtype=np.int32),np.r_[np.full(64,512,dtype=np.int32),np.int32(1024)],service)
   assert lowered['offsets'].shape==(65,2,2) and np.all(lowered['input_columns'][:64]==[1536,512]) and lowered['input_columns'][64].tolist()==[1536,1024]
   result=engine.run(lowered,routes[:,:,layer].reshape(-1,6),experts=64,shared=1,service=service,trace=True)
   costs=result['cost'].reshape(16,16);all_costs[case][layer]=costs
   with np.load(REPLAY/f'L{layer}.{case}.npz') as f:expected=f['cost'];old_bound=f['native_chip_bound']
   assert np.array_equal(costs,expected) and np.array_equal(costs,independent[case][layer]),(layer,case,int(np.max(np.abs(costs-expected))))
   assert np.all(result['lower_bound'].reshape(16,16)>=old_bound)
   trace=json.loads((REPLAY/f'L{layer}.{case}.trace.json').read_text())['trace'];native=result['trace'];assert len(trace)==len(native)==result['nodes']
   for a,b in zip(trace,native):
    expected_row=[NAMES.index(a['resource']),a['end_ns']-a['start_ns'],a['start_ns'],a['end_ns'],a['phase'],a['expert'],a['chip'],a['job']]
    assert b.tolist()==expected_row,(layer,case,a['index'])
   total_node_checks+=len(trace)
   records.append({'layer':layer,'case':case,'events':256,'max_abs_cost_difference_ns':0,'old_graph_and_independent_vectors_exact':True,'first_query_trace_nodes_exact':len(trace),'lowered_plan_jobs':len(lowered['jobs'])})
  if layer%10==9:print('Generic backend checked layers',layer+1,'elapsed',round(time.monotonic()-start,2),flush=True)
 totals=[]
 for case,array in all_costs.items():
  summed=array.sum(axis=0);assert np.array_equal(summed,np.load(REPLAY/f'{case}.sum40.npy'));np.save(OUT/f'{case}.generic_backend_layers.npy',array)
  totals.append({'case':case,'mean_ns':float(summed.mean()),'p99_ns':float(np.percentile(summed,99))})
 for p,h in hashes.items():assert sha(p)==h,p
 for p,h in sources.items():assert sha(p)==h,p
 assert sha(engine.provenance['binary'])==engine.provenance['binary_sha256']
 report={'status':'PASS','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'crosscheck_protocol_sha256':sha(protocol_path),'native_build':engine.provenance,'service':asdict(service),'source_sha256':sources,'inputs_sha256':hashes,'experts':64,'shared':1,'cost_cases_exact':20480,'mismatched_cost_cases':0,'max_abs_cost_difference_ns':0,'sealed_old_Graph_and_independent_vectors_both_exact':True,'first_query_traces_exact':80,'node_records_exact':total_node_checks,'all_full40_sum_arrays_exact':True,'totals':totals,'records':records,'wall_seconds':time.monotonic()-start,'limits':['Regression crosscheck of the software ordinary-read service engine; not hardware/quality validation.','Fixed defaultService and output_fringe0 only for this fresh regression; no fitting or source modifications.']}
 write(report_path,report);print(json.dumps({k:v for k,v in report.items() if k not in ['records','inputs_sha256','source_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
