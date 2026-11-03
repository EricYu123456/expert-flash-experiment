#!/usr/bin/env python3
"""Frozen geometry-only lowering experiment; no routes or fullFFN replay."""
from pathlib import Path
from functools import lru_cache
from datetime import datetime,timezone
import hashlib
import importlib.util
import itertools
import json
import sys
import lowering as lower

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def loadpipe():
 sp=importlib.util.spec_from_file_location('fixed_lower_pipe',PIPE);p=importlib.util.module_from_spec(sp);sys.modules[sp.name]=p;sp.loader.exec_module(p);return p

def freeze():
 assert not (HERE/'protocol.frozen.json').exists()
 paths=[Path(__file__),HERE/'lowering.py',HERE/'plan.frozen.json',PIPE]
 dump(HERE/'protocol.frozen.json',dict(status='frozen_before_run',utc=datetime.now(timezone.utc).isoformat(),inputs_sha256={str(p):sha(p) for p in paths},
    cases='Candidate plus7RR phase compositions ×native/hybrid shared geometries, eachfixed/permutation. Exact minimum objective plus bounded one-extra-wave timing controls on first optimal type order.',
    bounded_timing_search='After exact minimum witness, fixed source inserts one complete cyclic rotation before each useful slot, excluding empty waves; permutation splits each current wave at all useful prefix cuts. No additional order or timing search; report best feasible among this finite set, not a global latency optimum.',
    source='Fixed0123 has no PB reorder; permutation is separate conditional capability control. Both use actualunselected-page drop beforeECC conditionally.',
    independent_checks='Brute force all labelled-owner macro topological orders for K<=2/S<=2 versuscount DP; independent earliestmatching cyclicscan / distinctplane segmentation; independentfinite2slot maxplus reference on every saved minimum/best trace.',
    interpretation='No TRAIN, heldout, fullFFN, relocation, replication, addedinputcontext, FIFO or arbitraryidle hold.'))

def independent_slots(patterns,source):
 if not patterns:return 0
 seq=[p for pattern in patterns for p in pattern]
 if source=='fixed':
  n=0
  for p in seq:
   n+=((p-n)%4)+1
  return ((n+3)//4)*4
 # Independent interval-DP over the fixed stream, allowing every legal cut.
 @lru_cache(None)
 def part(i):
  if i==len(seq):return 0
  seen=set();best=10**9
  for j in range(i,len(seq)):
   if seq[j] in seen:break
   seen.add(seq[j]);best=min(best,1+part(j+1))
  return best
 return 4*part(0)

def brute_orders(problem):
 experts=[e for t in problem['types'] for e in t['experts']];ph={e:t['patterns'] for t in problem['types'] for e in t['experts']}
 S=problem['shared_patterns'];progress={e:0 for e in experts}
 def go(s,patterns):
  if all(v==2 for v in progress.values()) and s==len(S):yield patterns;return
  for e in experts:
   if progress[e]<2:
    j=progress[e];progress[e]+=1;yield from go(s,patterns+[ph[e][j]]);progress[e]-=1
  if s<len(S):yield from go(s+1,patterns+[S[s]])
 yield from go(0,[])

def small_checks():
 records=[]
 for kind in ['candidate','RR']:
  for K in range(3):
   for n in (range(K+1) if kind=='RR' else [0]):
    for S in range(3):
     p=lower.build_problem(kind,n,K,S)
     orders=list(brute_orders(p))
     for source in ['fixed','permutation']:
      brute=min(independent_slots(q,source) for q in orders)
      sol=lower.solve_macros(p,source);assert sol['optimal_slots']==brute,(kind,K,n,S,source,brute,sol)
      records.append(dict(kind=kind,K=K,even_count=n,shared_macros=S,source=source,labelled_orders=len(orders),optimal_slots=brute,status='PASS'))
 return records

def independent_timing(waves,source,trace,copy=500):
 cache=[0]*4;read_release=[0]*4;ecc=0;hazards=[];expected=[]
 for wi,w in enumerate(waves):
  start=0 if wi==0 else 28000+(wi-1)*9700
  ready=28000+wi*9700;overwrite=start if wi==0 else start+6040
  for plane in range(4):
   if read_release[plane]>overwrite:hazards.append((wi,plane,read_release[plane]-overwrite))
  true={p['plane']:p for p in w['pages']}
  for plane in range(4):
   if plane not in true:read_release[plane]=ready
  for plane in (list(range(4)) if source=='fixed' else w['source_order']):
   if plane not in true:continue
   cend=max(ready,cache[plane])+copy;end=max(cend,ecc)+2560
   read_release[plane]=cend;cache[plane]=end;ecc=end
   expected.append((true[plane]['page_id'],wi,plane,end))
 actual=[(p['page_id'],p['wave'],p['plane'],p['end_ns']) for p in trace['page_completions']]
 assert expected==actual
 reported=[(v['wave'],v['plane'],v['excess_ns']) for v in trace['violations'] if v['kind']=='read_latch_overwrite']
 assert sorted(hazards)==sorted(reported)
 assert trace['timing_feasible']==(not hazards)
 if not hazards:assert trace['makespan_ns']==max(28000+(len(waves)-1)*9700,ecc)
 return dict(status='PASS',page_completions=len(expected),overwrite_violations=len(hazards),independent_makespan_ns=None if hazards else max(28000+(len(waves)-1)*9700,ecc))

def summary(r):
 keys=['feasible','functional_ok','timing_feasible','makespan_ns','hypothetical_makespan_ns','waves','pages','physical_read_pages','ecc_pages','cold_waves','cr_waves']
 return {k:r[k] for k in keys if k in r}|dict(violations=r['violations'])

def candidate_plane2_bound():
 return dict(useful_plane_pages=[12,12,12,12],mandatory_internal_dummy_pages_by_plane=[4,4,6,0],lower_bound_waves=18,
   proof='Six B=[0,1,3] macros each force one plane2dummy between1and3. Four shared323/232 macros each force planes0and1 between3and2. An owner macro cannot interleave another useful owner. Thus plane2 must have12useful+6dummy slots, independently requiring>=18waves.')

def run():
 protocol=read(HERE/'protocol.frozen.json')
 for path,h in protocol['inputs_sha256'].items():assert sha(path)==h,path
 p=loadpipe();checks=small_checks();dump(HERE/'small_bruteforce.json',checks)
 (HERE/'traces').mkdir(exist_ok=True)
 cases=[('candidate',lower.build_problem())]
 cases += [(f'RR.native.even{n}',lower.build_problem('RR',n)) for n in range(7)]
 cases += [(f'RR.hybridShared.even{n}',lower.build_problem('RR',n,shared_geometry='candidate_patterns')) for n in range(7)]
 records=[];attempts=[]
 for name,problem in cases:
  for source in ['fixed','permutation']:
   solution=lower.solve_macros(problem,source);waves=lower.slots_to_waves(problem,solution['slots'],source)
   cert=lower.validate_lowering(problem,solution,waves,source)
   r=p.simulate(waves,cache_copy_ns=500,source_policy=source,unselected_policy='drop_before_ecc_conditional')
   check=independent_timing(waves,source,r)
   stem=f'{name}.{source}';dump(HERE/'traces'/f'{stem}.minimum.json',dict(problem=problem,solution=solution,waves=waves,certificate=cert,trace=r,independent_timing=check))
   best=(r['makespan_ns'],None,solution['slots'],waves,r) if r['feasible'] else None
   count=0;feasible=0
   for variant,slots in lower.one_extra_wave_variants(solution,source):
    ww=lower.slots_to_waves(problem,slots,source);cc=lower.validate_lowering(problem,solution,ww,source)
    rr=p.simulate(ww,cache_copy_ns=500,source_policy=source,unselected_policy='drop_before_ecc_conditional')
    independent_timing(ww,source,rr);count+=1;feasible+=int(rr['feasible'])
    attempts.append(dict(case=name,source=source,variant=variant,certificate=cc,**summary(rr)))
    if rr['feasible'] and (best is None or (rr['makespan_ns'],json.dumps(variant,sort_keys=True))<(best[0],json.dumps(best[1],sort_keys=True))):best=(rr['makespan_ns'],variant,slots,ww,rr)
   if best:
    dump(HERE/'traces'/f'{stem}.best_tested.json',dict(problem=problem,minimum_solution=solution,variant=best[1],slots=best[2],waves=best[3],trace=best[4],certificate=lower.validate_lowering(problem,solution,best[3],source),independent_timing=independent_timing(best[3],source,best[4])))
   rec=dict(case=name,source=source,kind=problem['kind'],shared_geometry=problem['shared_geometry'],optimal_slots=solution['optimal_slots'],optimal_waves=solution['optimal_waves'],states=solution['states'],optimal_type_orders=solution['optimum_type_orders'],actions=solution['actions'],minimum_certificate=cert,minimum_timing=summary(r),one_extra_wave_attempts=count,one_extra_wave_feasible=feasible,
     best_tested=None if best is None else dict(latency_ns=best[0],waves=len(best[3]),variant=best[1],physical_pages=4*len(best[3]),dummy_pages=4*len(best[3])-48),
     minimum_wave_timing_impossibility_proven=bool(solution['optimal_waves']==12 and not r['feasible']),
     minimum_failure_scope='If12fullwaves: universal whole-page two-cache bound2c+10*540<=6040 fails atc500. Otherwise failed witness alone does not reject alloptimalmacroorders.')
   records.append(rec);print(json.dumps({k:rec[k] for k in ['case','source','optimal_waves','best_tested']}),flush=True)
 dump(HERE/'attempts.json',attempts)
 report=dict(status='PASS',source_sha256=sha(HERE/'lowering.py'),experiment_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.frozen.json'),small_bruteforce_cases=len(checks),independent_timing_checks='Everyminimumandboundedvariant',candidate_fixed_plane_bound=candidate_plane2_bound(),records=records,
    limitations=['Pre-ECC dummy drop remainsconditional, not verifiednativehardware.','Minimumwavecount proven fordeclaredmacro/slotcontract; besttestedlatency is only boundedwitness search, not globaltimingoptimum.','Programmablesource controls are separate fromfixed0123 source.','No fullFFN, routes, numericchanges, replicas, extraPBcontext, FIFO oridlehold.','Hybrid RR sharedgeometry is separatelylabelled; native RR usesglobalpageplacement.'])
 dump(HERE/'report.json',report)
 lines=['# Whole-row macro lowering with a fixed PB source','','Every physical wave reads four real resident pages; dummy weights are conditionally dropped before ECC. All useful three-page macros remain contiguous after drop and preserve per-owner order. No coefficient is moved or copied.','','Candidate fixed0123 needs at least18waves: each of six B=[0,1,3] macros requires a plane2dummy, on top of12usefulplane2pages. The exact DP below checks attainability.','','| Layout | Source | Exact minimum waves | Minimum witness strict500 | Best tested feasible waves | Best latency µs |','|---|---|---:|---|---:|---:|']
 for r in records:
  b=r['best_tested'];lines.append(f'| {r["case"]} | {r["source"]} | {r["optimal_waves"]} | {"pass" if r["minimum_timing"]["feasible"] else "fail"} | {b["waves"] if b else "—"} | {b["latency_ns"]/1000 if b else "—"} |')
 lines += ['',f'Independent labelled-order brute force passed {len(checks)} small cases. A separate finite2slot recurrence matched every tested minimum and bounded one-extra-wave trace, including all latch-deadline violations.', '',
   'The DP proves minimum cyclic slots (fixed0123) or distinct-plane waves (permutation) for the declared whole-row macro contract. The timing compiler additionally tests one full cyclic rotation before each useful fixed slot, or every one-wave split for permutation; best tested latency is not a general timing optimum. A12wave full-useful chain atcopy500 is universally infeasible in this cache-only model, so a feasible13wave witness is wave-optimal under that timing contract.', '',
   'Native global RR uses E even=[012][301], odd=[230][123], and shared=[012][301][230][123]. The hybrid shared-geometry rows are a separately labelled stronger control with candidate-style shared patterns, not the native RR layout. All seven phase compositions are covered.', '',
   'Pre-ECC drop and programmable source order remain conditional capabilities. Fixed-source results remove source permutation only; they still require same-block independent plane addresses and whole-row owner switching already assumed by the IPR model. These are service-envelope/compiler witnesses, not measured hardware or fullFFN gains.']
 (HERE/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
