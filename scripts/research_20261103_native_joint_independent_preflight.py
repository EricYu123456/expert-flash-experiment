#!/usr/bin/env python3
"""Independent <=100-proposal TRAIN-only cache/family/physical contract audit."""
import collections,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'native-joint-refinement/independent-preflight'
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261103_native_joint_refine as target
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
class ForcedGlobal:
 def __init__(self,paired,p,a,b):self.paired=paired;self.p=p;self.a=a;self.b=b;self.calls=0
 def random(self):
  self.calls+=1
  if self.calls==1:return 0.
  if self.calls==2:return 0. if self.paired else .75
  return 0.
 def choice(self,n,size,replace=False):assert n==16 and size==2;return np.asarray([self.a,self.b])
 def integers(self,n):assert n==2;return self.p
def main():
 began=time.monotonic();OUT.mkdir(exist_ok=False);source=pathlib.Path(target.__file__);sourcehash=sha(source);checks=[];alltraces={};inputs={str(source):sourcehash};proposals=0
 for model,copies,steps in [('granite4tiny',1,24),('deepseek',4,8)]:
  case=BASE/f'shared-counterbalanced/{model}.L0.input{copies}/result.json';rec=json.loads(case.read_text());inputs[str(case)]=sha(case);train=pathlib.Path(rec['training']['path']);assert sha(train)==rec['training']['sha256'];inputs[str(train)]=sha(train)
  for fi,(family,kind) in enumerate([('joint_cyclic','cyclic'),('independent_permutation','independent'),('counterbalanced','coupled')]):
   w=rec['winners'][family];art=w[w['selected']];p=pathlib.Path(art['arrays']);assert sha(p)==art['sha256'];inputs[str(p)]=sha(p)
   with np.load(p) as z:rows=z['rows'];routes=z['routes'];original=z['cost']
   inherited=w['fit']['final_global_permutations'] if w['selected']=='final' else None
   fit=target.ExactFit(rows,rec['shape'],rec['service'],routes,kind,20261107+fi,inherited);assert fit.initial_total==int(original.sum())
   # Independently deduplicate ordered tuples via a Python counter.
   counter=collections.Counter(map(tuple,routes.tolist()));assert {tuple(r):int(n) for r,n in zip(fit.unique,fit.weights)}==counter
   initialperm=fit.permutations.copy();bases=[rows[0,p,np.argsort(initialperm[p])].copy() for p in range(2)];expected_multisets=np.sort(rows,axis=2);initialjoined=((rows[:,0]*rec['shape'][0]+65535)//65536)+((rows[:,1]*np.asarray([rec['shape'][1]]*64+[rec['shape'][5]])[:,None]+65535)//65536)
   checkpoints=accepted=rejected=0
   def checkpoint():
    nonlocal checkpoints
    rebuilt=fit.lower()
    for k in ['jobs','offsets','input_columns','total_rows']:assert np.array_equal(fit.plan[k],rebuilt[k]),(model,family,k)
    full,bound=fit.native.run(rebuilt,fit.unique,rec['service']);assert np.array_equal(full,fit.values) and np.array_equal(bound,fit.bounds);assert int(full@fit.weights)==fit.total
    assert np.array_equal(np.sort(fit.rows,axis=2),expected_multisets) and np.array_equal(fit.rows[-1],rows[-1])
    if kind=='cyclic':
     for phase in range(2):
      expected=np.unique(np.stack([np.roll(bases[phase],j)[fit.permutations[phase]] for j in range(16)]),axis=0)
      assert {tuple(r) for r in expected}=={tuple(r) for r in fit.variants[phase]}
      assert all(tuple(r) in {tuple(v) for v in expected} for r in fit.rows[:-1,phase])
    if kind=='coupled':
     joined=((fit.rows[:,0]*rec['shape'][0]+65535)//65536)+((fit.rows[:,1]*np.asarray([rec['shape'][1]]*64+[rec['shape'][5]])[:,None]+65535)//65536)
     assert np.array_equal(np.sort(joined,axis=1),np.sort(initialjoined,axis=1))
    checkpoints+=1
   checkpoint()
   for it in range(steps):
    oldrows=fit.rows.copy();oldaccept=fit.accepted;fit.proposal(0. if it%2==0 else 500.)
    accepted+=fit.accepted>oldaccept;rejected+=fit.accepted==oldaccept and not np.array_equal(oldrows,fit.rows);proposals+=1;checkpoint()
   forced=0
   if kind=='cyclic':
    # Deliberately accept nontrivial common chip permutations, then validate
    # the entire cyclic orbit independently in the original coordinate system.
    for paired in [True,False]:
     phase=next(p for p in [1,0] if any(np.any(fit.rows[:-1,p,a]!=fit.rows[:-1,p,b]) for a in range(16) for b in range(a+1,16)))
     a,b=next((a,b) for a in range(16) for b in range(a+1,16) if np.any(fit.rows[:-1,phase,a]!=fit.rows[:-1,phase,b]));rng=fit.rng;fit.rng=ForcedGlobal(paired,phase,a,b);oldaccept=fit.accepted;fit.proposal(float('inf'));fit.rng=rng;assert fit.accepted==oldaccept+1;forced+=1;proposals+=1;checkpoint()
   # Final rebuild is the CLI path. The object is not reused after run().
   plan,values,bounds,stats=fit.run(0,0.)
   independent=fit.backend.physical_addresses(plan,fit.service);assert independent['row_coverage_and_address_nonoverlap']
   j=plan['jobs'].astype(np.int64);assert np.array_equal(j[:,3],(j[:,1]*j[:,2]+16383)//16384);assert np.array_equal(j[:,4],(j[:,3]+3)//4)
   assert np.all(j[:,4]<=192)
   # Every phase/chip still has at most one job; zero-chip ownership preserved.
   for phases in plan['offsets']:
    for lo,hi in phases:assert len(set(map(int,j[lo:hi,0])))==hi-lo
   emitted=j[:,3]*16384//j[:,2] if fit.service.output_fringe==1 else j[:,1]
   assert np.all(emitted>=j[:,1]);ledger=dict(useful_rows=int(j[:,1].sum()),emitted_complete_rows=int(emitted.sum()),extra_complete_rows=int((emitted-j[:,1]).sum()),requested_pages=int(j[:,3].sum()),reserved_waves=int(j[:,4].sum()),useful_weight_bytes=int((j[:,1]*j[:,2]).sum()))
   graphchecks=[]
   for q in [0,len(routes)-1]:
    graph=fit.backend.ReferenceGraph.build(plan,np.r_[routes[q],64],fit.service);g=graph.run();assert g['cost']==int(values[q]);label=f'{model}.{family}.q{q}';alltraces[label]=g['trace'];graphchecks.append(dict(query=q,native_ns=int(values[q]),ReferenceGraph_ns=int(g['cost']),nodes=len(g['trace'])))
   checks.append(dict(model=model,copies=copies,family=family,seed=20261107+fi,random_proposals=steps,forced_accepted_global_moves=forced,full_recompute_checkpoints=checkpoints,valid=fit.valid,accepted=fit.accepted,inherited_common_permutation_nonidentity=bool(np.any(initialperm!=np.arange(16))),ordered_unique_routes=len(counter),sorted_set_unique_routes=len(set(map(tuple,np.sort(routes,axis=1).tolist()))),all_incremental_values_bounds_totals_exact=True,cyclic_orbit_legal_after_common_permutations=True if kind=='cyclic' else None,coupled_joined_histograms_preserved=True if kind=='coupled' else None,same_projection_row_multisets=True,shared_unchanged=True,onejob_per_phase_chip=True,ledger=ledger,ReferenceGraph_checks=graphchecks,initial_mean_ns=float(original.mean()),final_mean_ns=float(values.mean()),native_query_evaluations=fit.evaluated_queries))
 assert proposals==100 and sha(source)==sourcehash
 assert inputs=={p:sha(p) for p in inputs};np.savez_compressed(OUT/'reference_graph_traces.npz',**alltraces)
 report=dict(status='PASS',target_source_sha256=sourcehash,script_sha256=sha(__file__),total_proposals=proposals,full_recompute_checkpoints=sum(c['full_recompute_checkpoints'] for c in checks),ReferenceGraph_checks=sum(len(c['ReferenceGraph_checks']) for c in checks),checks=checks,inputs_sha256=inputs,seconds=time.monotonic()-began,review_findings=['No blocking correctness issue in current CLI path.','Incremental plan logical-row metadata is stale during search, but native execution consumes rebuilt job arrays; final CLI fully lowers before physical/output artifact construction.','run() restores best rows but leaves self.plan/self.values/self.bounds/self.permutations at terminal state. Current CLI uses returned final_plan/values and best_permutations, so it is correct; the fit object should not be reused without resynchronization.','Matched proposal counts are a search-budget control, not equal native-query/CPU budgets: valid/no-op, local/global and family proposal distributions differ; statistics expose counts.','Independent and cyclic controls can propose paired moves; coupled preserves common phase transpositions. This is an adequately disclosed standard heuristic, not equal search spaces or a new optimizer.'],scope='100 proposals only across G4 input1 and Deep input4, TRAIN-only. No campaign fitting, model inference, heldout access, source mutation or new hardware contract.')
 dump(OUT/'report.json',report);(OUT/'reviewed_target.py').write_bytes(source.read_bytes());(OUT/'audit_source.py').write_bytes(pathlib.Path(__file__).read_bytes());dump(OUT/'completion.json',dict(status='PASS',artifact_sha256={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}));print(json.dumps({k:report[k] for k in ['status','total_proposals','full_recompute_checkpoints','ReferenceGraph_checks','seconds','review_findings']},indent=2))
if __name__=='__main__':main()
