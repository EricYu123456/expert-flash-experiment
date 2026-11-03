#!/usr/bin/env python3
"""TRAIN pair-graph clique closure necessary-capacity screen, never a fit."""
import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import ortools
from ortools.sat.python import cp_model

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/'results/research-2026-11-02/round2'
OUT=ROUND/'m32-pair-closure-placement-screen'
TRAIN=ROUND/'w8-format-training/granite4tiny/decode.routes.npz'
PLAN=ROUND/'chip-count-regime-screen/g4-M32-coupled-cp/native/incumbent4.plan.npz'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def members(mask):
    result=[]
    while mask:
        bit=mask&-mask;result.append(bit.bit_length()-1);mask^=bit
    return result


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    files=[Path(__file__),TRAIN,PLAN]
    refs=[]
    for path in files:
        item=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        if path.suffix=='.py':
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        refs.append(item)
    p=dict(status='frozen_before_TRAIN_graph_and_screen',utc=datetime.now(timezone.utc).isoformat(),
      source_sha256=sha(__file__),references=refs,
      graph='64experts, undirected edge iff at leastone oforiginal512TRAIN L0 top6routes contains thepair; threshold exactly1, no dev/fresh access.',
      closure='Every6vertexclique inthatpairgraph, not allpossible6sets. AllTRAINroutes are included byconstruction. Syntheticjoint closure is a conservative enlargement, not asserted realizable router support.',
      characterization='Enumerate maximalcliques exactly with deterministic bitsetBronKerbosch. For C size>=6 and cap1/2/3, all6subsets satisfying sumX<=cap iff sumC X<=cap. Any cap+1markedvertices extendto6 insideC.',
      capacity='For each cap1/2/3 separately maximize sum64BooleansX subjectto eachmaximalclique>=6 sumX<=cap. This relaxes allcross-chip O/perexpertXdegree constraints, so provides perchip upperbounds, not a placement.',
      peak10='For fixedcoupledX5/O7, everyquery chipwaves=6+sumselectedX+sharedS. ActualS histogram12x3,17x2,3x1. Need320totalX incidences; if12U1+17U2+3U3<320, closure/profile/shared peak10 impossible.',
      solver=dict(version=ortools.__version__,seconds_per_cap=30,workers=8,seed=20261116,
        integer_bound='Pure Maximize sumX uses original objective scaling=-1,offset0,coeffs=-1. Use exact response inner_objective_lower_bound negated only for FEASIBLE/OPTIMAL and crosscheck external bound/incumbent; UNKNOWN conservatively uses64. Save model/response. No noninteger rounding needed.',
        fallback='If integer response crosscheck fails, do not trust it; use trivial64. No UNKNOWN-as-infeasible.'),
      enumeration_budget='Maximalclique recursion<=50s. Optional exact6clique count<=65s totalgeometrytime. If enumeration incomplete, discoveredvalidcliques still yield a relaxed safeupperbound, but counts/coverage-by-maximal reportedpartial.',
      limits='No full placement fitting, native/inference, backend change, T11/12 study, dev graph/coverage or threshold selection. Pair-unseen new routes are outsideprotection; no universal routing robustness ornew graph theorem claim.')
    dump(OUT/'protocol.frozen.json',p)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__)),indent=2))


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
    for ref in p['references']:assert sha(ref['path'])==ref['sha256']
    began=time.monotonic()
    with np.load(TRAIN) as z:
        assert str(z['split'])=='train' and z['completed'].all()
        routes=z['routes'][:,:,0].reshape(512,6).copy()
    assert np.all((routes>=0)&(routes<64)) and np.all(np.diff(np.sort(routes,axis=1),axis=1)>0)
    present=np.zeros((512,64),np.int64);present[np.arange(512)[:,None],routes]=1
    counts=present.T@present;adj=[]
    for e in range(64):adj.append(sum(1<<j for j in range(64) if e!=j and counts[e,j]>0))
    allmask=(1<<64)-1;route_masks=[sum(1<<int(e) for e in route) for route in routes]
    for route in routes:
        assert all((adj[int(a)]>>int(b))&1 for a in route for b in route if a!=b)
    with np.load(PLAN) as z:shared_rows=z['rows'][64]
    shared=np.zeros(32,np.int64)
    for phase,c in [(0,1536),(1,1024)]:
        for chip,n in enumerate(shared_rows[phase]):
            if n:shared[chip]+=((int(n)*c+16383)//16384+3)//4
    assert Counter(shared.tolist())==Counter({3:12,2:17,1:3})
    maximal=[];calls=0;enumeration_complete=True
    class Deadline(Exception):pass
    def bronk(R,P,X):
        nonlocal calls
        calls+=1
        if calls%1024==0 and time.monotonic()>began+50:raise Deadline()
        if P==0 and X==0:maximal.append(R);return
        pool=P|X
        pivot=max(members(pool),key=lambda u:(P&adj[u]).bit_count()) if pool else None
        choices=P if pivot is None else P&~adj[pivot]&allmask
        while choices:
            bit=choices&-choices;v=bit.bit_length()-1
            bronk(R|bit,P&adj[v],X&adj[v]);P&=~bit;X|=bit;choices&=~bit
    try:bronk(0,allmask,0)
    except Deadline:enumeration_complete=False
    assert len(maximal)==len(set(maximal))
    for mask in maximal:
        C=members(mask)
        assert all((adj[a]>>b)&1 for a in C for b in C if a!=b)
        assert not any(all((adj[u]>>v)&1 for v in C) for u in members(allmask^mask))
    constraints=[mask for mask in maximal if mask.bit_count()>=6]
    maximal_cover=[]
    if enumeration_complete:
        for mask in route_masks:
            covering=next(j for j,C in enumerate(maximal) if mask&C==mask);maximal_cover.append(covering)
    six_count=None;six_complete=True;six_calls=0
    def count_six(P,k):
        nonlocal six_calls
        six_calls+=1
        if six_calls%1024==0 and time.monotonic()>began+65:raise Deadline()
        if P.bit_count()<k:return 0
        if k==1:return P.bit_count()
        total=0
        while P.bit_count()>=k:
            bit=P&-P;P^=bit;total+=count_six(P&adj[bit.bit_length()-1],k-1)
        return total
    try:six_count=count_six(allmask,6)
    except Deadline:six_complete=False
    graph=dict(vertices=64,edges=int(sum(x.bit_count() for x in adj)//2),degree=[x.bit_count() for x in adj],
        maximal_enumeration_complete=enumeration_complete,maximal_cliques_count=len(maximal),
        maximal_size_histogram={str(k):v for k,v in sorted(Counter(x.bit_count() for x in maximal).items())},
        constrained_maximal_cliques=len(constraints),maximal_size=max(x.bit_count() for x in maximal),
        exact_six_clique_count=six_count if six_complete else None,six_count_complete=six_complete,
        TRAIN_routes=512,TRAIN_unique_sets=len(set(route_masks)),all_TRAIN_routes_in_closure=True,
        TRAIN_maximal_cover_witness= maximal_cover if enumeration_complete else None,
        geometry_seconds=time.monotonic()-began,enumeration_recursive_calls=calls)
    dump(OUT/'graph_summary.json',graph)
    dump(OUT/'maximal_cliques.json',[dict(index=i,experts=members(mask),mask_hex=hex(mask)) for i,mask in enumerate(maximal)])
    np.savez_compressed(OUT/'graph_arrays.npz',coactivation_counts=counts,adjacency_masks=np.asarray(adj,np.uint64),
                        route_masks=np.asarray(route_masks,np.uint64),routes=routes,shared_waves=shared)
    results=[]
    for cap in (1,2,3):
        model=cp_model.CpModel();x=[model.NewBoolVar('x'+str(e)) for e in range(64)]
        for mask in constraints:model.Add(sum(x[e] for e in members(mask))<=cap)
        for var in x:model.AddHint(var,0)
        model.Maximize(sum(x));objective=model.Proto().objective
        assert objective.scaling_factor==-1 and objective.offset==0 and list(objective.coeffs)==[-1]*64
        model.ExportToFile(str(OUT/f'cap{cap}.model.pbtxt'))
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=30
        solver.parameters.num_search_workers=8;solver.parameters.random_seed=20261116
        status=solver.Solve(model);response=solver.ResponseProto();(OUT/f'cap{cap}.response.pbtxt').write_text(str(response))
        incumbent=None;chosen=None;upper=64;bound_source='trivial64_no_trusted_integer_response'
        if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
            chosen=[e for e in range(64) if solver.Value(x[e])];incumbent=len(chosen)
            assert all(sum(bool((mask>>e)&1) for e in chosen)<=cap for mask in constraints)
            integer_upper=-int(response.inner_objective_lower_bound)
            # Objective is integer minimization of -sumX internally, so this is an exact upper bound.
            if incumbent<=integer_upper<=64 and abs(float(integer_upper)-solver.BestObjectiveBound())<1e-9:
                upper=integer_upper;bound_source='exact_negated_int64_inner_objective_lower_bound'
            if status==cp_model.OPTIMAL:assert upper==incumbent
        result=dict(cap=cap,status=solver.StatusName(status),incumbent_count=incumbent,incumbent_experts=chosen,
                    valid_integer_upper_bound=upper,bound_source=bound_source,
                    inner_objective_lower_bound_int64=int(response.inner_objective_lower_bound),
                    displayed_external_best_bound=float(solver.BestObjectiveBound()),wall_seconds=solver.WallTime(),
                    constraints=len(constraints),objective_scaling=-1,objective_offset=0,
                    full_closure_constraints_enumerated=enumeration_complete)
        results.append(result);print(json.dumps(result),flush=True)
    upper={r['cap']:r['valid_integer_upper_bound'] for r in results}
    capacity=12*upper[1]+17*upper[2]+3*upper[3];demand=64*5
    impossible=capacity<demand
    report=dict(status='necessary_capacity_screen_complete',protocol_sha256=sha(OUT/'protocol.frozen.json'),graph=graph,perchip_capacity=results,
        fixed_shared_wave_histogram={'3':12,'2':17,'1':3},aggregate_X_upper_capacity=capacity,required_X_incidences=demand,
        peak10_impossible_for_this_pair_closure=impossible,
        certificate=f'12*{upper[1]} + 17*{upper[2]} + 3*{upper[3]} = {capacity} '+('<' if impossible else '>=')+' 320',
        interpretation='Necessary aggregate capacity only. An upper capacity>=320 does not prove a joint physical map exists; incomplete enumeration, ifany, only weakens this safe upper bound.',
        protection='Only all6cliques oforiginalTRAIN pairgraph. Routes with a pair absent fromTRAIN are unprotected. No global routing robustness claim.',
        no_dev_read=True,no_placement_fit=True,no_native_or_inference=True,total_seconds=time.monotonic()-began)
    dump(OUT/'report.json',report)
    (OUT/'report.md').write_text('TRAIN pair-closure peak10 necessary screen\n\n'+report['certificate']+'\n\n'+
      ('Peak10 is impossible for this closure, coupledX5/O7 profile and fixed shared layout.\n' if impossible else 'This screen cannot rule out peak10; it does not construct or prove a feasible joint placement.\n')+
      '\nThe graph and thresholds use TRAIN only. All TRAIN routes are included. The closure adds hypothetical combinations; a route with a pair unseen in TRAIN is outside protection. No dev graph, new inference, fit or native replay was used.\n')
    for ref in p['references']:assert sha(ref['path'])==ref['sha256']
    dump(OUT/'completion.json',dict(status='sealed_necessary_screen',utc=datetime.now(timezone.utc).isoformat(),
         files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(OUT.rglob('*')) if x.is_file() and not x.name.endswith('.log')]))
    print(json.dumps(dict(certificate=report['certificate'],impossible=impossible,total_seconds=report['total_seconds'],
                         completion_sha256=sha(OUT/'completion.json')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));args=parser.parse_args();{'freeze':freeze,'run':run}[args.mode]()
