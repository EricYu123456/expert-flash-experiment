#!/usr/bin/env python3
"""Bounded W8 TRAIN diagnostic: arbitrary one-copy integer row geometries.

The MIP minimizes joint mandatory chip work, not actual causal FFN latency.
All newly found rows are lowered/replayed through the sealed native backend.
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize._highspy import _core as highs
from scipy.sparse import coo_matrix

from research_20261103_topology_placement import sha
from research_20261103_topology_placement_replay import Replay, load_python

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/ordinary-read-compiler'
DEST = ROOT / 'results/research-2026-11-02/round2/arbitrary-row-headroom'


def row_cost(rows, columns, service, useful_only=False):
    rows=np.asarray(rows,dtype=np.int64)
    pages=(rows*columns+service['page_bytes']-1)//service['page_bytes']
    waves=(pages+service['planes']-1)//service['planes']
    emitted=rows if useful_only else pages*service['page_bytes']//columns
    assert service['output_fringe']==1 and service['input_bytes']==1 and service['output_bytes']==4
    return waves*service['ordinary_read_ns']+(rows>0)*(service['command_ns']+columns//2)+2*emitted


def extract_rows(plan):
    rr=np.zeros((len(plan['offsets']),2,16),np.int64)
    for e in range(len(rr)):
        for p in range(2):
            lo,hi=plan['offsets'][e,p]
            for chip,r,*_ in plan['jobs'][lo:hi]:rr[e,p,chip]+=r
    return rr


def joint_mean(rows,routes,columns,shared,service,useful_only=False):
    cost=np.stack([row_cost(rows[:,p],columns[p],service,useful_only) for p in range(2)],axis=1)
    return float((cost[routes].sum((1,2))+shared).max(1).mean())


def refit_uniform(initial,routes,columns,shared,service,seconds,seed):
    """Strong control: same two phase geometries, free histogram, joint target.

    50% single-expert cyclic relabel / 50% expert-pair swaps. This control does
    not claim a new optimizer and does not preserve the old label histogram.
    """
    rng=np.random.default_rng(seed);experts=len(initial)
    variants=[];labels=[];tables=[]
    for p in range(2):
        base=initial[0,p]
        vv=np.unique(np.stack([np.roll(base,k) for k in range(16)]),axis=0)
        ll=[]
        for rr in initial[:,p]:
            match=np.flatnonzero(np.all(vv==rr,axis=1));assert len(match)==1
            ll.append(match[0])
        variants.append(vv);labels.append(np.array(ll,dtype=np.int32))
        tables.append(row_cost(vv,columns[p],service))
    labels=np.asarray(labels);members=np.zeros((experts,len(routes)),bool)
    for t,r in enumerate(routes):members[r,t]=True
    loads=sum(tables[p][labels[p]][routes].sum(1) for p in range(2))+shared
    costs=loads.max(1);total=int(costs.sum());best=total;bestlabels=labels.copy()
    begin=time.monotonic();proposals=valid=accepted=0
    while time.monotonic()-begin<seconds:
        proposals+=1;p=int(rng.integers(2));a=int(rng.integers(experts))
        la=int(labels[p,a]);b=-1
        if rng.integers(2):
            lb=int(rng.integers(len(variants[p])));affected=np.flatnonzero(members[a]);sign=np.ones(len(affected),np.int64)
        else:
            b=int(rng.integers(experts));lb=int(labels[p,b]);affected=np.flatnonzero(members[a]^members[b]);sign=np.where(members[a,affected],1,-1)
        if la==lb or not len(affected):continue
        valid+=1;delta=tables[p][lb]-tables[p][la]
        proposed=loads[affected]+sign[:,None]*delta
        pc=proposed.max(1);change=int((pc-costs[affected]).sum())
        elapsed=time.monotonic()-begin
        temp=2000*max(1e-4,1-elapsed/seconds)**3
        if change<=0 or rng.random()<math.exp(-change/(len(routes)*temp)):
            loads[affected]=proposed;costs[affected]=pc;total+=change;labels[p,a]=lb
            if b>=0:labels[p,b]=la
            accepted+=1
            if total<best:best=total;bestlabels=labels.copy()
    witness=np.stack([variants[p][bestlabels[p]] for p in range(2)],axis=1)
    check=joint_mean(witness,routes,columns,shared,service)
    assert abs(check-best/len(routes))<1e-6
    return witness,dict(seconds=time.monotonic()-begin,proposals=proposals,valid=valid,accepted=accepted,
                        best_exact_joint_chip_ns=check,label_histogram_preserved=False,seed=seed)


def mip(initial,routes,columns,shared,service,seconds):
    """Rows/waves/active integer; max-query chip work continuous, shared fixed.

    Useful output rows are a relaxation of page-fringe drain. Waves/active
    charges and full-input reload are mandatory exact terms for the new rows.
    """
    E,P,M=initial.shape;unique,counts=np.unique(np.sort(routes,axis=1),axis=0,return_counts=True)
    N=len(unique);cells=E*P*M
    rid=np.arange(cells).reshape(E,P,M);wid=rid+cells;aid=rid+2*cells;yid=np.arange(N)+3*cells;nv=3*cells+N
    objective=np.zeros(nv);objective[yid]=counts/len(routes)
    lower=np.zeros(nv);upper=np.full(nv,np.inf)
    integrality=np.ones(nv,bool);integrality[yid]=False
    rowtot=initial.sum(2)
    for e in range(E):
        for p,C in enumerate(columns):
            upper[rid[e,p]]=rowtot[e,p]
            upper[wid[e,p]]=min(service['block_wordlines'],math.ceil(rowtot[e,p]*C/65536))
            upper[aid[e,p]]=1
    ii=[];jj=[];vv=[];lb=[];ub=[]
    def add(terms,lo=-np.inf,hi=np.inf):
        k=len(lb)
        for j,v in terms:ii.append(k);jj.append(int(j));vv.append(float(v))
        lb.append(lo);ub.append(hi)
    for e in range(E):
        for p,C in enumerate(columns):
            add([(j,1) for j in rid[e,p]],int(rowtot[e,p]),int(rowtot[e,p]))
            for c in range(M):
                r,w,a=rid[e,p,c],wid[e,p,c],aid[e,p,c]
                add([(r,C),(w,-65536)],hi=0)
                add([(r,1),(a,-rowtot[e,p])],hi=0)
                add([(a,1),(r,-1)],hi=0)
                add([(w,1),(a,-upper[w])],hi=0)
    for q,route in enumerate(unique):
        for c in range(M):
            terms=[(yid[q],1)]
            for e in route:
                for p,C in enumerate(columns):
                    terms.extend([(wid[e,p,c],-service['ordinary_read_ns']/1000),
                                  (aid[e,p,c],-(service['command_ns']+C/2)/1000),
                                  (rid[e,p,c],-.002)])
            add(terms,lo=float(shared[c])/1000)
    A=coo_matrix((vv,(ii,jj)),shape=(len(lb),nv)).tocsc()
    lp=highs.HighsLp();lp.num_col_=nv;lp.num_row_=len(lb)
    lp.col_cost_=objective;lp.col_lower_=lower;lp.col_upper_=upper
    lp.row_lower_=np.asarray(lb);lp.row_upper_=np.asarray(ub)
    lp.integrality_=[highs.HighsVarType.kInteger if k else highs.HighsVarType.kContinuous for k in integrality]
    lp.a_matrix_.format_=highs.MatrixFormat.kColwise
    lp.a_matrix_.start_=A.indptr.astype(np.int32);lp.a_matrix_.index_=A.indices.astype(np.int32);lp.a_matrix_.value_=A.data
    h=highs._Highs();h.setOptionValue('output_flag',False);h.setOptionValue('time_limit',float(seconds));h.setOptionValue('threads',4)
    h.setOptionValue('random_seed',20261103);h.setOptionValue('mip_rel_gap',0.0001)
    assert h.passModel(lp)==highs.HighsStatus.kOk
    x0=np.zeros(nv);x0[rid]=initial
    for p,C in enumerate(columns):x0[wid[:,p]]=(initial[:,p]*C+65535)//65536
    x0[aid]=(initial>0)
    warmcost=sum(row_cost(initial[:,p],C,service,True)[unique].sum(1) for p,C in enumerate(columns))+shared
    x0[yid]=warmcost.max(1)/1000
    assert np.all(A@x0>=np.asarray(lb)-1e-7) and np.all(A@x0<=np.asarray(ub)+1e-7)
    warm=highs.HighsSolution();warm.col_value=x0;warm.value_valid=True
    assert h.setSolution(warm)==highs.HighsStatus.kOk
    start=time.monotonic();h.run();elapsed=time.monotonic()-start;info=h.getInfo();sol=h.getSolution()
    x=np.asarray(sol.col_value);assert sol.value_valid and len(x)==nv
    violation=max(float(np.max(np.asarray(lb)-A@x)),float(np.max(A@x-np.asarray(ub))),0)
    assert violation<1e-4,violation
    witness=np.rint(x[rid]).astype(np.int64)
    assert np.array_equal(witness.sum(2),rowtot) and np.all(witness>=0)
    assert max(math.ceil(int(r)*columns[p]/65536) for e in range(E) for p in range(P) for r in witness[e,p])<=192
    return witness,dict(status=h.modelStatusToString(h.getModelStatus()),time_limit_seconds=seconds,
                         solve_seconds=elapsed,variables=nv,integer_variables=int(integrality.sum()),constraints=len(lb),
                         nonzeros=A.nnz,unique_route_sets=N,objective='mean_query_max_joint_mandatory_chip_us; fixed_shared; useful-output relaxation',
                         incumbent_mean_joint_chip_ns=float(info.objective_function_value*1000),
                         dual_bound_mean_joint_chip_ns=float(info.mip_dual_bound*1000),gap=float(info.mip_gap),
                         mip_nodes=int(info.mip_node_count),max_constraint_violation=violation,
                         warm_start_mean_joint_chip_ns=float(objective@x0*1000))


def summary(cost):
    return dict(mean_ns=float(np.mean(cost)),p99_ns=float(np.percentile(cost,99)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['granite4tiny','deepseek'],required=True)
    ap.add_argument('--mip-seconds',type=float,default=90);ap.add_argument('--control-seconds',type=float,default=5)
    args=ap.parse_args();started=time.monotonic();out=DEST/args.model;out.mkdir(parents=True,exist_ok=True)
    summary_path=BASE/'full-campaign-v5-W8-training/summary.json';campaign=json.loads(summary_path.read_text())
    rec=next(r for r in campaign['records'] if r['model']==args.model and r['layer']==0 and r['command_ns']==250)
    with np.load(rec['training']['path'],allow_pickle=False) as z:
        assert z['split'].item()=='train' and z['completed'].all()
        routes=np.ascontiguousarray(z['routes'][:,:,0].reshape(-1,6),np.int32)
    assert sha(rec['training']['path'])==rec['training']['sha256']
    service=rec['service'];D,I,E,K,L,SI=rec['shape'];columns=[D,I]
    sharedrows=np.asarray(rec['shared_rows']);shared=sum(row_cost(sharedrows[p],C,service) for p,C in enumerate([D,SI]))
    native=Replay();backend=load_python();plans={};records={};saved={}
    incumbents=[]
    for family in ['disjoint','cyclic','whole_rows']:
        art=next(a for a in rec['artifacts'] if a['family']==family)
        assert sha(ROOT/art['arrays'])==art['arrays_sha256']
        with np.load(ROOT/art['arrays'],allow_pickle=False) as z:plan={k:z[k] for k in z.files}
        rr=extract_rows(plan);cost,_=native.run(plan,routes,service)
        records[family]=dict(**summary(cost),joint_chip_work_ns=joint_mean(rr[:-1],routes,columns,shared,service))
        saved[family+'__cost']=cost;saved[family+'__rows']=rr
        plans[family]=plan;incumbents.append((float(cost.mean()),family,rr[:-1]))
    # A direct fixed-geometry joint-objective control, including free relabels.
    for family in ['cyclic','whole_rows']:
        initial=saved[family+'__rows'][:-1]
        rr,fit=refit_uniform(initial,routes,columns,shared,service,args.control_seconds,20261103)
        allrows=np.concatenate([rr,sharedrows[None]],axis=0)
        plan=backend.lower_rows(allrows[:,0],allrows[:,1],D,[I]*E+[SI],backend.Service(**service))
        cost,_=native.run(plan,routes,service)
        label=family+'_joint_free_labels';records[label]=dict(**summary(cost),fit=fit)
        saved[label+'__cost']=cost;saved[label+'__rows']=allrows;plans[label]=plan
        incumbents.append((float(cost.mean()),label,rr))
    # Use the strongest actual-service incumbent; all controls remain available.
    _,warm_family,initial=min(incumbents,key=lambda x:x[0])
    rr,fit=mip(initial,routes,columns,shared,service,args.mip_seconds)
    allrows=np.concatenate([rr,sharedrows[None]],axis=0)
    plan=backend.lower_rows(allrows[:,0],allrows[:,1],D,[I]*E+[SI],backend.Service(**service))
    cost,bounds=native.run(plan,routes,service)
    footprint=backend.physical_addresses(plan,backend.Service(**service))
    label='arbitrary_rows_mip';records[label]=dict(**summary(cost),fit=fit,footprint=footprint,
                            warm_family=warm_family,exact_joint_chip_ns=joint_mean(rr,routes,columns,shared,service),
                            relaxed_joint_chip_ns=joint_mean(rr,routes,columns,shared,service,True))
    saved[label+'__cost']=cost;saved[label+'__rows']=allrows;saved[label+'__native_bound']=bounds
    for k in ['offsets','jobs','input_columns','total_rows']:saved[label+'__'+k]=plan[k]
    # Small direct Python checks after native lowering, including all deps.
    checks=[]
    for q in [0,255,511]:
        selected=np.r_[routes[q],E]
        result=backend.ReferenceGraph.build(plan,selected,backend.Service(**service)).run()
        assert result['cost']==cost[q]
        checks.append(dict(query=q,cost_ns=int(cost[q]),nodes=len(result['trace'])))
    base=min(records[k]['mean_ns'] for k in records if k!='arbitrary_rows_mip')
    fit['valid_elapsed_mean_lower_bound_ns']=max(0,fit['dual_bound_mean_joint_chip_ns']+service['host_submit_ns']-1)
    result=dict(status='bounded_train_only_diagnostic',model=args.model,layer=0,physical_layer=rec['training']['physical_model_layer'],
                queries=len(routes),shape=rec['shape'],service=service,training=rec['training'],summary_path=str(summary_path),
                summary_sha256=sha(summary_path),records=records,checks=checks,
                mip_vs_best_control_mean_gain_pct=100*(1-float(cost.mean())/base),
                global_arbitrary_geometry_possible_mean_gain_upper_pct=100*(1-fit['valid_elapsed_mean_lower_bound_ns']/base),
                elapsed_seconds=time.monotonic()-started,script_sha256=sha(__file__),
                scope='Same onecopy complete I8 rows, shared fixed, atmost one native blockjob perexpert/phase/chip. Arbitrary row allocations perexpert, no fixed rowtype histogram. Joint chipwork MIP is a scheduling relaxation, not an achievable FFN-time claim. All512 W8TRAIN queries, no heldout.',
                attribution='Joint free-label controls preserve phase geometry but allow single-expert cyclic relabels and swaps. Any MIP gain over original phase-fit alone can therefore reflect joint refitting; comparison to these controls is mandatory.')
    np.savez_compressed(out/'arrays.npz',routes=routes,**saved)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['model','elapsed_seconds','mip_vs_best_control_mean_gain_pct','global_arbitrary_geometry_possible_mean_gain_upper_pct']},indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a in ['mean_ns','p99_ns','fit']} for k,v in records.items()},indent=2))


if __name__=='__main__':main()
