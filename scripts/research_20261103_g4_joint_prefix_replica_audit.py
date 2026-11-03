#!/usr/bin/env python3
"""Independent joint rational-dual/CP/row-ownership/event audit adapter.

Does not import either joint producer or the native/compiler backend.
The independent scheduler is reused unchanged from the preceding sealed audit.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
JOINT=BASE/'g4-joint-prefix-replica-screen';OUT=JOINT/'independent'
GU=BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix';DOWN=BASE/'g4-down-paired-merge-screen'
SCHED=ROOT/'scripts/research_20261103_g4_independent_prefix_audit.py'
DEN=10**9


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def stats(x):return dict(mean_ns=float(np.mean(x)),p99_ns=float(np.percentile(x,99)))


def freeze():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),SCHED,GU/'independent/completion.json',JOINT/'protocol.frozen.json',
           JOINT/'work_arrays.npz',JOINT/'LP_arrays.npz',JOINT/'work_report.json',JOINT/'cp32/report.json',JOINT/'cp32/arrays.npz',
           JOINT/'native-v1/protocol.frozen.json',JOINT/'native-v1/arrays.npz',JOINT/'native-v1/row_owners.npz',
           GU/'rowid_ledgers.json',GU/'contract_arrays.npz',DOWN/'arrays.npz']
    dump(OUT/'protocol.frozen.json',dict(status='frozen_before_independent_checks',source_sha256=sha(__file__),
        inputs=[dict(path=str(p),sha256=sha(p)) for p in files],scope='Independent actualintegerLPcertificate,CP32rowcost,chosen512owners/counters and6nativeeventtraces. No fit/native replay/newdata.',
        scheduler='Reuse unchanged independentlywritten genericready-list scheduler from sealedv3audit, no backend/producer imports.'))
    print(sha(OUT/'protocol.frozen.json'))


def gu_rows(original,entry,u):
    rr=original.copy()
    for i,c in enumerate(entry['sources']):rr[c]=128 if u[i] else rr[c]
    for i,c in enumerate(entry['low_destinations']):rr[c]=0 if u[i] else 40
    c=entry['high_destination'];rr[c]=original[c] if not u[7] else entry['high_prefix_rows'] if not all(u) else 0
    return rr


def own_rows(rows,route,u,v,ledger,edges):
    result=rows[np.r_[route,64]].copy()
    for ordinal,e in enumerate(route):
        result[ordinal,0]=gu_rows(rows[e,0],ledger[e],u[ordinal])
        for i,(source,dest) in enumerate(edges[e]):
            if v[ordinal,i]:result[ordinal,1,source]=0;result[ordinal,1,dest]=256
    return result


def actual_bill(local):
    widths=np.full((7,2,1),1536,np.int64);widths[:6,1]=512;widths[6,1]=1024
    active=local>0;pages=(local*widths+16383)//16384;waves=(pages+3)//4;emitted=pages*16384//widths
    chipwork=(active*(250+(widths+1)//2)+waves*28000+emitted*2).sum((0,1))
    counter=np.stack([active.sum((0,2)),(local*widths).sum((0,2)),pages.sum((0,2)),waves.sum((0,2)),
                      (active*widths).sum((0,2)),(emitted*4).sum((0,2)),
                      (active*((widths+1)//2)+emitted*2).sum((0,2))],axis=1)
    return chipwork,counter


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for x in p['inputs']:assert sha(x['path'])==x['sha256']
    assert not (OUT/'completion.json').exists()
    # Establish that the imported independent scheduler is the previouslysealed
    # source, rather than whichever live script might now have that name.
    seal=json.loads((GU/'independent/completion.json').read_text())
    anchor=[x for x in seal['files'] if x['path']==str(SCHED)]
    assert len(anchor)==1 and sha(SCHED)==anchor[0]['sha256']
    spec=importlib.util.spec_from_file_location('sealed_independent_event_scheduler',SCHED)
    scheduler=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheduler)
    w,lp=load(JOINT/'work_arrays.npz'),load(JOINT/'LP_arrays.npz')
    cp=load(JOINT/'cp32/arrays.npz');cpr=json.loads((JOINT/'cp32/report.json').read_text())
    n=load(JOINT/'native-v1/arrays.npz');owners=load(JOINT/'native-v1/row_owners.npz')
    gu=load(GU/'contract_arrays.npz');down=load(DOWN/'arrays.npz')
    ledger=json.loads((GU/'rowid_ledgers.json').read_text())[3]['experts']
    rows,routes=w['rows_B'],w['routes']
    assert np.array_equal(routes,n['routes']) and np.array_equal(routes,owners['routes'])
    dg,dz,dd=np.zeros_like(w['DG']),np.zeros_like(w['DZ']),np.zeros_like(w['DD'])
    gu_image=np.full((64,16,128),-1,np.int16);down_image=np.full((64,16,256),-1,np.int16)
    for e in range(64):
        entry=ledger[e]
        for i,c in enumerate(entry['sources']):
            dg[e,i,c]=28086
            if i<7:dg[e,i,entry['low_destinations'][i]]=-29102
            else:dg[e,i,entry['high_destination']]=-28086
        dz[e,entry['high_destination']]=-29102
        for c,ids in entry['resident_prefix_image'].items():gu_image[e,int(c),:len(ids)]=ids
        starts=np.r_[0,np.cumsum(rows[e,1])]
        di={c:list(range(int(starts[c]),int(starts[c+1]))) for c in range(16)}
        for j,(source,dest) in enumerate(w['Down_edges'][e]):
            dd[e,j,source]=-28762;dd[e,j,dest]=28256;di[int(dest)]+=di[int(source)]
        for c,ids in di.items():down_image[e,c,:len(ids)]=ids
    assert np.array_equal(dg,w['DG']) and np.array_equal(dd,w['DD']) and np.array_equal(dz,w['DZ'])
    assert np.array_equal(w['Down_edges'],down['layouts'][14])
    # Exact arbitrary-precision integer box-dual arithmetic, no floatingfun.
    for q,route in enumerate(routes):
        A=np.zeros((70,91),np.int64);rhs=np.zeros(70,np.int64)
        A[:16,:48]=dg[route].reshape(48,16).T;A[:16,48:84]=dd[route].reshape(36,16).T
        A[:16,84:90]=dz[route].T;A[:16,90]=-1;rhs[:16]=-w['base'][q]
        for e in range(6):
            for i in range(8):A[16+8*e+i,8*e+i]=-1;A[16+8*e+i,84+e]=1
            A[64+e,8*e:8*e+8]=1;A[64+e,84+e]=-1;rhs[64+e]=7
        y=lp['dual_numerators'][q];assert np.all(y<=0)
        c=np.zeros(91,dtype=object);c[-1]=DEN
        residual=c-A.astype(object).T@y.astype(object)
        assert residual[-1]>=0 and residual[-1]==lp['T_residual_numerator'][q]
        numerator=sum(int(yy)*int(bb) for yy,bb in zip(y,rhs))+sum(min(0,int(r)) for r in residual[:90])
        assert numerator//DEN==lp['work_LB'][q]
    counters=np.zeros_like(n['counters']);actual=np.zeros(512,np.int64)
    for q,route in enumerate(routes):
        local=own_rows(rows,route,w['chosen_u'][q],w['chosen_v'][q],ledger,w['Down_edges'])
        assert np.array_equal(local,n['compact_rows'][q])
        chip,counters[q]=actual_bill(local);actual[q]=chip.max()
        for ordinal,e in enumerate(route):
            for phase,key,length,image in [(0,'GU',1024,gu_image),(1,'Down',1536,down_image)]:
                mapping=owners[key][q,ordinal];cc,position=mapping.T
                assert np.all(position>=0) and np.all(position<local[ordinal,phase,cc])
                assert np.array_equal(image[e,cc,position],np.arange(length))
            assert local[ordinal,0].sum()>=1024 and local[ordinal,1].sum()==1536
    assert np.array_equal(actual,w['chosen_work']) and np.array_equal(actual,n['actual_work'])
    assert np.array_equal(counters,n['counters']) and np.all(lp['work_LB']<=actual)
    assert np.array_equal(n['baseline'],w['baseline_native'])
    # All eight policy costs/retention are also independently row-accounted.
    for order in range(2):
        for start in range(4):
            for q,route in enumerate(routes):
                local=own_rows(rows,route,w['policy_u'][order,start,q],w['policy_v'][order,start,q],ledger,w['Down_edges'])
                assert actual_bill(local)[0].max()==w['policy_values'][order,start,q]
    flat=w['policy_values'].reshape(8,512)
    assert np.array_equal(flat.argmin(0),w['chosen_policy']) and np.array_equal(flat.min(0),actual)
    assert np.array_equal(cp['indices'],np.linspace(0,511,32,dtype=int))
    for i,q in enumerate(cp['indices']):
        local=own_rows(rows,routes[q],cp['u'][i],cp['v'][i],ledger,w['Down_edges'])
        record=cpr['cases'][i];value=int(actual_bill(local)[0].max())
        assert value==record['best_work_ns']<=w['chosen_work'][q]
        assert lp['work_LB'][q]<=record['best_bound_ns']<=value
        if record['status']=='OPTIMAL':assert value==record['best_bound_ns']
    traces=[]
    for q in (0,255,511):
        for tag in ('baseline','joint'):
            plan=load(JOINT/f'native-v1/q{q}.{tag}.npz')
            assert np.array_equal(plan['total_rows'],plan['rows'].sum(2))
            value,bound,trace=scheduler.independent_graph(plan,np.arange(7))
            assert value==plan['cost'][0] and bound==plan['lower_bound'][0] and np.array_equal(trace,plan['trace'])
            traces.append(dict(query=q,case=tag,nodes=len(trace),cost_ns=value))
    bcost,cost=n['baseline'],n['cost']
    gain=dict(mean_pct=100*(1-cost.mean()/bcost.mean()),p99_pct=100*(1-np.percentile(cost,99)/np.percentile(bcost,99)))
    incremental=dict(mean_pct=100*(1-cost.mean()/n['Down_single'].mean()),
                     p99_pct=100*(1-np.percentile(cost,99)/np.percentile(n['Down_single'],99)))
    report=dict(status='PASS_staged_native_gate_only_controls_pending',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),
                exact_rational_LP_certificates=512,rational_denominator=DEN,chosen_joint_rows_checked=512,
                GU_unique_ownership_maps=3072,Down_unique_ownership_maps=3072,all8policy_row_cost_checks=4096,
                CP32_actualrow_costs_checked=32,CP32_status=dict(optimal=cpr['optimal'],feasible=cpr['feasible']),
                independent_event_traces=traces,baseline=stats(bcost),joint=stats(cost),improvement=gain,
                incremental_vs_Down_only=incremental,
                stage_gate_passed=bool(gain['mean_pct']>=2 or gain['p99_pct']>=5),
                limitations=['OnlyfixedGU3+Down14 and one8pathjointgreedy policy onTRAINL0.',
                             'Zero software/selector/scatter cost is optimistic; no productionselector or samecapacity strongcontrols yet.',
                             'Passing2/5stagegate does not establish novelty, physicalexecution, modelquality, wholemodel orheldoutgain.',
                             'The strongest singlebranch has smaller P99; jointmean gain relative toDownonly is about0.208%.',
                             'CP32FEASIBLE cases are notoptimal; theirbits did notreplacegreedy512.'])
    dump(OUT/'audit.json',report)
    text=f'''The fixed GU-layout3 + Down-graph14 joint model passes independent correctness checks and the mean stage gate, with strong controls still pending.

| Case | Mean (ms) | Empirical P99 (ms) |
|---|---:|---:|
| Frozen CP0107 B | {bcost.mean()/1e6:.9f} | {np.percentile(bcost,99)/1e6:.9f} |
| GU-only greedy | {n['GU_single'].mean()/1e6:.9f} | {np.percentile(n['GU_single'],99)/1e6:.9f} |
| Down-only graph14 | {n['Down_single'].mean()/1e6:.9f} | {np.percentile(n['Down_single'],99)/1e6:.9f} |
| Joint fixed image/policy | {cost.mean()/1e6:.9f} | {np.percentile(cost,99)/1e6:.9f} |

The joint sensitivity improves mean {gain['mean_pct']:.6f}% and P99 {gain['p99_pct']:.6f}% versus B. Its mean increment versus Down-only is only {incremental['mean_pct']:.6f}%; its P99 is worse ({incremental['p99_pct']:.6f}% change). Branch gains cannot be added. There are 36 regressing queries, with maximum regression 29,118 ns.

Independent checks rederived all per-bit costs, verified 512 exact rational LP box-dual certificates with Python integer arithmetic, recomputed all 4,096 policy row-job costs and 512 chosen chip vectors, checked every routed expert GU1024/Down1536 consumed-row map, and matched all six representative native event traces node by node. Actual GU computed totals may exceed1024; the unchanged compiler guard uses computed totals while a separate row-ID map proves logical consumption. Original B's512 costs/bounds bridge exactly. CP32 reports six OPTIMAL and26 FEASIBLE cases, mean feasible improvement803.4375 ns, max1692 ns; no CP bits replaced nativegreedy512 and nonzero gaps are retained.

The images add35,638,272 GU useful bytes plus25,165,824 Down bytes =60,804,096 bytes (57.987305 MiB), but programmed growth is896waves=56MiB due reused GU padding. The static joint BFD allocation is32LSBblocks/384MiB versus B16/192MiB; counts cover full routed/shared FFN coefficients in this layer, exclude attention/dense/FTL/spares/hostmetadata, and do not demonstrate device programming. The separate independent-contract audit checks row coverage, all view aliases and this physical inventory.

This is synthetic ordinary-read TRAIN L0 timing with zero selector/scatter/software overhead. It passes only the mean resource-allocation gate for deciding whether to run production selector and strong zero-copy/same-budget wholeGU+Down/partial-RDA controls. It does not establish a new algorithm, numerical quality, actual ABI execution, full-model speedup or heldout behavior. No such expansion was run in this stage.
'''
    (OUT/'report.md').write_text(text)
    files=[OUT/'protocol.frozen.json',OUT/'audit.json',OUT/'report.md',Path(__file__),
           JOINT/'work_report.json',JOINT/'LP_arrays.npz',JOINT/'work_arrays.npz',JOINT/'cp32/report.json',JOINT/'cp32/arrays.npz',
           JOINT/'native-v1/protocol.frozen.json',JOINT/'native-v1/report.json',JOINT/'native-v1/arrays.npz',JOINT/'native-v1/row_owners.npz']
    dump(OUT/'completion.json',dict(status='sealed_stage_controls_pending',files=[dict(path=str(p),sha256=sha(p)) for p in files]))
    print(json.dumps(dict(status=report['status'],gain=gain,incremental=incremental,completion_sha256=sha(OUT/'completion.json')),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
