#!/usr/bin/env python3
"""Fixed GU-layout3 + Down-graph14 joint TRAIN prefix-view work screen."""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

from research_20261103_topology_placement_replay import CPP, PYTHON, load_python

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'g4-joint-prefix-replica-screen'
GU=BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix'
DOWN=BASE/'g4-down-paired-merge-screen'
B=BASE/'compact-coupled-solver/continuation/native/0107.npz'
CFG=BASE/'compact-coupled-solver/protocol.frozen.json'
DEN=10**9


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def stats(x):return dict(mean_ns=float(np.mean(x)),p99_ns=float(np.percentile(x,99)))
def ceiling(lb,b):return dict(LB=stats(lb),mean_pct=100*(1-np.mean(lb)/np.mean(b)),p99_pct=100*(1-np.percentile(lb,99)/np.percentile(b,99)))


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.frozen.json').exists()
    (OUT/'source-snapshot').mkdir(exist_ok=True)
    paths=[Path(__file__),B,CFG,GU/'protocol.frozen.json',GU/'contract_arrays.npz',GU/'rowid_ledgers.json',
           GU/'greedy_arrays.npz',GU/'independent/final-completion.json',GU/'native-v1/arrays.npz',
           DOWN/'arrays.npz',DOWN/'native-graph14/arrays.npz',DOWN/'completion.json',
           PYTHON,CPP,ROOT/'scripts/research_20261103_topology_placement_replay.py',
           ROOT/'scripts/research_20261102_whole_row_compiler.py']
    inputs=[]
    for path in paths:
        item=dict(path=str(path),sha256=sha(path))
        if path.suffix in ('.py','.cpp'):
            snapshot=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name)
            shutil.copy2(path,snapshot);item['snapshot']=str(snapshot)
        inputs.append(item)
    a,d,b=load(GU/'contract_arrays.npz'),load(DOWN/'arrays.npz'),load(B)
    assert np.array_equal(a['routes'],d['routes']) and np.array_equal(a['routes'],b['routes'])
    assert np.array_equal(a['base'],d['base']) and np.array_equal(a['rows'],d['rows'])
    p=dict(status='frozen_before_joint_work',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
           inputs=inputs,service=json.loads(CFG.read_text())['service'],fixed_GU_layout=3,fixed_Down_graph=14,
           scope='OnlyCP0107 G4L0 actualW8 TRAIN512. No image/source-layout fitting, heldout/newweights/alllayers.',
           model='84 runtime bits:48GUu+36Downv;6 z=AND8u. Chipwork=CPbase+GU_D*u+GU_Z*z+Down_effect*v. Fullordinary commands/input/read/fringeoutput exact.',
           LP=dict(variables=91,inequalities=70,solver='scipy.linprog HiGHS',scale=28000,
                   certificate='Nonpositive solverduals rationalized towardzero atdenominator1e9 inactualns. ExactPythoninteger boxdual; nonnegativeT residual. No rawfloatprimal used asLB.'),
           feasible=dict(initial_classes=['allB','allA_allmerged','frozen_GU_greedy_DownB','GUB_frozen_Down_unitflow'],
                         orders=['all48GU_then36Down','perexpert_interleavedGU_Down'],passes=2,
                         objective='lexicographic(maxchipwork,sumSq,descendingchiploads)',
                         runtime_policy='Bestof8 deterministic greedy paths perquery; onefixedGU+Downimage. Softwaretimezero optimistic, notmeasured.'),
           CP32=dict(indices=np.linspace(0,511,32,dtype=int).tolist(),workers=1,time_limit_s=0.5,seed=20261113,
                     incumbent='jointgreedybits retained andhinted',scope='Integerworkdiagnostic only; doesnotreplacegreedy512 ortriggerCP512.'),
           capacity='SumuniqueGUcopies35638272B+Downcopies25165824B=60804096B. Increment512+384=896waves/56MiB, not additiveperformance. FullcombinedstaticphysicalBFD separately.',
           gate=dict(mean_pct=2,p99_pct=5,action='Ifhost+feasiblework potentialpasses,evaluateonefixedjointpolicy512native+Bbridge under separatelyfrozenwrapper. Ifnativefails,stopselectorbench; else reportbefore productionselector/strongzero-copy/samebudgetwholeGU+Down/partialRDA.'),
           limits='Work/LPnotnativeperformance. ComputedGUrows mayexceed1024,butlogicaluniqueGU1024;Down1536unique fullprivate512input. AllGUjobs->fullA8->Down unchanged;onecontext/chip. GenericLP/replication/coordinate principles,nonewalgorithmclaim.')
    dump(OUT/'protocol.frozen.json',p)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__))))


def inputs():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for x in p['inputs']:assert sha(x['path'])==x['sha256']
    assert sha(__file__)==p['source_sha256']
    return p,load(GU/'contract_arrays.npz'),load(GU/'greedy_arrays.npz'),load(DOWN/'arrays.npz'),load(DOWN/'native-graph14/arrays.npz')


def actual_matrix(dg,dd,z,base):
    A=np.zeros((70,91),np.int64);rhs=np.zeros(70,np.int64)
    A[:16,:48]=dg.reshape(48,16).T;A[:16,48:84]=dd.reshape(36,16).T
    A[:16,84:90]=z.T;A[:16,90]=-1;rhs[:16]=-base
    for e in range(6):
        for i in range(8):A[16+8*e+i,84+e]=1;A[16+8*e+i,8*e+i]=-1
        A[64+e,8*e:8*e+8]=1;A[64+e,84+e]=-1;rhs[64+e]=7
    return A,rhs


def integer_dual(A,rhs,y):
    y=np.minimum(y,0).copy();mass=-y[:16].sum()
    if mass:y*=min(1.0,(1-1e-10)/mass)
    actual=y.copy();actual[16:]*=28000
    numerator=np.trunc(actual*DEN).astype(np.int64)
    residual=[(DEN if j==90 else 0)-sum(int(A[i,j])*int(numerator[i]) for i in range(70)) for j in range(91)]
    assert np.all(numerator<=0) and residual[-1]>=0
    bound=sum(int(numerator[i])*int(rhs[i]) for i in range(70))+sum(min(0,r) for r in residual[:90])
    return bound//DEN,numerator,residual[-1]


def run():
    p,a,g,d,dn=inputs();assert not (OUT/'work_report.json').exists();t0=time.monotonic()
    routes,base,native=a['routes'],a['base'],a['baseline_native']
    DG,DZ,DD=a['D'][3],a['Z'],d['effects'][14]
    objective=np.zeros(91);objective[-1]=1
    bounds=np.zeros(512,np.int64);primal=np.zeros(512);dual=np.zeros((512,70),np.int64)
    rt=np.zeros(512,np.int64)
    for q,selected in enumerate(routes):
        A,rhs=actual_matrix(DG[selected],DD[selected],DZ[selected],base[q])
        scaled=A.astype(float);br=rhs.astype(float);scaled[:16,:90]/=28000;br[:16]/=28000
        result=linprog(objective,A_ub=scaled,b_ub=br,bounds=[(0,1)]*90+[(0,None)],method='highs')
        assert result.success
        bounds[q],dual[q],rt[q]=integer_dual(A,rhs,result.ineqlin.marginals)
        primal[q]=result.fun*28000
        assert -1e-3<=primal[q]-bounds[q]<2
    np.savez_compressed(OUT/'LP_arrays.npz',primal_work=primal,work_LB=bounds,dual_numerators=dual,T_residual_numerator=rt)
    print(json.dumps(dict(stage='LP',necessary=ceiling(bounds+2000,native),seconds=time.monotonic()-t0)),flush=True)

    # Eight deterministic paths, one identical image and one runtime multi-start policy.
    values=np.zeros((2,4,512),np.int64);all_u=np.zeros((2,4,512,6,8),np.int8)
    all_v=np.zeros((2,4,512,6,6),np.int8);initial_values=values.copy();changes=np.zeros((2,4,512,2),np.int16)
    stage=[('u',e,i) for e in range(6) for i in range(8)]+[('v',e,i) for e in range(6) for i in range(6)]
    inter=[item for e in range(6) for i in range(8) for item in ([('u',e,i),('v',e,i)] if i<6 else [('u',e,i)])]
    def key(x):return (int(x.max()),int(x@x),*sorted(map(int,x),reverse=True))
    for order,moves in enumerate([stage,inter]):
        for start in range(4):
            for q,selected in enumerate(routes):
                u=np.ones((6,8),np.int8) if start==1 else g['feasible_bits'][3,q].copy() if start==2 else np.zeros((6,8),np.int8)
                v=np.ones((6,6),np.int8) if start==1 else dn['merge_bits'][q].reshape(6,6).copy() if start==3 else np.zeros((6,6),np.int8)
                dg,dz,dd=DG[selected],DZ[selected],DD[selected]
                loads=base[q]+np.einsum('ei,eic->c',u,dg)+(u.all(1)[:,None]*dz).sum(0)+np.einsum('ei,eic->c',v,dd)
                initial_values[order,start,q]=loads.max()
                for pi in range(2):
                    count=0
                    for phase,e,i in moves:
                        if phase=='u':
                            oldz=int(u[e].all());sign=1-2*int(u[e,i]);u[e,i]^=1
                            proposal=loads+sign*dg[e,i]+(int(u[e].all())-oldz)*dz[e]
                        else:
                            sign=1-2*int(v[e,i]);v[e,i]^=1;proposal=loads+sign*dd[e,i]
                        if key(proposal)<key(loads):loads=proposal;count+=1
                        elif phase=='u':u[e,i]^=1
                        else:v[e,i]^=1
                    changes[order,start,q,pi]=count
                full=base[q]+np.einsum('ei,eic->c',u,dg)+(u.all(1)[:,None]*dz).sum(0)+np.einsum('ei,eic->c',v,dd)
                assert np.array_equal(full,loads)
                values[order,start,q]=loads.max();all_u[order,start,q]=u;all_v[order,start,q]=v
            print(json.dumps(dict(order=order,start=start,initial=stats(initial_values[order,start]),final=stats(values[order,start]))),flush=True)
    flat=values.reshape(8,512);winner=flat.argmin(0);qindex=np.arange(512)
    chosen=flat[winner,qindex];chosen_u=all_u.reshape(8,512,6,8)[winner,qindex];chosen_v=all_v.reshape(8,512,6,6)[winner,qindex]
    assert np.all(chosen<=base.max(1)) and np.all(chosen>=bounds)
    # Physical inventory is independent of runtime selection and jointly repacked.
    resident=a['resident_rows'].copy()
    for e in range(64):
        for src,dest in d['layouts'][14,e]:resident[e,1,dest]=256
    backend=load_python();s=backend.Service(**p['service'])
    storage=backend.lower_rows(resident[:,0],resident[:,1],1536,[512]*64+[1024],s)
    physical=backend.physical_addresses(storage,s)
    original=backend.physical_addresses(backend.lower_rows(a['rows'][:,0],a['rows'][:,1],1536,[512]*64+[1024],s),s)
    addresses=np.asarray([[x[k] for k in ('chip','block','wordline','reserved_wordlines','pages','first_plane')] for x in physical.pop('addresses')],np.int32)
    original.pop('addresses')
    assert physical['useful_bytes']-original['useful_bytes']==60804096
    assert physical['reserved_wave_bytes']-original['reserved_wave_bytes']==896*65536
    assert physical['programmed_page_bytes']-original['programmed_page_bytes']==896*65536
    np.savez_compressed(OUT/'work_arrays.npz',routes=routes,rows_B=a['rows'],base=base,baseline_native=native,
                        DG=DG,DZ=DZ,DD=DD,Down_edges=d['layouts'][14],policy_values=values,
                        policy_initial_values=initial_values,policy_changes=changes,policy_u=all_u,policy_v=all_v,
                        chosen_policy=winner,chosen_work=chosen,chosen_u=chosen_u,chosen_v=chosen_v,
                        resident_rows=resident,resident_offsets=storage['offsets'],resident_jobs=storage['jobs'],resident_addresses=addresses)
    potential=ceiling(chosen+2000,native)
    passed=potential['mean_pct']>=2 or potential['p99_pct']>=5
    report=dict(status='bounded_joint_work_not_native',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),
                LP_necessary=ceiling(bounds+2000,native),LP_certificates=512,LP_max_integer_floor_gap_ns=float(np.max(primal-bounds)),
                baseline_native=stats(native),baseline_work=stats(base.max(1)),feasible_joint_work=stats(chosen),
                host_plus_feasible_work_potential=potential,gate_passed=bool(passed),
                policy='Perquery bestof8 fixed deterministic two-pass greedy paths; no runtime physicalgraph choice.',
                policy_results=[dict(order=o,start=st,initial=stats(initial_values[o,st]),final=stats(values[o,st]),changed_bits=changes[o,st].sum(0).tolist()) for o in range(2) for st in range(4)],
                chosen_policy_histogram={str(i):int(np.sum(winner==i)) for i in range(8)},
                flips_attempted=8*512*2*84,budget=dict(GU_useful_copy_bytes=35638272,Down_useful_copy_bytes=25165824,
                    total_useful_copy_bytes=60804096,extra_waves=896,extra_pages=3584,extra_programmed_bytes=896*65536,
                    baseline_physical=original,joint_resident_physical=physical),seconds=time.monotonic()-t0,
                limits='No native performance from work. Zero selector/scatter cost sensitivity only. Fixed sourceimages; computedGUrowcount mayinclude duplicates;Down rows1536unique. Jointbenefit notsumofsinglebranches. Strongcontrols after nativegate,notnovelalgorithm.')
    dump(OUT/'work_report.json',report);print(json.dumps(report,indent=2))


def cp32():
    from ortools.sat.python import cp_model
    import ortools
    p,a,g,d,dn=inputs();w=load(OUT/'work_arrays.npz');lp=load(OUT/'LP_arrays.npz')
    dest=OUT/'cp32';dest.mkdir(exist_ok=False)
    cp_proto=dict(status='frozen_before_integer32',source_sha256=sha(__file__),work_protocol_sha256=sha(OUT/'protocol.frozen.json'),
                  work_arrays_sha256=sha(OUT/'work_arrays.npz'),LP_arrays_sha256=sha(OUT/'LP_arrays.npz'),ortools_version=ortools.__version__,
                  indices=p['CP32']['indices'],time_limit_s=0.5,workers=1,seed=20261113,
                  scope='Diagnosticonly; do not replacegreedy512 ortriggernativeCP/heldout.')
    dump(dest/'protocol.frozen.json',cp_proto)
    records=[];us=[];vs=[]
    for q in cp_proto['indices']:
        selected=w['routes'][q];dg,dd,dz=w['DG'][selected],w['DD'][selected],w['DZ'][selected]
        m=cp_model.CpModel();u=[[m.NewBoolVar(f'u{e}_{i}') for i in range(8)] for e in range(6)]
        v=[[m.NewBoolVar(f'v{e}_{i}') for i in range(6)] for e in range(6)];z=[m.NewBoolVar(f'z{e}') for e in range(6)]
        inc=int(w['chosen_work'][q]);lower=int(lp['work_LB'][q]);T=m.NewIntVar(lower,inc,'T')
        for e in range(6):
            for i in range(8):m.Add(z[e]<=u[e][i]);m.AddHint(u[e][i],int(w['chosen_u'][q,e,i]))
            m.Add(z[e]>=sum(u[e])-7);m.AddHint(z[e],int(w['chosen_u'][q,e].all()))
            for i in range(6):m.AddHint(v[e][i],int(w['chosen_v'][q,e,i]))
        m.AddHint(T,inc)
        for c in range(16):m.Add(int(w['base'][q,c])+sum(int(dg[e,i,c])*u[e][i] for e in range(6) for i in range(8))+sum(int(dd[e,i,c])*v[e][i] for e in range(6) for i in range(6))+sum(int(dz[e,c])*z[e] for e in range(6))<=T)
        m.Minimize(T);solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=0.5
        solver.parameters.num_search_workers=1;solver.parameters.random_seed=20261113
        status=solver.Solve(m);uu=w['chosen_u'][q].copy();vv=w['chosen_v'][q].copy();best=inc
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            uu=np.asarray([[solver.Value(x) for x in row] for row in u],np.int8)
            vv=np.asarray([[solver.Value(x) for x in row] for row in v],np.int8)
            loads=w['base'][q]+np.einsum('ei,eic->c',uu,dg)+(uu.all(1)[:,None]*dz).sum(0)+np.einsum('ei,eic->c',vv,dd)
            best=int(loads.max());assert best==solver.Value(T)<=inc
        bound=max(lower,int(np.floor(solver.BestObjectiveBound())));assert bound<=best
        rec=dict(query=q,status=solver.StatusName(status),greedy_work_ns=inc,best_work_ns=best,best_bound_ns=bound,
                 improvement_ns=inc-best,gap_ns=best-bound,wall_seconds=solver.WallTime())
        records.append(rec);us.append(uu);vs.append(vv);print(json.dumps(rec),flush=True)
    np.savez_compressed(dest/'arrays.npz',indices=np.asarray(cp_proto['indices']),u=np.stack(us),v=np.stack(vs))
    dump(dest/'report.json',dict(status='bounded_joint_integer_work_only',cases=records,
        optimal=sum(r['status']=='OPTIMAL' for r in records),feasible=sum(r['status']=='FEASIBLE' for r in records),
        mean_improvement_ns=float(np.mean([r['improvement_ns'] for r in records])),max_improvement_ns=max(r['improvement_ns'] for r in records),
        cases_gt3us=sum(r['improvement_ns']>3000 for r in records),limits='32fixedTRAINindices,not512or nativeperformance. No automaticnewselector/placement.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run','cp32']);args=ap.parse_args()
    {'freeze':freeze,'run':run,'cp32':cp32}[args.mode]()
