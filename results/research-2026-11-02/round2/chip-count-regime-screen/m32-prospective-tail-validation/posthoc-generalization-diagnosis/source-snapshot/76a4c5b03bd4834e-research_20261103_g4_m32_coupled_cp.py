#!/usr/bin/env python3
"""Bounded standard coupled-coloring CP control; zero replicas, TRAIN only."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model
import ortools
from research_20261103_topology_placement_replay import CPP,PYTHON,load_python

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
STAGE=BASE/'chip-count-regime-screen/stage-b';OUT=BASE/'chip-count-regime-screen/g4-M32-coupled-cp'
SOURCE=STAGE/'granite4tiny.M32.minimum_depth.plan.npz'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def stats(x):return dict(mean_ns=float(np.mean(x)),p99_ns=float(np.percentile(x,99)))


def rows_from(X,O,shared):
    rows=np.zeros((65,2,32),np.int64)
    for e in range(64):
        active=np.flatnonzero(~O[e]);assert len(active)==25 and X[e].sum()==5 and O[e].sum()==7 and not np.any(X[e]&O[e])
        rows[e,0,active]=41;rows[e,0,active[0]]=40
        rows[e,1,X[e]|O[e]]=128
    rows[64]=shared
    assert np.all(rows[:64,0].sum(1)==1024) and np.all(rows[:64,1].sum(1)==1536)
    return rows


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    z=load(SOURCE);rows=z['rows'];assert rows.shape==(65,2,32) and z['routes'].shape==(512,6)
    assert np.all(np.sort(rows[:64,0],axis=1)==[0]*7+[40]+[41]*24)
    assert np.all(np.sort(rows[:64,1],axis=1)==[0]*20+[128]*12)
    O=rows[:64,0]==0;X=np.zeros((64,32),bool)
    for e in range(64):
        active=np.flatnonzero(~O[e]);X[e,np.roll(active,-(e%25))[:5]]=True
    hint=rows_from(X,O,rows[64]);np.savez_compressed(OUT/'hint.frozen.npz',X=X,O=O,rows=hint,routes=z['routes'],uncoupled_rows=rows)
    files=[Path(__file__),SOURCE,STAGE/'protocol.frozen.json',STAGE/'cost_arrays.npz',STAGE/'report.json',OUT/'hint.frozen.npz',CPP,PYTHON,
           ROOT/'scripts/research_20261103_topology_placement_replay.py',ROOT/'scripts/research_20261102_whole_row_compiler.py']
    inputs=[]
    for path in files:
        item=dict(path=str(path),sha256=sha(path))
        if path.suffix in ('.py','.cpp'):
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        inputs.append(item)
    old=json.loads((STAGE/'protocol.frozen.json').read_text());service=dict(old['service_base'])
    service.update(chips=32,channels=8,input_channel_copies=1,down_first=0)
    p=dict(status='frozen_before_standardCP_fit',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),inputs=inputs,
           service=service,model='G4actualW8 L0 TRAIN512',geometry='GU24x41+1x40 rows/25onewavejobs;Down12x128/12onewavejobs. Fixedshared fromstageBminimum_depth.',
           variables='For each64experts:32X/32O Booleans,sumX5,sumO7,X+O<=1. GU=!O,Down=X|O.',
           exact_ownedwork='29102+28762X-340O perexpert/chip,plusfrozenactualsharedchipwork. Input1/fringe1/cmd250; validatedviaunchangedbackend.',
           geometry_floor='295totalreadwaves perquery => at least10ownedreadwaves/chip maximum. 280000ns isnecessary ownedworkfloor,notnativeattainable.',
           hint='OriginalGUmask,Down=7GUomits+5GUactive rotatedbyexpertID;canonical40row onlowestactivechip. Sourceuncoupled retained separately.',
           solver=dict(version=ortools.__version__,workers=8,seed=20261114,peak_feasibility_seconds=30,mean_seconds=90,
                       peak='NANDreadload6+sumX_selected+sharedwaves<=10; purefeasibility,no incumbentmeanconstraint.',
                       mean='Removepeakconstraint,minimize sumqueryT. Retainhintglobalobjectivecap,notperquerycap. UNKNOWNnotUNSAT.'),
           retention='Atmost7distinctmean-improvingorpeakcontrols plushint; callbacks savebestactualworkmaps notslackT. Peakmap mayberetainedforfeasibilityevenifnotbestmean.',
           native='Everyretainedmap plushint input1 GF/DF512; uncoupledsourcecosts reusedandbridge. LastTRAINnativebestmap only getsinput4 GF/DF,nonewfit. <=9216nativecosts,unchangedcore.',
           scope='Standardjoint hypergraph/multicolor CP strongcontrol,notnewmethod. No extra geometry/copies/sharedfit/nativekernel/weights/heldout/alllayers. SameMhardware comparisons only.')
    dump(OUT/'protocol.frozen.json',p);print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__))))


def inputs():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for x in p['inputs']:assert sha(x['path'])==x['sha256']
    assert sha(__file__)==p['source_sha256']
    return p,load(SOURCE),load(OUT/'hint.frozen.npz')


def fit():
    p,z,hint=inputs();assert not (OUT/'fit_report.json').exists()
    backend=load_python();s=backend.Service(**p['service']);routes=z['routes'];shared=z['rows'][64]
    hintplan=backend.lower_rows(hint['rows'][:,0],hint['rows'][:,1],1536,[512]*64+[1024],s)
    table=backend.occupancy_table(hintplan,s)['chip_ns'];sharedwork=table[64].sum(0)
    sharedwaves=np.zeros(32,np.int64)
    for phase in range(2):
        for c,nr,nc,pg,wa in hintplan['jobs'][slice(*hintplan['offsets'][64,phase])]:sharedwaves[c]+=wa
    def actual(X,O):return 6*29102+sharedwork+28762*X[routes].sum(1)-340*O[routes].sum(1)
    hintwork=actual(hint['X'],hint['O']);assert np.array_equal(hintwork,table[routes].sum((1,2))+sharedwork)
    assert int(sharedwaves.sum())==73
    pool={};events=[];hint_score=int(hintwork.max(1).sum());best_seen=hint_score;t0=time.monotonic()
    def retain(X,O,stage,force=False,solver_objective=None):
        nonlocal best_seen
        X=np.asarray(X,bool);O=np.asarray(O,bool);rows=rows_from(X,O,shared);loads=actual(X,O);score=int(loads.max(1).sum())
        key=X.tobytes()+O.tobytes()
        improved=score<best_seen
        if improved:best_seen=score
        events.append(dict(stage=stage,actual_score_ns=score,improved=improved,solver_objective=solver_objective,elapsed_seconds=time.monotonic()-t0))
        if score<hint_score or force:
            pool[key]=dict(X=X.copy(),O=O.copy(),rows=rows,work=loads.max(1),score=score,stage=stage,forced=force)
            chosen=sorted(pool,key=lambda k:pool[k]['score'])[:7]
            if force and key not in chosen:chosen=chosen[:6]+[key]
            poolkeys=list(pool)
            for k in poolkeys:
                if k not in chosen:del pool[k]
        return score
    def build(peak,initial):
        m=cp_model.CpModel();X=[[m.NewBoolVar(f'X{e}_{c}') for c in range(32)] for e in range(64)]
        O=[[m.NewBoolVar(f'O{e}_{c}') for c in range(32)] for e in range(64)]
        high=int(6*29102+sharedwork.max()+6*28762)
        T=[m.NewIntVar(280000,high,f'T{q}') for q in range(512)]
        for e in range(64):
            m.Add(sum(X[e])==5);m.Add(sum(O[e])==7)
            for c in range(32):m.Add(X[e][c]+O[e][c]<=1);m.AddHint(X[e][c],int(initial['X'][e,c]));m.AddHint(O[e][c],int(initial['O'][e,c]))
        initialwork=actual(initial['X'],initial['O']).max(1)
        for q,route in enumerate(routes):
            for c in range(32):
                xs=sum(X[int(e)][c] for e in route);os=sum(O[int(e)][c] for e in route)
                m.Add(6*29102+int(sharedwork[c])+28762*xs-340*os<=T[q])
                if peak:m.Add(6+xs+int(sharedwaves[c])<=10)
            m.AddHint(T[q],int(initialwork[q]))
        if not peak:m.Add(sum(T)<=hint_score);m.Minimize(sum(T))
        return m,X,O,T
    stages=[];initial=dict(X=hint['X'],O=hint['O'])
    for peak,limit in [(True,30),(False,90)]:
        model,X,O,T=build(peak,initial)
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=limit
        solver.parameters.num_search_workers=8;solver.parameters.random_seed=p['solver']['seed']
        class Callback(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                xx=np.asarray([[self.Value(v) for v in row] for row in X],bool)
                oo=np.asarray([[self.Value(v) for v in row] for row in O],bool)
                retain(xx,oo,'peak' if peak else 'mean',force=peak,solver_objective=float(self.ObjectiveValue()))
        callback=Callback();status=solver.Solve(model,callback)
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            xx=np.asarray([[solver.Value(v) for v in row] for row in X],bool);oo=np.asarray([[solver.Value(v) for v in row] for row in O],bool)
            retain(xx,oo,'peak_final' if peak else 'mean_final',force=peak,solver_objective=float(solver.ObjectiveValue()))
            if peak:initial=dict(X=xx,O=oo)
        stages.append(dict(stage='peak10_feasibility' if peak else 'mean_no_peak',status=solver.StatusName(status),wall_seconds=solver.WallTime(),
                           objective=float(solver.ObjectiveValue()) if not peak else None,
                           best_objective_bound=float(solver.BestObjectiveBound()) if not peak else None,
                           limits='UNKNOWN isnotinfeasible; onlyOPTIMAL/INFEASIBLE status maycertifytheirrespectivefixedmodel.'))
        print(json.dumps(stages[-1]),flush=True)
    candidates=[dict(id='coupled_hint',rows=hint['rows'],X=hint['X'],O=hint['O'],work=hintwork.max(1),score=hint_score,stage='hint')]
    for j,item in enumerate(sorted(pool.values(),key=lambda r:r['score'])):candidates.append(dict(id=f'incumbent{j}',**item))
    assert len(candidates)<=8
    saved={}
    for c in candidates:
        for k in ('rows','X','O','work'):saved[c['id']+'.'+k]=c[k]
    saved['routes']=routes;saved['sharedwork']=sharedwork;saved['sharedwaves']=sharedwaves
    np.savez_compressed(OUT/'candidates.frozen.npz',**saved)
    report=dict(status='bounded_standardCP_control_fit',protocol_sha256=sha(OUT/'protocol.frozen.json'),stages=stages,events=events,
                candidates=[{k:v for k,v in c.items() if k not in ('rows','X','O','work')}|dict(work=stats(c['work'])) for c in candidates],
                hint_mean_work_ns=float(hintwork.max(1).mean()),best_mean_work_ns=float(min(c['score'] for c in candidates)/512),
                seconds=time.monotonic()-t0,limitations='StaticTRAINcoloringonly,no newgeometry/replicas,worknotnative and heuristicnotoptimalunlesscertified. Noheldout.')
    dump(OUT/'fit_report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('events','candidates')},indent=2))


def native():
    p,z,hint=inputs();assert (OUT/'fit_report.json').exists() and not (OUT/'native_report.json').exists()
    candidates=load(OUT/'candidates.frozen.npz');fit=json.loads((OUT/'fit_report.json').read_text());routes=candidates['routes']
    ids=[x['id'] for x in fit['candidates']];assert len(ids)<=8
    protocol=dict(status='frozen_before_native',source_sha256=sha(__file__),fit_report_sha256=sha(OUT/'fit_report.json'),
                  candidates_sha256=sha(OUT/'candidates.frozen.npz'),ids=ids,priorities=['GF','DF'],queries=512,
                  main_queries=len(ids)*1024,extra_input4_queries=1024,total_queries=(len(ids)+1)*1024,scope='NoCP/newfitforinput4; finalinput1nativebestmap only.')
    assert protocol['total_queries']<=10000;dump(OUT/'native_protocol.frozen.json',protocol)
    backend=load_python();base=backend.Service(**p['service']);native_dir=OUT/'native';native_dir.mkdir()
    binary=native_dir/'native.so';subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(CPP),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));fn=lib.eval_ordinary_read;i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    fn.argtypes=[i32,i32,i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64,i64,i64,ctypes.c_void_p,i64]
    def evaluate(plan,rr,s,trace=False):
        backend.validate_plan(plan,65,s);rr=np.ascontiguousarray(rr,np.int32);cost=np.zeros(len(rr),np.int64);bound=cost.copy();nlog=np.zeros(1,np.int64);tr=np.zeros((6000 if trace else 1,8),np.int64)
        fn(plan['offsets'],plan['jobs'],plan['input_columns'],rr,64,1,len(plan['jobs']),len(rr),6,s.array(),cost,bound,
           ctypes.c_void_p(tr.ctypes.data) if trace else None,nlog)
        assert np.all(cost>=bound)
        return cost,bound,tr[:nlog[0]].copy() if trace else None
    outputs={};records=[];best=None;physical={};trace_checks=[]
    for ident in ids:
        rows=candidates[ident+'.rows'];plan=backend.lower_rows(rows[:,0],rows[:,1],1536,[512]*64+[1024],base)
        assert np.array_equal(rows[64],z['rows'][64])
        phys=backend.physical_addresses(plan,base);addresses=phys.pop('addresses');physical[ident]=phys
        assert phys['chip_blocks']==[1]*32
        np.savez_compressed(native_dir/(ident+'.plan.npz'),rows=rows,routes=routes,offsets=plan['offsets'],jobs=plan['jobs'],input_columns=plan['input_columns'],
                            total_rows=plan['total_rows'],addresses=np.asarray([[a[k] for k in ['chip','block','wordline','reserved_wordlines','pages','first_plane']] for a in addresses],np.int32))
        for priority in (0,1):
            cfg=dict(p['service']);cfg['down_first']=priority;s=backend.Service(**cfg);cost,bound,_=evaluate(plan,routes,s)
            key=ident+'.'+('DF' if priority else 'GF');outputs[key+'.cost']=cost;outputs[key+'.bound']=bound
            qcost,qlb,tr=evaluate(plan,routes[:1],s,True);py=backend.ReferenceGraph.build(plan,np.r_[routes[0],64],s).run()
            assert py['cost']==qcost[0]==cost[0] and np.array_equal(tr,py['trace']);outputs[key+'.q0trace']=tr
            trace_checks.append(dict(id=key,nodes=len(tr),all_nodes_exact=True))
            record=dict(id=ident,priority='DF' if priority else 'GF',input_copies=1,cost=stats(cost),bound=stats(bound));records.append(record)
            if best is None or cost.mean()<best['cost']['mean_ns']:best=record.copy()
    # Source uncoupled retained exactly; its native costsvectors alreadysealed.
    old=load(STAGE/'cost_arrays.npz');oldkeys=[f'granite4tiny.M32.minimum_depth.{priority}.cost' for priority in ('GF','DF')]
    source_best=min(oldkeys,key=lambda k:old[k].mean());best_source=stats(old[source_best])
    winning_rows=candidates[best['id']+'.rows'];winning_id=best['id']
    if best_source['mean_ns']<best['cost']['mean_ns']:winning_rows=z['rows'];winning_id='uncoupled_source'
    dump(OUT/'input4_selection.frozen.json',dict(winner=winning_id,rows_sha256=hashlib.sha256(winning_rows.tobytes()).hexdigest(),
         rule='Lowestinput1nativeTRAINmeanamongretainedcoupledcontrolsandoriginaluncoupled; no input4fitting.'))
    plan=backend.lower_rows(winning_rows[:,0],winning_rows[:,1],1536,[512]*64+[1024],base)
    for priority in (0,1):
        cfg=dict(p['service']);cfg.update(input_channel_copies=4,down_first=priority);s=backend.Service(**cfg)
        cost,bound,_=evaluate(plan,routes,s);key='selected_input4.'+('DF' if priority else 'GF')
        outputs[key+'.cost']=cost;outputs[key+'.bound']=bound
        qcost,qlb,tr=evaluate(plan,routes[:1],s,True);py=backend.ReferenceGraph.build(plan,np.r_[routes[0],64],s).run()
        assert py['cost']==qcost[0]==cost[0] and np.array_equal(tr,py['trace']);outputs[key+'.q0trace']=tr
        trace_checks.append(dict(id=key,nodes=len(tr),all_nodes_exact=True));records.append(dict(id=winning_id,priority='DF' if priority else 'GF',input_copies=4,cost=stats(cost),bound=stats(bound)))
    np.savez_compressed(OUT/'native_arrays.npz',**outputs)
    dump(OUT/'native_report.json',dict(status='standardcoupledcontrol_native_sensitivity',protocol_sha256=sha(OUT/'protocol.frozen.json'),native_protocol_sha256=sha(OUT/'native_protocol.frozen.json'),
         records=records,best_coupled_input1=best,source_uncoupled=best_source,source_priority=source_best,final_selected_id=winning_id,
         mean_improvement_vs_uncoupled_pct=100*(1-best['cost']['mean_ns']/best_source['mean_ns']),
         physical=physical,traces=trace_checks,native_costs=(len(ids)+1)*1024,
         limits='Samegeometry labels-only TRAINCP strongcontrol,not newmethod/replica/hardwaregain. GF/DFequalfor all. Peak/meanUNKNOWNnotproof. Noheldout/alllayers/input4refit.'))
    print(json.dumps(dict(best=best,uncoupled=best_source,selected=winning_id,native_costs=(len(ids)+1)*1024),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','fit','native']);a=ap.parse_args()
    {'freeze':freeze,'fit':fit,'native':native}[a.mode]()
