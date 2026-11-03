#!/usr/bin/env python3
"""Separate frozen 80-second continuation; never changes the original screen."""
import argparse,gzip,hashlib,importlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/research-2026-11-02/round2/compact-coupled-solver'
T=D/'continuation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,z):Path(p).write_text(json.dumps(z,indent=2)+'\n')
def main(action):
    prior=read(D/'summary.json');warm=min(prior['candidates'],key=lambda z:z['metric']['weighted_sum_ns'])
    if action=='freeze':
        assert not T.exists();T.mkdir();(T/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        files=[D/'summary.json',D/'protocol.frozen.json',D/'input.frozen.npz',Path(warm['path']),T/Path(__file__).name]
        files+=list((D/'snapshots').glob('*.py'))+list((D/'snapshots').glob('*.cpp'))
        write(T/'protocol.frozen.json',dict(source_sha256={str(p):sha(p) for p in files},warm=warm,
          solver_limit_seconds=80,workers=8,seed=20261103,reason='Original predetermined64callback cap stopped mean optimization at8.20sec;this continuation has no solution-count stop. Same mathematical epigraph model;do not pool as independent statistical runs.',
          constraint='Same X5/O4/disjoint/fixedshared;min weighted maxchip-owned occupancy,not native DAG latency. All distinct callbacks saved and replayed. No heldout.',
          earlier_solver_seconds=prior['solver_total_seconds']))
        print(json.dumps(dict(status='frozen',protocol_sha256=sha(T/'protocol.frozen.json'))));return
    p=read(T/'protocol.frozen.json');assert not (T/'started.json').exists()
    for f,h in p['source_sha256'].items():assert sha(f)==h
    write(T/'started.json',dict(protocol_sha256=sha(T/'protocol.frozen.json')))
    spec=importlib.util.spec_from_file_location('frozen_compact',D/'snapshots/research_20261103_compact_coupled_solver.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    sys.path.insert(0,str(D/'snapshots'));backend=importlib.import_module('research_20261103_ordinary_read_backend');backend.ROOT=ROOT
    from ortools.sat.python import cp_model
    s=backend.Service(**read(D/'protocol.frozen.json')['service']);z=b.load(D/'input.frozen.npz');rows=z['rows'];routes=z['routes']
    unique,weights=np.unique(np.sort(routes,axis=1),axis=0,return_counts=True)
    shared=np.array([sum(b.occupancy(int(rows[64,ph,c]),C,s) for ph,C in enumerate([1536,1024])) for c in range(16)])
    sw=np.array([sum((int(rows[64,ph,c])*C+s.wave_bytes-1)//s.wave_bytes for ph,C in enumerate([1536,1024])) for c in range(16)])
    a,x,o=57864,28086,-676
    def metric(xx,oo):
        v=(6*a+shared+x*xx[unique].sum(1)+o*oo[unique].sum(1)).max(1)
        w=(12+sw+xx[unique].sum(1)).max(1)
        return dict(weighted_sum_ns=int(v@weights),mean_ns=float(v@weights/512),max_ns=int(v.max()),NAND_peak_histogram={str(int(k)):int(n) for k,n in zip(*np.unique(np.repeat(w,weights),return_counts=True))})
    model=cp_model.CpModel();X=[[model.new_bool_var(f'X{e}_{c}') for c in range(16)] for e in range(64)];O=[[model.new_bool_var(f'O{e}_{c}') for c in range(16)] for e in range(64)]
    for e in range(64):
        model.add(sum(X[e])==5);model.add(sum(O[e])==4)
        for c in range(16):model.add(X[e][c]+O[e][c]<=1)
    ts=[]
    for q,route in enumerate(unique):
        t=model.new_int_var(0,1000000,f'T{q}');ts.append(t)
        for c in range(16):model.add(t>=(6*a+int(shared[c]))//2+(x//2)*sum(X[int(e)][c] for e in route)+(o//2)*sum(O[int(e)][c] for e in route))
    obj=sum(int(w)*t for w,t in zip(weights,ts));model.minimize(obj);model.add(obj<=warm['metric']['weighted_sum_ns']//2)
    hz=b.load(warm['path']);hx,ho=hz['X'],hz['O']
    for e in range(64):
        for c in range(16):model.add_hint(X[e][c],int(hx[e,c]));model.add_hint(O[e][c],int(ho[e,c]))
    hv=(6*a+shared+x*hx[unique].sum(1)+o*ho[unique].sum(1)).max(1)
    for t,v in zip(ts,hv):model.add_hint(t,int(v)//2)
    (T/'incumbents').mkdir();(T/'native').mkdir();records=[];seen=set()
    class Callback(cp_model.CpSolverSolutionCallback):
        def __init__(self):super().__init__();self.count=0
        def on_solution_callback(self):
            self.count+=1;xx=np.array([[self.value(v) for v in r] for r in X],dtype=bool);oo=np.array([[self.value(v) for v in r] for r in O],dtype=bool)
            key=b.digest(np.stack([xx,oo]))
            if key in seen:return
            seen.add(key);i=len(records);path=T/'incumbents'/f'{i:04d}.npz';np.savez_compressed(path,X=xx,O=oo)
            rec=dict(index=i,path=str(path),sha256=sha(path),mask_sha256=key,metric=metric(xx,oo),reported_objective_half_ns=self.objective_value)
            records.append(rec);print('INCUMBENT',i,rec['metric']['mean_ns'],flush=True)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=80;solver.parameters.num_search_workers=8;solver.parameters.random_seed=20261103
    cb=Callback();t0=time.monotonic();status=solver.solve(model,cb);elapsed=time.monotonic()-t0
    report=dict(status=solver.status_name(status),elapsed_seconds=elapsed,solver_wall_time=solver.wall_time,objective_half_ns=solver.objective_value,best_bound_half_ns=solver.best_objective_bound,
      reported_solutions=cb.count,distinct_masks=len(records),response_stats=solver.response_stats())
    write(T/'solver.json',report);print('SOLVER',json.dumps(report),flush=True)
    assert elapsed<=90 and elapsed+p['earlier_solver_seconds']<300
    native=backend.NativeReplay(D/'build');replayed=[];baseline=prior['candidates'][0]
    for rec in records:
        m=b.load(rec['path']);rr=b.rowmap(m['X'],m['O'],rows);plan=backend.lower_rows(rr[:,0],rr[:,1],1536,[512]*64+[1024],s)
        phy=backend.physical_addresses(plan,s);audit=b.physicalcheck(plan,phy,s);actual=native.run(plan,routes,64,1,s,trace=True)
        own=backend.occupancy_table(plan,s)['chip_ns'].sum(1);assert np.array_equal(own[:64],a+x*m['X']+o*m['O']) and np.array_equal(own[64],shared)
        assert int((own[routes].sum(1)+shared).max(1).sum())==rec['metric']['weighted_sum_ns']
        traces={};checks=[]
        for q in [0,255,511]:
            out=actual if q==0 else native.run(plan,routes[q:q+1],64,1,s,trace=True)
            graph=backend.ReferenceGraph.build(plan,routes[q].tolist()+[64],s);ref=graph.run()
            assert int(out['cost'][0])==ref['cost'] and np.array_equal(out['trace'],ref['trace'])
            checks.append(dict(query=q,**b.tracecheck(graph,out['trace'])));traces[f'trace_q{q}']=out['trace']
        path=T/'native'/f'{rec["index"]:04d}.npz'
        np.savez_compressed(path,rows=rr,**{k:plan[k] for k in ['offsets','jobs','input_columns','total_rows']},routes=routes,cost=actual['cost'],lower_bound=actual['lower_bound'],**traces)
        pp=T/'native'/f'{rec["index"]:04d}.physical.json.gz'
        with gzip.GzipFile(filename=str(pp),mode='wb',mtime=0) as f:f.write(json.dumps(dict(logical=plan['logical'],**phy),separators=(',',':')).encode())
        replayed.append(dict(**rec,arrays=str(path),arrays_sha256=sha(path),physical=str(pp),physical_sha256=sha(pp),physical_audit=audit,reference_trace_checks=checks,
          native_mean_ns=float(actual['cost'].mean()),native_p99_ns=float(np.percentile(actual['cost'],99)),native_mean_gain_vs_v2=1-float(actual['cost'].mean())/baseline['native_mean_ns'],
          allocated_bytes_delta_vs_v2=audit['allocated_lsb_block_bytes']-baseline['physical_audit']['allocated_lsb_block_bytes']))
    best=min(replayed,key=lambda r:r['native_mean_ns']) if replayed else None
    write(T/'summary.json',dict(status='complete',protocol_sha256=sha(T/'protocol.frozen.json'),solver=report,candidates=replayed,best_native=best,
      total_solver_seconds=elapsed+p['earlier_solver_seconds'],prior_best_native=prior['candidates'][prior['best_native_index']],
      limits=['Epigraph incumbent objective can contain slack;exact mask work is independently recomputed,solver bound pertains the surrogate optimum.','TRAIN-only single-layer timing model;no native output numeric ABI or global all-layer capacity proof.']))
    for f,h in p['source_sha256'].items():assert sha(f)==h
    write(T/'completion.json',dict(status='complete',outputs_sha256={str(f):sha(f) for f in sorted(T.rglob('*')) if f.is_file() and f.name!='completion.json' and '__pycache__' not in str(f)}))
    print(json.dumps(dict(status='complete',candidates=len(replayed),best_native_mean_gain=None if best is None else best['native_mean_gain_vs_v2'],summary_sha256=sha(T/'summary.json'))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);main(p.parse_args().action)
