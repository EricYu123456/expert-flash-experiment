#!/usr/bin/env python3
"""Bounded CP-SAT control for a frozen one-copy ordinary-read row geometry."""
import argparse,dataclasses,gzip,hashlib,importlib,json,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
DEST=BASE/'compact-coupled-solver'
VENV='/tmp/codex-compact-coupled-20261103/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def freeze():
    assert not DEST.exists(),'Never overwrite a frozen solver experiment.'
    DEST.mkdir();snap=DEST/'snapshots';snap.mkdir()
    summary=BASE/'shared-counterbalanced/independent-v2/summary.json';report=read(summary)
    case=next(r for r in report['cases'] if 'v2-fix1/granite4tiny.L0.input1/' in r['path'])
    original=Path(case['families']['independent_paired']['arrays']);assert sha(case['path'])==case['sha256']
    result=read(case['path']);service=result['service'];service.setdefault('input_channel_copies',1)
    assert service['input_channel_copies']==1 and service['output_fringe']==1
    z=load(original);rows=z['rows'];routes=z['routes']
    train=result['training'];assert sha(train['path'])==train['sha256']
    archive=load(train['path']);assert archive['split'].item()=='train' and archive['completed'].all()
    assert np.array_equal(routes,archive['routes'][:,:,0].reshape(512,6))
    assert rows.shape==(65,2,16) and routes.shape==(512,6)
    high=rows[:64,0]>=82;down=rows[:64,1]>0;X=high&down;O=high&~down
    assert np.all(X.sum(1)==5) and np.all(O.sum(1)==4) and not np.any(X&O)
    assert not np.any(~high&~down)
    for e in range(64):
        assert sorted(rows[e,0].tolist())==[40]*7+[82]*3+[83]*6
        assert sorted(rows[e,1].tolist())==[0]*4+[128]*12
    np.savez_compressed(DEST/'input.frozen.npz',**z)
    sources=[Path(__file__),ROOT/'scripts/research_20261103_ordinary_read_backend.py',ROOT/'scripts/research_20261103_ordinary_read_native.cpp',ROOT/'scripts/research_20261102_whole_row_compiler.py']
    for p in sources:(snap/p.name).write_bytes(p.read_bytes())
    import ortools
    from ortools.sat.python import cp_model
    protocol=dict(frozen_utc=utc(),model='granite4tiny',moe_layer=0,raw_layer=0,input_channel_copies=1,
      source_case=case['path'],source_arrays=str(original),source_mean_native_ns=case['families']['independent_paired']['mean_ns'],
      inputs_sha256={str(p):sha(p) for p in [summary,Path(case['path']),original,Path(train['path'])]},
      snapshots_sha256={str(p):sha(p) for p in snap.iterdir()},frozen_array_sha256=sha(DEST/'input.frozen.npz'),
      service=service,train=train,ordered_routes_sha256=digest(routes),shared_rows_sha256=digest(rows[64]),
      solver=dict(name='OR-Tools CP-SAT',version=ortools.__version__,python=sys.executable,venv=VENV,
                  official_docs=['https://developers.google.com/optimization/cp/cp_solver','https://developers.google.com/optimization/cp/cp_tasks'],
                  workers=8,random_seed=20261103,time_limits_seconds={'peak19':60,'peak20':60,'mean_work':90},total_solver_limit_seconds=300,
                  mean_solution_callback_limit=64),
      formulation='X[e,c] both-high,exact5 perexpert;O[e,c] GUhigh/Downzero,exact4;X+O<=1. Remaining7 Glow/Downhigh. Mean-work uses exact chip-owned command+fullinputONFI+ordinaryread+all-pageoutputONFI occupancy;not native DAG optimum.',
      scope='Static one-copy complete rows;frozen shared geometry;no heldout/inference. No extra capacity constraint beyond one-block native jobs is fabricated;report all physical blocks versus incumbent and flag capacity differences.',
      run_policy='First all-TRAIN peak19 thenpeak20 feasibility;then unconstrained weightedmean maxownedwork with baseline/best feasible mask hints. Retain baseline regardless native outcome. Stop each<=90sec,total<=300sec;UNKNOWN isnot UNSAT. Save/replay every unique reported improving incumbent;stop callback at64solutions.',
      lowering='Each expert nine GUhigh chips get six83rows and three82rows,otherseven40;Down128on12chips,0onfour. Retain original83positions if entire mask equals incumbent;otherwise deterministic firstsix highchips. Shared rows unchanged. Physical address certificates are virtual,not programmed weights or native numerical execution.')
    write(DEST/'protocol.frozen.json',protocol)
    print(json.dumps(dict(status='frozen',protocol_sha256=sha(DEST/'protocol.frozen.json'),source_arrays_sha256=sha(original),solver=ortools.__version__),indent=2))

def occupancy(r,C,s):
    if not r:return 0
    pages=(r*C+s.page_bytes-1)//s.page_bytes;waves=(pages+s.planes-1)//s.planes
    emitted=pages*s.page_bytes//C
    return s.command_ns+s.transfer(C*s.input_bytes*s.input_channel_copies,s.channel_bytes_per_second)+waves*s.ordinary_read_ns+s.transfer(emitted*s.output_bytes,s.channel_bytes_per_second)
def masks(rows):
    gu=rows[:64,0]>=82;down=rows[:64,1]>0
    return gu&down,gu&~down
def rowmap(X,O,base):
    assert X.shape==O.shape==(64,16) and not np.any(X&O)
    assert np.all(X.sum(1)==5) and np.all(O.sum(1)==4)
    out=base.copy()
    for e in range(64):
        high=X[e]|O[e]
        if np.array_equal(high,base[e,0]>=82):out[e,0]=base[e,0]
        else:
            out[e,0]=40;hh=np.flatnonzero(high);out[e,0,hh]=82;out[e,0,hh[:6]]=83
        out[e,1]=np.where(O[e],0,128)
    assert np.all(out[:64,0].sum(1)==1024) and np.all(out[:64,1].sum(1)==1536)
    return out
def tracecheck(graph,tr):
    assert len(graph.nodes)==len(tr);res=defaultdict(list);owners=defaultdict(list);jobs=defaultdict(list)
    for i,(x,t) in enumerate(zip(graph.nodes,tr)):
        r,d,a,b,p,e,c,j=map(int,t)
        assert (r,d,p,e,c,j)==tuple(x[k] for k in ['resource','duration','phase','expert','chip','job']) and b-a==d
        assert all(tr[z,3]<=a for z in x['deps']);res[r].append((a,b))
        if j>=0:jobs[j].append(i)
    for rs in res.values():
        v=sorted((a,b) for a,b in rs if a<b);assert all(a[1]<=b[0] for a,b in zip(v,v[1:]))
    for j,ii in jobs.items():
        starts=[i for i in ii if graph.nodes[i]['acquire']];ends=[i for i in ii if graph.nodes[i]['release']]
        assert len(starts)==len(ends)==1;a=tr[starts[0],2];b=tr[ends[0],3];c=graph.nodes[starts[0]]['chip'];owners[c].append((a,b))
        assert all(a<=tr[i,2]<=tr[i,3]<=b for i in ii)
    for rs in owners.values():
        v=sorted(rs);assert all(a[1]<=b[0] for a,b in zip(v,v[1:]))
    return dict(nodes=len(tr),jobs=len(jobs),resources=len(res),chip_owner_intervals=sum(map(len,owners.values())))
def physicalcheck(plan,physical,s):
    assert len(plan['logical'])==len(plan['jobs'])==len(physical['addresses'])
    occupied=defaultdict(list);coverage=defaultdict(list);useful=pages=waves=0
    for i,(a,l,j) in enumerate(zip(physical['addresses'],plan['logical'],plan['jobs'])):
        c,r,C,p,w=map(int,j);assert (a['job'],a['chip'],a['reserved_wordlines'],a['pages'],a['first_plane'])==(i,c,w,p,0)
        assert 0<=a['wordline']<a['wordline']+w<=s.block_wordlines
        assert p==(r*C+s.page_bytes-1)//s.page_bytes and w==(p+s.planes-1)//s.planes
        assert all(l[k]==v for k,v in dict(chip=c,rows=r,columns=C,pages=p,waves=w,useful_bytes=r*C).items())
        start=(a['block']*s.block_wordlines+a['wordline'])*s.planes
        occupied[c].append((start,start+w*s.planes));coverage[l['expert'],l['phase']].append((l['row_start'],l['row_start']+r))
        useful+=r*C;pages+=p;waves+=w
    for rs in occupied.values():
        rs.sort();assert all(a[1]<=b[0] for a,b in zip(rs,rs[1:]))
    for key,rs in coverage.items():
        rs.sort();assert rs[0][0]==0 and rs[-1][1]==plan['total_rows'][key]
        assert all(a[1]==b[0] for a,b in zip(rs,rs[1:]))
    assert useful==64*3*1536*512+3*1536*1024==physical['useful_bytes']
    assert pages*s.page_bytes==physical['programmed_page_bytes'] and waves*s.wave_bytes==physical['reserved_wave_bytes']
    for c,rs in occupied.items():assert max(b for a,b in rs)<=physical['chip_blocks'][c]*s.block_wordlines*s.planes
    return dict(jobs=len(plan['jobs']),useful_bytes=useful,pages=pages,waves=waves,chip_blocks=physical['chip_blocks'],allocated_lsb_block_bytes=physical['allocated_lsb_block_bytes'])

def run():
    proto=read(DEST/'protocol.frozen.json');assert not (DEST/'started.json').exists()
    for p,h in proto['inputs_sha256'].items():assert sha(p)==h
    for p,h in proto['snapshots_sha256'].items():assert sha(p)==h
    assert sha(DEST/'input.frozen.npz')==proto['frozen_array_sha256']
    sys.path.insert(0,str(DEST/'snapshots'))
    backend=importlib.import_module('research_20261103_ordinary_read_backend');backend.ROOT=ROOT
    from ortools.sat.python import cp_model
    import ortools
    assert ortools.__version__==proto['solver']['version']
    s=backend.Service(**proto['service']);s.validate();data=load(DEST/'input.frozen.npz');rows=data['rows'];routes=data['routes']
    unique,weights=np.unique(np.sort(routes,axis=1),axis=0,return_counts=True);Q=len(unique)
    X0,O0=masks(rows);shared=np.array([sum(occupancy(int(rows[64,p,c]),C,s) for p,C in enumerate([1536,1024])) for c in range(16)])
    sharedwaves=np.array([sum((int(rows[64,p,c])*C+s.wave_bytes-1)//s.wave_bytes for p,C in enumerate([1536,1024])) for c in range(16)])
    a=occupancy(40,1536,s)+occupancy(128,512,s);x=occupancy(83,1536,s)+occupancy(128,512,s)-a;o=occupancy(83,1536,s)-a
    assert occupancy(82,1536,s)==occupancy(83,1536,s)
    assert (a,x,o)==(57864,28086,-676)
    assert np.array_equal(shared,np.where(sharedwaves==4,114164,142292))
    def metric(X,O):
        load=a*6+shared+x*X[unique].sum(1)+o*O[unique].sum(1)
        values=load.max(1);wv=(12+sharedwaves+X[unique].sum(1)).max(1)
        return dict(weighted_sum_ns=int(values@weights),mean_ns=float(values@weights/512),max_ns=int(values.max()),
                    p99_ns=float(np.percentile(np.repeat(values,weights),99)),NAND_peak_histogram={str(int(v)):int(n) for v,n in zip(*np.unique(np.repeat(wv,weights),return_counts=True))}),values
    baseline_metric,baseline_values=metric(X0,O0)
    derivation=dict(GUlow=occupancy(40,1536,s),GUhigh=occupancy(83,1536,s),Downhigh=occupancy(128,512,s),Downzero=0,
      base=a,X_coefficient=x,O_coefficient=o,shared_owned_ns=shared.tolist(),shared_wave_counts=sharedwaves.tolist(),
      base_owned_metric=baseline_metric,unique_query_sets=Q,ordered_queries=512,
      ownership='Command+fullinputONFI+ordinarywaves+all-pageoutputONFI,excludingPCIe andhost exceptreportednativeengine. Workobjective is necessary occupancy,not feasibleDAGlatency.')
    write(DEST/'derivation.json',derivation);write(DEST/'started.json',dict(utc=utc(),protocol_sha256=sha(DEST/'protocol.frozen.json')))
    (DEST/'incumbents').mkdir();(DEST/'native').mkdir()
    candidates=[];seen=set();solver_seconds=0;reports=[]
    def store(X,O,stage,objective=None):
        key=digest(np.stack([X,O]));m,v=metric(X,O)
        if key in seen:return
        seen.add(key);i=len(candidates);path=DEST/'incumbents'/f'{i:03d}.{stage}.npz'
        np.savez_compressed(path,X=X,O=O,mean_work_values_ns=v)
        r=dict(index=i,stage=stage,path=str(path),sha256=sha(path),mask_sha256=key,metric=m,reported_objective_half_ns=objective)
        candidates.append(r);print('INCUMBENT',i,stage,m['mean_ns'],flush=True)
    store(X0,O0,'frozen_v2')
    for stage,peak,limit in [('peak19',19,60),('peak20',20,60),('mean_work',None,90)]:
        assert solver_seconds<300
        model=cp_model.CpModel();X=[[model.new_bool_var(f'X{e}_{c}') for c in range(16)] for e in range(64)];O=[[model.new_bool_var(f'O{e}_{c}') for c in range(16)] for e in range(64)]
        for e in range(64):
            model.add(sum(X[e])==5);model.add(sum(O[e])==4)
            for c in range(16):model.add(X[e][c]+O[e][c]<=1)
        tv=[]
        for q,route in enumerate(unique):
            if peak is None:tv.append(model.new_int_var(0,1000000,f'T{q}'))
            for c in range(16):
                nx=sum(X[int(e)][c] for e in route)
                if peak is not None:model.add(nx<=peak-12-int(sharedwaves[c]))
                else:
                    no=sum(O[int(e)][c] for e in route)
                    model.add(tv[q]>=(a*6+int(shared[c]))//2+(x//2)*nx+(o//2)*no)
        if peak is None:
            model.minimize(sum(int(w)*t for w,t in zip(weights,tv)))
            model.add(sum(int(w)*t for w,t in zip(weights,tv))<=baseline_metric['weighted_sum_ns']//2)
        hint=min(candidates,key=lambda c:c['metric']['weighted_sum_ns']) if peak is None else candidates[0]
        hz=load(hint['path']);hx=hz['X'];ho=hz['O']
        for e in range(64):
            for c in range(16):model.add_hint(X[e][c],int(hx[e,c]));model.add_hint(O[e][c],int(ho[e,c]))
        if tv:
            _,hv=metric(hx,ho)
            for t,v in zip(tv,hv):model.add_hint(t,int(v)//2)
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=min(limit,300-solver_seconds)
        solver.parameters.num_search_workers=8;solver.parameters.random_seed=20261103
        solver.parameters.log_search_progress=False
        if peak is not None:solver.parameters.stop_after_first_solution=True
        class Capture(cp_model.CpSolverSolutionCallback):
            def __init__(self):super().__init__();self.n=0;self.stopped_by_limit=False
            def on_solution_callback(self):
                xx=np.array([[self.value(v) for v in row] for row in X],dtype=bool)
                oo=np.array([[self.value(v) for v in row] for row in O],dtype=bool)
                store(xx,oo,stage,self.objective_value if peak is None else None);self.n+=1
                if peak is None and self.n>=64:self.stopped_by_limit=True;self.stop_search()
        cb=Capture();began=time.monotonic();status=solver.solve(model,cb);elapsed=time.monotonic()-began;solver_seconds+=elapsed
        report=dict(stage=stage,status=solver.status_name(status),limit_seconds=limit,elapsed_seconds=elapsed,solver_wall_time=solver.wall_time,
          reported_solutions=cb.n,stopped_by_solution_limit=cb.stopped_by_limit,objective_half_ns=solver.objective_value if status in (cp_model.FEASIBLE,cp_model.OPTIMAL) else None,
          best_bound_half_ns=solver.best_objective_bound if peak is None else None,branches=solver.num_branches,conflicts=solver.num_conflicts,
          response_stats=solver.response_stats(),hint_mask_sha256=hint['mask_sha256'],hint_satisfies_peak=(None if peak is None else max(map(int,hint['metric']['NAND_peak_histogram']))<=peak))
        reports.append(report);write(DEST/(stage+'.solver.json'),report)
        print('SOLVER',json.dumps(report),flush=True)
    # Replay every distinct recorded incumbent with actual ordered routes.
    native=backend.NativeReplay(DEST/'build');replayed=[]
    for candidate in candidates:
        mask=load(candidate['path']);rr=rowmap(mask['X'],mask['O'],rows)
        plan=backend.lower_rows(rr[:,0],rr[:,1],1536,[512]*64+[1024],s)
        phy=backend.physical_addresses(plan,s);physical_audit=physicalcheck(plan,phy,s)
        real=native.run(plan,routes,64,1,s,trace=True)
        if candidate['stage']=='frozen_v2':assert np.array_equal(real['cost'],data['cost'])
        traces={};trace_checks=[]
        for q in [0,255,511]:
            actual=real if q==0 else native.run(plan,routes[q:q+1],64,1,s,trace=True)
            graph=backend.ReferenceGraph.build(plan,routes[q].tolist()+[64],s);ref=graph.run()
            assert int(actual['cost'][0])==ref['cost'] and np.array_equal(actual['trace'],ref['trace'])
            trace_checks.append(dict(query=q,**tracecheck(graph,actual['trace'])));traces[f'trace_q{q}']=actual['trace']
        # Direct nativejob occupancy must equal the compact affine formula.
        own=np.zeros((65,16),dtype=np.int64)
        for e,phases in enumerate(plan['offsets']):
            for p,(lo,hi) in enumerate(phases):
                for chip,r,C,_,_ in plan['jobs'][lo:hi]:own[e,chip]+=occupancy(int(r),int(C),s)
        want=a+x*mask['X']+o*mask['O'];assert np.array_equal(own[:64],want) and np.array_equal(own[64],shared)
        values=(own[routes].sum(1)+shared).max(1)
        expected=(6*a+shared+x*mask['X'][routes].sum(1)+o*mask['O'][routes].sum(1)).max(1)
        assert np.array_equal(values,expected)
        assert int(values.sum())==candidate['metric']['weighted_sum_ns']
        stem=f'{candidate["index"]:03d}.{candidate["stage"]}'
        path=DEST/'native'/(stem+'.npz');np.savez_compressed(path,rows=rr,**{k:plan[k] for k in ['offsets','jobs','input_columns','total_rows']},routes=routes,cost=real['cost'],lower_bound=real['lower_bound'],**traces)
        pp=DEST/'native'/(stem+'.physical.json.gz')
        with gzip.GzipFile(filename=str(pp),mode='wb',mtime=0) as f:f.write(json.dumps(dict(logical=plan['logical'],**phy),separators=(',',':')).encode())
        replayed.append(dict(**candidate,arrays=str(path),arrays_sha256=sha(path),physical=str(pp),physical_sha256=sha(pp),physical_audit=physical_audit,reference_trace_checks=trace_checks,
          native_mean_ns=float(real['cost'].mean()),native_p99_ns=float(np.percentile(real['cost'],99)),native_min_ns=int(real['cost'].min()),native_max_ns=int(real['cost'].max()),mean_native_LB_ns=float(real['lower_bound'].mean())))
    baseline=replayed[0];best=min(replayed,key=lambda x:x['native_mean_ns'])
    for item in replayed:
        item['native_mean_gain_vs_v2']=1-item['native_mean_ns']/baseline['native_mean_ns']
        item['native_p99_gain_vs_v2']=1-item['native_p99_ns']/baseline['native_p99_ns']
        item['allocated_bytes_delta_vs_v2']=item['physical_audit']['allocated_lsb_block_bytes']-baseline['physical_audit']['allocated_lsb_block_bytes']
    result=dict(status='complete',completed_utc=utc(),protocol_sha256=sha(DEST/'protocol.frozen.json'),solver_total_seconds=solver_seconds,
      solver_reports=reports,derivation=derivation,candidates=replayed,best_native_index=best['index'],best_native_mean_gain_vs_v2=best['native_mean_gain_vs_v2'],
      native=native.provenance,limits=['TRAIN512only,one layer,one cost profile;no heldout or inference.','Thecompactgeometryforcesminimumhighintersection5 and excludesboth-low;notallfreewhole-rowmapping.','CP-SATobjective is maxchip-ownedoccupancy,not complete nativeDAG latency or a nativeoptimum.','Every mask hasonecopy complete rows andfixedshared;globalmodelcapacity andnative numericalABI are not proven by these virtual one-layer addresses.'])
    write(DEST/'summary.json',result)
    for p,h in proto['inputs_sha256'].items():assert sha(p)==h
    write(DEST/'completion.json',dict(status='complete',outputs_sha256={str(p):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='completion.json' and '__pycache__' not in str(p)}))
    print(json.dumps(dict(status='complete',candidates=len(replayed),solver_seconds=solver_seconds,best_native_mean_gain=best['native_mean_gain_vs_v2'],summary_sha256=sha(DEST/'summary.json')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.action=='freeze' else run()
