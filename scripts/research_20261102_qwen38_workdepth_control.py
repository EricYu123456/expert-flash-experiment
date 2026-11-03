#!/usr/bin/env python3
"""Matched joint work/depth geometry controls; no external data opened."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

import numpy as np
from research_20261102_qwen38_fit import TRAIN,TRAIN_SHA,sha
from research_20261102_qwen38_replay import BASE,inventories,build
from research_20261102_qwen38_fast_replay import encode,evaluate
from research_20261102_qwen38_compiled_campaign import library
from research_20261102_residue_masks import get_rows,shared_cost,items_for

DEST=BASE/'qwen38-placement/workdepth-control'
INITIAL=BASE/'qwen38-placement/joint-smoke'


def costs(rows):
    result=[]
    for p,c in enumerate((2560,640)):
        r=rows[p].astype(np.int64)
        result.append(np.where(r>0,((r*c+65535)//65536)*28000+250+(c+1)//2+2*r,0))
    return np.asarray(result)


def value(rows,selected,weights,shared):
    cost=costs(rows);load=[cost[p][selected].sum(axis=1)+shared[p] for p in range(2)]
    phase=load[0].max(axis=1)+load[1].max(axis=1);joined=(load[0]+load[1]).max(axis=1)
    return float((joined*1000+phase)@weights/weights.sum()),float(joined@weights/weights.sum())


def templates():
    g=json.loads((BASE/'qwen38-placement/geometry_certificates.json').read_text());result=[]
    for p,n,widths in zip(g['phases'],(1280,2560),((10,13,16),(5,13,16))):
        t={f'g{w}':np.pad(np.asarray(next(x for x in p['equal_row_all_width_controls'] if x['width']==w)['rows'],dtype=np.int32),(0,16-w)) for w in widths}
        t['balanced']=np.asarray(p['balanced_whole_wave']['rows'],dtype=np.int32)
        # Same depth as balanced, but fewer positive jobs: nondominated in
        # job-count yet potentially worse for active-query chip imbalance.
        d=4 if n==1280 else 2
        t['depth_minjobs']=np.asarray(next(x for x in p['exact_capacity_dp_frontier'] if x['max_job_waves']==d)['rows'],dtype=np.int32)
        result.append(t)
    return result


def worker(task):
    layer,gu,down,budget,output=task;target=Path(output);ts=templates()
    planpath=INITIAL/f'L{layer}.whole.global_complement.r1000.plan.json';plan=json.loads(planpath.read_text())
    original,partial=encode(plan,inventories());shared=shared_cost(original)
    labels=np.asarray(plan['plans']['gate_up']['labels'])
    with np.load(TRAIN) as f:
        fit=f['routes'][:,::4,layer].reshape(-1,10);steps=np.unique(np.linspace(0,798,32,dtype=int));smoke=f['routes'][:,steps,layer].reshape(-1,10)
    selected,weights=np.unique(np.sort(fit,axis=1),axis=0,return_counts=True)
    selected=np.ascontiguousarray(selected,dtype=np.int32);weights=np.ascontiguousarray(weights,dtype=np.int64)
    controls=[]
    for shift in range(16):
        rows=np.stack([np.asarray([np.roll(ts[0][gu],int(l)) for l in labels]),np.asarray([np.roll(ts[1][down],int(l)+shift) for l in labels])])
        controls.append((value(rows,selected,weights,shared),shift,rows))
    best,shift,initial=min(controls,key=lambda x:x[0][0]);best_rows=initial.copy();runs=[]
    lib=ctypes.CDLL(str(target/'workdepth_profiles.so'));p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.fit_workdepth_profiles.argtypes=[p32,p32,p64,p32,p64]+[ctypes.c_int]*6+[ctypes.c_uint64,ctypes.c_double,p64];lib.fit_workdepth_profiles.restype=ctypes.c_double
    for seed,temp in enumerate((0.,2000.)):
        rows=initial.copy();stats=np.zeros(3,dtype=np.int64)
        score=lib.fit_workdepth_profiles(rows,np.array([2560,640],dtype=np.int32),shared,selected,weights,512,len(selected),10,0,1,budget,20261102+layer*1009+seed,temp,stats)
        got=value(rows,selected,weights,shared);assert abs(score-got[0])<1e-5
        assert np.array_equal(np.sort(rows,axis=2),np.sort(initial,axis=2)) and np.array_equal(np.sort(rows,axis=1),np.sort(initial,axis=1))
        runs.append(dict(temperature_ns=temp,stats=stats.tolist(),mean_bound_ns=got[1]))
        if got[0]<best[0]:best,best_rows=got,rows.copy()
    desc=original.copy()
    for e in range(512):
        for p,c in enumerate((2560,640)):
            for chip,r in enumerate(best_rows[p,e]):desc[e,p,chip]=(chip,r,c,(int(r)*c+65535)//65536)
    native=evaluate(library(),desc,partial,smoke,1000,0,False)
    occupied=costs(best_rows).sum(axis=0);bound=(occupied[smoke].sum(axis=1)+shared.sum(axis=0)).max(axis=1)+2000
    assert np.all(native>=bound)
    # items_for includes zero-row entries; omit them as the native descriptor
    # interpreter does, so no fictitious zero-size commands are submitted.
    traces=[]
    for i in (0,len(smoke)//2,len(smoke)-1):
        items=items_for(best_rows,plan,smoke[i])
        for item in items:
            for p in ('gate_up','down'):item[p]=[j for j in item[p] if j['rows']]
        got=build(items,1000,0).run(False,trace=i==0);assert got['makespan_ns']==int(native[i])
        if i==0:traces=dict(layer=layer,selected=smoke[i].tolist(),items=items,**got)
    name=f'L{layer}.GU_{gu}.D_{down}';np.savez_compressed(target/(name+'.npz'),rows=best_rows,descriptor=desc,partial=partial,cost=native,native_chip_bound=bound,token_steps=steps)
    (target/(name+'.trace.json')).write_text(json.dumps(traces,indent=2)+'\n')
    result=dict(layer=layer,gate_up=gu,down=down,label=name,global_shift=shift,mean_ns=float(native.mean()),p99_ns=float(np.percentile(native,99)),mean_bound_ns=float(bound.mean()),fit_bound_ns=best[1],runs=runs,
                total_waves=[int(((ts[p][s].astype(np.int64)*(2560,640)[p]+65535)//65536).sum()) for p,s in enumerate((gu,down))],
                artifacts_sha256={ext:sha(target/(name+'.'+ext)) for ext in ('npz','trace.json')})
    (target/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(name,result['mean_ns']/1e6,flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layers',type=int,default=3);ap.add_argument('--valid-budget',type=int,default=24000);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--output',type=Path,default=DEST)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);assert sha(TRAIN)==TRAIN_SHA
    cpp=Path(__file__).with_name('research_20261102_workdepth_profiles.cpp');subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(args.output/'workdepth_profiles.so')],check=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),arguments={**vars(args),'output':str(args.output)},source_inputs_sha256={str(p):sha(p) for p in (Path(__file__),cpp,TRAIN)},
                  fit='3200train tokens/layer,16same-taskrequests.16globalshifts then2 matched24000valid profile-swap starts. Shared location fixed to same complement plan.',
                  limits='Work/depth and complementary geometries are known controls, not new method. NominalI8 on NVFP4 routes, first3layers only, no external opened.')
    (args.output/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    ts=templates();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker,(l,g,d,args.valid_budget,str(args.output))) for l in range(args.layers) for g in ts[0] for d in ts[1]]):records.append(f.result())
    totals=[]
    for g in ts[0]:
        for d in ts[1]:
            rr=sorted((r for r in records if r['gate_up']==g and r['down']==d),key=lambda r:r['layer']);cost=np.stack([np.load(args.output/(r['label']+'.npz'))['cost'] for r in rr]).sum(axis=0)
            totals.append(dict(gate_up=g,down=d,mean_ns=float(cost.mean()),p99_ns=float(np.percentile(cost,99))))
    totals.sort(key=lambda r:r['mean_ns'])
    assert all(sha(p)==h for p,h in protocol['source_inputs_sha256'].items())
    (args.output/'summary.json').write_text(json.dumps(dict(status='complete matched work/depth controls',records=records,totals=totals,protocol_sha256=sha(args.output/'protocol.frozen.json')),indent=2)+'\n')
    print(json.dumps(totals,indent=2),flush=True)


if __name__=='__main__':main()
