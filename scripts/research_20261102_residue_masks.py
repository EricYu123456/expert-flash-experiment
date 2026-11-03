#!/usr/bin/env python3
"""Train-only free-mask pilot against matched capacity-preserving controls.

The external task has previously been examined. This script does not open it;
future reuse is development validation, not a fresh untouched test.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from research_20261102_qwen38_fit import TRAIN, TRAIN_SHA, sha
from research_20261102_qwen38_replay import BASE, inventories, build
from research_20261102_qwen38_fast_replay import encode, evaluate
from research_20261102_qwen38_compiled_campaign import library

DEST = BASE/'qwen38-placement/residue-mask-pilot'
INITIAL = BASE/'qwen38-placement/joint-smoke'


def get_rows(desc):
    rows = np.zeros((2,512,16), dtype=np.int32)
    for e in range(512):
        for p in range(2):
            for chip, r, c, w in desc[e,p]:
                rows[p,e,chip] = r
    return rows


def describe(rows, original):
    desc = original.copy()
    for e in range(512):
        for p, columns in enumerate((2560,640)):
            for c, r in enumerate(rows[p,e]):
                desc[e,p,c] = (c,r,columns,(int(r)*columns+65535)//65536)
    return desc


def occupancy(rows):
    result = []
    for p, columns in enumerate((2560,640)):
        r = rows[p].astype(np.int64)
        result.append(((r*columns+65535)//65536)*28000+250+(columns+1)//2+2*r)
    return np.asarray(result)


def shared_cost(desc):
    result = np.zeros((2,16), dtype=np.int64)
    for p in range(2):
        for chip,r,c,w in desc[512,p]:
            if r:
                result[p,chip] += int(w)*28000+250+(int(c)+1)//2+2*int(r)
    return result


def score(rows, selected, weights, shared, objective):
    cost = occupancy(rows)
    loads = [cost[p][selected].sum(axis=1)+shared[p] for p in range(2)]
    phase = loads[0].max(axis=1)+loads[1].max(axis=1)
    joined = (loads[0]+loads[1]).max(axis=1)
    value = phase*1000+joined if objective == 0 else joined*1000+phase
    return float(value@weights/weights.sum()),float(phase@weights/weights.sum()),float(joined@weights/weights.sum())


def items_for(rows, plan, selected):
    inv = inventories(); items = []
    for e in selected:
        item = dict(gu_partial=False,down_partial=False)
        for p,name in enumerate(('gate_up','down')):
            cursor=0;jobs=[]
            for chip,r in enumerate(rows[p,e]):
                jobs.append(dict(chip=chip,row_start=cursor,rows=int(r),column_start=0,columns=(2560,640)[p]));cursor+=int(r)
            assert cursor==(1280,2560)[p];item[name]=jobs
        items.append(item)
    item = dict(gu_partial=False,down_partial=False)
    for name,r in plan['fixed_shared'].items():item[name]=inv[name][r['label']]['variants'][r['color']]
    items.append(item);return items


def worker(task):
    layer,mode,objective,budget,output = task
    target=Path(output);lib=ctypes.CDLL(str(target/'residue_masks.so'))
    p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.fit_residue_masks.argtypes=[p32,p32,p64,p32,p64]+[ctypes.c_int]*6+[ctypes.c_uint64,ctypes.c_double,p64]
    lib.fit_residue_masks.restype=ctypes.c_double
    path=INITIAL/f'L{layer}.whole.global_complement.r1000.plan.json'
    plan=json.loads(path.read_text());desc,partial=encode(plan,inventories());initial=get_rows(desc)
    shared=shared_cost(desc)
    with np.load(TRAIN) as f:
        fit=f['routes'][:,::4,layer].reshape(-1,10)
        steps=np.unique(np.linspace(0,798,32,dtype=int));smoke=f['routes'][:,steps,layer].reshape(-1,10)
    selected,weights=np.unique(np.sort(fit,axis=1),axis=0,return_counts=True)
    selected=np.ascontiguousarray(selected,dtype=np.int32);weights=np.ascontiguousarray(weights,dtype=np.int64)
    before=score(initial,selected,weights,shared,objective);best=before[0];best_rows=initial.copy();runs=[]
    for seed,temperature in enumerate((0.,2000.)):
        rows=initial.copy();stats=np.zeros(3,dtype=np.int64)
        value=lib.fit_residue_masks(rows,np.array([2560,640],dtype=np.int32),shared,selected,weights,512,len(selected),10,mode,objective,budget,20261102+layer*1009+seed,temperature,stats)
        got=score(rows,selected,weights,shared,objective);assert abs(value-got[0])<1e-5
        # Each expert's job multiset and every chip's exact row-type inventory.
        assert np.array_equal(np.sort(rows,axis=2),np.sort(initial,axis=2))
        assert np.array_equal(np.sort(rows,axis=1),np.sort(initial,axis=1))
        runs.append(dict(temperature_ns=temperature,attempts=int(stats[0]),valid=int(stats[1]),accepted=int(stats[2]),bound=got))
        if value<best:best=value;best_rows=rows.copy()
    final=score(best_rows,selected,weights,shared,objective)
    encoded=describe(best_rows,desc);costs=evaluate(library(),encoded,partial,smoke,1000,0,False)
    occupied=occupancy(best_rows).sum(axis=0)
    bounds=(occupied[smoke].sum(axis=1)+shared.sum(axis=0)).max(axis=1)+2000
    assert np.all(costs>=bounds)
    # Several independent Python Graph constructions check the new adapter.
    for i in (0,len(smoke)//2,len(smoke)-1):
        assert build(items_for(best_rows,plan,smoke[i]),1000,0).run(False)['makespan_ns']==int(costs[i])
    trace=dict(layer=layer,selected=smoke[0].tolist(),items=items_for(best_rows,plan,smoke[0]))
    trace.update(build(trace['items'],1000,0).run(False,trace=True))
    name=f'L{layer}.m{mode}.o{objective}'
    np.savez_compressed(target/(name+'.npz'),rows=best_rows,initial_rows=initial,descriptor=encoded,partial=partial,cost=costs,native_chip_bound=bounds,token_steps=steps)
    (target/(name+'.trace.json')).write_text(json.dumps(trace,indent=2)+'\n')
    report=dict(layer=layer,mode=mode,objective=objective,label=name,initial_plan=str(path),initial_plan_sha256=sha(path),initial_bound=before,final_bound=final,runs=runs,
                mean_ns=float(costs.mean()),p99_ns=float(np.percentile(costs,99)),mean_bound_ns=float(bounds.mean()),artifact_sha256={ext:sha(target/(name+'.'+ext)) for ext in ('npz','trace.json')})
    (target/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(name,final,'native',report['mean_ns']/1e6,flush=True);return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layers',type=int,default=3);ap.add_argument('--valid-budget',type=int,default=24000)
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--output',type=Path,default=DEST);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert sha(TRAIN)==TRAIN_SHA
    cpp=Path(__file__).with_suffix('.cpp');subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(args.output/'residue_masks.so')],check=True)
    hashes={str(p):sha(p) for p in (Path(__file__),cpp,TRAIN)}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),arguments={**vars(args),'output':str(args.output)},source_inputs_sha256=hashes,
                  moves={0:'Swap complete two-phase cyclic profiles (strong matched-search control)',1:'Degree-preserving two-expert/two-chip trades of complete GU/down row pairs',2:'Degree-preserving trades of one phase'},
                  fit='3200 training tokens per layer, all16requests steps0,4,...796. Two starts, same valid-move budget; original complement shared location held fixed.',
                  prior='Ordinary balanced declustering/local search baseline. Free masks and dependent rounding are known; no novelty claim.',
                  evaluation='512 samples from the same task, first3layers. Existing external task is development data and is not opened by this script.')
    (args.output/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker,(l,m,1,args.valid_budget,str(args.output))) for l in range(args.layers) for m in (0,1,2)]):records.append(f.result())
    totals=[]
    for mode in (0,1,2):
        rr=sorted((r for r in records if r['mode']==mode),key=lambda r:r['layer']);costs=np.stack([np.load(args.output/(r['label']+'.npz'))['cost'] for r in rr]).sum(axis=0)
        totals.append(dict(mode=mode,mean_ns=float(costs.mean()),p99_ns=float(np.percentile(costs,99))))
    assert all(sha(p)==h for p,h in hashes.items())
    (args.output/'summary.json').write_text(json.dumps(dict(status='complete capacity-preserving free-mask baseline pilot',protocol_sha256=sha(args.output/'protocol.frozen.json'),records=records,totals=totals),indent=2)+'\n')
    print(json.dumps(totals,indent=2),flush=True)


if __name__=='__main__':main()
