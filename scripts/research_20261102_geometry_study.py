#!/usr/bin/env python3
"""New-model screen of legal whole-row geometry, fitted on training routes only.

Every candidate uses the same four-plane ordinary-read GEMV and integer rows.
This evaluates routed projection service, not full-model or physical latency.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

import aif_model as flash
from materialize_plan import WaveAllocator
from run_corpus_traces import ROOT, geometry, sha256
from research_20261102_diagnostics import row_template
from research_20261102_projection_study import equal_rows, evaluate, scalar_cost

DEST = ROOT / 'results/research-2026-11-02/round2/geometry-study'


def load_library():
    DEST.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('research_20261102_geometry_solver.cpp')
    target = DEST / 'geometry.so'
    subprocess.run(['g++', '-std=c++17', '-O3', '-march=native', '-shared',
                    '-fPIC', str(source), '-o', str(target)], check=True)
    lib = ctypes.CDLL(str(target))
    ptr32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    ptr64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    lib.optimize_geometry.argtypes = [ptr32,ptr32,ptr64]+[ctypes.c_int]*6+[
        ctypes.c_uint64,ctypes.c_double,ptr64]
    lib.optimize_geometry.restype = ctypes.c_double
    return lib


def candidate_rows(rows, columns, experts):
    for g in (1,2,4,8,16):
        yield 'group_g'+str(g), equal_rows(rows,experts,g)
    template = row_template(rows,columns)
    yield 'wave_rotations', np.asarray([np.roll(template,e%16) for e in range(experts)])
    # Replicate a small set of balanced layouts: two complementary rotations
    # may avoid the maximum across sixteen unrelated load profiles.
    yield 'wave_two_rotations', np.asarray([np.roll(template,8*(e%2)) for e in range(experts)])
    for omit in (1,2,4,8):
        width = 16-omit
        if rows % width:
            continue
        value = np.zeros((experts,16),dtype=np.int32)
        for e in range(experts):
            value[e] = rows//width
            first = (e % (16//omit))*omit
            value[e,first:first+omit] = 0
        yield 'complement_g'+str(width),value


def is_disjoint_equal_group(rows):
    """Equivalent parameterizations must strengthen the old-group baseline."""
    rows=np.asarray(rows)
    masks=[]
    for row in rows:
        nonzero=row[row>0]
        if not len(nonzero) or not np.all(nonzero==nonzero[0]):
            return False
        mask=tuple(np.flatnonzero(row))
        if len(mask) not in (1,2,4,8,16):
            return False
        if mask not in masks:masks.append(mask)
    return all(not(set(a)&set(b)) for i,a in enumerate(masks) for b in masks[i+1:])


def materialize(phases, experts, target):
    alloc = [WaveAllocator() for _ in range(16)]
    streams=[]
    for phase in phases:
        for e in range(experts):
            cursor=0
            for chip,count in enumerate(phase['rows'][e]):
                count=int(count)
                if count:
                    p,w,n,s=alloc[chip].allocate(count*phase['columns'])
                    streams.append([phase['layer'],phase['name'],e,chip,cursor,count,
                                    phase['columns'],p,w,n,s])
                cursor+=count
            assert cursor==phase['total_rows']
    doc=dict(kind='virtual physical addresses, no programmed NAND',streams=streams,
             chips=[a.summary(i) for i,a in enumerate(alloc)],
             stream_columns=['layer','phase','expert','chip','row_start','rows','columns',
                             'pages','waves','block_commands','segments'])
    with gzip.GzipFile(filename=str(target),mode='wb',mtime=0) as f:
        f.write(json.dumps(doc,separators=(',',':')).encode())
    return dict(path=str(target),sha256=sha256(target),
                reserved_lsb_bytes=sum(c['reserved_lsb_bytes'] for c in doc['chips']),
                physical_pages=sum(c['physical_pages'] for c in doc['chips']),
                max_chip_reserved_lsb_bytes=max(c['reserved_lsb_bytes'] for c in doc['chips']))


def fit(args):
    lib=load_library()
    meta=json.loads(args.metadata.read_text())
    layers,experts,active=geometry(meta)
    dims=flash.tensor_dimensions(meta)
    with np.load(args.routes) as f:
        routes=f['routes']; ids=f['prompt_ids'].tolist()
        assert f['layers'].tolist()==layers
    assert all(x.startswith('train_') for x in ids), 'Fit accepts only original-training prompts'
    assert routes.shape[2:]==(len(layers),active)
    result=[]
    for li,layer in enumerate(layers):
        if args.layers and layer not in args.layers:
            continue
        unique,weights=np.unique(np.sort(routes[:,:,li].reshape(-1,active),axis=1),
                                 axis=0,return_counts=True)
        unique=np.ascontiguousarray(unique,dtype=np.int32)
        weights=np.ascontiguousarray(weights,dtype=np.int64)
        for name,total,columns,common in [('gate_up',sum(dims[layer][p][0] for p in ('gate','up')),
                                         dims[layer]['up'][1],True),
                                        ('down',*dims[layer]['down'],False)]:
            candidates=[]
            for ci,(label,initial) in enumerate(candidate_rows(total,columns,experts)):
                best=np.ascontiguousarray(initial,dtype=np.int32)
                initial_mean=float(evaluate(unique,best,columns,common)['model_ns']@weights/weights.sum())
                best_mean=initial_mean; stats_records=[]
                for temperature in args.temperatures:
                    if np.all(initial==initial[0]):
                        break
                    fitted=np.ascontiguousarray(initial,dtype=np.int32)
                    stats=np.zeros(2,dtype=np.int64)
                    cost=lib.optimize_geometry(fitted,unique,weights,experts,len(unique),active,columns,
                        int(common),args.iterations,20261102+li*1009+ci*31+int(temperature),temperature,stats)
                    values=evaluate(unique,fitted,columns,common)
                    reference=float(values['model_ns']@weights/weights.sum())
                    assert abs(cost-reference)<1e-6,(label,cost,reference)
                    # Swapping whole row vectors preserves the complete per-chip inventory.
                    np.testing.assert_array_equal(np.sort(fitted,axis=0),np.sort(initial,axis=0))
                    assert np.all(fitted.sum(axis=1)==total)
                    for t in sorted({0,len(unique)//2,len(unique)-1}):
                        scalar=scalar_cost(unique[t],fitted,columns,common)
                        for k,v in scalar.items():assert int(values[k][t])==v
                    stats_records.append(dict(temperature_ns=temperature,valid=int(stats[0]),
                                              accepted=int(stats[1]),mean_ns=reference))
                    if reference<best_mean:best_mean,best=reference,fitted.copy()
                item=dict(label=label,group_equivalent=is_disjoint_equal_group(initial),
                          mean_ns=best_mean,initial_mean_ns=initial_mean,
                          rows=best.tolist(),search=stats_records)
                candidates.append(item)
            # All phase candidates share host barriers and projected arithmetic.
            winner=min(candidates,key=lambda c:c['mean_ns'])
            group_best=min((c for c in candidates if c['group_equivalent']),
                           key=lambda c:c['mean_ns'])
            baseline=next(c for c in candidates if c['label']=='group_g16')
            phase=dict(layer=layer,name=name,total_rows=total,columns=columns,common=common,
                candidates=candidates,chosen_label=winner['label'],rows=winner['rows'],
                group_label=group_best['label'],group_rows=group_best['rows'],
                group_ns=group_best['mean_ns'],baseline_ns=baseline['mean_ns'],chosen_ns=winner['mean_ns'])
            result.append(phase)
            print(layer,name,'best',winner['label'],'vs phase group',
                  round(100*(1-winner['mean_ns']/group_best['mean_ns']),5),'percent',flush=True)
            (DEST/(args.name+'.partial.json')).write_text(json.dumps(result,indent=2)+'\n')
    # Freeze the entire model before opening any new-model evaluation routes.
    doc=dict(model=args.name,phases=result,layers=layers,experts=experts,active=active,
        train_ids=ids,training_routes=str(args.routes),training_routes_sha256=sha256(args.routes),
        metadata_path=str(args.metadata),metadata_sha256=sha256(args.metadata),
        frozen_utc=datetime.now(timezone.utc).isoformat(),test_routes_opened=False,
        scope='routed FFN service only; shared FFN, SSM/attention and full-model quality excluded',
        status='exploratory training screen, not a paper novelty or hardware result',
        iterations=args.iterations,temperatures=args.temperatures,
        source_sha256={p.name:sha256(p) for p in [Path(__file__),
            Path(__file__).with_name('research_20261102_geometry_solver.cpp')]})
    if len(result)==len(layers)*2:
        doc['chosen_physical']=materialize(result,experts,DEST/(args.name+'.physical.json.gz'))
        group_phases=[dict(p,rows=p['group_rows']) for p in result]
        doc['phase_group_physical']=materialize(group_phases,experts,DEST/(args.name+'.group_physical.json.gz'))
    doc['training_means_ms']={key:sum(p[key+'_ns'] for p in result)/1e6 for key in ('baseline','group','chosen')}
    (DEST/(args.name+'.frozen.json')).write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(doc['training_means_ms']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='granite4tiny')
    p.add_argument('--metadata',type=Path,default=Path('/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-metadata.json'))
    p.add_argument('--routes',type=Path,default=ROOT/'results/research-2026-11-02/round2/neuron-layout-prior/granite4tiny-routes/screen/routes.npz')
    p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--temperatures',type=float,nargs='+',default=[0.,1000.])
    p.add_argument('--layers',type=int,nargs='*')
    fit(p.parse_args())
