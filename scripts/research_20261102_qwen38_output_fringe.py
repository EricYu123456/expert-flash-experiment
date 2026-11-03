#!/usr/bin/env python3
"""Charge potentially emitted padding outputs for every frozen placement.

The AiF page-count command does not establish a useful-row stop. Three
software contracts are separated: useful rows only, all complete rows in
requested pages, and all complete rows in the reserved four-plane waves.
No new row-stop or controller filtering capability is assumed for the latter.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np

from research_20261102_qwen38_fit import sha
from research_20261102_qwen38_replay import BASE,build,inventories
from research_20261102_qwen38_fast_replay import encode
from research_20261102_qwen38_compiled_campaign import items_for

ROOT=Path(__file__).resolve().parents[1]
SOURCE=BASE/'qwen38-placement/native-full48'
EXTERNAL=BASE/'qwen38-placement/external-doc-grounded'
ROUTES=BASE/'qwen38-trace-audit/external-doc-grounded-stride1-short/external_decode.npz'
DEST=BASE/'qwen38-placement/output-fringe'


def emitted(rows,columns,mode):
    if mode==0:return rows
    useful=rows*columns
    bytes_=((useful+16383)//16384)*16384 if mode==1 else ((useful+65535)//65536)*65536
    return bytes_//columns


def python_graph(items,mode):
    g=build(items,1000,0)
    ordered=[j for p in ('gate_up','down') for item in items for j in item[p]]
    assert len(ordered)==len(g.jobs)
    for node,job in zip(g.jobs,ordered):
        r=emitted(job['rows'],job['columns'],mode)
        g.nodes[node['release']]['duration']=r*2
        g.nodes[node['host']]['duration']=(r*4+7)//8
    return g


def worker(task):
    rec,output=task;target=Path(output);inv=inventories();path=Path(rec['plan'])
    assert sha(path)==rec['plan_sha256'];plan=json.loads(path.read_text());desc,partial=encode(plan,inv)
    with np.load(ROUTES) as f:selected=np.ascontiguousarray(f['routes'][:,rec['layer']],dtype=np.int32)
    lib=ctypes.CDLL(str(target/'native_output_fringe.so'));p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.eval_native_ffn_fringe.argtypes=[p32,p32,p32]+[ctypes.c_int]*3+[ctypes.c_int64]*2+[ctypes.c_int]*2+[p64];lib.eval_native_ffn_fringe.restype=None
    rows=[]
    for mode in (0,1,2):
        chosen=selected[:1] if mode==0 else selected
        costs=np.zeros(len(chosen),dtype=np.int64)
        lib.eval_native_ffn_fringe(desc,partial,chosen,512,len(chosen),10,1000,0,0,mode,costs)
        with np.load(EXTERNAL/(rec['label']+'.npz')) as f:old=f['cost'].copy()
        if mode==0:assert int(costs[0])==int(old[0]);continue
        # A slower node-duration contract need not make a greedy priority
        # schedule pointwise slower (resource-order changes can improve it).
        # Report signed differences without assuming monotonic makespan.
        items=items_for(plan,inv,selected[0]);trace=python_graph(items,mode).run(False,trace=True)
        assert trace['makespan_ns']==int(costs[0])
        label=rec['label']+f'.mode{mode}'
        np.savez_compressed(target/(label+'.npz'),cost=costs,difference_vs_useful=costs-old)
        doc=dict(layer=rec['layer'],case=rec['case'],mode=mode,selected=selected[0].tolist(),items=items,**trace)
        (target/(label+'.trace.json')).write_text(json.dumps(doc,indent=2)+'\n')
        rows.append(dict(layer=rec['layer'],case=rec['case'],label=label,mode=mode,mean_ns=float(costs.mean()),p99_ns=float(np.percentile(costs,99)),mean_difference_ns=float((costs-old).mean()),
                         artifacts_sha256={ext:sha(target/(label+'.'+ext)) for ext in ('npz','trace.json')}))
    return rows


def main():
    DEST.mkdir(parents=True,exist_ok=True);frozen=SOURCE/'plans.frozen.json';f=json.loads(frozen.read_text())
    cpp=Path(__file__).with_name('research_20261102_native_output_fringe.cpp')
    subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(DEST/'native_output_fringe.so')],check=True)
    inputs=[frozen,ROUTES,EXTERNAL/'summary.json'];sources=[Path(__file__),cpp,DEST/'native_output_fringe.so']
    hashes={str(p):sha(p) for p in inputs+sources}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_inputs_sha256=hashes,
                  modes={0:'original useful-row output contract',1:'drain every complete row in programmed16KiB pages',2:'drain every complete row in all reserved64KiB four-plane waves'},
                  controls='All240frozenplans unchanged, no refit; sharedincluded;1uspartials;0activation/combine; full external5519decode.',
                  limitation='Output-count sensitivity under fixed4B result width. No verified native row-stop/output ABI and no silicon calibration. Both channel and PCIe drain all emitted results.')
    (DEST/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n');records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,done in enumerate(as_completed([pool.submit(worker,(r,str(DEST))) for r in f['records']]),1):
            records+=done.result()
            if i%40==0:print('complete',i,'/240',flush=True)
    totals=[]
    baseline={r['case']:r['token_weighted_mean_ns'] for r in json.loads((EXTERNAL/'summary.json').read_text())['totals']}
    for mode in (1,2):
        for case in sorted(baseline):
            rr=[r for r in records if r['mode']==mode and r['case']==case]
            values=np.stack([np.load(DEST/(r['label']+'.npz'))['cost'] for r in rr]).sum(axis=0)
            totals.append(dict(mode=mode,case=case,mean_ns=float(values.mean()),p99_ns=float(np.percentile(values,99)),relative_mean_change_vs_useful=float(values.mean()/baseline[case]-1)))
    assert all(sha(p)==h for p,h in hashes.items())
    (DEST/'summary.json').write_text(json.dumps(dict(status='complete native output-fringe contract sensitivity',protocol_sha256=sha(DEST/'protocol.frozen.json'),records=records,totals=totals),indent=2)+'\n')
    print(json.dumps(totals,indent=2),flush=True)


if __name__=='__main__':main()
