#!/usr/bin/env python3
"""Same frozen placements under explicit command/host/read sensitivities."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import numpy as np

from research_20261102_qwen38_fit import sha
from research_20261102_qwen38_replay import BASE,inventories
from research_20261102_qwen38_fast_replay import encode
from research_20261102_qwen38_compiled_campaign import items_for
from research_20261102_qwen38_output_fringe import python_graph

ROOT=Path(__file__).resolve().parents[1]
SOURCE=BASE/'qwen38-placement/native-full48'
FRINGE=BASE/'qwen38-placement/output-fringe'
ROUTES=BASE/'qwen38-trace-audit/external-doc-grounded-stride1-short/external_decode.npz'
DEST=BASE/'qwen38-placement/parameter-sensitivity'
PROFILES=[dict(label=f'command_{v}',command_ns=v,host_ns=2000,read_ns=28000) for v in (0,250,1000,2000,4000,5000,10000)]
PROFILES+=[dict(label=f'host_{v}',command_ns=250,host_ns=v,read_ns=28000) for v in (0,10000)]
PROFILES+=[dict(label=f'read_{v}',command_ns=250,host_ns=2000,read_ns=v) for v in (37000,56000)]


def worker(task):
    rec,steps,output=task;target=Path(output);path=Path(rec['plan']);assert sha(path)==rec['plan_sha256']
    plan=json.loads(path.read_text());inv=inventories();desc,partial=encode(plan,inv)
    with np.load(ROUTES) as f:selected=np.ascontiguousarray(f['routes'][steps,rec['layer']],dtype=np.int32)
    lib=ctypes.CDLL(str(target/'native_parameters.so'));p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.eval_native_ffn_parameters.argtypes=[p32,p32,p32]+[ctypes.c_int]*3+[ctypes.c_int64]*2+[ctypes.c_int]*2+[ctypes.c_int64]*3+[p64]
    lib.eval_native_ffn_parameters.restype=None
    reports=[];costs=[]
    for profile in PROFILES:
        out=np.zeros(len(selected),dtype=np.int64)
        lib.eval_native_ffn_parameters(desc,partial,selected,512,len(selected),10,1000,0,0,1,profile['command_ns'],profile['host_ns'],profile['read_ns'],out)
        if profile['label']=='command_250':
            with np.load(FRINGE/(rec['label']+'.mode1.npz')) as f:assert np.array_equal(out,f['cost'][steps])
        items=items_for(plan,inv,selected[0]);g=python_graph(items,1)
        for n in g.nodes:
            if n['acquire']:n['duration']=profile['command_ns']
            elif n['resource']==0 and n['duration']==2000:n['duration']=profile['host_ns']
            elif n.get('kind')=='nonpreemptive_native_compute':
                assert n['duration']%28000==0;n['duration']=n['duration']//28000*profile['read_ns']
        got=g.run(False,trace=True);assert got['makespan_ns']==int(out[0])
        label=rec['label']+'.'+profile['label'];(target/(label+'.trace.json')).write_text(json.dumps(dict(layer=rec['layer'],case=rec['case'],profile=profile,selected=selected[0].tolist(),items=items,**got),indent=2)+'\n')
        reports.append(dict(profile=profile['label'],layer=rec['layer'],case=rec['case'],label=label,mean_ns=float(out.mean()),p99_ns=float(np.percentile(out,99)),trace_sha256=sha(target/(label+'.trace.json'))));costs.append(out)
    np.savez_compressed(target/(rec['label']+'.npz'),cost=np.stack(costs),profiles=np.asarray([p['label'] for p in PROFILES]),external_indices=steps)
    return reports


def main():
    DEST.mkdir(parents=True,exist_ok=True);frozen=SOURCE/'plans.frozen.json';f=json.loads(frozen.read_text())
    cpp=Path(__file__).with_name('research_20261102_native_parameters.cpp');subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(DEST/'native_parameters.so')],check=True)
    with np.load(ROUTES) as data:steps=np.unique(np.linspace(0,len(data['routes'])-1,256,dtype=np.int64))
    hashes={str(p):sha(p) for p in (Path(__file__),cpp,DEST/'native_parameters.so',frozen,ROUTES,FRINGE/'summary.json')}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_inputs_sha256=hashes,profiles=PROFILES,external_indices=steps.tolist(),
                  controls='Same240frozenviews,48FFNlayers inclshared,256fixeduniformindices from the already-used externalsession. No refit,1uspartialreconstruction,0activation/combine,requested-pagecomplete-rowoutputdrain.',
                  limits='Sensitivity of frozen schemes, not optimized layouts on newly calibrated hardware. Timings remain assumptions; MAC/ECC overlap and4Boutput contract unverified. Not a new heldout task.')
    (DEST/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n');records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,done in enumerate(as_completed([pool.submit(worker,(r,steps,str(DEST))) for r in f['records']]),1):
            records+=done.result()
            if i%40==0:print('complete',i,'/240',flush=True)
    totals=[]
    for p,profile in enumerate(PROFILES):
        for case in sorted({r['case'] for r in f['records']}):
            rr=[r for r in f['records'] if r['case']==case]
            values=np.stack([np.load(DEST/(r['label']+'.npz'))['cost'][p] for r in rr]).sum(axis=0)
            totals.append(dict(**profile,case=case,mean_ns=float(values.mean()),p99_ns=float(np.percentile(values,99))))
    assert all(sha(p)==h for p,h in hashes.items())
    (DEST/'summary.json').write_text(json.dumps(dict(status='complete frozen-plan timing-parameter sensitivity',protocol_sha256=sha(DEST/'protocol.frozen.json'),records=records,totals=totals),indent=2)+'\n')
    for profile in PROFILES:
        rr={r['case']:r for r in totals if r['label']==profile['label']}
        print(profile['label'],'whole/overlap ms',rr['whole_row']['mean_ns']/1e6,rr['overlap_best']['mean_ns']/1e6,'gain',1-rr['whole_row']['mean_ns']/rr['overlap_best']['mean_ns'],flush=True)


if __name__=='__main__':main()
