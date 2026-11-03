#!/usr/bin/env python3
"""Mirror existing 16-chip strong anchor geometries across input-fanout costs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import research_20261103_ordinary_read_backend as backend

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
STAGE=BASE/'chip-count-regime-screen/stage-b';OUT=STAGE/'anchor-fanout-addendum'
ANCHORS={'granite4tiny':(BASE/'compact-coupled-solver/continuation/native/0107.npz',4,1536,512,1024),
         'deepseek':(BASE/'native-joint-refinement/pilot-v1/deepseek.L0.input4.independent_paired.seed20261106/arrays.npz',1,2048,1408,2816)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}


def freeze():
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),STAGE/'protocol.frozen.json',STAGE/'report.json',ROOT/'scripts/research_20261103_ordinary_read_backend.py',
           ROOT/'scripts/research_20261103_ordinary_read_native.cpp']+[x[0] for x in ANCHORS.values()]
    dump(OUT/'protocol.frozen.json',dict(source_sha256=sha(__file__),inputs=[dict(path=str(p),sha256=sha(p)) for p in paths],
        reason='Samefrozen16chipanchorrows evaluated underotherinputcopycontract. Notnewgeometry/search. PreserveStageBoriginalreport.',
        cases=[dict(model=m,path=str(x[0]),input_copies=x[1]) for m,x in ANCHORS.items()],priorities=['GF','DF'],
        costs=2048,total_including_stageB=29696,limit=32768,
        scope='No label fit/replica/heldout/changednativecore. AllothercostsandfullGU/A8dependenciesunchanged.'))
    print(sha(OUT/'protocol.frozen.json'))


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for x in p['inputs']:assert sha(x['path'])==x['sha256']
    assert not (OUT/'report.json').exists()
    svc=json.loads((STAGE/'protocol.frozen.json').read_text())['service_base'];native=backend.NativeReplay(OUT/'build')
    records=[];arrays={}
    for model,(path,copies,H,I,SI) in ANCHORS.items():
        z=load(path);base=dict(svc);base.update(chips=16,channels=8,input_channel_copies=copies)
        plan=backend.lower_rows(z['rows'][:,0],z['rows'][:,1],H,[I]*64+[SI],backend.Service(**base))
        for priority in (0,1):
            config=dict(base);config['down_first']=priority;s=backend.Service(**config)
            result=native.run(plan,z['routes'],64,1,s)
            check=native.run(plan,z['routes'][:1],64,1,s,trace=True)
            py=backend.ReferenceGraph.build(plan,np.r_[z['routes'][0],64],s).run()
            assert py['cost']==check['cost'][0]==result['cost'][0] and np.array_equal(py['trace'],check['trace'])
            tag=f'{model}.M16.legacy_anchor.input{copies}.'+('DF' if priority else 'GF')
            arrays[tag+'.cost']=result['cost'];arrays[tag+'.bound']=result['lower_bound'];arrays[tag+'.q0trace']=check['trace']
            records.append(dict(id=tag,model=model,chips=16,input_copies=copies,priority='DF' if priority else 'GF',
                cost=dict(mean_ns=float(result['cost'].mean()),p99_ns=float(np.percentile(result['cost'],99))),
                service=config,source_anchor=str(path),source_sha256=sha(path),nodes=len(check['trace']),full_q0_trace_exact=True))
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    dump(OUT/'report.json',dict(status='PASS_strong_anchor_mirror_contract',protocol_sha256=sha(OUT/'protocol.frozen.json'),
         records=records,queries=2048,no_geometry_or_fit_change=True,native=native.provenance))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);a=ap.parse_args()
    freeze() if a.mode=='freeze' else run()
