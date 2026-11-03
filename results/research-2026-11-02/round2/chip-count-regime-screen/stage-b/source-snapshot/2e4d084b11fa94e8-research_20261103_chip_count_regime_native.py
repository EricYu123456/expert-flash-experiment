#!/usr/bin/env python3
"""Bounded unchanged-native replay of frozen chip-count geometry controls."""
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
from research_20261103_topology_placement_replay import CPP,PYTHON,load_python

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
STAGE=BASE/'chip-count-regime-screen/stage-a';OUT=BASE/'chip-count-regime-screen/stage-b'
CFG=BASE/'compact-coupled-solver/protocol.frozen.json'
ANCHORS={'granite4tiny':(BASE/'compact-coupled-solver/continuation/native/0107.npz',1),
         'deepseek':(BASE/'native-joint-refinement/pilot-v1/deepseek.L0.input4.independent_paired.seed20261106/arrays.npz',4)}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def stats(x):return dict(mean_ns=float(np.mean(x)),p99_ns=float(np.percentile(x,99)))


def occupancy(rows,shape,copies):
    widths=np.full((65,2,1),shape['H'],np.int64);widths[:64,1]=shape['I'];widths[64,1]=shape['SI']
    active=rows>0;pages=(rows*widths+16383)//16384;waves=(pages+3)//4;emit=pages*16384//widths
    return active*(250+(widths*copies+1)//2)+waves*28000+emit*2


def extreme_plan(geos,kind,routes,copies,shape):
    M=geos['GU']['chips'];rows=np.zeros((65,2,M),np.int64)
    for phase,routed,shared in [(0,'GU','sharedGU'),(1,'Down','sharedDown')]:
        for tag,experts in [(routed,range(64)),(shared,[64])]:
            menu=geos[tag]['menu']
            key=(lambda p:(p['waves'],p['depth'],p['positive_jobs'])) if kind=='minimum_work' else (lambda p:(p['depth'],p['waves'],p['positive_jobs']))
            base=np.asarray(min(menu,key=key)['rows'])
            for e in experts:rows[e,phase]=np.roll(base,0 if e==64 else e%M)
    table=occupancy(rows,shape,copies);gu=table[routes,0].sum(1)+table[64,0]
    down=table[routes,1].sum(1)+table[64,1]
    scores=np.asarray([(gu+np.roll(down,shift,axis=1)).max(1).mean() for shift in range(M)])
    shift=int(scores.argmin());rows[:,1]=np.roll(rows[:,1],shift,axis=1)
    return rows,dict(Down_global_shift=shift,proxy_mean_ns=float(scores[shift]))


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    p=json.loads((STAGE/'protocol.frozen.json').read_text());r=json.loads((STAGE/'report.json').read_text())
    arrays=load(STAGE/'candidate_arrays.npz');geometry=json.loads((STAGE/'geometry.json').read_text())
    cases=[];snapshots={}
    for model,shape in p['models'].items():
        for M in p['chips']:
            evidence=[x for x in r['cases'] if x['model']==model and x['chips']==M]
            if max(x['owned_work_optimization_ceiling_pct'] for x in evidence)<=5:
                continue
            route=arrays[f'{model}.M{M}.input1.routes'];geo=next(x['shapes'] for x in geometry if x['model']==model and x['chips']==M)
            candidates=[]
            for copies in (1,4):candidates.append((f'finite_proxy_input{copies}',copies,arrays[f'{model}.M{M}.input{copies}.rows'],dict(source='stageAfrozenfiniteproxy')))
            extra=[('minimum_work',1),('minimum_depth',1)]
            if M==16:
                path,copies=ANCHORS[model];z=load(path)
                assert np.array_equal(z['routes'],route)
                candidates.append(('legacy_anchor',copies,z['rows'],dict(path=str(path),sha256=sha(path))))
                extra=extra[:1]
            for kind,copies in extra:
                rr,meta=extreme_plan(geo,kind,route,copies,shape);candidates.append((kind,copies,rr,meta))
            seen=set()
            for label,copies,rows,meta in candidates:
                key=(copies,np.asarray(rows,np.int64).tobytes())
                if key in seen:continue
                seen.add(key);ident=f'{model}.M{M}.{label}'
                snapshots[ident+'.rows']=rows;snapshots[ident+'.routes']=route
                cases.append(dict(id=ident,model=model,chips=M,channels=min(8,M),input_copies=copies,label=label,shape=shape,metadata=meta))
            assert len(seen)<=4
    np.savez_compressed(OUT/'plans.frozen.npz',**snapshots)
    paths=[Path(__file__),STAGE/'protocol.frozen.json',STAGE/'report.json',STAGE/'geometry.json',STAGE/'candidate_arrays.npz',
           OUT/'plans.frozen.npz',CFG,CPP,PYTHON,ROOT/'scripts/research_20261103_topology_placement_replay.py',
           ROOT/'scripts/research_20261102_whole_row_compiler.py']+[x[0] for x in ANCHORS.values()]
    inputs=[]
    for path in paths:
        item=dict(path=str(path),sha256=sha(path))
        if path.suffix in ('.py','.cpp'):
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        inputs.append(item)
    protocol=dict(status='frozen_before_bounded_native',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
                  inputs=inputs,cases=cases,priorities=[dict(name='GF',down_first=0),dict(name='DF',down_first=1)],
                  native_query_budget=len(cases)*2*512,max_budget=32768,
                  service_base=json.loads(CFG.read_text())['service'],
                  anchors='16chipCP0107G4/input1 and strongestseed06Deep/input4; GF mustmatchall512oldcosts/bounds. Notweakseed05.',
                  control_selection='Allwidth/cyclic/disjoint/simplefrontier family searchedonlyvia frozenstageA finiteownedworkproxy. Extremesareadditionalrouteblindprofilecontrols plusuniformcostDownrotations. No labelsfit ornativeperquerycandidatechoice.',
                  scope='Only8/16/32/64chip scalingregime onactualW8TRAINL0;GF/DFequalcapabilityforall. No replicas/inference/newmodels/heldout/nativecorechanges.',
                  limits='UnmodifiednativeGraph andnominal28us service;8/32/64 arePE-scalevariants,notprovenAiFimplementations. No acrosshardwareplacementgain claim. Finiteprofiles/heuristicsnotSOTA/globaloptimum;futurefree-mask/hypergraphmustcomparematchednative/capacitycontrols.')
    assert protocol['native_query_budget']<=32768
    dump(OUT/'protocol.frozen.json',protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),cases=len(cases),native_queries=protocol['native_query_budget'])))


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for item in p['inputs']:assert sha(item['path'])==item['sha256']
    assert sha(__file__)==p['source_sha256'] and not (OUT/'report.json').exists()
    backend=load_python();plans=load(OUT/'plans.frozen.npz');t0=time.monotonic()
    binary=OUT/'native.so';subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(CPP),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));fn=lib.eval_ordinary_read
    i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    fn.argtypes=[i32,i32,i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64,i64,i64,ctypes.c_void_p,i64]
    def evaluate(plan,rr,s,trace=False):
        backend.validate_plan(plan,65,s);rr=np.ascontiguousarray(rr,np.int32)
        result=np.zeros(len(rr),np.int64);bounds=result.copy();nlog=np.zeros(1,np.int64);log=np.zeros((6000 if trace else 1,8),np.int64)
        fn(plan['offsets'],plan['jobs'],plan['input_columns'],rr,64,1,len(plan['jobs']),len(rr),6,s.array(),result,bounds,
           ctypes.c_void_p(log.ctypes.data) if trace else None,nlog)
        assert np.all(result>=bounds)
        if trace:assert nlog[0]<=len(log)
        return result,bounds,log[:nlog[0]].copy() if trace else None
    records=[];outputs={};traces=[];anchors=[]
    for case in p['cases']:
        ident=case['id'];rows=plans[ident+'.rows'];routes=plans[ident+'.routes'];shape=case['shape']
        config=dict(p['service_base']);config.update(chips=case['chips'],channels=case['channels'],input_channel_copies=case['input_copies'])
        base_service=backend.Service(**config)
        plan=backend.lower_rows(rows[:,0],rows[:,1],shape['H'],[shape['I']]*64+[shape['SI']],base_service)
        assert np.array_equal(plan['total_rows'][:64],np.tile([2*shape['I'],shape['H']],(64,1)))
        assert np.array_equal(plan['total_rows'][64],[2*shape['SI'],shape['H']])
        physical=backend.physical_addresses(plan,base_service);addresses=physical.pop('addresses')
        np.savez_compressed(OUT/(ident+'.plan.npz'),rows=rows,routes=routes,offsets=plan['offsets'],jobs=plan['jobs'],input_columns=plan['input_columns'],
                            total_rows=plan['total_rows'],addresses=np.asarray([[x[k] for k in ['chip','block','wordline','reserved_wordlines','pages','first_plane']] for x in addresses],np.int32))
        for priority in p['priorities']:
            svc=dict(config);svc['down_first']=priority['down_first'];s=backend.Service(**svc)
            values,bounds,_=evaluate(plan,routes,s)
            expected,bl,tr=evaluate(plan,routes[:1],s,True)
            py=backend.ReferenceGraph.build(plan,np.r_[routes[0],64],s).run()
            assert expected[0]==values[0]==py['cost'] and np.array_equal(tr,py['trace'])
            if case['label']=='legacy_anchor' and priority['name']=='GF':
                anchor=load(Path(case['metadata']['path']))
                assert np.array_equal(values,anchor['cost']) and np.array_equal(bounds,anchor['lower_bound'])
                anchors.append(dict(id=ident,values_exact=512,bounds_exact=512,mean_ns=float(values.mean())))
            key=ident+'.'+priority['name'];outputs[key+'.cost']=values;outputs[key+'.bound']=bounds;outputs[key+'.q0trace']=tr
            traces.append(dict(id=key,nodes=len(tr),cost_ns=int(values[0]),all_nodes_exact=True))
            record=dict(**case,priority=priority['name'],cost=stats(values),native_bound=stats(bounds),service=svc,
                        physical=physical,commands_mean=float(((plan['offsets'][:,:,1]-plan['offsets'][:,:,0])[routes].sum((1,2))+
                            (plan['offsets'][64,:,1]-plan['offsets'][64,:,0]).sum()).mean()))
            records.append(record);print(json.dumps(dict(id=key,cost=record['cost'])),flush=True)
    np.savez_compressed(OUT/'cost_arrays.npz',**outputs)
    necessary={(c['model'],c['chips']):c['necessary'] for c in json.loads((STAGE/'report.json').read_text())['cases']}
    summary=[]
    for model,M in sorted(set((r['model'],r['chips']) for r in records)):
        for copies in (1,4):
            choices=[r for r in records if (r['model'],r['chips'],r['input_copies'])==(model,M,copies)]
            if not choices:continue
            best=min(choices,key=lambda r:r['cost']['mean_ns']);bound=necessary[model,M]['universal_necessary_ns']
            summary.append(dict(model=model,chips=M,input_copies=copies,best_id=best['id'],priority=best['priority'],cost=best['cost'],necessary_LB_ns=bound,
                                optimistic_remaining_mean_pct=100*(1-bound/best['cost']['mean_ns']),
                                optimistic_remaining_p99_pct=100*(1-bound/best['cost']['p99_ns']),
                                warning='Looseuniversalnecessarybound,notattainableor evidenceofnoveladaptivegain. BestonlyfinitecontrolsandGF/DF.'))
    report=dict(status='bounded_frozen_profiles_native_scale_sensitivity',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),
                cpp_sha256=sha(CPP),binary_sha256=sha(binary),native_costs=len(records)*512,records=records,summary=summary,
                representative_checks=traces,legacy_anchor_bridges=anchors,seconds=time.monotonic()-t0,
                limits='No routinglabelsfit/replicas/CP/newnativecore/inference/heldout.8/32/64PE-scalevariantsnotAiFhardwareproof. No acrossMplacementgain. Beforeanymethodclaim,strongnative/capacity/prior controlsandmatchedfree-mask/hypergraphneedseparatestudy.')
    dump(OUT/'report.json',report);print(json.dumps(dict(status=report['status'],costs=report['native_costs'],summary=summary,seconds=report['seconds']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
