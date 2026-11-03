#!/usr/bin/env python3
"""Independent TRAIN-only geometry, work and native replay audit."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/'results/research-2026-11-02/round2'
BASE=ROUND/'chip-count-regime-screen'
TARGET=BASE/'m32-tail-directed-rotation-control'
OUT=TARGET/'independent'
SNAP=ROUND/'ordinary-read-compiler/full-campaign-v6-W8-input4/source-snapshot'
CPP=SNAP/'ef7994fa2ba37a4b-research_20261103_ordinary_read_native.cpp'
PYTHON=SNAP/'7c61f09756e24f11-research_20261103_ordinary_read_backend.py'
TRAIN=ROUND/'w8-format-training/granite4tiny/decode.routes.npz'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def stats(x):return dict(mean_ns=float(np.mean(x)),p99_ns=float(np.percentile(x,99)),max_ns=int(np.max(x)))


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    files=[Path(__file__),ROOT/'scripts/research_20261103_m32_tail_rotation_control.py',CPP,PYTHON,
           ROOT/'scripts/research_20261102_whole_row_compiler.py',TRAIN,
           BASE/'stage-b/granite4tiny.M32.minimum_depth.plan.npz',BASE/'g4-M32-coupled-cp/native_arrays.npz',
           BASE/'m32-prospective-tail-validation/protocol.frozen.json']
    files += [TARGET/name for name in ('protocol.frozen.json','inputs.frozen.npz','candidates.frozen.npz','fit_report.json',
                                      'native_protocol.frozen.json','native_arrays.npz','native_report.json','completion.json')]
    files += sorted((TARGET/'native').glob('*.plan.npz'))
    references=[]
    for path in files:
        item=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        if path.suffix in ('.py','.cpp'):
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        references.append(item)
    dump(OUT/'protocol.frozen.json',dict(status='frozen_before_independent_recomputation',utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),references=references,
        geometry='Derivecanonical GU/Downexpert0 orbits independently fromStageB. Checklabels/rowmasks/40slot/coverage, rebuildraggednativearrays fromrows withoutproducerlowerer, savedaddresses oneblock/nonoverlap/page/wave.',
        work='Elementarycommand+fullinputreload+readwaves+pagefringeoutput perchip; bothcopies; primarydirect29102G+28762D. Compareevery512chipvector andstoredwork.',
        native='Freshcompile unchangedCPP; independentlyrebuiltplans, all6mapsx2copiesxGF/DFx512=12288newcosts. Then24newC++/Pythonfulltraces: q0/511 oneachmap forinput1GF/input4DF.',
        no_producer_as_oracle='Rootstoredcosts are onlycomparison afterfreshCPP output. Producerfit/generator neverimported orrun. FrozenPythonReferenceGraph independentlychecks24fulltraces.',
        chronology='Supplementarycontrolfrozeafternewpromptacquisition butbefore rootviewedfreshoutcomes, asparentattests. Not originalprospectiveprimary, no claimwholeteamneveraccessednewpayload. Thisauditoropensnonewmanifest/prompt/capture/routes.',
        exclusions='No fitting, source/plan change, newcapture read, all-layer, inference or numericalcampaign. Existingprospectiveprimary andthreeplans remainunchanged.'))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__)),indent=2))


def own_plan(rows):
    offsets=np.zeros((65,2,2),np.int32);jobs=[];logical=[];columns=np.tile([1536,512],(65,1)).astype(np.int32);columns[64,1]=1024
    for e in range(65):
        for phase in (0,1):
            cursor=0;offsets[e,phase,0]=len(jobs);c=int(columns[e,phase])
            for chip in range(32):
                n=int(rows[e,phase,chip])
                if not n:continue
                pages=(n*c+16383)//16384;waves=(pages+3)//4
                assert 0<waves<=192
                jobs.append([chip,n,c,pages,waves])
                logical.append(dict(expert=e,phase=phase,chip=chip,row_start=cursor,rows=n,columns=c,useful_bytes=n*c,pages=pages,waves=waves));cursor+=n
            assert cursor==int(rows[e,phase].sum())
            offsets[e,phase,1]=len(jobs)
    return dict(offsets=offsets,jobs=np.asarray(jobs,np.int32),logical=logical,input_columns=columns,total_rows=rows.sum(2))


def run():
    protocol=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==protocol['source_sha256']
    for x in protocol['references']:assert sha(x['path'])==x['sha256']
    original_seal=json.loads((TARGET/'completion.json').read_text())
    for x in original_seal['files']:assert sha(x['path'])==x['sha256']
    p=json.loads((TARGET/'protocol.frozen.json').read_text());z=load(TARGET/'inputs.frozen.npz');candidates=load(TARGET/'candidates.frozen.npz')
    old=load(TARGET/'native_arrays.npz');fit=json.loads((TARGET/'fit_report.json').read_text())
    ids=[x['id'] for x in fit['retained']];assert len(ids)==6
    with np.load(TRAIN) as archive:
        assert str(archive['split'])=='train' and archive['completed'].all()
        routes=archive['routes'][:,:,0].reshape(512,6).copy()
    assert np.array_equal(routes,z['routes']) and np.array_equal(routes,candidates['routes'])
    template=load(BASE/'stage-b/granite4tiny.M32.minimum_depth.plan.npz')['rows']
    gu_orbit=np.asarray([np.roll(template[0,0],shift) for shift in range(32)])
    down_orbit=np.asarray([np.roll(template[0,1],shift) for shift in range(32)])
    assert np.array_equal(gu_orbit,z['gu_rotations']) and np.array_equal(down_orbit,z['down_rotations'])
    build=OUT/'build';build.mkdir();binary=build/'native.so'
    subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(CPP),'-o',str(binary)],check=True)
    spec=importlib.util.spec_from_file_location('independent_tail_audit_frozen_backend',PYTHON)
    backend=importlib.util.module_from_spec(spec);sys.modules[spec.name]=backend;spec.loader.exec_module(backend)
    lib=ctypes.CDLL(str(binary));fn=lib.eval_ordinary_read
    i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    fn.argtypes=[i32,i32,i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64,i64,i64,ctypes.c_void_p,i64];fn.restype=None
    def evaluate(plan,rr,service,trace=False):
        rr=np.ascontiguousarray(rr,np.int32);cost=np.zeros(len(rr),np.int64);bounds=cost.copy();nlog=np.zeros(1,np.int64);log=np.zeros((6000 if trace else 1,8),np.int64)
        fn(plan['offsets'],plan['jobs'],plan['input_columns'],rr,64,1,len(plan['jobs']),len(rr),6,service.array(),cost,bounds,ctypes.c_void_p(log.ctypes.data) if trace else None,nlog)
        assert np.all(cost>=bounds)
        if trace:assert nlog[0]<=6000
        return cost,bounds,log[:nlog[0]].copy() if trace else None
    output={};geometry=[];records=[];traces=[]
    for ident in ids:
        rows=candidates[ident+'.rows'];labels=candidates[ident+'.labels']
        assert labels.shape==(64,2) and np.all((labels>=0)&(labels<32))
        assert np.array_equal(rows[:64,0],gu_orbit[labels[:,0]]) and np.array_equal(rows[:64,1],down_orbit[labels[:,1]])
        assert np.array_equal(rows[64],template[64]) and np.array_equal(rows.sum(2),np.asarray([[1024,1536]]*64+[[2048,1536]]))
        plan=own_plan(rows);saved=load(TARGET/'native'/(ident+'.plan.npz'))
        for key in ('offsets','jobs','input_columns','total_rows'):assert np.array_equal(plan[key],saved[key])
        occupied=set();perchipwaves=np.zeros(32,np.int64)
        for job,address in zip(plan['jobs'],saved['addresses']):
            chip,n,columns,pages,waves=map(int,job);ac,block,wl,span,ap,plane=map(int,address)
            assert (chip,waves,pages)==(ac,span,ap) and block==plane==0 and wl+span<=192
            for at in range(wl,wl+span):
                key=chip,at;assert key not in occupied;occupied.add(key)
            perchipwaves[chip]+=waves
        assert np.all(perchipwaves<=192)
        overlap=((rows[:64,0]>0)&(rows[:64,1]>0)).sum(1)
        geometry.append(dict(id=ident,exact_phase_orbits_and40row=True,onecopy=True,unchanged_shared=True,
            useful_bytes=sum(j['useful_bytes'] for j in plan['logical']),pages=int(plan['jobs'][:,3].sum()),waves=int(perchipwaves.sum()),
            allocated_blocks=32,perchip_waves=perchipwaves.tolist(),overlap_hist={str(int(k)):int(v) for k,v in zip(*np.unique(overlap,return_counts=True))}))
        for copies in (1,4):
            table=np.zeros((65,2,32),np.int64);wave_table=np.zeros_like(table)
            for j in plan['logical']:
                emitted=j['pages']*16384//j['columns']
                table[j['expert'],j['phase'],j['chip']]+=250+j['columns']*copies//2+28000*j['waves']+2*emitted
                wave_table[j['expert'],j['phase'],j['chip']]+=j['waves']
            work=table[routes].sum((1,2))+table[64].sum(0)
            gucoef,downcoef=(29102,28762) if copies==1 else (31406,29530)
            direct=gucoef*(rows[:64,0]>0)[routes].sum(1)+downcoef*(rows[:64,1]>0)[routes].sum(1)+table[64].sum(0)
            assert np.array_equal(work,direct)
            if copies==1:
                assert np.array_equal(work.max(1),candidates[ident+'.work'])
                assert np.array_equal(table[64].sum(0),candidates['shared_work'])
                assert np.array_equal(wave_table[64].sum(0),candidates['shared_waves'])
            output[f'input{copies}.{ident}.chip_work']=work
            for priority in (0,1):
                service=backend.Service(**dict(p['service'],input_channel_copies=copies,down_first=priority));backend.validate_plan(plan,65,service)
                cost,bounds,_=evaluate(plan,routes,service);label='DF' if priority else 'GF';key=f'input{copies}.{ident}.{label}'
                assert np.array_equal(cost,old[key+'.cost']) and np.array_equal(bounds,old[key+'.bound'])
                output[key+'.cost'],output[key+'.bound']=cost,bounds
                records.append(dict(id=ident,input_copies=copies,priority=label,cost=stats(cost)))
                if (copies,priority) in ((1,0),(4,1)):
                    for q in (0,511):
                        costq,bq,trace=evaluate(plan,routes[q:q+1],service,True)
                        py=backend.ReferenceGraph.build(plan,np.r_[routes[q],64],service).run()
                        assert costq[0]==cost[q]==py['cost'] and np.array_equal(trace,py['trace']) and np.array_equal(trace,old[key+f'.q{q}trace'])
                        output[key+f'.q{q}trace']=trace
                        traces.append(dict(id=key,query=q,nodes=len(trace),all_nodes_exact=True))
    selected=min((r for r in records if r['input_copies']==1 and r['priority']=='GF'),key=lambda r:(r['cost']['p99_ns'],r['cost']['mean_ns']))
    reported=json.loads((TARGET/'native_report.json').read_text());assert selected==reported['selected_train_tail_control']
    free=load(BASE/'g4-M32-coupled-cp/native_arrays.npz')['incumbent4.GF.cost'];best=output['input1.'+selected['id']+'.GF.cost'];diff=best-free
    gains=dict(mean_pct=100*(1-free.mean()/best.mean()),p99_pct=100*(1-np.percentile(free,99)/np.percentile(best,99)),
               query_free_improves=int((diff>0).sum()),query_ties=int((diff==0).sum()),query_free_regresses=int((diff<0).sum()),max_free_regression_ns=int(max(0,-diff.min())))
    assert abs(gains['mean_pct']-reported['free_vs_tail_control_mean_gain_pct'])<1e-12 and abs(gains['p99_pct']-reported['free_vs_tail_control_p99_gain_pct'])<1e-12
    np.savez_compressed(OUT/'recomputed_arrays.npz',**output)
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.frozen.json'),geometry=geometry,
        native_fresh_recomputed_costs=12288,native_trace_validation_calls=len(traces),python_full_traces=traces,
        total_python_trace_nodes=sum(x['nodes'] for x in traces),all_saved_costs_bounds_exact=True,selected_train_tail_control=selected,
        records=records,free_vs_selected_train_tail_control=gains,cpp_source_sha256=sha(CPP),binary_sha256=sha(binary),
        producer_generator_imported=False,new_capture_payload_read=False,refit=False,
        chronology=protocol['chronology'],limits='SupplementaryTRAIN finitework-proxy control only, notnewprimary oroptimalcyclicproof. Existingprospective3maps/analysis untouched.')
    dump(OUT/'audit.json',result)
    (OUT/'notes.md').write_text('Independent TRAIN audit: PASS\n\nAll6maps preserveexactphaseorbits (including40row),fullonecopyrows/shared, pages/waves andsingle-blockaddresses. All12288freshCPPcosts andbounds match savedresults;24newPythonfulltraces match every node.\n\nSupplementarycontrol frozeafternewpromptacquisition andbeforerootviewedfreshoutcomes, perparentattestation. It isnot theoriginalprospectiveprimary; no claimthatallteam memberswereunexposed. Thisauditorreadnonewcapture/prompt/routes.\n')
    for item in protocol['references']:assert sha(item['path'])==item['sha256']
    dump(OUT/'completion.json',dict(status='sealed_independent_PASS',utc=datetime.now(timezone.utc).isoformat(),
         files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(OUT.rglob('*')) if x.is_file() and not x.name.endswith('.log')]))
    print(json.dumps(dict(status='PASS',completion_sha256=sha(OUT/'completion.json'),selected=selected,gains=gains,python_traces=len(traces)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));args=parser.parse_args();{'freeze':freeze,'run':run}[args.mode]()
