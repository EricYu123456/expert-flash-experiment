#!/usr/bin/env python3
"""One fixed joint greedy policy replay with alias/consumed-row ledgers."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from research_20261103_topology_placement_replay import CPP, PYTHON, load_python

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
JOINT=BASE/'g4-joint-prefix-replica-screen';OUT=JOINT/'native-v1'
GU=BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix'
DOWN=BASE/'g4-down-paired-merge-screen'
B=BASE/'compact-coupled-solver/continuation/native/0107.npz'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def stats(x):return dict(mean_ns=float(np.mean(x)),p99_ns=float(np.percentile(x,99)))


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    files=[Path(__file__),JOINT/'protocol.frozen.json',JOINT/'work_arrays.npz',JOINT/'work_report.json',
           B,GU/'rowid_ledgers.json',GU/'native-v1/arrays.npz',DOWN/'native-graph14/arrays.npz',CPP,PYTHON,
           ROOT/'scripts/research_20261103_topology_placement_replay.py',ROOT/'scripts/research_20261102_whole_row_compiler.py']
    inputs=[]
    for path in files:
        item=dict(path=str(path),sha256=sha(path))
        if path.suffix in ('.cpp','.py'):
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        inputs.append(item)
    p=json.loads((JOINT/'protocol.frozen.json').read_text())
    assert json.loads((JOINT/'work_report.json').read_text())['gate_passed']
    dump(OUT/'protocol.frozen.json',dict(status='frozen_before_joint_native',utc=datetime.now(timezone.utc).isoformat(),
         source_sha256=sha(__file__),inputs=inputs,service=p['service'],fixed_GU=3,fixed_Down=14,queries=512,
         selection='Exactlychosen_u/chosen_v in sealedjointwork_arrays; CP32doesnotreplaceanybits. One8pathgreedypolicy, notnewfitting.',
         lowering='ComputedGUrowcounts may>1024; total_rows=actualcomputed. Separateownership arrays proveGU1024unique andDown1536unique. Sharedunchanged. Noexistingguardweakened.',
         resident='Readviews aliasprefixes in combinedstaticGU/Downimage, oneBFDrepacking. No perquerycoefficientmovement/newcopies. All commands oneblock.',
         dependencies='UnmodifiednativeGraph allcomputed/emittedGUoutputs->hostfullA8->originalprivate512Down; onecontextthroughchanneldrain.',
         checks='B512cost/LBbridge; B/jointq0,255,511fullPythonnode equality;3072routedviews eachGU/Down consumedIDs;all512actualwork/counters.',
         software_cost='Selector/hostscatter/extra mapping software0 optimistic sensitivity only. No unchargedNAND/output/input/commands.',
         gate='Ifmean<2% andP99<5%,stopselectorbench. Else reportrootbefore productionselector/strongzero-copy/samebudgetwholeGU+Down/partialRDA.',
         scope='TRAINL0only,nativecostmodel notdevice orquality. NoCP-selectednative/heldout/alllayers.'))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__))))


def run():
    proto=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==proto['source_sha256']
    for item in proto['inputs']:assert sha(item['path'])==item['sha256']
    assert not (OUT/'report.json').exists();t0=time.monotonic()
    w,b=load(JOINT/'work_arrays.npz'),load(B);rows,routes=w['rows_B'],w['routes']
    gu_ledger=json.loads((GU/'rowid_ledgers.json').read_text())[3]['experts']
    backend=load_python();s=backend.Service(**proto['service'])
    binary=OUT/'native.so';subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(CPP),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));call=lib.eval_ordinary_read
    p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    call.argtypes=[p32,p32,p32,p32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,p64,p64,p64,ctypes.c_void_p,p64]
    def evaluate(plan,rr,experts,trace=False):
        backend.validate_plan(plan,experts+1,s);rr=np.ascontiguousarray(rr,np.int32)
        result=np.zeros(len(rr),np.int64);bounds=result.copy();nlog=np.zeros(1,np.int64);log=np.zeros((5000 if trace else 1,8),np.int64)
        call(plan['offsets'],plan['jobs'],plan['input_columns'],rr,experts,1,len(plan['jobs']),len(rr),6,s.array(),result,bounds,
             ctypes.c_void_p(log.ctypes.data) if trace else None,nlog)
        assert np.all(result>=bounds)
        if trace:assert nlog[0]<=len(log)
        return result,bounds,log[:nlog[0]].copy() if trace else None
    bplan={k:b[k] for k in ('offsets','jobs','input_columns','total_rows')}
    baseline,baselineLB,_=evaluate(bplan,routes,64)
    assert np.array_equal(baseline,b['cost']) and np.array_equal(baselineLB,b['lower_bound'])
    resident={}
    for e in range(65):
        for phase in range(2):
            for j in range(*w['resident_offsets'][e,phase]):
                resident[e,phase,int(w['resident_jobs'][j,0])]=w['resident_addresses'][j]
    down_images=[]
    for e in range(64):
        starts=np.r_[0,np.cumsum(rows[e,1])]
        original={c:list(range(int(starts[c]),int(starts[c+1]))) for c in range(16)}
        image={c:rr.copy() for c,rr in original.items()}
        for src,dest in w['Down_edges'][e]:image[int(dest)]+=original[int(src)]
        assert sum(map(len,image.values()))==2304
        down_images.append((original,image))
    owner_gu=np.zeros((512,6,1024,2),np.int16);owner_down=np.zeros((512,6,1536,2),np.int16)
    compact=np.zeros((512,7,2,16),np.int64);cost=np.zeros(512,np.int64);lbs=cost.copy();work=cost.copy()
    counters=np.zeros((512,2,7),np.int64);trace_checks=[];summaries=[]
    def lower(local,selected):
        plan=backend.lower_rows(local[:,0],local[:,1],1536,[512]*6+[1024],s)
        assert np.array_equal(plan['total_rows'],local.sum(2))
        aliases=np.zeros((len(plan['jobs']),6),np.int32)
        for ordinal,e in enumerate(selected):
            for phase in range(2):
                for j in range(*plan['offsets'][ordinal,phase]):
                    c,nr,nc,pages,waves=map(int,plan['jobs'][j]);addr=resident[int(e),phase,c]
                    assert pages<=addr[4] and waves<=addr[3] and addr[2]+waves<=192
                    aliases[j]=addr
        return plan,aliases
    for q,route in enumerate(routes):
        selected=np.r_[route,64];local=rows[selected].copy();entries=[]
        for ordinal,e in enumerate(route):
            e=int(e);entry=gu_ledger[e];u=w['chosen_u'][q,ordinal];v=w['chosen_v'][q,ordinal]
            image={int(c):rr for c,rr in entry['resident_prefix_image'].items()}
            original={int(c):rr for c,rr in entry['B_original_rows'].items()}
            owners={}
            for i,c in enumerate(entry['sources']):
                nr=128 if u[i] else int(rows[e,0,c]);local[ordinal,0,c]=nr
                for pos,rid in enumerate(image[c][:nr]):assert rid not in owners;owners[rid]=(c,pos)
            for i,c in enumerate(entry['low_destinations']):
                local[ordinal,0,c]=0 if u[i] else 40
                if not u[i]:
                    for pos,rid in enumerate(original[c]):assert rid not in owners;owners[rid]=(c,pos)
            hd=entry['high_destination'];nr=int(rows[e,0,hd]) if not u[7] else entry['high_prefix_rows'] if not u.all() else 0
            local[ordinal,0,hd]=nr
            for pos,rid in enumerate(original[hd][:nr]):
                if rid not in owners:owners[rid]=(hd,pos)
            assert sorted(owners)==list(range(1024))
            for rid,(c,pos) in owners.items():assert pos<local[ordinal,0,c] and image[c][pos]==rid
            owner_gu[q,ordinal]=np.asarray([owners[i] for i in range(1024)],np.int16)
            down_original,down_image=down_images[e]
            for i,(src,dest) in enumerate(w['Down_edges'][e]):
                if v[i]:local[ordinal,1,src]=0;local[ordinal,1,dest]=256
            down_owners={}
            for c,nr in enumerate(local[ordinal,1]):
                for pos,rid in enumerate(down_image[c][:int(nr)]):assert rid not in down_owners;down_owners[rid]=(c,pos)
            assert sorted(down_owners)==list(range(1536)) and local[ordinal,1].sum()==1536
            owner_down[q,ordinal]=np.asarray([down_owners[i] for i in range(1536)],np.int16)
            entries.append(dict(expert=e,GU_computed_rows=int(local[ordinal,0].sum()),GU_consumed_unique=1024,
                                Down_computed_rows=1536,Down_consumed_unique=1536))
        assert np.array_equal(local[6],rows[64]);plan,aliases=lower(local,selected)
        values,lower_bound,_=evaluate(plan,np.arange(6,dtype=np.int32)[None],6)
        cost[q],lbs[q],compact[q]=values[0],lower_bound[0],local
        occupancy=np.zeros(16,np.int64)
        for ordinal in range(7):
            for phase in range(2):
                for c,nr,nc,pages,waves in plan['jobs'][slice(*plan['offsets'][ordinal,phase])]:
                    c,nr,nc,pages,waves=map(int,(c,nr,nc,pages,waves));emit=pages*16384//nc
                    occupancy[c]+=250+(nc+1)//2+waves*28000+emit*2
                    counters[q,phase]+=[1,nr*nc,pages,waves,nc,emit*4,(nc+1)//2+emit*2]
        work[q]=occupancy.max();assert work[q]==w['chosen_work'][q]
        summaries.append(dict(query=q,selected=route.tolist(),experts=entries))
        if q in (0,255,511):
            bp,ba=lower(rows[selected],selected)
            for tag,pp,aa,lr,expected in [('baseline',bp,ba,rows[selected],baseline[q]),('joint',plan,aliases,local,cost[q])]:
                values,bound,tr=evaluate(pp,np.arange(6,dtype=np.int32)[None],6,True)
                py=backend.ReferenceGraph.build(pp,np.arange(7),s).run()
                assert values[0]==expected==py['cost'] and np.array_equal(py['trace'],tr)
                np.savez_compressed(OUT/f'q{q}.{tag}.npz',rows=lr,offsets=pp['offsets'],jobs=pp['jobs'],input_columns=pp['input_columns'],
                                    total_rows=pp['total_rows'],aliases=aa,trace=tr,cost=values,lower_bound=bound)
                trace_checks.append(dict(query=q,case=tag,nodes=len(tr),cost_ns=int(expected)))
    gain=dict(mean_pct=100*(1-cost.mean()/baseline.mean()),p99_pct=100*(1-np.percentile(cost,99)/np.percentile(baseline,99)))
    gu_single=load(GU/'native-v1/arrays.npz')['greedy'];down_single=load(DOWN/'native-graph14/arrays.npz')['cost']
    np.savez_compressed(OUT/'arrays.npz',routes=routes,baseline=baseline,baseline_LB=baselineLB,cost=cost,lower_bound=lbs,
                        actual_work=work,compact_rows=compact,counters=counters,GU_single=gu_single,Down_single=down_single)
    np.savez_compressed(OUT/'row_owners.npz',GU=owner_gu,Down=owner_down,routes=routes)
    dump(OUT/'query_row_summary.json',summaries)
    result=dict(status='zero_software_joint_native_model_sensitivity',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),
                cpp_sha256=sha(CPP),binary_sha256=sha(binary),baseline=stats(baseline),GU_single=stats(gu_single),Down_single=stats(down_single),
                joint=stats(cost),improvement=gain,gate_passed=bool(gain['mean_pct']>=2 or gain['p99_pct']>=5),
                paired=dict(improved=int(np.sum(cost<baseline)),equal=int(np.sum(cost==baseline)),worsened=int(np.sum(cost>baseline)),
                            max_saving_ns=int((baseline-cost).max()),max_regression_ns=int((cost-baseline).max())),
                checks=dict(B_costs_exact=512,B_bounds_exact=512,GU_ownership_checks=3072,Down_ownership_checks=3072,actual_work_checks=512,
                            representative_traces=trace_checks),counter_columns=['jobs','computed_coefficient_bytes','pages','waves','fullinputbytes','emittedoutputbytes','channel_ns'],
                mean_counters_by_phase=counters.mean(0).tolist(),seconds=time.monotonic()-t0,
                limits='OnlyfixedGU3+Down14 jointgreedy TRAIN512. Zero selector/scatter/mappingsoftware is optimistic. Not sumofsinglebranch gains; no newcontext/MAC/FIFO/inputformat. RowIDsymbolic notnumericABI proof. Strongstatic/samebudgetcopies andproductionselector pending ifgatepasses; no heldout/alllayer.')
    dump(OUT/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
