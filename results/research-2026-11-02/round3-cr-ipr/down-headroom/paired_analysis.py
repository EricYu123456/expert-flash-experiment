#!/usr/bin/env python3
"""Descriptive paired TRAIN analysis; selection remains the frozen screen choice."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE/'final-run'
OLD=HERE.parent/'full-ffn/asymmetric-down/TRAIN512/service_arrays.npz'
LAYOUTS=['candidate','RR','g8_fixed']
INCUMBENT={'candidate':'original_g4_shared16','RR':'original_g4_shared16','g8_fixed':'original_g2_shared16'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def stats(x):
    a=np.asarray(x,dtype=np.float64).ravel()
    return dict(n=int(a.size),mean_ns=float(a.mean()),median_ns=float(np.median(a)),p95_ns=float(np.percentile(a,95)),p99_ns=float(np.percentile(a,99)),max_ns=float(a.max()))
def compare(a,b):
    assert a.shape==b.shape==(3,512) and np.all(a>=0) and np.all(b>=0)
    d=a-b;sa=a.sum(axis=0);sb=b.sum(axis=0)
    return dict(candidate=stats(a),incumbent=stats(b),mean_gain=float(1-a.mean()/b.mean()),
                p99_gain=float(1-np.percentile(a,99)/np.percentile(b,99)),
                faster=int(np.count_nonzero(d<0)),equal=int(np.count_nonzero(d==0)),slower=int(np.count_nonzero(d>0)),
                delta_ns=stats(d),worst_regression_ns=int(d.max()),best_improvement_ns=int(-d.min()),
                per_layer=[dict(layer=l,mean_gain=float(1-a[l].mean()/b[l].mean()),candidate=stats(a[l]),incumbent=stats(b[l])) for l in range(3)],
                three_layer_query_sum=dict(candidate=stats(sa),incumbent=stats(sb),mean_gain=float(1-sa.mean()/sb.mean()),p99_gain=float(1-np.percentile(sa,99)/np.percentile(sb,99)),faster=int(np.count_nonzero(sa<sb)),equal=int(np.count_nonzero(sa==sb)),slower=int(np.count_nonzero(sa>sb))))

def freeze():
    p=RUN/'analysis.plan.frozen.json'
    assert not p.exists()
    dump(p,dict(status='frozen_before_reading_full512_outcomes',utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=sha(__file__),selection_sha256=sha(RUN/'selection.frozen.json'),
                protocol_sha256=sha(RUN/'protocol.frozen.json'),old_incumbent_arrays_sha256=sha(OLD),
                primary='For each GU, use exact selected best and incumbent priority from frozen TRAIN64 selection. Repeat same priorities for input4; no TRAIN512 reselection.',
                secondary='Report fixed-map GF-to-GF and DF-to-DF pairs for input1/4, and compare selected candidate versus selected RR/g8 under identical screen-fixed priorities.',
                checks='All 24 arrays must contain 1536 completed PB-valid events and FIFO<=128; exact incumbent cost/FIFO bridge to sealed asymmetricDown arrays; all traces are retained for independent audit.',
                statistics='TRAIN descriptive arithmetic mean and percentile99, event and aligned three-layer query sum; faster/equal/slower and worst paired regression. No heldout claim, no selection adjustment, no new fit.'))

def run():
    plan=read(RUN/'analysis.plan.frozen.json');assert sha(__file__)==plan['source_sha256']
    assert sha(RUN/'selection.frozen.json')==plan['selection_sha256']
    assert sha(OLD)==plan['old_incumbent_arrays_sha256']
    sel=read(RUN/'selection.frozen.json');raw=read(RUN/'TRAIN512/report.json')
    z=np.load(RUN/'TRAIN512/service_arrays.npz');old=np.load(OLD)
    assert raw['status']=='PASS' and len(raw['records'])==24
    validity=[]
    for r in raw['records']:
        a=z[r['name']+'.cost'];fifo=z[r['name']+'.fifo']
        good=a.shape==(3,512) and bool(np.all(a>=0)) and bool(np.all(fifo<=128))
        assert good,r['name']
        validity.append(dict(name=r['name'],events=int(a.size),min_ns=int(a.min()),peak_FIFO_bytes=int(fifo.max()),status='PASS'))
    bridge=[]
    for g in LAYOUTS:
        oldmenu='g2_reused_TRAIN_labels' if g=='g8_fixed' else 'g4_reused_TRAIN_labels'
        for cp in [1,4]:
            for pr in ['GF','DF']:
                for suffix in ['cost','fifo']:
                    nk=f'{g}.{INCUMBENT[g]}.input{cp}.{pr}.{suffix}'
                    ok=f'{g}.{oldmenu}.input{cp}.{pr}.{suffix}'
                    assert np.array_equal(z[nk],old[ok]),nk
                    bridge.append(dict(new=nk,old=ok,status='EXACT'))
    primary={};same_priority=[];fixed={}
    for g in LAYOUTS:
        best=sel['comparisons'][g]['best'];anchor=sel['comparisons'][g]['incumbent']
        fixed[g]=(best,anchor)
        for cp in [1,4]:
            akey=best.replace('.input1.',f'.input{cp}.');bkey=anchor.replace('.input1.',f'.input{cp}.')
            primary[f'{g}.input{cp}']=dict(candidate_key=akey,incumbent_key=bkey,**compare(z[akey+'.cost'],z[bkey+'.cost']))
            for pr in ['GF','DF']:
                akey=f'{g}.{sel["winners"][g]}.input{cp}.{pr}';bkey=f'{g}.{INCUMBENT[g]}.input{cp}.{pr}'
                same_priority.append(dict(GU=g,input_copies=cp,priority=pr,candidate_key=akey,incumbent_key=bkey,**compare(z[akey+'.cost'],z[bkey+'.cost'])))
    cross=[]
    for cp in [1,4]:
        akey=fixed['candidate'][0].replace('.input1.',f'.input{cp}.')
        for g in ['RR','g8_fixed']:
            bkey=fixed[g][0].replace('.input1.',f'.input{cp}.')
            cross.append(dict(input_copies=cp,other_GU=g,candidate_key=akey,incumbent_key=bkey,**compare(z[akey+'.cost'],z[bkey+'.cost'])))
    screen=read(RUN/'screen/report.json');static=[]
    for g in LAYOUTS:
        for m in ['dedicated_shared4_learned','dedicated_shared4_static','all64_shared4_learned_to0','all64_shared4_learned_to1','all64_shared4_learned_to2','all64_shared4_static_to0','all64_shared4_static_to1','all64_shared4_static_to2']:
            rr=[r for r in screen['records'] if r['GU']==g and r['menu']==m]
            if rr:
                r=min(rr,key=lambda x:(x['mean_ns'],x['priority']))
                static.append(dict(GU=g,menu=m,screen_only=True,best_priority=r['priority'],mean_ns=r['mean_ns'],p99_ns=r['cost_ns']['p99'],completed=r['completed'],valid_128B=r['valid_FIFO_capacities']['128']))
    paths=[Path(__file__),RUN/'analysis.plan.frozen.json',RUN/'protocol.frozen.json',RUN/'selection.frozen.json',RUN/'TRAIN512/report.json',RUN/'TRAIN512/service_arrays.npz',OLD,HERE/'two-block-addendum/physical_capacity_audit.json']
    result=dict(status='PASS',inputs_sha256={str(p):sha(p) for p in paths},events=24*1536,configurations=24,
                validity=validity,incumbent_bridge=bridge,primary=primary,same_priority=same_priority,cross_GU=cross,
                static_and_capacity_controls_TRAIN64=static,
                interpretation='Standard fixed resource partition and reoptimized generic whole-row baseline, not a new mapping algorithm or measured hardware speedup. TRAIN fitting and descriptive TRAIN replay only. Conditional PB/CR service envelope remains inherited.',
                capacity='Common two blocks/chip, 32 blocks/layer. all64 22/21/21 routed groups plus dedicated sharedg4 is legal with one whole 12-page private job in second block on four chips; fresh cold and command counts unchanged. No cross-block job. Exact one-copy row coverage.',
                source_gate_erratum='Preserved parent naming gate error; final continuation re-evaluated unchanged numerical >1% gate. No row map, winner, threshold, or event result changed.')
    dump(RUN/'paired_report.json',result)
    lines=['# Fixed shared-Down partition: TRAIN512 causal replay','',
           'All 24 configurations completed 1,536 layer/query events with finite PB and a 128-byte output FIFO. The 12 incumbent profiles reproduce the sealed asymmetric-Down cost and FIFO arrays exactly. Selection and priorities remain the frozen TRAIN64 choices.','',
           '| GU | input copies | baseline mean µs | partition mean µs | mean reduction | event P99 change | 3-layer P99 change |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for g in LAYOUTS:
        for cp in [1,4]:
            r=primary[f'{g}.input{cp}']
            lines.append(f'| {g} | {cp} | {r["incumbent"]["mean_ns"]/1000:.3f} | {r["candidate"]["mean_ns"]/1000:.3f} | {100*r["mean_gain"]:.3f}% | {100*r["p99_gain"]:.3f}% | {100*r["three_layer_query_sum"]["p99_gain"]:.3f}% |')
    lines += ['', 'Positive percentages mean lower latency. The primary pair uses screen-selected DF for the partition and GF for the incumbent; same-priority controls are in paired_report.json. Per-query regressions and tail changes must accompany mean gains.', '',
              'The map reserves chips 12–15 for shared Down. Routed Down uses three disjoint groups of four chips with 22/21/21 experts; every routed job has 384 complete output rows, 12 pages, one cold start and three CR waves. Shared jobs have 384 rows, 24 pages and six waves. GU kernels are unchanged. A whole private job may occupy block 1 under the common two-block budget; no job crosses blocks or incurs an extra cold start.', '',
              'This is a stronger standard resource-partition baseline supported by the existing generic whole-row compiler. It does not establish algorithmic novelty. All fitting and replay use TRAIN, conditional PB/CR assumptions are inherited, and no heldout or measured hardware performance is claimed.', '',
              'TRAIN64 static grouping and 63+1 one-block variants remain separate screen controls; they were not selected again using TRAIN512. The inherited gate-name metadata error is documented in gate_metadata_erratum.json.']
    (RUN/'paired_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',primary={k:dict(mean_gain=v['mean_gain'],p99_gain=v['p99_gain']) for k,v in primary.items()})),flush=True)

if __name__=='__main__':
    import sys
    freeze() if sys.argv[1]=='freeze' else run()
