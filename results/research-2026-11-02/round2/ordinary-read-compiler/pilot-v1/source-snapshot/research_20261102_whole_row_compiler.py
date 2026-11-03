#!/usr/bin/env python3
"""Portable whole-row geometry certificates for ordinary-read IFP.

The optimizer is standard integer covering/knapsack. Its purpose is to expose
hardware work/depth constraints and emit executable row bands, not to claim a
new DP or a globally optimal routing-aware placement/schedule.
"""
import argparse
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from math import gcd
from pathlib import Path
import json
import numpy as np


def ceildiv(a,b):return (a+b-1)//b


@dataclass(frozen=True)
class Capability:
    chips:int=16
    planes:int=4
    page_bytes:int=16384
    block_wordlines:int=192
    weight_bytes:int=1

    @property
    def wave_bytes(self):return self.planes*self.page_bytes

    def validate(self):
        assert min(asdict(self).values())>0


def intervals(waves,row_bytes,wave_bytes):
    w=np.asarray(waves,dtype=np.int64)
    lo=np.where(w>0,(w-1)*wave_bytes//row_bytes+1,0)
    hi=w*wave_bytes//row_bytes
    return lo,hi


def materialize(total_rows,columns,waves,cap):
    """Complete rows in the given ceil intervals, then emit logical row bands."""
    cap.validate();row_bytes=columns*cap.weight_bytes;w=np.asarray(waves,dtype=np.int64)
    assert len(w)==cap.chips and np.all(w>=0)
    lo,hi=intervals(w,row_bytes,cap.wave_bytes)
    if np.any(lo>hi) or int(lo.sum())>total_rows or int(hi.sum())<total_rows:return None
    rows=lo.copy();remaining=total_rows-int(rows.sum())
    while remaining:
        eligible=np.flatnonzero(rows<hi)
        assert len(eligible)
        quantum=max(1,remaining//len(eligible))
        take=np.minimum(hi[eligible]-rows[eligible],quantum)
        if int(take.sum())>remaining:
            take[:]=0;take[:remaining]=1
        rows[eligible]+=take;remaining-=int(take.sum())
    assert int(rows.sum())==total_rows
    assert np.array_equal((rows*row_bytes+cap.wave_bytes-1)//cap.wave_bytes,w)
    jobs=[];cursor=0
    for chip,r in enumerate(rows):
        if r:
            useful=int(r)*row_bytes;pages=ceildiv(useful,cap.page_bytes)
            jobs.append(dict(chip=chip,row_start=cursor,rows=int(r),column_start=0,columns=columns,
                             useful_bytes=useful,pages=pages,waves=int(w[chip]),block_commands=ceildiv(int(w[chip]),cap.block_wordlines)))
        cursor+=int(r)
    assert cursor==total_rows
    return dict(rows=rows.tolist(),waves=w.tolist(),jobs=jobs,total_waves=int(w.sum()),max_job_waves=int(w.max()),positive_jobs=len(jobs),
                single_block_jobs=all(j['block_commands']==1 for j in jobs))


def minimum_work(total_rows,columns,depth,cap):
    """Exact minimum aggregate waves with <=chips jobs of depth <=depth.

    DP[n,s] is maximum row capacity using n positive jobs and s waves. Only
    realizable whole-row ceil intervals are admitted. At the minimum feasible
    s over n, its parent intervals necessarily contain total_rows: otherwise
    removing rows from their lower endpoints would yield a cheaper feasible
    allocation, contradicting minimality. This is also checked on the witness.
    """
    cap.validate();assert total_rows>0 and columns>0 and depth>0
    rb=columns*cap.weight_bytes;b=cap.wave_bytes;m=min(cap.chips,total_rows)
    if cap.chips*(depth*b//rb)<total_rows:return None
    equal=np.full(m,total_rows//m,dtype=np.int64);equal[:total_rows%m]+=1
    assert np.all((equal*rb+b-1)//b<=depth)
    budget=int(((equal*rb+b-1)//b).sum())
    dp=np.full((m+1,budget+1),-1,dtype=np.int64);parent=np.full((m+1,budget+1),-1,dtype=np.int32);dp[0,0]=0
    valid=[]
    for w in range(1,min(depth,budget)+1):
        lo=(w-1)*b//rb+1;hi=w*b//rb
        if lo<=hi:valid.append((w,hi))
    for n in range(1,m+1):
        for w,hi in valid:
            old=dp[n-1,:budget+1-w];candidate=old+hi;target=dp[n,w:]
            improve=(old>=0)&(candidate>target)
            target[improve]=candidate[improve];parent[n,w:][improve]=w
    feasible=np.argwhere(dp>=total_rows)
    assert len(feasible)
    work=int(feasible[:,1].min())
    # Maximize positive jobs only as a deterministic tie-break. No inference
    # that it minimizes active-query imbalance or native FFN service follows.
    n=int(feasible[feasible[:,1]==work,0].max());chosen=n;left=work;waves=[]
    while n:
        w=int(parent[n,left]);assert w>0;waves.append(w);left-=w;n-=1
    assert left==0
    waves=sorted(waves,reverse=True)+[0]*(cap.chips-chosen)
    witness=materialize(total_rows,columns,waves,cap);assert witness is not None
    assert witness['total_waves']==work and witness['max_job_waves']<=depth
    return dict(optimal_total_waves=work,requested_max_job_waves=depth,witness=witness,
                proof='Maximum-capacity integer DP over every legal wave/job count; materialized parent ceil intervals cover the requested rows.')


def certificate(total_rows,columns,cap):
    rb=columns*cap.weight_bytes;b=cap.wave_bytes
    minimum_depth=ceildiv(ceildiv(total_rows,cap.chips)*rb,b)
    useful_work_floor=ceildiv(total_rows*rb,b);period=rb//gcd(rb,b)
    equal=np.full(cap.chips,total_rows//cap.chips,dtype=np.int64);equal[:total_rows%cap.chips]+=1
    equal_w=(equal*rb+b-1)//b;baseline=materialize(total_rows,columns,equal_w,cap)
    # baseline materialization may redistribute equal rows inside the same
    # legal wave profile. Keep the actual equal rows explicitly for comparison.
    baseline['original_equal_rows']=equal.tolist()
    balanced=None
    for work in range(useful_work_floor,int(equal_w.sum())+1):
        w=np.full(cap.chips,work//cap.chips,dtype=np.int64);w[:work%cap.chips]+=1
        balanced=materialize(total_rows,columns,w,cap)
        if balanced is not None:break
    # For R<M or a row larger than one wave, a globally balanced wave vector
    # need not exist (e.g. R=1,C=3,B=2,M=4). It is an optional control, not a
    # prerequisite for a legal complete-row geometry.
    exact_fill=total_rows*rb%b==0
    arithmetic=dict(row_bytes=rb,wave_bytes=b,wave_period=period,exact_fill=exact_fill,
                    minimum_job_depth=minimum_depth,useful_aggregate_wave_floor=useful_work_floor)
    depths={minimum_depth}
    if exact_fill:
        assert useful_work_floor%period==0
        units=useful_work_floor//period
        depth_at_floor=period*ceildiv(units,cap.chips)
        u=np.full(cap.chips,units//cap.chips,dtype=np.int64);u[:units%cap.chips]+=1
        floor_witness=materialize(total_rows,columns,u*period,cap);assert floor_witness is not None
        assert floor_witness['total_waves']==useful_work_floor and floor_witness['max_job_waves']==depth_at_floor
        arithmetic.update(exact_work_minimum_job_depth=depth_at_floor,exact_work_witness=floor_witness,
                          lemma='At exact pooled fill, sum of nonnegative job padding is zero; every job is fully packed. Thus its waves are divisible by row_bytes/gcd(row_bytes,wave_bytes). Distributing integer period units proves the stated min-max depth.')
        depths.add(depth_at_floor)
    constrained=[minimum_work(total_rows,columns,d,cap) for d in sorted(depths)]
    return dict(rows=total_rows,columns=columns,arithmetic=arithmetic,equal_wave_profile=baseline,balanced_wave_profile=balanced,depth_constrained_certificates=constrained,
                scope='Complete output rows and coefficient bytes only. No output-column partials, quantization change, replicas, CR-read, dynamic relocation or native scheduling optimality.')


SHAPES={
    'granite3':(1024,512),'olmoe':(2048,1024),'deepseek':(2048,1408),'qwen3':(2048,768),
    'mixtral':(4096,14336),'granite4tiny':(1536,512),'qwen38':(2560,640),
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rows',type=int);ap.add_argument('--columns',type=int)
    ap.add_argument('--chips',type=int,default=16);ap.add_argument('--planes',type=int,default=4);ap.add_argument('--page-bytes',type=int,default=16384)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/research-2026-11-02/round2/whole-row-compiler/certificates.json')
    args=ap.parse_args();cap=Capability(chips=args.chips,planes=args.planes,page_bytes=args.page_bytes)
    records=[]
    if args.rows is not None or args.columns is not None:
        assert args.rows and args.columns;records=[dict(model='specified',phase='projection',**certificate(args.rows,args.columns,cap))]
    else:
        for name,(hidden,intermediate) in SHAPES.items():
            for phase,r,c in (('gate_up',2*intermediate,hidden),('down',hidden,intermediate)):
                rec=dict(model=name,phase=phase,**certificate(r,c,cap));records.append(rec)
                a=rec['arithmetic'];balanced=rec['balanced_wave_profile'];print(name,phase,'period',a['wave_period'],'depth',a['minimum_job_depth'],a.get('exact_work_minimum_job_depth'),'equal/balanced',rec['equal_wave_profile']['total_waves'],None if balanced is None else balanced['total_waves'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import hashlib
    doc=dict(status='complete hardware-parametric whole-row work/depth certificates',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),capability=asdict(cap),records=records,
             method='Standard integer covering DP and elementary divisibility lemma; new-algorithm or publication-priority claims are not made.',
             numerical_contract='If native GEMV returns an opaque row-separable function of the original full row, full input, original dimensions and row/expert metadata, moving whole rows and scattering their outputs preserves it. No raw-I32 partial-output contract is needed.',
             limits=['This generates logical row-band geometry, not a calibrated full-model time or native PE numerical implementation.',
                     'Coefficient-only row bytes exclude quantization/scale/address metadata; a backend must account for those separately or adjust physical footprint.',
                     'Jobs larger than one block are explicitly marked and require command splitting and supported continuation/host combination before use.',
                     'Balanced wave profiles and maximum-capacity parents are different controls; scalar Pareto dominance does not establish query-schedule dominance.',
                     'A globally balanced wave profile may not exist. It is then null; the exact depth-constrained integer DP still returns a legal witness.'])
    args.output.write_text(json.dumps(doc,indent=2)+'\n')


if __name__=='__main__':main()
