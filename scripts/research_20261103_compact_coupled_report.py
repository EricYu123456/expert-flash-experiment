#!/usr/bin/env python3
"""Independent scalar/interval checks of the compact solver's saved artifacts."""
import gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R/'results/research-2026-11-02/round2/compact-coupled-solver'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def write(p,z):Path(p).write_text(json.dumps(z,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
    first=rd(D/'summary.json');second=rd(D/'continuation/summary.json');records=first['candidates']+second['candidates'];base=records[0]
    frozen=load(D/'input.frozen.npz');routes=frozen['routes'];checks=[];nodes=0;inputs={}
    for rec in records:
        assert sha(rec['arrays'])==rec['arrays_sha256'] and sha(rec['physical'])==rec['physical_sha256']
        inputs[rec['arrays']]=rec['arrays_sha256'];inputs[rec['physical']]=rec['physical_sha256']
        z=load(rec['arrays']);rows=z['rows'];jobs=z['jobs'];offsets=z['offsets'];assert np.array_equal(z['routes'],routes) and np.array_equal(rows[64],frozen['rows'][64])
        X=(rows[:64,0]>=82)&(rows[:64,1]>0);O=(rows[:64,0]>=82)&(rows[:64,1]==0)
        assert np.all(X.sum(1)==5) and np.all(O.sum(1)==4) and not np.any(X&O)
        own=np.zeros((65,16),dtype=np.int64);waves=np.zeros_like(own);coverage=defaultdict(list)
        with gzip.open(rec['physical'],'rt') as f:phy=json.load(f)
        occ=defaultdict(list)
        for e in range(65):
            for ph in range(2):
                lo,hi=map(int,offsets[e,ph]);cursor=0;C=1536 if ph==0 else (512 if e<64 else 1024)
                for i in range(lo,hi):
                    c,n,c0,p,w=map(int,jobs[i]);a=phy['addresses'][i];l=phy['logical'][i]
                    assert c0==C and p==(n*C+16383)//16384 and w==(p+3)//4 and 0<w<=192
                    assert n==rows[e,ph,c] and (l['expert'],l['phase'],l['row_start'],l['rows'],l['columns'])==(e,ph,cursor,n,C)
                    assert (a['job'],a['chip'],a['pages'],a['reserved_wordlines'],a['first_plane'])==(i,c,p,w,0)
                    assert 0<=a['wordline']<a['wordline']+w<=192
                    occ[c,a['block']].append((a['wordline'],a['wordline']+w));cursor+=n
                    own[e,c]+=250+(C+1)//2+28000*w+((p*16384//C)*4+1)//2;waves[e,c]+=w
                assert cursor==int(rows[e,ph].sum())==int(z['total_rows'][e,ph])
        for intervals in occ.values():
            intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        assert np.array_equal(own[:64],57864+28086*X-676*O)
        work=(own[routes].sum(1)+own[64]).max(1);peak=(waves[routes].sum(1)+waves[64]).max(1)
        assert int(work.sum())==rec['metric']['weighted_sum_ns'] and np.all(z['cost']>=work+2000)
        assert float(z['cost'].mean())==rec['native_mean_ns'] and float(np.percentile(z['cost'],99))==rec['native_p99_ns']
        assert sum(int(n*C) for _,n,C,_,_ in jobs)==155713536 and int(jobs[:,3].sum())==9760 and int(jobs[:,4].sum())==2440
        assert len(jobs)==1824 and phy['chip_blocks']==[1]*16 and phy['allocated_lsb_block_bytes']==201326592
        for q in [0,255,511]:
            tr=z[f'trace_q{q}'];assert tr.shape[1]==8 and np.all(tr[:,3]-tr[:,2]==tr[:,1]);assert tr[:,3].max()==z['cost'][q];nodes+=len(tr)
            for resource in np.unique(tr[:,0]):
                intervals=sorted((int(t[2]),int(t[3])) for t in tr if t[0]==resource and t[1]>0)
                assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
            expected=[]
            for ph in range(2):
                for ei,e in enumerate(list(routes[q])+[64]):
                    lo,hi=offsets[e,ph];expected.extend((ei,ph,jobs[i]) for i in range(lo,hi))
            owners=defaultdict(list)
            for j,(ei,ph,job) in enumerate(expected):
                t=tr[tr[:,7]==j];c,n,C,p,w=map(int,job)
                assert t.shape==(4,8) and np.all(t[:,4]==ph) and np.all(t[:,5]==ei) and np.all(t[:,6]==c)
                assert t[:,0].tolist()==[1,3+c%8,11+c,3+c%8]
                assert t[:,1].tolist()==[250,(C+1)//2,28000*w,((p*16384//C)*4+1)//2]
                assert np.all(t[:-1,3]<=t[1:,2]);owners[c].append((int(t[0,2]),int(t[-1,3])))
                if ph==1:
                    gu=tr[(tr[:,0]==2)&(tr[:,4]==0)&(tr[:,5]==ei)]
                    assert t[0,2]>=gu[:,3].max()+2000
            for intervals in owners.values():
                intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        checks.append(dict(arrays=rec['arrays'],all512cost_statistics_exact=True,scalar_complete_trace_checks=3,physical_geometry_and_same_capacity=True))
    best=min(records,key=lambda z:z['native_mean_ns']);workbest=min(records,key=lambda z:z['metric']['mean_ns'])
    result=dict(status='passed',scope='G4L0 actualW8 TRAIN512, inputcopy1, frozen independent_paired shared geometry;no heldout.',
      evidence_inputs_sha256=inputs,checks=checks,recorded_masks=len(records),distinct_masks=len(set(z['mask_sha256'] for z in records)),native_costs_checked=len(records)*512,
      full_scalar_traces_checked=len(records)*3,trace_nodes_checked=nodes,full_reference_trace_comparisons=len(records)*3,
      baseline_native_mean_ns=base['native_mean_ns'],baseline_native_p99_ns=base['native_p99_ns'],best_native=best,best_work=workbest,
      best_native_mean_gain_percent=100*(1-best['native_mean_ns']/base['native_mean_ns']),best_native_p99_gain_percent=100*(1-best['native_p99_ns']/base['native_p99_ns']),
      solver_seconds=second['total_solver_seconds'],peak19_status=first['solver_reports'][0]['status'],peak20_status='SAT;CP-SAT reports OPTIMAL for feasibility,not a latency optimum',
      mean_work_status=second['solver']['status'],mean_work_bound_ns=2*second['solver']['best_bound_half_ns']/512,
      interpretation='Compact standard cardinality/packing formulation produces feasible masks but only a small TRAIN mean gain,not a demonstrated escape from the heuristic barrier. No further search/heldout justified here.',
      limitations=['19-wave UNKNOWN is not infeasibility. Neither native nor surrogate global optimum is certified.','Peak20 feasibility masks are slower in mean than incumbent;optimizing worst NAND waves is not mean-native optimization.','All records use identical one-layer payload155713536B,9760pages,2440waves,1824jobs and16allocatedLSBblocks192MiB;full-model per-chip residency remains outside scope.','Native replay is the frozen software cost model,not manufactured-device timing or numerical output ABI validation.','First mean stage stopped at64callbacks after8.20sec;separate unchanged-input80sec continuation is explicitly archived. Epigraph callback objectives may contain slack;mask work was independently recomputed.'])
    write(D/'notes.json',result)
    write(D/'report.receipt.json',dict(status='passed',source_sha256=sha(__file__),notes_sha256=sha(D/'notes.json'),input_summaries_sha256={str(p):sha(p) for p in [D/'summary.json',D/'continuation/summary.json']}))
    print(json.dumps({k:result[k] for k in ['status','recorded_masks','distinct_masks','native_costs_checked','full_scalar_traces_checked','trace_nodes_checked','best_native_mean_gain_percent','best_native_p99_gain_percent','solver_seconds']},indent=2))
if __name__=='__main__':main()
