#!/usr/bin/env python3
"""Independent metadata/DP checks of the train-only GU work screen."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from research_20261102_qwen38_independent_audit import templates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round2/gu-coalescing'
BASE=OUT.parent/'qwen38-placement'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def python_dp(layout,selected,chip,align):
    spans=[]
    for e in [int(x) for x in selected]+[512]:
        for b,s,t in layout[e,chip]:
            if t>s:spans.append((int(b),int(s),int(t)))
    spans.sort();dp=[(0,0,0,0)]+[None]*len(spans)
    for end in range(1,len(spans)+1):
        candidates=[]
        for begin in range(end):
            if spans[begin][0]!=spans[end-1][0]:continue
            start=spans[begin][1]-spans[begin][1]%align
            n=spans[end-1][2]-start;w=(n*2560+65535)//65536;o=w*65536//2560
            old=dp[begin];candidates.append((old[0]+w*28000+1530+2*o,old[1]+w,old[2]+1,old[3]+o))
        dp[end]=min(candidates,key=lambda x:x[:3])
    return dp[-1]


def main():
    doc=read(OUT/'screen.json');protocol=read(OUT/'protocol.frozen.json');inv=templates()
    for p,h in doc['source_sha256'].items():assert sha(p)==h
    frozen=read(BASE/'native-full48/plans.frozen.json');fr={(r['layer'],r['case']):r for r in frozen['records']}
    with np.load(OUT.parent/'qwen38-trace-audit/train_decode.npz') as f:routes=f['routes'][:,::4,:3].reshape(-1,3,10)
    records=[];checks=0
    for rec in doc['records']:
        prefix=OUT/(rec['label']+'.npz');assert sha(prefix)==rec['npz_sha256']
        with np.load(prefix) as f:layout=f['layout'].copy();cost=f['cost'].copy();old=f['original_context_ns'].copy();waves=f['original_waves'].copy()
        assert cost.shape==(3200,16,4) and layout.shape==(513,16,2,3)
        p=ROOT/fr[rec['layer'],rec['case']]['plan'];assert sha(p)==fr[rec['layer'],rec['case']]['plan_sha256']
        plan=read(p);expected=np.zeros((513,16),dtype=np.int32)
        for e in range(513):
            spec=plan['fixed_shared']['gate_up'] if e==512 else plan['plans']['gate_up']
            color=spec['color'] if e==512 else spec['labels'][e]
            for j in inv['gate_up'][spec['label']]['variants'][color]:
                assert j['columns']==2560;expected[e,j['chip']]+=j['rows']
        assert np.array_equal((layout[:,:,:,2]-layout[:,:,:,1]).sum(axis=2),expected)
        for chip in range(16):
            spans=sorted(tuple(map(int,v)) for v in layout[:,chip].reshape(-1,3) if v[2]>v[1])
            for a,b in zip(spans,spans[1:]):assert a[0]!=b[0] or a[2]<=b[1]
            assert sum(t-s for _,s,t in spans)*2560==rec['inventory'][chip]['useful_bytes']
            assert max(b for b,_,_ in spans)+1==rec['inventory'][chip]['blocks']
        value=cost[:,:,0].max(axis=1)
        assert abs(value.mean()-rec['context_ns']['mean'])<1e-7
        assert abs(np.percentile(value,99)-rec['context_ns']['p99'])<1e-7
        assert abs(cost[:,:,1].max(axis=1).mean()-rec['waves']['mean'])<1e-7
        for query in [0,1599,3199]:
            for chip in range(16):
                answer=python_dp(layout,routes[query,rec['layer']],chip,rec['alignment_rows'])
                assert tuple(cost[query,chip])==answer;checks+=1
        if rec['case']=='whole_row':
            chosen=np.column_stack((routes[:,rec['layer']],np.full(3200,512,dtype=np.int32)))
            selected=expected[chosen].astype(np.int64)*2560
            assert np.array_equal((selected.sum(axis=1)+65535)//65536,((selected+65535)//65536).sum(axis=1))
        records.append(dict(label=rec['label'],onecopy_row_counts_exact=True,all_physical_intervals_disjoint=True,
                            all_query_stats_exact=True,python_cpp_DP_cases=48))
    report=dict(status='PASS independent GU work-screen audit',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),screen_sha256=sha(OUT/'screen.json'),records=len(records),
        independent_Python_CPP_DP_cases=checks,query_chip_zero_NAND_headroom_checks=3*3200*16,
        all_static_rows_onecopy=True,all_query_summary_scalars_recomputed=True,
        alignment_erratum='32row branch is an optimistic window relaxation: waves ignore initial startPage mod4 WL span, in addition to unverified arbitrary-plane command start. It is not a legal ordinary-multiplane implementation.128row starts are integral5WL positions.',
        scope='Perchip aggregate nativework lowerbounds, not exact multi-resource FFN schedules or measured silicon. Three orderings/fixed ownership do not exhaust cluster packing or joint reshards.',
        external_or_heldout_opened=False,checks=records)
    p=OUT/'independent_audit.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(path=str(p),sha256=sha(p),checks=checks),indent=2))


if __name__=='__main__':main()
