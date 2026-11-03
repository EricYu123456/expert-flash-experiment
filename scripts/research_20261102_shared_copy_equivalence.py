#!/usr/bin/env python3
"""Constructive control: paired rows equal two full sharedGU copies + views.

This concerns the *measured selector* in shared_pipeline.py, which sends every
pair to the same half when imbalanced. It is not an impossibility result for
independent per-pair selection or for an implemented NAND compiler/runtime.
"""
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path

import numpy as np

from research_20261102_shared_pipeline import shared_gu_ranges,build_jobs
from run_corpus_traces import ROOT,sha256


def view(name):
    if name=='A':return [(p,256*p,256) for p in range(8)]
    if name=='B':return [(p+8,256*p,256) for p in range(8)]
    return [entry for p in range(8) for entry in [(p,256*p,128),(p+8,256*p+128,128)]]


def dense_rows(ranges):
    out=np.zeros(16,dtype=np.int64)
    for c,start,count in ranges:out[c]+=count
    return out


def main():
    dest=ROOT/'results/research-2026-11-02/round2/novelty-reverse-review';dest.mkdir(parents=True,exist_ok=True)
    states=[]
    for n in range(7):
        selected_view='A' if n<3 else 'B' if n>3 else 'hybrid'
        control=view(selected_view);measured=shared_gu_ranges(n,True)
        assert sorted(control)==sorted(measured)
        rows=dense_rows(control);assert rows.sum()==2048
        ordered=sorted((start,start+count) for _,start,count in control)
        assert ordered[0][0]==0 and ordered[-1][1]==2048 and all(a[1]==b[0] for a,b in zip(ordered,ordered[1:]))
        # Every read is a contiguous subrange of a stored six-wave256row copy.
        for chip,start,count in control:
            p=chip%8;assert 256*p<=start<start+count<=256*(p+1)
            assert (start-256*p)*1536%65536==0 and count*1536%65536==0
        items=[]
        for e in range(6):
            gu=np.zeros(16,dtype=np.int64);gu[:8] = 128 if e<n else 0;gu[8:]=128 if e>=n else 0
            down=np.full(16,128);down[(e%4)*4:(e%4)*4+4]=0
            items.append(dict(gu_rows=gu.tolist(),down_rows=down.tolist(),down_columns=512))
        shared_down=[128]*8+[64]*8
        a=items+[dict(gu_rows=dense_rows(measured).tolist(),down_rows=shared_down,down_columns=1024)]
        b=items+[dict(gu_rows=rows.tolist(),down_rows=shared_down,down_columns=1024)]
        ga=build_jobs(a);gb=build_jobs(b);assert ga.nodes==gb.nodes and ga.jobs==gb.jobs
        for priority in [True,False]:assert ga.run(priority)['makespan_ns']==gb.run(priority)['makespan_ns']
        states.append(dict(n_a=n,selected_full_tensor_view=selected_view,logical_ranges=control,
            native_jobs=len(control),chip_waves=int(((rows*1536+65535)//65536).sum()),
            graph_nodes_byte_identical=True,both_priority_makespans_identical=True))
    traces=[]
    for path in sorted((ROOT/'results/research-2026-11-02/round2/shared-pipeline').glob('*.rep1.*.trace.json')):
        d=json.loads(path.read_text());items=d['items'];n=sum(sum(x['gu_rows'][:8])>0 for x in items[:-1])
        name='A' if n<3 else 'B' if n>3 else 'hybrid'
        assert sorted(view(name))==sorted(tuple(x) for x in d['shared_gu_ranges'])
        np.testing.assert_array_equal(dense_rows(view(name)),items[-1]['gu_rows'])
        traces.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),n_a=n,matching_view=name))
    result=dict(status='PASS constructive equivalence for the currently measured sharedGU replica selector',created_utc=datetime.now(timezone.utc).isoformat(),
        storage='CopyA places contiguous logical rows[256p,256(p+1)) onchip p; copyB places the same rows onchip p+8, for p0..7. Each half stores a full2048x1536 sharedGU matrix. The hybrid reads the first128rows fromA and second128rows fromB in each pair.',
        budget=dict(one_tensor_bytes=2048*1536,total_two_copy_bytes=2*2048*1536,extra_bytes_full40=40*2048*1536),
        exhaustive_selector_states=states,actual_replica_trace_checks=traces,
        conclusion='For the measured nA-only rule, a same-budget two-full-GU-copy baseline with A/B/hybrid views emits identical logical ranges, native jobs, dependency graph and schedules. Paired-row replica selection is not a distinct algorithm or physical layout in these experiments.',
        what_remains_open=['Independent choices across chip pairs were not used by the measured selector; a new policy must beat the full three-view baseline and a generic native-tile replica selector.',
            'A compiler/runtime that lowers these existing ideas to legal NAND streams may still be a systems contribution, subject to implementation and evidence.',
            'This equivalence is not a claim that generic expressibility makes all future systems work unpublishable.'],
        root_source_sha256=sha256(ROOT/'scripts/research_20261102_shared_pipeline.py'),source_sha256=sha256(Path(__file__)))
    path=dest/'shared_copy_equivalence.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],states=len(states),actual_traces=len(traces),sha256=sha256(path)),indent=2))


if __name__=='__main__':main()
