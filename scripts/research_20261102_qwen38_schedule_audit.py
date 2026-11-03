#!/usr/bin/env python3
"""Independently reschedule representative ledgers with set-based queues.

Only our separate specification checker is imported.  The production Graph,
matrix builder and priority scheduler are not imported or called.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_qwen38_independent_audit import BASE, RESOURCES, expected_nodes, read, sha


def schedule(nodes,down_first):
    remaining=[len(n['deps']) for n in nodes]
    children=[[] for _ in nodes]
    available={r:set() for r in RESOURCES}
    for i,n in enumerate(nodes):
        for d in n['deps']:children[d].append(i)
        if not remaining[i]:available[n['resource']].add(i)
    starts=np.full(len(nodes),-1,dtype=np.int64);ends=starts.copy()
    running={};owner=[None]*16;now=0;completed=0
    priority=[((1-n['phase']) if down_first else n['phase'],n['expert'],i) for i,n in enumerate(nodes)]
    def eligible(i):
        n=nodes[i]
        if n['job']<0:return True
        if n['acquire']:return owner[n['chip']] is None
        return owner[n['chip']]==n['job']
    while completed<len(nodes):
        for resource in RESOURCES:
            if resource in running:continue
            options=[i for i in available[resource] if eligible(i)]
            if not options:continue
            i=min(options,key=lambda j:priority[j]);available[resource].remove(i)
            n=nodes[i]
            if n['acquire']:
                assert owner[n['chip']] is None;owner[n['chip']]=n['job']
            starts[i]=now;running[resource]=(now+n['duration'],i)
        assert running
        now=min(t for t,_ in running.values())
        finished=sorted((i,r) for r,(t,i) in running.items() if t==now)
        for i,resource in finished:
            del running[resource];n=nodes[i];ends[i]=now;completed+=1
            if n['release']:
                assert owner[n['chip']]==n['job'];owner[n['chip']]=None
            for child in children[i]:
                remaining[child]-=1
                if remaining[child]==0:available[nodes[child]['resource']].add(child)
    assert not running and owner==[None]*16
    return starts,ends


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',default=['native-smoke','native-smoke-reduce1us'])
    ap.add_argument('--output',type=Path,default=BASE/'independent-audit/representative-reschedule.json');args=ap.parse_args()
    records=[]
    for name in args.runs:
        screen=read(BASE/name/'screen.json')
        for r in screen['records']:
            p=BASE/name/(r['label']+'.trace.json');trace=read(p)
            assert sha(p)==r['artifact_sha256']['trace.json']
            expected=expected_nodes(trace['items'],r['reduction_ns'],r['activation_ns'])
            starts,ends=schedule(expected,r['down_first'])
            actual_starts=np.asarray([n['start_ns'] for n in trace['trace']])
            actual_ends=np.asarray([n['end_ns'] for n in trace['trace']])
            assert np.array_equal(starts,actual_starts) and np.array_equal(ends,actual_ends)
            records.append(dict(run=name,label=r['label'],trace_sha256=sha(p),nodes=len(expected),
                all_start_end_times_exact=True,makespan_ns=int(ends.max())))
    out=dict(status='complete independent representative rescheduling',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),specification_source_sha256=sha(Path(__file__).with_name('research_20261102_qwen38_independent_audit.py')),
        records=records,total_nodes=sum(r['nodes'] for r in records),
        method='Set-based ready queues and running-resource completion sweep, fixed phase/expert/index priority, single job owner from command to output-channel end. Root generator and Graph are not imported.',
        scope='Representative saved trace for every case/layer; all-query mean/p99 and lower bounds are separately checked in the specification audit.',
        external_task_payloads_opened=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),traces=len(records),total_nodes=out['total_nodes']),indent=2))


if __name__=='__main__':main()
