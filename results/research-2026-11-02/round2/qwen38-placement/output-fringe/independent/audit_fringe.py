#!/usr/bin/env python3
"""Independent fringe adapter; imports only our sealed specification/scheduler.

Does not import the production Graph, encoder, fringe source or C++ library.
Every saved query scalar/aggregate/bound is inspected. Event replay covers
the saved first query and one distinct last query of each static plan/mode.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents
          if (p/'scripts/research_20261102_qwen38_schedule_audit.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from research_20261102_qwen38_independent_audit import (
    RESOURCES, SHAPES, coverage, expected_nodes, read, sha, templates)
from research_20261102_qwen38_schedule_audit import schedule

BASE=ROOT/'results/research-2026-11-02/round2/qwen38-placement'
RUN=BASE/'output-fringe'
OUT=RUN/'independent'
OLD=BASE/'external-doc-grounded'
FROZEN=BASE/'native-full48/plans.frozen.json'
FROZEN_SHA='9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505'
DATA=BASE.parent/'qwen38-trace-audit/external-doc-grounded-stride1-short/external_decode.npz'
DATA_SHA='2111de845e4e8938e2158d0fc778f7910c48879184aeb1900759cd764d5f4736'
NAMES=['group_g16','group_best','overlap_best','whole_row','rect_min_io4']


def emitted(r,c,mode):
    if mode==0:return int(r)
    unit=16384 if mode==1 else 65536
    return ((int(r)*int(c)+unit-1)//unit*unit)//int(c)


def counters(jobs,mode):
    # Useful bytes/pages/waves/commands/inputs/emitted rows/partial tail bytes,
    # then sixteen native-context occupancies, eight channels and controller,
    # PCIe-output work. The native context runs command->channel-output end.
    a=np.zeros(33,dtype=np.int64)
    for j in jobs:
        r,c,ch=j['rows'],j['columns'],j['chip'];n=r*c
        er=emitted(r,c,mode);w=(n+65535)//65536;p=(n+16383)//16384
        budget=n if mode==0 else (p*16384 if mode==1 else w*65536)
        a[:7]+=[n,p,w,1,c,er,budget-er*c]
        a[7+ch]+=250+(c+1)//2+28000*w+2*er
        a[23+ch%8]+=(c+1)//2+2*er
        a[31]+=250;a[32]+=(er+1)//2
    return a


def items_from_plan(plan,inv,selected):
    result=[]
    for e in [int(v) for v in selected]+[512]:
        row={}
        for p in ['gate_up','down']:
            if e==512:
                spec=plan['fixed_shared'][p];color=spec['color']
            else:
                spec=plan['plans'][p];color=spec['labels'][e]
            candidate=inv[p][spec['label']]
            row[p]=candidate['variants'][color]
            row['gu_partial' if p=='gate_up' else 'down_partial']=bool(candidate['conditional_partial_sum'])
        result.append(row)
    return result


def fringe_nodes(items,mode):
    result=expected_nodes(items,1000,0)
    ordered=[j for p in ['gate_up','down'] for item in items for j in item[p]]
    seen=set()
    for i,n in enumerate(result):
        if n['kind']=='output_channel':
            j=ordered[n['job']];er=emitted(j['rows'],j['columns'],mode)
            assert n['job'] not in seen;seen.add(n['job'])
            n['duration']=2*er
            assert result[i+1]['kind']=='output_pcie' and result[i+1]['deps']==[i]
            result[i+1]['duration']=(er+1)//2
    assert len(seen)==len(ordered)
    return result


def event_worker(task):
    rec,inv,selected,last_selected,expected_first,expected_last=task
    plan=read(ROOT/rec['plan']);mode=rec['mode'];items=items_from_plan(plan,inv,selected)
    trace=read(RUN/(rec['label']+'.trace.json'))
    assert trace['items']==items and trace['selected']==selected
    assert trace['layer']==rec['layer'] and trace['case']==rec['case'] and trace['mode']==mode
    result=fringe_nodes(items,mode)
    assert len(result)==trace['nodes']==len(trace['trace'])
    work={r:0 for r in RESOURCES}
    for x,n in zip(result,trace['trace']):
        for key,value in x.items():
            if key!='duration':assert n[key]==value,(rec['label'],key,n.get(key),value)
        assert n['end_ns']-n['start_ns']==x['duration']
        work[x['resource']]+=x['duration']
    starts,ends=schedule(result,False)
    assert np.array_equal(starts,[n['start_ns'] for n in trace['trace']])
    assert np.array_equal(ends,[n['end_ns'] for n in trace['trace']])
    assert int(ends.max())==trace['makespan_ns']==expected_first
    assert [work[r] for r in RESOURCES]==trace['resource_work_ns']
    # One more query tests C++ costs beyond the same representative query
    # already cross-checked by the production Python implementation.
    last=fringe_nodes(items_from_plan(plan,inv,last_selected),mode)
    _,last_ends=schedule(last,False)
    assert int(last_ends.max())==expected_last
    # Mode0 remains the independently verified original useful-row contract.
    zero=fringe_nodes(items,0);_,zero_ends=schedule(zero,False)
    with np.load(OLD/(rec['base_label']+'.npz')) as z:old_first=int(z['cost'][0])
    assert int(zero_ends.max())==old_first
    return dict(label=rec['label'],nodes=len(result),first_query_all_node_times_exact=True,
                resource_work_exact=True,first_cost_ns=expected_first,last_cost_ns=expected_last,
                last_query_index=5518,last_query_independent_cost_exact=True,
                useful_mode_first_query_exact=True)


def stats(x,req):
    means=[float(x[req==r].mean()) for r in np.unique(req)]
    return dict(mean_ns=float(x.mean()),p99_ns=float(np.percentile(x,99)),
                equal_request_mean_ns=float(np.mean(means)))


def comparisons(costs,req):
    result=[];ids=np.unique(req)
    for bi,b in enumerate(NAMES):
        for c in NAMES[bi+1:]:
            x,y=costs[b],costs[c]
            gains=np.asarray([100*(1-y[req==r].mean()/x[req==r].mean()) for r in ids])
            result.append(dict(baseline=b,candidate=c,
                paired_48layer_mean_reduction_pct=100*(1-y.mean()/x.mean()),
                paired_48layer_p99_reduction_pct=100*(1-np.percentile(y,99)/np.percentile(x,99)),
                equal_request_mean_reduction_pct=100*(1-np.mean([y[req==r].mean() for r in ids])/np.mean([x[req==r].mean() for r in ids])),
                requests_worse=int((gains<0).sum()),requests_equal=int((gains==0).sum()),
                min_request_gain_pct=float(gains.min()),max_request_gain_pct=float(gains.max()),
                tokens_worse=int((y>x).sum()),tokens_equal=int((y==x).sum())))
    return result


def main():
    assert sha(FROZEN)==FROZEN_SHA and sha(DATA)==DATA_SHA
    freeze=read(FROZEN);protocol=read(RUN/'protocol.frozen.json');summary=read(RUN/'summary.json')
    assert len(freeze['records'])==240 and len(summary['records'])==480
    assert sha(RUN/'protocol.frozen.json')==summary['protocol_sha256']
    for p,h in protocol['source_inputs_sha256'].items():assert sha(p)==h
    prior=read(BASE/'independent-audit/audit-external.json')
    assert prior['all240plan_hashes_unchanged'] and prior['frozen240plans_sha256_before_after']==FROZEN_SHA
    with np.load(DATA) as a:data={k:a[k].copy() for k in a.files}
    routes=data['routes'];req=data['req_ids']
    assert routes.shape==(5519,48,10) and np.array_equal(np.unique(req),np.arange(32,73))
    assert np.all(data['phase']==1) and np.all(~data['warmup']) and len(np.unique(data['session_ids']))==1
    inv=templates();tables={m:{p:{n:np.stack([counters(j,m) for j in v['variants']])
                               for n,v in cc.items()} for p,cc in inv.items()} for m in [0,1,2]}
    for p,cc in inv.items():
        for v in cc.values():
            for jobs in v['variants']:coverage(jobs,SHAPES[p])
    sources=[Path(__file__),ROOT/'scripts/research_20261102_qwen38_independent_audit.py',
             ROOT/'scripts/research_20261102_qwen38_schedule_audit.py']
    original_sha={str(p):sha(p) for p in sources}
    frozen={(r['layer'],r['case']):r for r in freeze['records']}
    costs={(m,c):[] for m in [0,1,2] for c in NAMES};records=[];tasks=[];inventories=[];seen=set()
    inventory_examples={}
    for rec in sorted(summary['records'],key=lambda r:(r['layer'],r['case'],r['mode'])):
        layer,case,mode=rec['layer'],rec['case'],rec['mode'];key=(layer,case,mode)
        assert key not in seen;seen.add(key)
        source=frozen[layer,case];path=ROOT/source['plan'];assert sha(path)==source['plan_sha256']
        plan=read(path);base_label=source['label'];label=base_label+f'.mode{mode}';assert rec['label']==label
        for ext,h in rec['artifacts_sha256'].items():assert sha(RUN/(label+'.'+ext))==h
        with np.load(RUN/(label+'.npz')) as a:value=a['cost'].copy();diff=a['difference_vs_useful'].copy()
        with np.load(OLD/(base_label+'.npz')) as a:old=a['cost'].copy()
        assert value.shape==diff.shape==old.shape==(5519,) and np.array_equal(diff,value-old)
        assert abs(value.mean()-rec['mean_ns'])<1e-7
        assert abs(np.percentile(value,99)-rec['p99_ns'])<1e-7
        assert abs(diff.mean()-rec['mean_difference_ns'])<1e-7
        selected=routes[:,layer];counter=np.zeros((5519,33),dtype=np.int64)
        inventory=np.zeros((33,),dtype=np.int64);partials=0
        for p,spec in plan['plans'].items():
            labels=np.asarray(spec['labels']);assert np.array_equal(np.bincount(labels,minlength=16),np.full(16,32))
            tt=tables[mode][p][spec['label']];counter+=tt[labels][selected].sum(axis=1);inventory+=tt[labels].sum(axis=0)
            partials+=10*bool(inv[p][spec['label']]['conditional_partial_sum'])
            sp=plan['fixed_shared'][p];shared=tables[mode][p][sp['label']][sp['color']]
            counter+=shared[None,:];inventory+=shared
            partials+=bool(inv[p][sp['label']]['conditional_partial_sum'])
            if layer==0:
                useful=sum(j['rows'] for j in inv[p][spec['label']]['variants'][0])
                emit=tables[mode][p][spec['label']][:,5]
                inventory_examples[f'{case}/{p}/mode{mode}']=dict(useful_output_rows_per_expert=useful,
                    emitted_output_rows_per_expert_by_color=emit.tolist(),extra_output_rows_by_color=(emit-useful).tolist(),
                    candidate=spec['label'])
        native_bound=2000+counter[:,7:23].max(axis=1)
        pcie_bound=320+11*80+counter[:,32]
        all_resource_bound=np.maximum.reduce([native_bound,counter[:,23:31].max(axis=1),counter[:,31],pcie_bound,
                                               np.full(5519,24000+1000*partials,dtype=np.int64)])
        assert np.all(value>=all_resource_bound)
        assert inventory[0]==513*(1280*2560+2560*640)
        inventories.append(dict(layer=layer,case=case,mode=mode,total_useful_bytes=int(inventory[0]),
            actual_pages=int(inventory[1]),waves=int(inventory[2]),commands=int(inventory[3]),
            input_bytes=int(inventory[4]),emitted_rows=int(inventory[5]),unemitted_incomplete_tail_bytes=int(inventory[6])))
        costs[mode,case].append((layer,value));
        if mode==1:costs[0,case].append((layer,old))
        records.append(dict(layer=layer,case=case,mode=mode,mean_ns=float(value.mean()),p99_ns=float(np.percentile(value,99)),
            all5519_native_and_resource_bounds_valid=True,mean_native_bound_ns=float(native_bound.mean()),
            mean_output_extra_cost_ns=float(diff.mean()),samples_with_lower_cost_than_useful=int((diff<0).sum()),
            min_signed_cost_change_ns=int(diff.min()),max_signed_cost_change_ns=int(diff.max()),
            plan_sha256=source['plan_sha256']))
        task=dict(rec,**{k:source[k] for k in ['plan','plan_sha256']},base_label=base_label)
        tasks.append((task,inv,selected[0].tolist(),selected[-1].tolist(),int(value[0]),int(value[-1])))
    assert seen=={(l,c,m) for l in range(48) for c in NAMES for m in [1,2]}
    totals=[];pairwise={};changes=[];summed={}
    for mode in [0,1,2]:
        current={}
        for case in NAMES:
            rr=costs[mode,case];assert len(rr)==48
            value=np.stack([a for _,a in sorted(rr,key=lambda v:v[0])]).sum(axis=0)
            current[case]=value;summed[mode,case]=value
            st=stats(value,req);totals.append(dict(mode=mode,case=case,**st))
            if mode:
                ref=next(t for t in summary['totals'] if t['mode']==mode and t['case']==case)
                assert abs(st['mean_ns']-ref['mean_ns'])<1e-7 and abs(st['p99_ns']-ref['p99_ns'])<1e-7
            else:
                ref=next(t for t in read(OLD/'summary.json')['totals'] if t['case']==case)
                assert abs(st['mean_ns']-ref['token_weighted_mean_ns'])<1e-7 and abs(st['p99_ns']-ref['p99_ns'])<1e-7
        pairwise[str(mode)]=comparisons(current,req)
    for mode in [1,2]:
        for c in NAMES:
            x,y=summed[0,c],summed[mode,c];d=y-x
            changes.append(dict(mode=mode,case=c,mean_change_ns=float(d.mean()),
                mean_change_pct=100*(y.mean()/x.mean()-1),p99_change_pct=100*(np.percentile(y,99)/np.percentile(x,99)-1),
                tokens_with_lower_greedy_makespan=int((d<0).sum()),tokens_equal=int((d==0).sum()),
                min_signed_change_ns=int(d.min()),max_signed_change_ns=int(d.max())))
    event_reports=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,done in enumerate(as_completed([pool.submit(event_worker,t) for t in tasks]),1):
            event_reports.append(done.result())
            if i%80==0:print(f'independent event audits {i}/480',flush=True)
    for p,h in original_sha.items():assert sha(p)==h
    for p,h in protocol['source_inputs_sha256'].items():assert sha(p)==h
    assert sha(FROZEN)==FROZEN_SHA and sha(DATA)==DATA_SHA
    for r in freeze['records']:assert sha(ROOT/r['plan'])==r['plan_sha256']
    out=dict(status='PASS independent output-fringe sensitivity audit',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=original_sha,protocol_sha256=sha(RUN/'protocol.frozen.json'),summary_sha256=sha(RUN/'summary.json'),
        frozen_sha256_before_after=FROZEN_SHA,external_npz_sha256_before_after=DATA_SHA,
        cases=NAMES,modes=[0,1,2],static_plans=240,new_fringe_records=480,
        scalar_samples_checked=480*5519,representative_schedules_checked=480*3,
        representative_saved_nodes_exact=sum(r['nodes'] for r in event_reports),
        first_and_last_query_independent_costs_exact=True,all_saved_query_bounds_and_aggregates_valid=True,
        exclusions=dict(decode_tokens=5519,requests=41,client_sessions=1,req_ids='32..72',all_phase1=True,no_warmup=True,
                        prior_provenance_audit_sha256=sha(BASE/'independent-audit/audit-external.json')),
        totals=totals,paired_comparisons=pairwise,changes_from_useful=changes,
        inventory_examples=inventory_examples,physical_inventories=inventories,records=records,
        event_reports=sorted(event_reports,key=lambda r:r['label']),
        assumptions=['All modes retain nominal I8 weights, A8 inputs and four-byte results; these are not established AiF output/scaling ABI.',
            'Mode1 drains complete C-wide rows in ceil(R*C/16KiB) pages; mode2 in ceil(R*C/64KiB) four-plane waves.',
            'Both floors omit any final incomplete row accumulator. Its flush/discard behavior is undocumented.',
            'Mode2 assumes all reserved-wave padding is sensed and emits dense outputs; command pagecount might instead exclude unused plane pages.',
            'Neither mode is a proven hardware upper bound or device measurement. Source pages, valid-row stop, unused-plane behavior and output framing require clarification.',
            'Native contexts stay owned from command through channel-output drain, not through host PCIe completion; host receives every emitted four-byte result.',
            'Compute waves, command counts, input reloads, one-copy coefficients, partial-reduction cost1us and zero activation/combine are unchanged.',
            'Fixed priority greedy schedules can become faster for isolated queries after node durations rise. No pointwise monotonic assertion was made.',
            'This independent event scheduler checks first/last queries for each plan/mode and useful-mode first query, not all2649120event schedules.',
            'P99 is empirical per-token summed48FFN service, not whole-model latency or41independent tasks.',
            'No fit/refit, new dataset access or plan change. Existing external NPZ only. Qwen2560/640 upload sizes must not be reused for Granite.'])
    target=OUT/'audit-fringe.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(OUT/'paired-48layer-costs.npz',**{f'mode{m}__{c}':v for (m,c),v in summed.items()})
    print(json.dumps(dict(path=str(target),sha256=sha(target),totals=totals,changes=changes,
                         comparisons=pairwise['2']),indent=2))


if __name__=='__main__':main()
