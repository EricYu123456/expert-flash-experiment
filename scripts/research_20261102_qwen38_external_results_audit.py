#!/usr/bin/env python3
"""Independent post-freeze external-result audit, with request-level contrasts."""
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from research_20261102_qwen38_independent_audit import (
    BASE,ROOT,SHAPES,coverage,ledger,templates,validate_trace,expected_nodes,sha,read)
from research_20261102_qwen38_schedule_audit import schedule
from research_20261102_qwen38_trace_audit import dtypes

FREEZE=BASE/'native-full48/plans.frozen.json'
FREEZE_SHA='9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505'
DATA=BASE.parent/'qwen38-trace-audit/external-doc-grounded-stride1-short'
EXTERNAL_SHA='2111de845e4e8938e2158d0fc778f7910c48879184aeb1900759cd764d5f4736'
RUN=BASE/'external-doc-grounded'


def source_rows():
    audit=read(DATA/'audit.json');before=read(DATA/'before_payload.json')
    assert sha(DATA/'external_decode.npz')==EXTERNAL_SHA
    assert audit['freeze_sha256_before_and_after']==FREEZE_SHA==before['freeze_sha256']
    assert datetime.fromisoformat(before['created_utc'])>datetime.fromisoformat(read(FREEZE)['created_utc'])
    source=Path(audit['main_stream_path']);assert sha(source)==audit['main_stream_sha256']
    _,_,td,_=dtypes();raw=np.memmap(source,mode='r',offset=5712,dtype=td,shape=(41532,))
    mask=(raw['step']['req_id']>=32)&(raw['step']['phase']==1);ix=np.flatnonzero(mask)
    assert len(ix)==5519
    with np.load(DATA/'external_decode.npz') as f:
        arrays={k:f[k].copy() for k in f.files}
    assert np.array_equal(arrays['source_file_rows'],ix)
    assert np.array_equal(arrays['routes'],raw['layers']['experts'][ix])
    assert np.array_equal(arrays['gates'],raw['layers']['gates'][ix])
    for key,sourcekey in [('req_ids','req_id'),('token_pos','tok_idx'),('tok_idx','tok_idx'),('token_ids','token_id'),('phase','phase')]:
        assert np.array_equal(arrays[key],raw['step'][sourcekey][ix])
    assert np.all(arrays['phase']==1) and np.all(~arrays['warmup'])
    assert len(np.unique(arrays['session_ids']))==1
    reqs=np.unique(arrays['req_ids']);assert np.array_equal(reqs,np.arange(32,73))
    for req in reqs:
        j=np.flatnonzero(arrays['req_ids']==req)
        assert np.array_equal(arrays['token_pos'][j],np.arange(len(j)))
    warm_ix=np.flatnonzero((raw['step']['req_id']<32)&(raw['step']['phase']==1));assert len(warm_ix)==11866
    assert len(np.intersect1d(warm_ix,ix))==0
    assert np.flatnonzero(raw['step']['req_id']>=32)[0]==14182
    assert sha(DATA/'warmup_decode.npz')==audit['views']['warmup_decode']['sha256']
    with np.load(DATA/'warmup_decode.npz') as f:
        assert np.array_equal(f['source_file_rows'],warm_ix) and np.all(f['req_ids']<32)
    return arrays,dict(external_tokens=5519,external_requests=41,independent_client_sessions=1,
        all_external_rows_exactly_main_stream_req_ge32_phase1=True,excluded_warmup_decode_rows=11866,
        first_nonwarmup_file_row=14182,no_warmup_or_prefill_rows_in_external_decode=True,
        external_sha256=EXTERNAL_SHA,acquisition_audit_sha256=sha(DATA/'audit.json'),
        before_payload_sha256=sha(DATA/'before_payload.json'))


def stats(values,req):
    request_means=np.asarray([values[req==r].mean() for r in np.unique(req)])
    return dict(token_weighted_mean_ns=float(values.mean()),p99_ns=float(np.percentile(values,99)),
        equal_request_mean_ns=float(request_means.mean()),min_request_mean_ns=float(request_means.min()),
        max_request_mean_ns=float(request_means.max()))


def contrasts(costs,req,token_pos,source_ix):
    reqs=np.unique(req);pairs=[];request_rows=[]
    names=['group_g16','group_best','overlap_best','whole_row','rect_min_io4']
    for q in reqs:
        ix=np.flatnonzero(req==q)
        request_rows.append(dict(req_id=int(q),decode_rows=len(ix),
            mean_ns={name:float(costs[name][ix].mean()) for name in names},
            p99_ns={name:float(np.percentile(costs[name][ix],99)) for name in names}))
    for bi,baseline in enumerate(names):
        for candidate in names[bi+1:]:
            b=costs[baseline];c=costs[candidate];gain=100*(1-c/b)
            request_gain=[]
            request_p99_gain=[]
            for row in request_rows:
                request_gain.append(100*(1-row['mean_ns'][candidate]/row['mean_ns'][baseline]))
                request_p99_gain.append(100*(1-row['p99_ns'][candidate]/row['p99_ns'][baseline]))
            request_gain=np.asarray(request_gain);request_p99_gain=np.asarray(request_p99_gain)
            worst=np.argsort(gain)[:5]
            negative_requests=[dict(req_id=int(reqs[i]),decode_rows=int((req==reqs[i]).sum()),gain_pct=float(g))
                               for i,g in enumerate(request_gain) if g<0]
            pairs.append(dict(candidate=candidate,baseline=baseline,
                token_weighted_cost_reduction_pct=100*(1-float(c.mean())/float(b.mean())),
                empirical_token_p99_cost_reduction_pct=100*(1-float(np.percentile(c,99))/float(np.percentile(b,99))),
                equal_request_cost_reduction_pct=100*(1-float(np.mean([r['mean_ns'][candidate] for r in request_rows]))/float(np.mean([r['mean_ns'][baseline] for r in request_rows]))),
                mean_request_gain_pct=float(request_gain.mean()),min_request_gain_pct=float(request_gain.min()),
                max_request_gain_pct=float(request_gain.max()),median_request_gain_pct=float(np.median(request_gain)),
                request_means_improved=int((request_gain>0).sum()),request_means_worsened=int((request_gain<0).sum()),request_means_equal=int((request_gain==0).sum()),
                request_p99_improved=int((request_p99_gain>0).sum()),request_p99_worsened=int((request_p99_gain<0).sum()),
                tokens_improved=int((c<b).sum()),tokens_worsened=int((c>b).sum()),tokens_equal=int((c==b).sum()),
                min_token_gain_pct=float(gain.min()),max_token_gain_pct=float(gain.max()),
                negative_request_means=negative_requests,
                worst_five_tokens=[dict(index=int(i),req_id=int(req[i]),token_pos=int(token_pos[i]),source_file_row=int(source_ix[i]),
                    candidate_ns=int(c[i]),baseline_ns=int(b[i]),gain_pct=float(gain[i])) for i in worst]))
    return pairs,request_rows


def main():
    import json
    assert sha(FREEZE)==FREEZE_SHA
    freeze=read(FREEZE);frozen={(r['layer'],r['case']):r for r in freeze['records']};assert len(frozen)==240
    for r in frozen.values():assert sha(ROOT/r['plan'])==r['plan_sha256']
    doc=read(RUN/'summary.json');assert doc['plans_sha256']==FREEZE_SHA and doc['external_sha256']==EXTERNAL_SHA
    for p,h in doc['source_sha256'].items():assert sha(p)==h
    data,exclusions=source_rows();req=data['req_ids'];routes=data['routes'];assert routes.shape==(5519,48,10)
    assert doc['tokens']==5519 and doc['work_requests']==41 and doc['excluded_warmup_decode']==11866
    inv=templates();tables={p:{name:np.stack([ledger(j) for j in c['variants']]) for name,c in cc.items()} for p,cc in inv.items()}
    for phase,cc in inv.items():
        for c in cc.values():
            for jobs in c['variants']:coverage(jobs,SHAPES[phase])
    all_costs={};records=[];nodes=0;seen=set()
    for r in doc['records']:
        key=(r['layer'],r['case']);assert key not in seen;seen.add(key);f=frozen[key]
        assert r['plan']==f['plan'] and r['plan_sha256']==f['plan_sha256']
        planpath=ROOT/r['plan'];assert sha(planpath)==r['artifact_sha256']['plan.json'];plan=read(planpath)
        prefix=RUN/r['label']
        for ext in ['npz','trace.json']:assert sha(str(prefix)+'.'+ext)==r['artifact_sha256'][ext]
        with np.load(str(prefix)+'.npz') as a:cost=a['cost'].copy();bound=a['native_chip_bound'].copy()
        assert cost.shape==bound.shape==(5519,) and np.all(cost>=bound)
        assert r['samples']==5519 and abs(float(cost.mean())-r['mean_ns'])<1e-7
        assert abs(float(np.percentile(cost,99))-r['p99_ns'])<1e-7 and abs(float(bound.mean())-r['mean_bound_ns'])<1e-7
        selected=routes[:,r['layer']];load=np.zeros((5519,16),dtype=np.int64)
        for phase,spec in plan['plans'].items():
            labels=np.asarray(spec['labels']);assert np.array_equal(np.bincount(labels,minlength=16),np.full(16,32))
            load+=tables[phase][spec['label']][labels,:,6][selected].sum(axis=1)
            shared=plan['fixed_shared'][phase];load+=tables[phase][shared['label']][shared['color'],None,:,6]
        assert np.array_equal(bound,2000+load.max(axis=1))
        # The existing specification checker expects a request/token view.
        # A singleton outer dimension preserves exact external file order.
        rr={**r,'reduction_ns':1000,'activation_ns':0,'down_first':False}
        report=validate_trace(Path(str(prefix)+'.trace.json'),plan,inv,routes[None],rr,cost,bound)
        trace=read(str(prefix)+'.trace.json');expected=expected_nodes(trace['items'],1000,0)
        starts,ends=schedule(expected,False)
        assert np.array_equal(starts,[n['start_ns'] for n in trace['trace']]) and np.array_equal(ends,[n['end_ns'] for n in trace['trace']])
        nodes+=len(expected)
        records.append(dict(layer=r['layer'],case=r['case'],all5519_native_chip_bounds_exact=True,
            representative_nodes=len(expected),all_representative_start_end_times_exact=True,
            mean_ns=r['mean_ns'],p99_ns=r['p99_ns'],plan_sha256=r['plan_sha256']))
        all_costs.setdefault(r['case'],[]).append((r['layer'],cost))
    assert seen==set(frozen)
    costs={case:np.stack([a for _,a in sorted(v,key=lambda x:x[0])]).sum(axis=0) for case,v in all_costs.items()}
    paired=RUN/'paired48layer_costs.npz';assert sha(paired)==doc['paired_costs_sha256']
    with np.load(paired) as a:
        assert np.array_equal(a['req_ids'],req)
        for case,value in costs.items():assert np.array_equal(a[case],value)
    totals=[]
    for old in doc['totals']:
        got=stats(costs[old['case']],req)
        for key,value in got.items():assert abs(value-old[key])<1e-7
        totals.append(dict(case=old['case'],**got))
    pairs,request_rows=contrasts(costs,req,data['token_pos'],data['source_file_rows'])
    train=read(BASE/'native-full48/screen.json');train_costs={r['case']:r['mean_ns'] for r in train['totals']}
    shifts=[]
    for p in pairs:
        b,c=p['baseline'],p['candidate']
        shifts.append(dict(candidate=c,baseline=b,train_cost_reduction_pct=100*(1-train_costs[c]/train_costs[b]),
            external_cost_reduction_pct=p['token_weighted_cost_reduction_pct']))
    assert sha(FREEZE)==FREEZE_SHA
    for r in frozen.values():assert sha(ROOT/r['plan'])==r['plan_sha256']
    source_paths=[Path(__file__),Path(__file__).with_name('research_20261102_qwen38_independent_audit.py'),
                  Path(__file__).with_name('research_20261102_qwen38_schedule_audit.py')]
    out=dict(status='PASS independent no-refit external results audit',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in source_paths},external_summary_sha256=sha(RUN/'summary.json'),
        frozen240plans_sha256_before_after=FREEZE_SHA,all240plan_hashes_unchanged=True,
        exclusions=exclusions,static_plans_checked=240,cost_samples_checked=240*5519,
        representative_nodes_independently_rescheduled=nodes,all_representative_start_end_times_exact=True,
        totals=totals,paired_comparisons=pairs,request_rows=request_rows,train_external_contrasts=shifts,records=records,
        interpretation=['group_best is the train-selected disjoint-width family, not an oracle best grouping on the external task. group_g16 remains a distinct mandatory baseline.',
            'Report token-weighted ratios, equal-request ratios and the range/count of request-level gains; they answer different weighting questions.',
            'A request is one forward request/turn in one client session, not an independent QA task. No task-level significance or41-independent-task confidence interval is supported.',
            'P99 is the empirical per-token sum of48FFN service costs, not request completion latency, queueing SLO or full-model inference time.',
            'No external refit, relabel, shared geometry selection or family change was performed by this auditor.',
            'Current rectangles remain conditional on native output/scale reconstruction; the data source is NVFP4 while service costs are nominalI8/A8/4B.',
            'Full-query chip bounds/scalars and all paired aggregates checked; independent full event replay covers one representative query for each static plan.'],
        extractor_source_collision='Original extractor was replaced by same-named root replay runner after extraction. NPZ hashes remain fixed. A separate provenance erratum/source snapshot will preserve the original executed extraction source; current replay source is untouched.')
    target=BASE/'independent-audit/audit-external.json';target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(target),sha256=sha(target),queries=240*5519,nodes=nodes,totals=totals,
        comparisons=[{k:p[k] for k in ['candidate','baseline','token_weighted_cost_reduction_pct','empirical_token_p99_cost_reduction_pct',
        'min_request_gain_pct','max_request_gain_pct','request_means_worsened','tokens_worsened']} for p in pairs]),indent=2))


if __name__=='__main__':main()
