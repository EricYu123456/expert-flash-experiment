#!/usr/bin/env python3
"""Adapter for the frozen48-layer schema; original auditors stay sealed."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_qwen38_independent_audit import (
    BASE,ROOT,TRAIN,TRAIN_SHA,SHAPES,RESOURCES,coverage,ledger,templates,validate_trace,expected_nodes,sha,read)
from research_20261102_qwen38_schedule_audit import schedule

FROZEN=BASE/'native-full48/plans.frozen.json'
FROZEN_SHA='9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505'


def check_fit48(fitdir,inv,routes):
    fit=read(fitdir/'fit.json')
    for p,h in fit['source_sha256'].items():assert sha(p)==h
    for p,h in fit['inputs_sha256'].items():assert sha(p)==h
    assert fit['arguments']['stride']==4 and fit['arguments']['layers']==48
    assert sha(fitdir/'templates.json')==fit['templates_sha256']==sha(BASE/'phase-fit3/templates.json')
    tables={p:{name:np.stack([ledger(j) for j in c['variants']]) for name,c in cc.items()} for p,cc in inv.items()}
    unique={}
    for layer in range(48):
        a=routes[:,::4,layer].reshape(-1,10)
        u,w=np.unique(np.sort(a,axis=1),axis=0,return_counts=True);assert int(w.sum())==3200
        unique[layer]=(u,w)
    reports=[]
    for result in fit['results']:
        layer=result['layer'];p=result['phase'];u,w=unique[layer]
        assert result['fit_stride']==4 and result['fit_tokens']==3200 and result['unique_routes']==len(u)
        for r in result['records']:
            labels=np.asarray(r['labels']);hist=np.bincount(labels,minlength=16)
            assert labels.shape==(512,) and np.array_equal(hist,np.full(16,32))
            table=tables[p][r['label']]
            assert np.array_equal(table[labels].sum(axis=0),table[np.arange(512)%16].sum(axis=0))
            learned=table[labels,:,6][u].sum(axis=1).max(axis=1)
            initial=table[np.arange(512)%16,:,6][u].sum(axis=1).max(axis=1)
            value=float(learned@w/w.sum());before=float(initial@w/w.sum())
            assert abs(value-r['mean_bound_ns'])<1e-7 and abs(before-r['initial_mean_bound_ns'])<1e-7
            reports.append(dict(layer=layer,phase=p,label=r['label'],histogram=hist.tolist(),mean_bound_ns=value))
    assert len(fit['results'])==96 and len(reports)==2112
    return dict(fit_json_sha256=sha(fitdir/'fit.json'),requests=16,steps_per_request=200,
        step_positions='0,4,...796',tokens_per_layer=3200,phases_checked=96,label_records_checked=2112,
        all_label_histograms_32_per_color=True,all_fitted_scores_independently_exact=True,
        records=reports),tables


def main():
    assert sha(FROZEN)==FROZEN_SHA and sha(TRAIN)==TRAIN_SHA
    freeze=read(FROZEN);assert len(freeze['records'])==240
    screenpath=BASE/'native-full48/screen.json';screen=read(screenpath)
    assert screen['plan_freeze_sha256']==FROZEN_SHA
    for p,h in screen['inputs_sha256'].items():assert sha(p)==h
    for p,h in screen['source_sha256'].items():assert sha(p)==h
    with np.load(TRAIN) as a:routes=a['routes'].copy()
    assert routes.shape==(16,799,48,10)
    inv=templates();variants=0
    for phase,cc in inv.items():
        for candidate in cc.values():
            for jobs in candidate['variants']:coverage(jobs,SHAPES[phase]);variants+=1
    fit,tables=check_fit48(ROOT/freeze['fit_directory'],inv,routes)
    frozen_records={(r['layer'],r['case']):r for r in freeze['records']}
    records=[];trace_reports=[];reschedule_reports=[];costs={};bounds={};inventories={}
    seen=set()
    for r in screen['records']:
        key=(r['layer'],r['case']);assert key not in seen;seen.add(key)
        f=frozen_records[key];assert r['plan']==f['plan'] and r['plan_sha256']==f['plan_sha256']
        planpath=ROOT/r['plan'];assert sha(planpath)==r['plan_sha256'];plan=read(planpath)
        assert sha(planpath)==r['artifact_sha256']['plan.json']
        prefix=BASE/'native-full48'/r['label']
        for ext in ['npz','trace.json']:assert sha(str(prefix)+'.'+ext)==r['artifact_sha256'][ext]
        with np.load(str(prefix)+'.npz') as a:
            cost=a['cost'].copy();bound=a['native_chip_bound'].copy();steps=a['token_steps'].copy()
        assert np.array_equal(steps,np.arange(799)) and cost.shape==bound.shape==(12784,)
        assert r['samples']==12784 and np.all(cost>=bound)
        assert abs(float(cost.mean())-r['mean_ns'])<1e-7
        assert abs(float(np.percentile(cost,99))-r['p99_ns'])<1e-7
        assert abs(float(bound.mean())-r['mean_bound_ns'])<1e-7
        selected=routes[:,:,r['layer']].reshape(-1,10);load=np.zeros((12784,16),dtype=np.int64)
        inventory=np.zeros((16,7),dtype=np.int64)
        for phase,spec in plan['plans'].items():
            labels=np.asarray(spec['labels']);assert np.array_equal(np.bincount(labels,minlength=16),np.full(16,32))
            table=tables[phase][spec['label']]
            load+=table[labels,:,6][selected].sum(axis=1);inventory+=table[labels].sum(axis=0)
            shared=plan['fixed_shared'][phase]
            sl=tables[phase][shared['label']][shared['color']]
            load+=sl[None,:,6];inventory+=sl
        assert np.array_equal(bound,2000+load.max(axis=1))
        assert int(inventory[:,0].sum())==513*(1280*2560+2560*640)
        tracepath=Path(str(prefix)+'.trace.json')
        trace_reports.append(validate_trace(tracepath,plan,inv,routes,r,cost,bound))
        trace=read(tracepath);expected=expected_nodes(trace['items'],r['reduction_ns'],r['activation_ns'])
        starts,ends=schedule(expected,r['down_first'])
        assert np.array_equal(starts,[n['start_ns'] for n in trace['trace']])
        assert np.array_equal(ends,[n['end_ns'] for n in trace['trace']])
        assert int(ends.max())==int(cost[0])
        reschedule_reports.append(dict(layer=r['layer'],case=r['case'],nodes=len(expected),all_start_end_times_exact=True))
        costs.setdefault(r['case'],[]).append((r['layer'],cost));bounds.setdefault(r['case'],[]).append((r['layer'],bound))
        inventories[r['case']]=inventories.get(r['case'],np.zeros((16,7),dtype=np.int64))+inventory
        records.append(dict(layer=r['layer'],case=r['case'],samples=12784,all_query_bounds_exact=True,
            mean_ns=float(cost.mean()),p99_ns=float(np.percentile(cost,99)),plan_sha256=r['plan_sha256']))
    assert seen==set(frozen_records)
    totals=[]
    for rr in screen['totals']:
        case=rr['case'];assert len(costs[case])==48
        value=np.stack([a for _,a in sorted(costs[case],key=lambda x:x[0])]).sum(axis=0)
        bound=np.stack([a for _,a in sorted(bounds[case],key=lambda x:x[0])]).sum(axis=0)
        assert abs(float(value.mean())-rr['mean_ns'])<1e-7
        assert abs(float(np.percentile(value,99))-rr['p99_ns'])<1e-7
        totals.append(dict(case=case,paired_48layer_mean_ns=float(value.mean()),paired_48layer_p99_ns=float(np.percentile(value,99)),
                           mean_sum_native_chip_bounds_ns=float(bound.mean()),samples=12784))
    capacity={case:dict(total_useful_bytes=int(a[:,0].sum()),per_chip_useful_bytes=a[:,0].tolist(),
        per_chip_actual_pages=a[:,1].tolist(),per_chip_reserved_wave_bytes=(a[:,2]*65536).tolist()) for case,a in inventories.items()}
    assert len({c['total_useful_bytes'] for c in capacity.values()})==1
    assert sha(FROZEN)==FROZEN_SHA
    sources=[Path(__file__),Path(__file__).with_name('research_20261102_qwen38_independent_audit.py'),
             Path(__file__).with_name('research_20261102_qwen38_schedule_audit.py')]
    output=dict(status='PASS independent frozen48-layer campaign audit',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in sources},freeze_sha256=FROZEN_SHA,screen_sha256=sha(screenpath),
        fit=fit,variants_coverage_checked=variants,static_plans_checked=240,
        cost_samples_aggregated=240*12784,all_query_native_chip_bounds_independently_exact=True,
        representative_nodes_independently_rescheduled=sum(x['nodes'] for x in reschedule_reports),
        all_representative_start_end_times_exact=True,records=records,totals=totals,
        representative_trace_checks=trace_reports,rescheduling_checks=reschedule_reports,physical_capacity=capacity,
        limitations=['All samples are the original one-task exploratory training stream, not the separately opened external QA task.',
            'All saved query scalars, bounds and aggregates checked; full independent event rescheduling is representative one query per static plan, not all3068160queries.',
            'Root compiled/Python equivalence audits are separate evidence. This audit imports neither production Graph nor C++ scheduler/generator.',
            '240plans comprise five ordinary geometry/stripe baseline families, not a novel algorithm claim.',
            'rect_min_io4 GU inherits grid-fitted labels; no claim of fully optimized rectangle assignment.',
            'Native output dtype/scales, I8 conversion accuracy and real PE/device service remain unverified; traces use sourceNVFP4 routed weights.',
            '48-layer FFN service includes one always-active shared expert; it excludes router/attention/SSM/HC/PLE/full-model walltime.',
            'Host activation and final combine remain zero defaults. Reconstruction is charged1000ns/expert/partial phase as synthetic sensitivity.',
            'Different layouts have identical useful weight payload but can have different actual page counts and reserved wave bytes; no vendor-capacity improvement is implied.'],
        external_payload_read_by_this_auditor=False,physical_plan_fit_performed=False)
    target=BASE/'independent-audit/audit-full48.json';target.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(path=str(target),sha256=sha(target),plans=240,queries=output['cost_samples_aggregated'],
        representative_nodes=output['representative_nodes_independently_rescheduled'],totals=totals),indent=2))


if __name__=='__main__':main()
