#!/usr/bin/env python3
"""Separate conditional read-phase cadence sensitivity; sealed v1 unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[4]
SOURCE = ROOT / 'scripts/research_20261102_cr_ipr_pipeline.py'
SEALED_SHA = 'ad35d08e17b4458cb5a0f54732aafce338d0526f5888fb02fcfdcc507ad99905'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SEALED_SHA
spec = importlib.util.spec_from_file_location('sealed_cadence_pipeline', SOURCE)
v1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v1
spec.loader.exec_module(v1)


def main():
    plan_path = DEST / 'plan.frozen.json'
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    layouts = dict(candidate12=v1.build_g4_candidate(),
                   packed_adjacent12=v1.build_ordinary_packed(),
                   packed_spaced15_keep_hot_drop=v1.with_keep_hot(v1.build_ordinary_packed([0,3,6,9,12,15])))
    records = []
    traces = {}
    for mode in ('fixed_recycle', 'fixed_evaluation'):
        for cr in (9700,10000,10240,10500,11000):
            recycle = 6040 if mode == 'fixed_recycle' else cr - 3660
            for copy in (100,500,1000,2000):
                for label, waves in layouts.items():
                    r = v1.simulate(waves, cr_ns=cr, recycle_ns=recycle, cache_copy_ns=copy,
                                    unselected_policy='drop_before_ecc_conditional')
                    assert r['issue_stall_ns'] == 0
                    assert all(a['idle_before_ns'] == 0 for a in r['array_waves'])
                    assert r['cold_waves'] == 1 and r['cr_waves'] == len(waves)-1
                    records.append(dict(mode=mode, layout=label, cr_ns=cr,
                                        recycle_ns=recycle, evaluation_ns=cr-recycle, copy_ns=copy,
                                        **v1._summary(r)))
                    if label == 'candidate12' and copy == 500 and cr in (9700,10000,10240,10500,11000):
                        traces[f'{mode}.cr{cr}.copy500'] = r
    counter = next(x for x in records if x['mode'] == 'fixed_recycle' and x['layout'] == 'candidate12'
                   and x['cr_ns'] == 10240 and x['copy_ns'] == 500)
    assert counter['feasible'] and counter['makespan_ns'] == 151380
    nominal = next(x for x in records if x['mode'] == 'fixed_recycle' and x['layout'] == 'candidate12'
                   and x['cr_ns'] == 9700 and x['copy_ns'] == 500)
    assert not nominal['feasible']
    report = dict(status='PASS', sealed_v1_sha256=SEALED_SHA,
                  frozen_plan_sha256=plan_hash,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cases=len(records), records=records,
                  priority_countercontrol=counter,
                  interpretation=['At copy500ns, candidate12 at10.24us cadence completes151.380us, faster than13wave fixed9.7us shaping155.140us.',
                                  'The10.24us read period equals the four-page6.4GB/s ECC service time; no accumulating540ns/wave backlog remains.',
                                  'Fixed-recycle and fixed-evaluation modes assign extra read duration to different phases explicitly.',
                                  'Every next wave starts at prior completion; no issue-idle bias hold is used.',
                                  'These are conditional timing parameters, not evidence that the native chip can legally stretch a phase or that new hardware is needed.',
                                  'The published9.7us raw minimum and6.4GB/s effective rate may already embody cadence/control overhead; neither is silently counted twice.',
                                  'Shaping benefit is restricted to a contract fixing9.7us where a slower legal read phase is unavailable or disadvantageous.',
                                  'All ordinary packed controls use the same capability set; any result is a service envelope ending at ECC, not a measured fullGEMV cost.'])
    (DEST / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for key, trace in traces.items():
        (DEST / (key+'.events.json')).write_text(json.dumps(trace, indent=2)+'\n')
    lines = ['# Conditional read-phase cadence control', '',
             '**A slower legal read phase can beat dummy shaping in this model.** With copy500ns, candidate12 at10.24us/read completes151.380us, versus155.140us for13wave shaping at fixed9.7us. The nominal9.7us candidate fails finite-PB timing.', '',
             'This is a source timing parameter sensitivity. No idle interval is inserted: every wave starts exactly when the previous array operation completes. Native legality or programmability of longer evaluation/recycling is unresolved; neither existing support nor a hardware requirement is established.', '',
             '| Phase assignment | CR us | Candidate copy100 ns | copy500 ns | copy1000 ns | copy2000 ns |',
             '|---|---:|---:|---:|---:|---:|']
    for mode in ('fixed_recycle','fixed_evaluation'):
        for cr in (9700,10000,10240,10500,11000):
            rr = [next(x for x in records if x['mode']==mode and x['layout']=='candidate12' and x['cr_ns']==cr and x['copy_ns']==c) for c in (100,500,1000,2000)]
            vals = [f"{x['makespan_ns']/1000:.3f}" if x['feasible'] else 'FAIL' for x in rr]
            lines.append('| '+mode+' | '+str(cr/1000)+' | '+' | '.join(vals)+' |')
    lines += ['', 'Entries are completion times in us. Fixed-recycle keeps6.04us and assigns the added duration to evaluation. Fixed-evaluation keeps3.66us and assigns it to recycling. Both modes preserve one cold start and continuous issue.', '',
              'The JSON also includes packed adjacent12 and spaced15 keep-hot/drop controls, for120cases total. Physical reads, useful/ECC bytes, latch deadlines, startup and final ECC drain remain explicit.', '',
              'A10.24us four-page arrival period matches shared ECC service and removes the accumulating540ns/wave backlog. Strict9.7us failure therefore cannot be a standalone novelty claim. Dummy shaping is useful only under the narrower fixed9.7us contract when legal phase extension is unavailable or otherwise worse.', '',
              'Published raw minimum latency and effective6.4GB/s may already incorporate cadence or overhead. This experiment does not infer a native timer interface, and it does not combine an effective-bandwidth approximation with a second raw array charge.']
    (DEST / 'report.md').write_text('\n'.join(lines)+'\n')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SEALED_SHA
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == plan_hash
    print(json.dumps(dict(status='PASS', cases=len(records), countercontrol=counter),indent=2))


if __name__=='__main__':
    main()
