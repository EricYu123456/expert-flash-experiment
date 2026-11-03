#!/usr/bin/env python3
"""Audit frozen persistent-copy artifacts and derive comparison tables."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((HERE / 'summary.json').read_text())
    coded = json.loads((HERE / 'coded_reference.json').read_text())
    archive = ROOT / 'results/research-2026-11-02/round2/coded-latency'
    completion = json.loads((archive / 'completion.json').read_text())
    assert len(summary['models']) == 3
    assert summary['source_sha256'] == sha(ROOT / 'scripts/research_20261102_persistent_replica.py')
    assert summary['coded_source_sha256'] == sha(ROOT / 'scripts/research_20261102_coded_latency.py')
    assert summary['coded_reference_sha256'] == sha(HERE / 'coded_reference.json')
    assert sha(HERE / 'coded_reference.json') == sha(archive / 'summary.json')
    for path, expected in completion['files_sha256'].items():
        assert sha(ROOT / path) == expected, path

    capacities = []
    comparisons = []
    no_copy_checks = 0
    clean_cap_checks = 0
    for model in summary['models']:
        name = model['model']
        reference = next(m for m in coded['models'] if m['model'] == name)
        assert model['phase_ledger_sha256'] == sha(archive / 'phase_ledgers' / (name + '.npz'))
        frozen = ROOT / f'results/research-2026-11-02/projection-study/{name}.frozen_plan.json'
        assert model['chosen_projection_sha256'] == sha(frozen)
        for key, receipt in model['plans'].items():
            path = Path(receipt['path'])
            assert receipt['sha256'] == sha(path)
            plan = json.loads(path.read_text())
            assert plan['base_projection_sha256'] == sha(frozen)
            total_bytes, total_pages = 0, 0
            for phase, cap in zip(plan['phases'], reference['capacity']['phases']):
                owners = phase['owners']
                actual_bytes = sum(tile['bytes'] for tile in owners)
                actual_pages = sum(tile['pages'] for tile in owners)
                assert actual_bytes == phase['actual_copy_bytes'] <= phase['budget_bytes']
                assert actual_pages == phase['actual_copy_pages'] <= phase['budget_pages']
                assert all(tile['source'] != tile['alternate_chip'] for tile in owners)
                identities = [(t['expert'], t['source'], t['row_start']) for t in owners]
                assert len(identities) == len(set(identities))
                if plan['budget'] == 'same_as_coded':
                    assert phase['budget_bytes'] == cap['parity_weight_bytes']
                    assert phase['budget_pages'] == cap['parity_physical_pages']
                total_bytes += actual_bytes
                total_pages += actual_pages
            assert receipt['actual_copy_bytes'] == total_bytes
            assert receipt['actual_copy_pages'] == total_pages
            capacities.append(dict(model=name, policy=key,
                actual_bytes=total_bytes, budget_bytes=receipt['budget_bytes'],
                actual_pages=total_pages, budget_pages=receipt['budget_pages'],
                exact_unit_wave_assignment=receipt['exact_all_tiles_unit_wave']))

        for scenario in model['scenarios']:
            matched = next(s for s in reference['scenarios']
                if s['erase_ms'] == scenario['erase_ms'] and s['duty'] == scenario['duty'])
            policies, ref = scenario['policies'], matched['policies']
            for cap in ('none', '0', '10000', '100000', '1000000'):
                np.testing.assert_allclose(policies['chosen_wait_cap' + cap]['per_token_ms'],
                    ref['chosen_wait_cap' + cap + '_decode0']['per_token_ms'], rtol=0, atol=1e-7)
                no_copy_checks += 1
                if scenario['duty'] == 0:
                    for key in model['plans']:
                        np.testing.assert_array_equal(policies[key + '_cap' + cap]['per_token_ms'],
                            policies[key + '_capnone']['per_token_ms'])
                        clean_cap_checks += 1
                for decode in ('0', '2000', '10000', 'measured_median', 'measured_p95'):
                    code_key = 'coded_cap' + cap + '_decode' + decode
                    if code_key not in ref:
                        continue
                    for policy in model['plans']:
                        copy = policies[policy + '_cap' + cap]
                        row = dict(model=name, erase_ms=scenario['erase_ms'], duty=scenario['duty'],
                            cap_ns=None if cap == 'none' else int(cap),
                            coded_decode=decode, uncoded_policy=policy)
                        for metric in ('mean_ms', 'p95_ms', 'p99_ms'):
                            row['coded_' + metric] = ref[code_key][metric]
                            row['uncoded_resource_' + metric] = copy[metric]
                            row['coded_gain_vs_uncoded_resource_' + metric + '_pct'] = 100 * (
                                1 - ref[code_key][metric] / copy[metric])
                        comparisons.append(row)

    limitations = [
        'The full-token replay values are resource-comparator outputs, not rigorous end-to-end lower bounds: autonomous policy clocks encounter different absolute background intervals.',
        'The flow minimum is exact for integral one-wave jobs; multi-wave whole-output-row tiles may be split in the relaxation. DeepSeek and Qwen therefore lack executable schedules.',
        'Original I/O and matrix-block command costs are held fixed after redirection. Actual copy input fanout, output channels, extra dispatch and routing must be rebuilt before claiming implementation latency.',
        'Only two conventional frequency-only fixed layouts are screened. Neither the best static-copy placement nor a coactivation-aware optimum is proved.',
        'Routes are 32 samples from the existing 200-prompt training traces; this is not held-out statistical validation. P99 is an interpolation over 32 values.',
        'The synthetic global one-chip erase calendar is not a calibrated device model. Read-only inference does not itself generate these erases.',
        'Read availability caps do not simulate suspend/resume steps, lost erase work, erase starvation, background maintenance progress or write service-level objectives.',
        'A new erase beginning inside a phase is counted as uncertainty and does not affect the phase, matching the shared phase-start admission model.',
        'No measured in-flash hardware, full-model end-to-end latency, production GC rates or system novelty claim follows from this experiment.'
    ]
    audit = dict(status='passed', source_sha256=summary['source_sha256'],
        coded_source_sha256=summary['coded_source_sha256'],
        summary_sha256=sha(HERE / 'summary.json'),
        coded_reference_sha256=summary['coded_reference_sha256'],
        independent_bruteforce_checks=summary['independent_unit_job_bruteforce_checks'],
        exact_no_copy_replay_checks=no_copy_checks, clean_cap_invariance_checks=clean_cap_checks,
        fixed_plan_count=len(capacities), all_phase_byte_page_budgets_respected=True,
        capacities=capacities, limitations=limitations)
    (HERE / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    findings = dict(status='No universal large-tail breakthrough established',
        interpretation='Coding has a capacity advantage during long nonpreemptive stalls in these two uncoded controls. The advantage does not survive the strong 100us-cap plus same-capacity uncoded resource screen. This is an elimination screen, not a proof of implementable uncoded superiority.',
        same_capacity='Both physical NAND pages and useful INT8 bytes are hard-capped per phase. DeepSeek matches the actual 26.911695% padded parity capacity, rather than calling it 25%.',
        coding_decode='Warm-memory host measured median/p95 values are imported from the frozen coded study; no new host measurement is made here.',
        clean_control='With the selected unequal/G8 source layouts, fixed copies can improve healthy load balance. The balanced-G16 zero-gain theorem does not apply to these different base loads.',
        statistical_scope='The same immutable placement and routes are replayed under fixed synthetic calendar parameters. No confidence interval or empirical workload frequency claim is made.',
        main_setting=[row for row in comparisons if row['erase_ms'] == 5 and row['duty'] == .25 and row['coded_decode'] == 'measured_p95'],
        comparisons=comparisons, limitations=limitations)
    (HERE / 'findings.json').write_text(json.dumps(findings, indent=2) + '\n')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('capacities', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
