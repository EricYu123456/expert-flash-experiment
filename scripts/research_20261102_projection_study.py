#!/usr/bin/env python3
"""Freeze projection-specific ordinary-read layouts, then evaluate old test routes.

This is an exploratory reuse of the existing corpus, not a new blind test set.
All weights have one immutable whole-output-row owner; no CR or plane contexts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import aif_model as flash
from materialize_plan import WaveAllocator
from run_corpus_traces import ROOT, DATA, MODELS, geometry, sha256

COMPONENTS = ('flash_ns', 'control_ns', 'host_ns', 'input_ns', 'output_ns')
DEST = ROOT / 'results/research-2026-11-02/projection-study'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def equal_rows(rows, experts, g=16, mapping=None):
    mapping = np.arange(experts) if mapping is None else np.asarray(mapping)
    result = np.zeros((experts, 16), dtype=np.int64)
    for expert, slot in enumerate(mapping):
        first = (int(slot) % (16 // g)) * g
        for shard in range(g):
            result[expert, first + shard] = rows // g + (shard < rows % g)
    assert np.all(result.sum(axis=1) == rows)
    return result


def evaluate(selected, rows, columns, common, use_cr=False):
    rows = np.asarray(rows, dtype=np.int64)
    pages = (rows * columns + 16383) // 16384
    waves = (pages + 3) // 4
    runs = (waves + 191) // 192
    wave_work = waves[selected].sum(axis=1)
    page_work = pages[selected].sum(axis=1)
    run_work = runs[selected].sum(axis=1)
    if use_cr:
        nand = (waves * 9700 + runs * 18300)[selected].sum(axis=1).max(axis=1)
    else:
        nand = wave_work.max(axis=1) * 28000
    inputs = ((rows[selected] > 0).any(axis=1) if common else
              (rows[selected] > 0).sum(axis=1)) * columns
    outputs = rows[selected].sum(axis=1) * 4
    host_input = columns * (1 if common else selected.shape[1])
    result = dict(
        flash_ns=np.maximum(nand, page_work.max(axis=1) * 2560),
        control_ns=run_work.sum(axis=1) * 250,
        host_ns=np.full(len(selected), 2000, dtype=np.int64),
        input_ns=np.maximum((host_input + 7) // 8,
                            (np.max(inputs[:, :8] + inputs[:, 8:], axis=1) + 1) // 2),
        output_ns=np.maximum((outputs.sum(axis=1) + 7) // 8,
                             (np.max(outputs[:, :8] + outputs[:, 8:], axis=1) + 1) // 2))
    result['model_ns'] = sum(result.values())
    return result


def scalar_cost(selected, rows, columns, common):
    """Independent scalar byte/page/chip/channel calculation for ordinary reads."""
    waves, pages, starts, inputs, outputs = ([0] * 16 for _ in range(5))
    for expert in selected:
        for chip in range(16):
            count = int(rows[expert][chip])
            if count == 0:
                continue
            nbytes = count * columns
            p = (nbytes + 16383) // 16384
            w = (p + 3) // 4
            pages[chip] += p
            waves[chip] += w
            starts[chip] += (w + 191) // 192
            inputs[chip] = columns if common else inputs[chip] + columns
            outputs[chip] += count * 4
    host_in = columns * (1 if common else len(selected))
    result = dict(
        flash_ns=max(max(waves) * 28000, max(pages) * 2560),
        control_ns=sum(starts) * 250, host_ns=2000,
        input_ns=max((host_in + 7) // 8,
                     max((inputs[c] + inputs[c + 8] + 1) // 2 for c in range(8))),
        output_ns=max((sum(outputs) + 7) // 8,
                      max((outputs[c] + outputs[c + 8] + 1) // 2 for c in range(8))))
    result['model_ns'] = sum(result.values())
    return result


def model_phases(dimensions, layer):
    d = dimensions[layer]
    return [('gate_up', d['gate'][0] + d['up'][0], d['up'][1], True),
            ('down', *d['down'], False)]


def materialize(name, phases, experts):
    allocators = [WaveAllocator() for _ in range(16)]
    records = []
    for phase in phases:
        rows = np.asarray(phase['chosen_rows'])
        for expert in range(experts):
            row_start = 0
            for chip in range(16):
                count = int(rows[expert, chip])
                if count:
                    p, w, nrun, segments = allocators[chip].allocate(count * phase['columns'])
                    records.append([phase['layer'], expert, phase['name'], chip,
                                    row_start, count, phase['columns'], p, w, nrun, segments])
                row_start += count
            assert row_start == phase['total_rows']
    summaries = [a.summary(c) for c, a in enumerate(allocators)]
    expected = sum(p['total_rows'] * p['columns'] * experts for p in phases)
    assert sum(s['useful_int8_bytes'] for s in summaries) == expected
    value = dict(model=name, artifact_type='virtual_addresses_only_no_programmed_weights',
                 columns=['layer', 'expert', 'phase', 'chip', 'row_start', 'row_count',
                          'input_columns', 'pages', 'waves', 'block_runs', 'segments'],
                 segment_columns=['block', 'wordline_offset', 'waves', 'page_offset',
                                  'pages', 'plane_page_counts'],
                 streams=records, chips=summaries)
    path = DEST / (name + '.physical_plan.json.gz')
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(value, separators=(',', ':')).encode())
    # Reparse the serialized plan and independently check rows and address ranges.
    with gzip.open(path, 'rt') as stream:
        reread = json.load(stream)
    coverage = {}
    last_address = [0] * 16
    allocated_pages = 0
    for layer, expert, pname, chip, start, count, cols, pages, waves, runs, segments in reread['streams']:
        key = (layer, expert, pname)
        assert start == coverage.get(key, 0)
        coverage[key] = start + count
        assert pages == (count * cols + 16383) // 16384
        assert waves == (pages + 3) // 4 and len(segments) == runs
        segment_pages = 0
        for block, wl, nwaves, page_offset, npages, counts in segments:
            first = block * 192 + wl
            assert first >= last_address[chip] and wl + nwaves <= 192
            assert page_offset == segment_pages
            assert sum(counts) == npages and max(counts) == nwaves
            last_address[chip] = first + nwaves
            segment_pages += npages
        assert segment_pages == pages
        allocated_pages += pages
    for phase in phases:
        for expert in range(experts):
            assert coverage[(phase['layer'], expert, phase['name'])] == phase['total_rows']
    return dict(path=str(path), sha256=sha256(path), streams=len(records),
                physical_pages=allocated_pages, useful_bytes=expected,
                reserved_lsb_bytes=sum(s['reserved_lsb_bytes'] for s in summaries),
                actual_page_bytes=sum(s['physical_page_bytes'] for s in summaries),
                row_coverage_once=True, nonoverlapping_chip_plane_wl=True)


def fit():
    diagnostics_path = DEST.parent / 'diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text())
    assert not diagnostics['test_routes_opened']
    assert diagnostics['source_sha256'] == sha256(ROOT / 'scripts/research_20261102_diagnostics.py')
    protocol = dict(
        purpose='projection-specific fixed whole-row placement without CR',
        candidate_rule='Each layer/phase chooses the lowest training mean among equal g16, '
                       'existing trained whole-row g1/2/4/8/16 groups (w4 only), '
                       'and the train-only wave-aligned row template. Ties prefer equal g16.',
        training_prompts=200, test_prompts=100, decode_steps=128,
        no_test_grid_or_test_selection=True,
        evaluation_scope='routed FFN projection service, not full-model latency',
        reused_test_warning='The corpus test was used by earlier studies and its aggregate results '
                            'were already known when proposing this method. This is exploratory '
                            'fixed-test reuse, not a new blind confirmation.',
        hardware='16 chips, 4 same-context planes/chip, 16KiB pages, 28us ordinary read, '
                 '6.4GB/s chip, 250ns/block-shard command, 2us/host phase; input INT8/output FP32',
        dependency='gate_up -> host reassembly and activation -> down; no inter-layer or next-token overlap',
        unmeasured='host row reassembly, activation/combine compute, physical PE/NAND and FTL',
        source_sha256={p: sha256(ROOT / 'scripts' / p) for p in
                       ('research_20261102_projection_study.py', 'research_20261102_diagnostics.py',
                        'materialize_plan.py', 'aif_model.py')},
        diagnostics_sha256=sha256(diagnostics_path))
    write_json(DEST / 'protocol.json', protocol)
    receipts = []
    for model in diagnostics['models']:
        name = model['model']
        legacy_path = Path(model['provenance']['frozen_plan_path'])
        assert sha256(legacy_path) == model['provenance']['frozen_plan_sha256']
        legacy = json.loads(legacy_path.read_text())
        metadata_path = DATA / 'models' / (name + '-metadata.json')
        assert sha256(metadata_path) == model['provenance']['metadata_sha256']
        metadata = json.loads(metadata_path.read_text())
        dimensions = flash.tensor_dimensions(metadata)
        layers, experts, active = geometry(metadata)
        train_path = Path(model['provenance']['train_cache_path'])
        assert sha256(train_path) == model['provenance']['train_cache_sha256']
        with np.load(train_path) as cached:
            routes = cached['routes']
            train_ids = cached['prompt_ids'].tolist()
        assert train_ids == model['provenance']['prompt_ids']
        legacy_winner = legacy['winners']['group_only_learned']
        assert legacy_winner['plane_width'] == 4
        phases = []
        training_totals = Counter()
        for li, layer in enumerate(layers):
            selected, weights = np.unique(np.sort(routes[:, :, li].reshape(-1, active), axis=1),
                                          axis=0, return_counts=True)
            for pname, total_rows, columns, common in model_phases(dimensions, layer):
                baseline = equal_rows(total_rows, experts)
                row_spec = model['row_shard_plans'][li][pname]
                row_variants = np.array([np.roll(row_spec['template_rows'], o) for o in range(16)])
                fractional = row_variants[np.array(row_spec['offsets'])]
                choices = [('equal_g16', baseline)]
                for cell in legacy['training_candidates']:
                    if cell['family'] == 'group_only_learned' and cell['plane_width'] == 4:
                        choices.append((cell['key'], equal_rows(total_rows, experts,
                                       cell['group_chips'], legacy['mappings'][cell['key']][str(layer)])))
                choices.append(('wave_rows', fractional))
                scores = []
                for label, candidate in choices:
                    values = evaluate(selected, candidate, columns, common)
                    mean = float(values['model_ns'] @ weights / weights.sum())
                    scores.append(dict(label=label, mean_ns=mean))
                chosen_index = min(range(len(scores)), key=lambda i: (scores[i]['mean_ns'], i))
                chosen_label, chosen = choices[chosen_index]
                chosen_cost = evaluate(selected, chosen, columns, common)
                for key, value in chosen_cost.items():
                    training_totals[key] += float(value @ weights / weights.sum())
                legacy_rows = equal_rows(total_rows, experts, legacy_winner['group_chips'],
                                        legacy['mappings'][legacy_winner['key']][str(layer)])
                phases.append(dict(layer=layer, name=pname, total_rows=total_rows,
                                   columns=columns, common=common, candidates=scores,
                                   chosen_label=chosen_label, chosen_rows=chosen.tolist(),
                                   baseline_rows=baseline.tolist(), legacy_rows=legacy_rows.tolist(),
                                   row_only_rows=fractional.tolist()))
        physical = materialize(name, phases, experts)
        # The source row partitions must store exactly one complete expert.
        plan = dict(model=name, phases=phases, train_ids=train_ids, layers=layers,
                    experts=experts, active=active, training_means_ns=dict(training_totals),
                    train_path=str(train_path), train_sha256=sha256(train_path),
                    metadata_path=str(metadata_path), metadata_sha256=sha256(metadata_path),
                    legacy_path=str(legacy_path), legacy_sha256=sha256(legacy_path),
                    test_path=str(train_path.with_name('test_routes.npz')),
                    physical=physical, frozen_utc=datetime.now(timezone.utc).isoformat(),
                    test_routes_opened_before_freeze=False)
        path = DEST / (name + '.frozen_plan.json')
        write_json(path, plan)
        receipts.append(dict(model=name, path=str(path), sha256=sha256(path)))
        print(name, 'train selected ms', training_totals['model_ns'] / 1e6,
              'choices', dict(Counter(p['chosen_label'] for p in phases)), flush=True)
    write_json(DEST / 'all_models_freeze_receipt.json',
               dict(plans=receipts, protocol_sha256=sha256(DEST / 'protocol.json'),
                    test_routes_opened=False, frozen_utc=datetime.now(timezone.utc).isoformat()))


def bootstrap(baseline, candidate, seed=20261102):
    # Resample complete prompts, retaining every decode step inside a prompt.
    b = baseline.mean(axis=1)
    c = candidate.mean(axis=1)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(b), (2000, len(b)))
    improvement = 1 - c[index].mean(axis=1) / b[index].mean(axis=1)
    return (100 * np.quantile(improvement, [0.025, 0.975])).tolist()


def test():
    receipt = json.loads((DEST / 'all_models_freeze_receipt.json').read_text())
    assert not receipt['test_routes_opened']
    assert receipt['protocol_sha256'] == sha256(DEST / 'protocol.json')
    protocol = json.loads((DEST / 'protocol.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        assert sha256(ROOT / 'scripts' / path) == digest
    for item in receipt['plans']:
        assert sha256(Path(item['path'])) == item['sha256']
    results = []
    for item in receipt['plans']:
        plan = json.loads(Path(item['path']).read_text())
        assert sha256(Path(plan['legacy_path'])) == plan['legacy_sha256']
        assert sha256(Path(plan['physical']['path'])) == plan['physical']['sha256']
        with np.load(plan['test_path']) as cached:
            routes = cached['routes']
            ids = cached['prompt_ids'].tolist()
            assert cached['layers'].tolist() == plan['layers']
        assert len(ids) == 100 and set(ids).isdisjoint(plan['train_ids'])
        assert all(s.startswith('test_') for s in ids)
        shape = routes.shape[:2]
        totals = {label: {k: np.zeros(shape, dtype=np.int64) for k in (*COMPONENTS, 'model_ns')}
                  for label in ('baseline', 'legacy', 'row_only', 'chosen')}
        checks = 0
        for phase in plan['phases']:
            li = plan['layers'].index(phase['layer'])
            selected = routes[:, :, li].reshape(-1, plan['active'])
            for label in totals:
                rows = np.array(phase[label + '_rows'])
                assert np.all(rows.sum(axis=1) == phase['total_rows'])
                values = evaluate(selected, rows, phase['columns'], phase['common'])
                for index in (0, len(selected) // 2, len(selected) - 1):
                    oracle = scalar_cost(selected[index], rows, phase['columns'], phase['common'])
                    for key in oracle:
                        assert int(values[key][index]) == oracle[key]
                    checks += 1
                for key in totals[label]:
                    totals[label][key] += values[key].reshape(shape)
        old_tokens = ROOT / 'results/plane-wave-v2/no-cr' / (plan['model'] + '.token_costs.npz')
        # Stored old arrays have policy-qualified keys; compare when available.
        with np.load(old_tokens) as saved:
            matching = [k for k in saved.files if k == 'joint_group_plane__model_ns']
            if matching:
                np.testing.assert_array_equal(totals['legacy']['model_ns'], saved[matching[0]])
        means = {label: {k: float(v.mean() / 1e6) for k, v in value.items()}
                 for label, value in totals.items()}
        value = dict(model=plan['model'], means_ms=means,
                     improvement_vs_baseline_pct=100 * (1 - means['chosen']['model_ns'] / means['baseline']['model_ns']),
                     improvement_vs_legacy_pct=100 * (1 - means['chosen']['model_ns'] / means['legacy']['model_ns']),
                     ci_vs_baseline_pct=bootstrap(totals['baseline']['model_ns'], totals['chosen']['model_ns']),
                     ci_vs_legacy_pct=bootstrap(totals['legacy']['model_ns'], totals['chosen']['model_ns']),
                     chosen_counts=dict(Counter(p['chosen_label'] for p in plan['phases'])),
                     scalar_phase_policy_checks=checks,
                     p95_ms={label: float(np.quantile(v['model_ns'], .95) / 1e6) for label, v in totals.items()},
                     test_ids=ids, test_sha256=sha256(Path(plan['test_path'])),
                     frozen_sha256=item['sha256'], physical=plan['physical'])
        np.savez_compressed(DEST / (plan['model'] + '.token_costs.npz'),
                            **{label + '__' + k: v for label, data in totals.items() for k, v in data.items()})
        results.append(value)
        print(plan['model'], 'test', json.dumps({k: value[k] for k in
              ('improvement_vs_baseline_pct', 'improvement_vs_legacy_pct', 'ci_vs_legacy_pct')}), flush=True)
    write_json(DEST / 'summary.json', dict(protocol=protocol, models=results,
               selection='all choices and weights frozen before this runner read any test routes',
               validation='exploratory reuse of existing held-out corpus; new external confirmation required'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fit', 'test'), required=True)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    (fit if args.stage == 'fit' else test)()


if __name__ == '__main__':
    main()
