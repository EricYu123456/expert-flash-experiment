#!/usr/bin/env python3
"""Summarize frozen layouts, capacity, fixed-plan sensitivity, and research plots."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from materialize_plan import WaveAllocator
from run_corpus_traces import ROOT, sha256
from research_20261102_projection_study import bootstrap

DEST = ROOT / 'results/research-2026-11-02'


def capacity(phases, key):
    allocators = [WaveAllocator() for _ in range(16)]
    for phase in phases:
        for expert in phase[key]:
            for chip, count in enumerate(expert):
                if count:
                    allocators[chip].allocate(count * phase['columns'])
    values = [a.summary(chip) for chip, a in enumerate(allocators)]
    keys = ('useful_int8_bytes', 'physical_page_bytes', 'reserved_lsb_bytes',
            'idle_plane_slot_bytes', 'intra_page_padding_bytes', 'boundary_padding_waves',
            'final_block_tail_waves')
    return {k: sum(s[k] for s in values) for k in keys}


def main():
    study = json.loads((DEST / 'projection-study/summary.json').read_text())
    diagnostics = json.loads((DEST / 'diagnostics.json').read_text())
    findings, sensitivity = [], []
    for model in study['models']:
        name = model['model']
        plan = json.loads((DEST / 'projection-study' / (name + '.frozen_plan.json')).read_text())
        with np.load(DEST / 'projection-study' / (name + '.token_costs.npz')) as costs:
            arrays = {k: costs[k] for k in costs.files}
        with np.load(plan['test_path']) as cache:
            routes = cache['routes']
        conservative = {}
        for label in ('baseline', 'legacy', 'row_only', 'chosen'):
            drained = np.zeros(routes.shape[:2], dtype=np.int64)
            overlapped = np.zeros_like(drained)
            for phase in plan['phases']:
                selected = routes[:, :, plan['layers'].index(phase['layer'])]
                rows = np.asarray(phase[label + '_rows'], dtype=np.int64)
                pages = (rows * phase['columns'] + 16383) // 16384
                waves = (pages + 3) // 4
                nand = waves[selected].sum(axis=2) * 28000
                ecc = pages[selected].sum(axis=2) * 2560
                # Read one four-plane wave, drain all its pages, then reuse PBs.
                # Summing those per-wave drains equals total page_work*2560.
                drained += (nand + ecc).max(axis=2)
                overlapped += np.maximum(nand, ecc).max(axis=2)
            np.testing.assert_array_equal(overlapped, arrays[label + '__flash_ns'])
            conservative[label] = arrays[label + '__model_ns'] - overlapped + drained
        capacities = {k: capacity(plan['phases'], k + '_rows')
                      for k in ('baseline', 'legacy', 'row_only', 'chosen')}
        assert all(c['useful_int8_bytes'] == capacities['baseline']['useful_int8_bytes']
                   for c in capacities.values())
        # All evaluated layouts use all four planes per stream. At >=14us/read,
        # even a full 4-page wave is slower than 4*2.56us ECC drain. Thus NAND
        # dominates every chip and scaling the saved NAND component is exact
        # within the same ideal-overlap analytical model.
        for read_us in (14, 28, 56):
            assert read_us >= 4 * 2.56
            for control_ns in (0, 250, 1000):
                result = dict(model=name, ordinary_read_us=read_us,
                              chip_command_ns=control_ns, layouts_remain_frozen=True)
                for label in ('legacy', 'chosen'):
                    ns = (arrays[label + '__flash_ns'] * (read_us / 28)
                          + arrays[label + '__control_ns'] * (control_ns / 250)
                          + arrays[label + '__host_ns'] + arrays[label + '__input_ns']
                          + arrays[label + '__output_ns'])
                    result[label + '_ms'] = float(ns.mean() / 1e6)
                result['improvement_pct'] = 100 * (1 - result['chosen_ms'] / result['legacy_ms'])
                sensitivity.append(result)
        means = model['means_ms']
        saved_ms = means['legacy']['model_ns'] - means['chosen']['model_ns']
        ctrl_difference = means['chosen']['control_ns'] - means['legacy']['control_ns']
        intercept = (means['legacy']['model_ns'] - means['legacy']['control_ns']
                     - means['chosen']['model_ns'] + means['chosen']['control_ns'])
        break_even = (250 * intercept / ctrl_difference) if ctrl_difference > 1e-9 else None
        findings.append(dict(model=name, capacity=capacities,
                             saved_ms_vs_legacy=saved_ms,
                             extra_unmodeled_cost_break_even_us_per_layer=saved_ms * 1000 / len(plan['layers']),
                             chip_command_break_even_ns_at_28us=break_even,
                             conservative_drain_ms={k: float(v.mean() / 1e6)
                                                    for k, v in conservative.items()},
                             conservative_drain_gain_vs_legacy_pct=100 * (1 - conservative['chosen'].mean() /
                                                                                 conservative['legacy'].mean()),
                             conservative_drain_ci_pct=bootstrap(conservative['legacy'], conservative['chosen']),
                             physical_bytes_ratio_vs_legacy=capacities['chosen']['physical_page_bytes'] /
                                                          capacities['legacy']['physical_page_bytes'],
                             reserved_lsb_ratio_vs_legacy=capacities['chosen']['reserved_lsb_bytes'] /
                                                        capacities['legacy']['reserved_lsb_bytes']))
    (DEST / 'analysis.json').write_text(json.dumps(dict(
        findings=findings, sensitivity=sensitivity,
        interpretation='Sensitivity changes costs of immutable base-trained layouts; '
                       'it does not reselect or retrain configurations at new costs.',
        source_sha256=sha256(Path(__file__))), indent=2) + '\n')
    with (DEST / 'sensitivity.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity[0]))
        writer.writeheader()
        writer.writerows(sensitivity)
    names = [m['model'] for m in study['models']]
    labels = ['Granite', 'OLMoE', 'DeepSeek', 'Qwen3', 'Mixtral']
    x = np.arange(len(names))
    headroom = [max(0, 100 * next(m for m in diagnostics['models'] if m['model'] == n)
                    ['pooled_nand_gap_fraction']) for n in names]
    improvements = [m['improvement_vs_legacy_pct'] for m in study['models']]
    cis = np.array([m['ci_vs_legacy_pct'] for m in study['models']])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    axes[0].bar(x, headroom, color='#6e859c', width=.62)
    axes[0].set_title('Remaining NAND work above pooled lower bound')
    axes[0].set_ylabel('Fraction of baseline NAND time (%)')
    axes[0].set_ylim(0, 29)
    axes[1].bar(x, improvements, color='#137f78', width=.62)
    axes[1].errorbar(x, improvements,
                    yerr=np.maximum(0, np.stack([np.array(improvements)-cis[:, 0],
                                                cis[:, 1]-np.array(improvements)])),
                    fmt='none', color='#1b2b34', capsize=4)
    axes[1].set_title('Projection-specific layout vs previous best')
    axes[1].set_ylabel('Routed FFN service reduction (%)')
    axes[1].set_ylim(0, 7.5)
    for ax, values in zip(axes, (headroom, improvements)):
        ax.set_xticks(x, labels)
        ax.grid(axis='y', alpha=.17)
        ax.set_axisbelow(True)
        for xi, value in zip(x, values):
            ax.text(xi, value + (.45 if ax is axes[0] else .18), f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9)
    fig.suptitle('Ordinary reads only: 16 chips, four planes per chip', fontsize=13, y=1.01)
    fig.text(.5, -.02,
             'Trace-driven cost model. Right: 100 reused test prompts/model, 128 decode steps; '
             '95% paired prompt bootstrap. Not end-to-end hardware latency.',
             ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(DEST / 'research_overview.png', dpi=180, bbox_inches='tight')
    fig.savefig(DEST / 'research_overview.pdf', bbox_inches='tight')
    print(json.dumps(findings, indent=2))


if __name__ == '__main__':
    main()
