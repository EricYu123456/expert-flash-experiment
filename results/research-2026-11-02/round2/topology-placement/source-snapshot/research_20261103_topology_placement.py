#!/usr/bin/env python3
"""Independent TRAIN-only fixed-plan ONFI-topology headroom certificate.

Common physical-chip permutations preserve per-chip native context work,
whole-expert dependence paths, controller/host/PCIe volume, and total ONFI
volume. This program does not fit labels or mutate existing plans/backend.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/ordinary-read-compiler'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def transfer(nbytes, rate):
    return (int(nbytes) * 1_000_000_000 + int(rate) - 1) // int(rate)


def stats(actual, lower):
    actual = np.asarray(actual); lower = np.asarray(lower)
    assert np.all(actual >= lower), (actual - lower).min()
    return dict(queries=len(actual), mean_ns=float(actual.mean()),
                lower_bound_mean_ns=float(lower.mean()),
                possible_mean_reduction_pct=float(100 * (1 - lower.mean() / actual.mean())),
                p99_ns=float(np.percentile(actual, 99)),
                lower_bound_p99_ns=float(np.percentile(lower, 99)),
                possible_p99_reduction_pct=float(100 * (1 - np.percentile(lower, 99) / np.percentile(actual, 99))),
                max_query_possible_reduction_pct=float(100 * np.max(1 - lower / actual)))


def certificate(plan, service):
    s = service
    chips, channels = s['chips'], s['channels']
    copies = s.get('input_channel_copies', 1)
    offsets, jobs = plan['offsets'], plan['jobs']
    E = len(offsets)
    work = np.zeros((E, 2, chips), np.int64)
    traffic = np.zeros_like(work)
    counts = np.zeros((E, 2), np.int64)
    pciout = np.zeros((E, 2), np.int64)
    # Dependency-only longest path through a phase, excluding input upload.
    phase = [[[] for _ in range(2)] for _ in range(E)]
    for e in range(E):
        for p in range(2):
            lo, hi = offsets[e, p]
            for chip, rows, columns, pages, waves in jobs[lo:hi]:
                chip, rows, columns, pages, waves = map(int, (chip, rows, columns, pages, waves))
                emitted = rows
                if s['output_fringe'] == 1:
                    emitted = pages * s['page_bytes'] // columns
                elif s['output_fringe'] == 2:
                    emitted = waves * s['page_bytes'] * s['planes'] // columns
                inp = transfer(columns * s['input_bytes'] * copies, s['channel_bytes_per_second'])
                out = transfer(emitted * s['output_bytes'], s['channel_bytes_per_second'])
                pci = transfer(emitted * s['output_bytes'], s['pcie_bytes_per_second'])
                compute = waves * s['ordinary_read_ns']
                work[e, p, chip] += s['command_ns'] + inp + compute + out
                traffic[e, p, chip] += inp + out
                counts[e, p] += 1
                pciout[e, p] += pci
                phase[e][p].append((inp, compute, out, pci))
    routes = plan['selection_routes']
    experts = E - 1
    chosen = np.c_[routes, np.full(len(routes), experts)]
    chip = s['host_submit_ns'] + work[chosen].sum((1, 2)).max(1)
    controller = s['host_submit_ns'] + counts[chosen].sum((1, 2)) * s['command_ns']
    common_upload = transfer(int(plan['input_columns'][0, 0]) * s['input_bytes'], s['pcie_bytes_per_second'])
    private_upload = np.array([transfer(int(c) * s['input_bytes'], s['pcie_bytes_per_second'])
                               for c in plan['input_columns'][:, 1]], np.int64)
    pcie = s['host_submit_ns'] + common_upload + private_upload[chosen].sum(1) + pciout[chosen].sum((1, 2))
    cp_expert = []
    for e in range(E):
        gu = s['host_submit_ns'] + max(s['command_ns'], common_upload)
        gu += max(sum(x) for x in phase[e][0])
        down = gu + s['activation_ns'] + s['host_submit_ns']
        down += max(s['command_ns'], int(private_upload[e]))
        down += max(sum(x) for x in phase[e][1]) + s['combine_ns']
        cp_expert.append(down)
    cp = np.asarray(cp_expert, np.int64)[chosen].max(1)
    host = np.full(len(routes), s['host_submit_ns'] + chosen.shape[1] *
                   (s['activation_ns'] + s['host_submit_ns']) + s['combine_ns'], np.int64)
    selected_traffic = traffic[chosen].sum((1, 2))
    # Pooling across channels is a relaxation for every physical pairing.
    channel_pool = s['host_submit_ns'] + (selected_traffic.sum(1) + channels - 1) // channels
    channel_current = s['host_submit_ns'] + selected_traffic.reshape(len(routes), chips // channels, channels).sum(1).max(1)
    lower = np.maximum.reduce((chip, controller, pcie, cp, host, channel_pool))
    actual = plan['cost']
    assert np.all(actual >= lower), dict(min_slack=int((actual-lower).min()))
    return dict(actual=actual, lower=lower, chip=chip, controller=controller,
                pcie=pcie, dependency_path=cp, channel_pool=channel_pool,
                channel_current=channel_current, chip_traffic=selected_traffic)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    anchor = BASE / 'W8_selected_plans.frozen.json'
    assert sha(anchor) == '00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00'
    inputs = [{'path': str(anchor), 'sha256': sha(anchor)}]
    grouped = defaultdict(list); records = []; arrays = {}
    for campaign in ['full-campaign-v5-W8-training', 'full-campaign-v6-W8-input4']:
        path = BASE / campaign / 'summary.json'; summary = json.loads(path.read_text())
        inputs.append(dict(path=str(path), sha256=sha(path)))
        for rec in summary['records']:
            for art in rec['artifacts']:
                p = ROOT / art['arrays']; assert sha(p) == art['arrays_sha256']
                with np.load(p, allow_pickle=False) as f:
                    plan = {k: f[k] for k in f.files}
                assert len(plan['selection_routes']) == 64
                c = certificate(plan, rec['service'])
                ident = f"{campaign}/{rec['model']}/cmd{rec['command_ns']}/{art['family']}"
                grouped[ident].append((rec['layer'], c))
                records.append(dict(campaign=campaign, model=rec['model'], layer=rec['layer'],
                                    command_ns=rec['command_ns'], family=art['family'],
                                    artifact=str(p), artifact_sha256=art['arrays_sha256'],
                                    **stats(c['actual'], c['lower']),
                                    channel_load_over_chip_max=float(np.max(c['channel_current']/c['chip'])),
                                    channel_load_dominates_chip_queries=int(np.sum(c['channel_current']>c['chip']))))
    totals = {}
    for ident, entries in grouped.items():
        entries.sort(key=lambda x: x[0]); actual = np.stack([c['actual'] for _, c in entries])
        lower = np.stack([c['lower'] for _, c in entries])
        totals[ident] = dict(layers=len(entries), **stats(actual.sum(0), lower.sum(0)),
                            max_layer_possible_mean_reduction_pct=max(stats(c['actual'],c['lower'])['possible_mean_reduction_pct'] for _, c in entries))
        key = ident.replace('/', '__')
        arrays[key + '__actual_ns'] = actual; arrays[key + '__lower_ns'] = lower
    np.savez_compressed(args.out / 'fixed_plan_bounds.npz', **arrays)
    result = dict(scope='TRAIN64 selection queries only; all40G4/all26Deep routed+shared FFN layers. No new fitting, no external data.',
                  theorem='For any common GU/Down/shared physical-chip permutation and any legal scheduler, elapsed time is at least the maximum of initial-submit plus per-chip native context work, controller volume, PCIe volume, dependency-only path, host volume, and pooled-channel volume. Thus current minus this bound upper-bounds possible pure-topology improvement for each frozen plan. It does not bound independent phase remapping or new row geometry.',
                  script_sha256=sha(__file__), inputs=inputs, totals=totals, records=records)
    (args.out / 'bounds.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
