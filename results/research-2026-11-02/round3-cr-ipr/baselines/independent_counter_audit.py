#!/usr/bin/env python3
"""Direct integer page/mask audit independent of the study's cost helpers."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'scripts/research_20261102_cr_ipr_baselines.py').exists())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


freeze = json.loads((OUT / 'protocol.frozen.json').read_text())
source = ROOT / 'scripts/research_20261102_cr_ipr_baselines.py'
assert sha(source) == freeze['source_sha256']
assert sha(freeze['TRAIN_archive']) == freeze['TRAIN_archive_sha256']
for filename, key in [
    ('frozen_TRAIN_phase_maps.npz', 'frozen_phase_file_sha256'),
    ('frozen_TRAIN_packed_expert_order.npz', 'frozen_packed_matching_sha256'),
    ('physical_page_ledgers.json', 'static_layouts_file_sha256'),
    ('sparse_mask_and_keep_hot_ledger.json', 'sparse_mask_keep_hot_ledger_sha256'),
]:
    assert sha(OUT / filename) == freeze[key]

with np.load(freeze['TRAIN_archive']) as z:
    routes = z['routes'].transpose(2, 0, 1, 3).reshape(40, 512, 6)
with np.load(OUT / 'frozen_TRAIN_packed_expert_order.npz') as z:
    orders = z['slot_to_expert']
with np.load(OUT / 'frozen_TRAIN_phase_maps.npz') as z:
    phases = z['learned']
with np.load(OUT / 'TRAIN_page_wave_arrays.npz') as z:
    values = {k: z[k].copy() for k in z.files}
with np.load(OUT / 'sparse_mask_and_keep_hot_arrays.npz') as z:
    gaps = {k: z[k].copy() for k in z.files}

ledgers = json.loads((OUT / 'physical_page_ledgers.json').read_text())
total_addresses = 0
for data in ledgers.values():
    for view in data if isinstance(data, list) else [data]:
        occupied, counts = set(), np.zeros(4, dtype=int)
        for e, row in enumerate(view['addresses']):
            assert row['expert'] == e
            assert len(row['pages']) == (12 if e == 64 else 6)
            for a in row['pages']:
                key = tuple(a[k] for k in ['plane', 'block', 'wordline'])
                assert key not in occupied and key[1] == 0 and 0 <= key[2] < 192
                occupied.add(key)
                counts[key[0]] += 1
        assert len(occupied) == 396 and max(counts) <= 192
        assert np.array_equal(np.sum(view['profiles'], axis=0), counts)
        total_addresses += len(occupied)

checks = 0
for layer in range(40):
    assert sorted(orders[layer].tolist()) == list(range(64))
    inverse = np.argsort(orders[layer])
    route = routes[layer]
    for name, positions in [
        ('sameWL_packed_skipCR_global_roundrobin', np.arange(64)),
        ('sameWL_packed_TRAIN_exact_mean_matching', inverse),
    ]:
        selected = positions[route]
        both = np.array([6 - len(set(row)) for row in selected // 2])
        waves = 15 - both
        assert np.array_equal(waves, values[name][layer])
        prefix = 'identity_' if 'global' in name else 'TRAIN_matching_'
        assert np.array_equal(4 * waves - 48, gaps[prefix + 'keep_hot_overfetch_pages'][layer])
        direct_runs = []
        for row in selected:
            flags = {}
            for start, n in [(int(e) * 6, 6) for e in row] + [(384, 12)]:
                for page in range(start, start + n):
                    flags[page // 4] = flags.get(page // 4, 0) | (1 << (page % 4))
            runs, previous = np.zeros(4, dtype=int), 0
            for _, mask in sorted(flags.items()):
                for plane in range(4):
                    runs[plane] += bool(mask & (1 << plane)) and not bool(previous & (1 << plane))
                previous = mask
            direct_runs.append(runs)
        assert np.array_equal(direct_runs, gaps[prefix + 'plane_cold_runs'][layer])
        checks += 512
    profiles = np.array([[1 + int((p - int(phase)) % 4 < 2) for p in range(4)]
                         for phase in phases[layer]])
    waves = (profiles[route].sum(axis=1) + 3).max(axis=1)
    assert np.array_equal(waves, values['IPR_TRAIN_learned_phase_coordinate_descent'][layer])
    assert np.all(values['IPR_joint_onecopy_shared_compensation'][layer] == 12)
    assert np.array_equal(values['IPR_joint_onecopy_shared_compensation'][layer],
                          values['IPR_generic_sparse_row_compiler_same_layout'][layer])
    checks += 512

result = dict(
    status='pass',
    scope='Independent direct integer page/mask reconstruction; TRAIN only; no native timeline audit.',
    physical_page_addresses_checked=total_addresses, route_control_checks=checks,
    freeze_sha256=sha(OUT / 'protocol.frozen.json'), report_sha256=sha(OUT / 'report.json'),
    source_sha256=sha(source), audit_source_sha256=sha(__file__),
    checks=['unique one-copy addresses; all G4 layouts one block',
            'independent pair-collision formula for identity and learned matching',
            'direct per-plane cold runs from integer mask transitions',
            'explicit keep-hot physical overfetch', 'independent learned phase load computation',
            'candidate equals generic compiler control'])
(OUT / 'independent_counter_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
