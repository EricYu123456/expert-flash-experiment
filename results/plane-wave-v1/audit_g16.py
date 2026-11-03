#!/usr/bin/env python3
"""Reconstruct fixed-g16 diagnostics from bin occupancy and paper constants."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import aif_model as flash

OUT = ROOT / 'results/plane-wave-v1'


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = read_json(OUT / 'fixed_g16_diagnostics.json')
    assert receipt['main_manifest_sha256'] == digest(OUT / 'manifest.json')
    assert receipt['command_file_sha256'] == digest(OUT / 'fixed_g16.command.json')
    assert receipt['selection_changed'] is False
    for name, expected in receipt['source_sha256'].items():
        assert digest(ROOT / 'scripts' / name) == expected
    source = read_json(OUT / 'static-bundles/manifest.json')['sources']
    checked = []
    with np.load(OUT / 'fixed_g16_token_costs.npz', allow_pickle=False) as saved:
        for item in receipt['provenance']:
            name = item['model']
            plan_path = OUT / (name + '.frozen_plan.json')
            assert digest(plan_path) == item['frozen_plan_sha256']
            plan = read_json(plan_path)
            assert digest(Path(item['test_cache']['path'])) == item['test_cache']['sha256']
            with np.load(item['test_cache']['path'], allow_pickle=False) as cache:
                routes, layers = cache['routes'], cache['layers'].tolist()
                assert saved['prompt_ids'].tolist() == cache['prompt_ids'].tolist()
            dimensions = flash.tensor_dimensions(read_json(Path(source[name]['metadata'])))
            active = routes.shape[-1]
            for width in (4, 2, 1):
                predicted = {timing: np.zeros(routes.shape[:2], dtype=np.int64)
                             for timing in ('ideal_overlap', 'retained_cr_schedule', 'conservative_drain')}
                for li, layer in enumerate(layers):
                    dims = dimensions[layer]
                    bins = np.asarray(plan['train_only_plane_assignments']['g16'][f'w{width}'][str(layer)])
                    selected_bins = bins[routes[:, :, li]]
                    peak = np.max(np.stack([(selected_bins == b).sum(axis=-1)
                                            for b in range(4 // width)]), axis=0)
                    for rows, columns, common in ((dims['gate'][0] + dims['up'][0], dims['up'][1], True),
                                                   (*dims['down'], False)):
                        assert rows % 16 == 0
                        pages = ((rows // 16) * columns + 16383) // 16384
                        waves = (pages + width - 1) // width
                        cold = (waves + 191) // 192
                        nand = waves * 9700 + cold * 18300
                        ecc = active * pages * 2560
                        # At g16 every chip participates. INT8 vector inputs
                        # saturate the two-chips/channel term; FP32 outputs
                        # saturate the PCIe term. Both are routing independent.
                        input_ns = columns * (1 if common else active)
                        output_ns = (active * rows * 4 + 7) // 8
                        other = active * 16 * cold * 250 + 2000 + input_ns + output_ns
                        predicted['ideal_overlap'] += np.maximum(peak * nand, ecc) + other
                        predicted['retained_cr_schedule'] += peak * nand + ecc + other
                        predicted['conservative_drain'] += peak * waves * 28000 + ecc + other
                for timing, values in predicted.items():
                    np.testing.assert_array_equal(saved[f'{name}__{timing}__width{width}__model_ns'], values)
                    checked.append(dict(model=name, width=width, timing=timing,
                                        every_token_bit_exact=True, tokens=int(values.size)))
    target = OUT / 'audit_g16.json'
    target.write_text(json.dumps(dict(scope='Independent fixed-g16 occupancy algebra, vectors and block command costs',
                                     arrays=checked), indent=2) + '\n')
    print(json.dumps(dict(arrays=len(checked), token_predictions=sum(c['tokens'] for c in checked),
                          result=str(target)), indent=2))


if __name__ == '__main__':
    main()
