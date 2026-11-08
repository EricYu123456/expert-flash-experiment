#!/usr/bin/env python3
"""Run spec2 diagnostics with independent frozen-C++ W8A8 model checks."""
import json
from pathlib import Path

import numpy as np

import research_20261109_spec2_diagnostics as diag
from research_20261109_shared_feature_targets import Numeric


def main():
    numeric = Numeric(8)
    original, checks = diag.canonical, []

    def checked(xtrain, xvalid, ytrain, ridge):
        model, pt, pv, stats = original(xtrain, xvalid, ytrain, ridge)
        x = np.ascontiguousarray(xvalid[:7], np.float32)
        out = np.empty((len(x), len(model['bias'])), np.float32)
        assert numeric.lib.sf_predict(*x.shape, len(model['bias']), x,
                                      np.ascontiguousarray(model['qweight']),
                                      np.ascontiguousarray(model['wscale']),
                                      np.ascontiguousarray(model['bias']), out) == 0
        expected = np.ascontiguousarray(pv[:len(x)])
        assert np.array_equal(out.view(np.uint32), expected.view(np.uint32))
        q, scale = numeric.quantize(x)
        pyq, pyscale = diag.fit.old.quantize_rows(x)
        assert np.array_equal(q, pyq)
        assert np.array_equal(scale.view(np.uint32), pyscale.view(np.uint32))
        checks.append({'input_width': x.shape[1], 'output_width': out.shape[1],
                       'frozen_CPP_prediction_floats_bitchecked': out.size,
                       'frozen_CPP_A8_codes_and_scale_bitchecked': True})
        return model, pt, pv, stats

    diag.canonical = checked
    diag.main()
    layer = int(__import__('sys').argv[__import__('sys').argv.index('--layer') + 1])
    report = {'status': 'passed', 'layer': layer, 'cases': checks,
              'numeric_binary_sha256': numeric.binary_sha256,
              'source_sha256': diag.fit.old.sha(Path(__file__)),
              'diagnostic_source_sha256': diag.fit.old.sha(Path(diag.__file__))}
    diag.fit.old.dump(diag.DATA / f'spec2-diagnostics/layer{layer}/canonical_numeric_audit.json', report)
    diag.fit.old.dump(diag.ROOT / f'results/research-2026-11-09-shared-features/spec2_numeric_layer{layer}.json', report)
    print(json.dumps({'event': 'canonical_numeric_pass', 'layer': layer, 'cases': len(checks)}), flush=True)


if __name__ == '__main__':
    main()
