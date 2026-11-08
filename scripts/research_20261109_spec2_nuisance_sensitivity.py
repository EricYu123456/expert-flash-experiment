#!/usr/bin/env python3
"""Hold layer26/E1 raw-x ridge and J fixed; vary only z-on-x ridge."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import research_20261109_spec2_diagnostics as d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    out = d.DATA / 'spec2-nuisance-sensitivity/layer26'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.monotonic()
    with threadpool_limits(limits=a.threads):
        caps, meta = d.load_inputs(26)
        rawx, rawz = caps['train'].get('x'), caps['train'].get('shared_z')
        validx, validz = caps['validation'].get('x'), caps['validation'].get('shared_z')
        yt, yv = caps['train'].get('expert_outputs')[:, 1], caps['validation'].get('expert_outputs')[:, 1]
        old = d.DATA / 'spec2-diagnostics/layer26'
        rY = np.load(old / 'oof_target_residual.f32.npy', mmap_mode='r')[:, 2048:]
        folds, _ = d.fold_ids(meta['train'][1], 3)
        oof = {ridge: np.empty_like(rawz) for ridge in (.01, .001)}
        for fold in range(3):
            held = folds == fold
            center, gain = d.fit.mean_std(rawx[~held])
            tx, vx = d.fit.transform(rawx[~held], center, gain), d.fit.transform(rawx[held], center, gain)
            gram = d.fit.prepare_gram(tx, np.ones(len(tx)))
            ym, rhs = d.fit.solve_components(gram, rawz[~held])
            for ridge in oof:
                model = d.fit.solve_one(gram, ym, rhs, ridge)
                oof[ridge][held] = rawz[held] - d.fp_predict(vx, model)
            print(json.dumps({'event': 'P_fold_complete', 'fold': fold}), flush=True)
        xc, xg = d.fit.mean_std(rawx)
        xt, xv = d.fit.transform(rawx, xc, xg), d.fit.transform(validx, xc, xg)
        gram = d.fit.prepare_gram(xt, np.ones(len(xt)))
        ym, rhs = d.fit.solve_components(gram, yt)
        base = d.fit.solve_one(gram, ym, rhs, 1.)
        bt, bv = d.fp_predict(xt, base), d.fp_predict(xv, base)
        a2t, a2v = d.fit.alpha_squared(caps['train'], 1), d.fit.alpha_squared(caps['validation'], 1)
        baseline = {'train': d.errors(yt, bt, a2t, meta['train'][2]),
                    'validation': d.errors(yv, bv, a2v, meta['validation'][2])}
        zm, zrhs = d.fit.solve_components(gram, rawz)
        rows = []
        for ridge, rz_oof in oof.items():
            P = d.fit.solve_one(gram, zm, zrhs, ridge)
            rztrain = rawz - d.fp_predict(xt, P)
            rzvalid = validz - d.fp_predict(xv, P)
            rc, rg = d.fit.mean_std(rz_oof)
            for width in (128, 256):
                indices = np.load(old / f'shared_indices{width}.i32.npy')
                ro = d.fit.transform(rz_oof[:, indices], rc[indices], rg[indices])
                rt = d.fit.transform(rztrain[:, indices], rc[indices], rg[indices])
                rv = d.fit.transform(rzvalid[:, indices], rc[indices], rg[indices])
                bg = d.fit.prepare_gram(ro, np.ones(len(ro)))
                bm, brhs = d.fit.solve_components(bg, rY)
                for b_ridge in (.03, .3, 3.):
                    B = d.fit.solve_one(bg, bm, brhs, b_ridge)
                    pt = np.add(bt, d.fp_predict(rt, B), dtype=np.float32)
                    pv = np.add(bv, d.fp_predict(rv, B), dtype=np.float32)
                    metrics = {'train': d.errors(yt, pt, a2t, meta['train'][2]),
                               'validation': d.errors(yv, pv, a2v, meta['validation'][2])}
                    gain = {name: (baseline['validation'][name]['selected_alpha2_nmse'] - v['selected_alpha2_nmse']) / baseline['validation'][name]['selected_alpha2_nmse']
                            if v['selected_alpha2_nmse'] is not None else None
                            for name, v in metrics['validation'].items()}
                    rows.append({'P_ridge': ridge, 'P_effective_df': float(np.sum(gram['values']/(gram['values']+ridge))),
                                 'width': width, 'B_ridge': b_ridge, 'metrics': metrics,
                                 'relative_VALID_selected_alpha2_NMSE_reduction': gain})
            print(json.dumps({'event': 'P_ridge_complete', 'ridge': ridge}), flush=True)
    report = {'status': 'complete', 'layer': 26, 'expert': 1, 'raw_x_ridge': 1., 'conditional_beta': 0,
              'baseline': baseline, 'records': rows, 'seconds': time.monotonic()-start,
              'source_sha256': d.fit.old.sha(Path(__file__)),
              'fixed_J': {str(w): {'path': str(old / f'shared_indices{w}.i32.npy'), 'sha256': d.fit.old.sha(old / f'shared_indices{w}.i32.npy')} for w in (128,256)},
              'scope': 'FP sensitivity diagnostic, original OOF rF and article folds/J fixed; only P ridge changed. TRAIN-only fit; untouched calibration-VALID. Ridge shrinkage is reduced, not mathematically eliminated. No TEST.'}
    d.fit.old.dump(out / 'report.json', report)
    d.fit.old.dump(d.ROOT / 'results/research-2026-11-09-shared-features/spec2_nuisance_sensitivity.json', report)
    print(json.dumps({'status': 'complete', 'seconds':report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
