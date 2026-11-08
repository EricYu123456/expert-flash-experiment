#!/usr/bin/env python3
"""Article-grouped TRAIN residual diagnostics; only legacy TRAIN/VALID allowed.

OOF nuisance residuals select features and fit the small residual readout.
Consequently TRAIN residual-readout scores are fitting diagnostics, not an
unbiased OOF estimate. Untouched calibration-VALID measures conditional gain.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import research_20261109_shared_feature_fit as fit
from research_20261109_shared_feature_targets import DATA, ROOT

OLD = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')
FIXED = {1: (14, 40), 13: (10, 56), 26: (0, 1)}
BETAS = (0, .5, .9, .99)
SEED = '20261109-spec2-article-oof-v1'


def metadata(cap, records):
    prompts = cap.prompts().astype(str)
    articles = np.array([records[p]['article_id'] for p in prompts])
    positions = np.asarray(cap.get('positions'))
    assert len(prompts) == len(positions)
    return prompts, articles, positions


def load_inputs(layer):
    corpus = json.loads((OLD / 'corpus/manifest.json').read_text())
    records = {r['id']: r for r in corpus['records'] if r['split'] in ('train', 'validation')}
    caps = {s: fit.old.Capture(OLD / 'captures' / s, layer) for s in ('train', 'validation')}
    meta = {s: metadata(c, records) for s, c in caps.items()}
    assert len(set(meta['train'][0])) == 600 and len(set(meta['validation'][0])) == 200
    assert not set(meta['train'][1]) & set(meta['validation'][1])
    assert all(c.expert_ids == list(FIXED[layer]) for c in caps.values())
    return caps, meta


def neff(w):
    w = np.asarray(w, np.float64)
    return float(w.sum() ** 2 / np.sum(w * w)) if np.any(w) else 0.


def group_support(w, groups):
    _, idx = np.unique(groups, return_inverse=True)
    mass = np.bincount(idx, weights=w)
    return {'groups': len(mass), 'nonzero_groups': int(np.sum(mass > 0)),
            'mass_aggregated_N_eff': neff(mass)}


def support(caps, meta, layer):
    rows = []
    for split, cap in caps.items():
        prompts, articles, positions = meta[split]
        for expert in FIXED[layer]:
            a2 = fit.alpha_squared(cap, expert)
            train_mean = float(fit.alpha_squared(caps['train'], expert).mean())
            for beta in BETAS:
                # VALID uses TRAIN's normalization, matching a frozen weight rule.
                w = (1 - beta) + beta * a2 / train_mean
                rows.append({'split': split, 'layer': layer, 'expert': expert, 'beta': beta,
                             'rows': len(w), 'weight_only_row_N_eff': neff(w),
                             'prompt': group_support(w, prompts), 'article': group_support(w, articles),
                             'selected_rows': int(np.sum(a2 > 0)),
                             'selected_prompts': len(set(prompts[a2 > 0])),
                             'selected_articles': len(set(articles[a2 > 0])),
                             'selected_alpha2_row_N_eff': neff(a2),
                             'selected_alpha2_prompt': group_support(a2, prompts),
                             'selected_alpha2_article': group_support(a2, articles),
                             'position_support': {name: {'rows': int(mask.sum()),
                                                        'selected_rows': int(np.sum((a2 > 0) & mask)),
                                                        'alpha2_sum': float(a2[mask].sum()),
                                                        'alpha2_N_eff': neff(a2[mask])}
                                                  for name, mask in slices(positions).items()}})
    return rows


def slices(positions):
    return {'all': np.ones(len(positions), bool), 'BOS_pos0': positions == 0,
            'nonBOS_pos1_15': positions > 0, 'early_pos1_7': (positions >= 1) & (positions <= 7),
            'late_pos8_15': positions >= 8}


def errors(y, pred, a2, positions):
    # Sum in FP64 and retain numerators, so slices/experts can be pooled correctly.
    error = np.sum((np.asarray(y, np.float64) - pred) ** 2, axis=1)
    energy = np.sum(np.asarray(y, np.float64) ** 2, axis=1)
    out = {}
    for name, mask in slices(positions).items():
        w = a2[mask]
        numerator, denominator = float(w @ error[mask]), float(w @ energy[mask])
        out[name] = {'rows': int(mask.sum()), 'selected_rows': int(np.sum(w > 0)),
                     'selected_alpha2_squared_error': numerator,
                     'selected_alpha2_target_energy': denominator,
                     'selected_alpha2_nmse': numerator / denominator if denominator > 0 else None,
                     'all_row_nmse': float(error[mask].sum() / max(energy[mask].sum(), 1e-300))}
    return out


def fp_predict(x, model):
    return fit.old.predict(x, model, 'fp')


def ridge_model(x, y, ridge):
    gram = fit.prepare_gram(x, np.ones(len(x)))
    ym, rhs = fit.solve_components(gram, y)
    model = fit.solve_one(gram, ym, rhs, ridge)
    diagnostics = {'effective_df': float(np.sum(gram['values'] / (gram['values'] + ridge))),
                   'Gram_rank': gram['numerical_rank']}
    return model, diagnostics


def fold_ids(articles, folds):
    unique = sorted(set(articles), key=lambda a: hashlib.sha256((SEED + a).encode()).digest())
    # Greedy article-size balancing preserves all rows/prompts of an article.
    counts, loads, assignment = Counter(articles), [0] * folds, {}
    for article in unique:
        f = min(range(folds), key=lambda j: (loads[j], j))
        assignment[article] = f
        loads[f] += counts[article]
    return np.array([assignment[a] for a in articles]), assignment


def canonical(xtrain, xvalid, ytrain, ridge):
    tx, vx = fit.encode(xtrain), fit.encode(xvalid)
    model, diag = ridge_model(tx['qdq'], ytrain, ridge)
    fit.recalibrate_bias(tx, model, ytrain, np.ones(len(ytrain)))
    return model, fit.predict_encoded(tx, model), fit.predict_encoded(vx, model), diag


def conditional(args, caps, meta, folder):
    start = time.monotonic()
    train, valid = caps['train'], caps['validation']
    rawx, rawz = np.asarray(train.get('x')), np.asarray(train.get('shared_z'))
    vxraw, vzraw = np.asarray(valid.get('x')), np.asarray(valid.get('shared_z'))
    targets = {s: np.asarray(c.get('expert_outputs')) for s, c in caps.items()}
    n, h, zwidth = len(rawx), rawx.shape[1], rawz.shape[1]
    y = np.concatenate([targets['train'][:, i] for i in range(2)], axis=1)
    joint = np.concatenate((y, rawz), axis=1)
    folds, assignment = fold_ids(meta['train'][1], args.folds)
    oof_y, oof_z = np.empty_like(y), np.empty_like(rawz)
    fold_records = []
    for fold in range(args.folds):
        held = folds == fold
        center, gain = fit.mean_std(rawx[~held])
        xt, xv = fit.transform(rawx[~held], center, gain), fit.transform(rawx[held], center, gain)
        nuisance, diag = ridge_model(xt, joint[~held], args.ridge_x)
        pred = fp_predict(xv, nuisance)
        oof_y[held], oof_z[held] = y[held] - pred[:, :2*h], rawz[held] - pred[:, 2*h:]
        fold_records.append({'fold': fold, 'fit_rows': int((~held).sum()), 'held_rows': int(held.sum()),
                             'fit_articles': len(set(meta['train'][1][~held])),
                             'held_articles': len(set(meta['train'][1][held])), **diag})
        print(json.dumps({'event': 'nuisance_fold_complete', 'layer': args.layer, **fold_records[-1]}), flush=True)
    np.save(folder / 'oof_target_residual.f32.npy', oof_y)
    np.save(folder / 'oof_shared_residual.f32.npy', oof_z)
    fit.old.dump(folder / 'article_fold_assignment.json', assignment)
    # F32 stored nuisance coefficients, FP64 dot/cast/F32 bias, matching legacy FP diagnostic.
    xc, xg = fit.mean_std(rawx)
    zc, zg = fit.mean_std(rawz)
    xt, xv = fit.transform(rawx, xc, xg), fit.transform(vxraw, xc, xg)
    nuisance, xdiag = ridge_model(xt, joint, args.ridge_x)
    full_pred, valid_pred = fp_predict(xt, nuisance), fp_predict(xv, nuisance)
    rztrain = rawz - full_pred[:, 2*h:]
    rzvalid = vzraw - valid_pred[:, 2*h:]
    rzc, rzg = fit.mean_std(oof_z)
    rz_oof_scaled = fit.transform(oof_z, rzc, rzg)
    # Equal expert contribution: sum each expert's normalized cross-covariance score.
    rz64 = np.asarray(oof_z, np.float64)
    rz64 -= rz64.mean(axis=0)
    variance = np.mean(rz64 ** 2, axis=0)
    score = np.zeros(zwidth)
    for i in range(2):
        response = np.asarray(oof_y[:, i*h:(i+1)*h], np.float64)
        response -= response.mean(axis=0)
        cross = rz64.T @ response / n
        score += np.sum(cross ** 2, axis=1) / np.maximum(variance * np.mean(np.sum(response ** 2, axis=1)), 1e-30)
    order = np.argsort(-score, kind='stable')
    np.savez(folder / 'feature_scores.npz', score=score, order=order, OOF_residual_variance=variance)
    records = []
    baseline_can = {}
    for i, expert in enumerate(FIXED[args.layer]):
        a2 = {s: fit.alpha_squared(c, expert) for s, c in caps.items()}
        base = {}
        for split, pred in (('train', full_pred[:, i*h:(i+1)*h]), ('validation', valid_pred[:, i*h:(i+1)*h])):
            base[split] = errors(targets[split][:, i], pred, a2[split], meta[split][2])
        records.append({'method': 'FP_raw_x_fixed_ridge', 'expert': expert, 'ridge': args.ridge_x, 'metrics': base, **xdiag})
        can, pt, pv, diag = canonical(xt, xv, targets['train'][:, i], args.ridge_x)
        baseline_can[expert] = (can, pt, pv)
        records.append({'method': 'W8A8_QDQ_raw_x_fixed_ridge', 'expert': expert, 'ridge': args.ridge_x,
                        'metrics': {s: errors(targets[s][:, i], p, a2[s], meta[s][2]) for s, p in (('train', pt), ('validation', pv))}, **diag})
    for width in args.widths:
        indices = np.sort(order[:width])
        np.save(folder / f'shared_indices{width}.i32.npy', indices.astype(np.int32))
        bh_train, bh_valid = fit.transform(rztrain[:, indices], rzc[indices], rzg[indices]), fit.transform(rzvalid[:, indices], rzc[indices], rzg[indices])
        raw_htrain = np.concatenate((xt, fit.transform(rawz[:, indices], zc[indices], zg[indices])), axis=1)
        raw_hvalid = np.concatenate((xv, fit.transform(vzraw[:, indices], zc[indices], zg[indices])), axis=1)
        for i, expert in enumerate(FIXED[args.layer]):
            a2 = {s: fit.alpha_squared(c, expert) for s, c in caps.items()}
            can, pt, pv, diag = canonical(raw_htrain, raw_hvalid, targets['train'][:, i], args.ridge_x)
            records.append({'method': 'W8A8_QDQ_joint_hybrid', 'expert': expert, 'width_shared': width, 'ridge': args.ridge_x,
                            'note': 'Jointly refit on actual whole-vector A8 QDQ; does not keep raw-x coefficients frozen.',
                            'metrics': {s: errors(targets[s][:, i], p, a2[s], meta[s][2]) for s, p in (('train', pt), ('validation', pv))}, **diag})
            for ridge in args.ridge_b:
                b, bdiag = ridge_model(rz_oof_scaled[:, indices], oof_y[:, i*h:(i+1)*h], ridge)
                correction_train, correction_valid = fp_predict(bh_train, b), fp_predict(bh_valid, b)
                fp_train = np.add(full_pred[:, i*h:(i+1)*h], correction_train, dtype=np.float32)
                fp_valid = np.add(valid_pred[:, i*h:(i+1)*h], correction_valid, dtype=np.float32)
                record = {'method': 'FP_frozen_raw_x_plus_conditional_shared_residual', 'expert': expert,
                          'width_shared': width, 'ridge_B': ridge, 'metrics':
                          {s: errors(targets[s][:, i], p, a2[s], meta[s][2]) for s, p in (('train', fp_train), ('validation', fp_valid))}, **bdiag}
                records.append(record)
                # Real-valued algebraic folding changes finite-precision operations.
                br = b['coefficient'].astype(np.float64) * rzg[indices]
                P = nuisance['coefficient'][2*h:][indices].astype(np.float64)
                cP = nuisance['bias'][2*h:][indices].astype(np.float64)
                A = nuisance['coefficient'][i*h:(i+1)*h].astype(np.float64)
                cb = nuisance['bias'][i*h:(i+1)*h].astype(np.float64)
                coeff = np.concatenate((A - br @ P, br / zg[indices]), axis=1).astype(np.float32)
                bias = (cb + b['bias'] + br @ (zc[indices] - cP - rzc[indices])).astype(np.float32)
                qw, sw = fit.old.quantize_rows(coeff)
                folded = {'coefficient': coeff, 'bias': bias, 'qweight': qw, 'wscale': sw}
                folded_fp = fp_predict(raw_hvalid, folded)
                encoded_train, encoded_valid = fit.encode(raw_htrain), fit.encode(raw_hvalid)
                fit.recalibrate_bias(encoded_train, folded, targets['train'][:, i], np.ones(n))
                qt, qv = fit.predict_encoded(encoded_train, folded), fit.predict_encoded(encoded_valid, folded)
                records.append({'method': 'W8A8_folded_conditional_no_joint_refit', 'expert': expert, 'width_shared': width,
                                'ridge_B': ridge, 'note': 'Folded FP residual model, row-W8; TRAIN-only bias recalibration. No QDQ coefficient refit.',
                                'FP_fold_vs_unfolded_VALID_max_abs_difference': float(np.max(np.abs(folded_fp - fp_valid))),
                                'FP_fold_vs_unfolded_VALID_relative_squared_error': float(np.sum((folded_fp.astype(np.float64) - fp_valid)**2) / max(np.sum(fp_valid.astype(np.float64)**2), 1e-300)),
                                'metrics': {s: errors(targets[s][:, i], p, a2[s], meta[s][2]) for s, p in (('train', qt), ('validation', qv))}})
        print(json.dumps({'event': 'conditional_width_complete', 'layer': args.layer, 'shared_width': width, 'seconds': time.monotonic()-start}), flush=True)
    by_method = {}
    for record in records:
        key = (record['expert'], 'W8A8' if record['method'].startswith('W8A8') else 'FP')
        if record['method'] in ('FP_raw_x_fixed_ridge', 'W8A8_QDQ_raw_x_fixed_ridge'):
            by_method[key] = record
    for record in records:
        kind = 'W8A8' if record['method'].startswith('W8A8') else 'FP'
        base = by_method[(record['expert'], kind)]
        record['VALID_relative_selected_alpha2_NMSE_reduction_vs_matched_raw_x'] = {}
        for name in slices(meta['validation'][2]):
            a = base['metrics']['validation'][name]['selected_alpha2_nmse']
            b = record['metrics']['validation'][name]['selected_alpha2_nmse']
            record['VALID_relative_selected_alpha2_NMSE_reduction_vs_matched_raw_x'][name] = (a-b)/a if a and b is not None else None
        a = record['metrics']['train']['all']['selected_alpha2_nmse']
        b = record['metrics']['validation']['all']['selected_alpha2_nmse']
        record['VALID_minus_TRAIN_selected_alpha2_NMSE'] = b-a if a is not None and b is not None else None
    return {'records': records, 'folds': fold_records, 'seconds': time.monotonic()-start,
            'TRAIN_OOF_shared_linear_residual_energy_fraction': float(np.sum(oof_z.astype(np.float64)**2)/np.sum(rawz.astype(np.float64)**2)),
            'selection': 'Layer-common indices from sum over two experts of normalized OOF r_z versus OOF r_F cross-covariance; beta0, fixed nuisance ridge. No VALID inputs to feature selection.',
            'TRAIN_metric_scope': 'B/indices fit all TRAIN OOF nuisance residuals; B TRAIN scores are fitting diagnostics, not an unbiased OOF generalization estimate.',
            'quality_scope': 'Calibration-VALID local expert-output reconstruction only; no full-model replacement, TEST, PPL or system-speed claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layer', type=int, required=True, choices=FIXED)
    p.add_argument('--support-only', action='store_true')
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--widths', type=int, nargs='+', default=[128, 256])
    p.add_argument('--ridge-x', type=float, default=1.)
    p.add_argument('--ridge-b', type=float, nargs='+', default=[.03, .3, 3.])
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    assert args.folds >= 2 and all(0 < w <= 768 for w in args.widths)
    assert args.ridge_x > 0 and all(v > 0 for v in args.ridge_b)
    with threadpool_limits(limits=args.threads):
        caps, meta = load_inputs(args.layer)
        folder = DATA / 'spec2-diagnostics' / f'layer{args.layer}'
        folder.mkdir(parents=True, exist_ok=True)
        report = {'status': 'support_complete', 'layer': args.layer, 'fixed_experts': FIXED[args.layer],
                  'source_sha256': fit.old.sha(Path(__file__)), 'capture_manifests': {s:c.identity() for s,c in caps.items()},
                  'seed': SEED, 'arguments': vars(args), 'support': support(caps, meta, args.layer),
                  'N_eff_scope': 'Weight-only Kish N_eff, not independent samples. Prompt/article N_eff aggregates row weights before applying Kish formula; correlated tokens do not create new articles.',
                  'numeric_contract': fit.CONTRACT}
        fit.old.dump(folder / 'support.json', report)
        fit.old.dump(ROOT / f'results/research-2026-11-09-shared-features/spec2_support_layer{args.layer}.json', report)
        print(json.dumps({'event':'support_complete', 'layer':args.layer}), flush=True)
        if not args.support_only:
            report['conditional'] = conditional(args, caps, meta, folder)
            report['status'] = 'complete'
            fit.old.dump(folder / 'conditional.json', report)
            fit.old.dump(ROOT / f'results/research-2026-11-09-shared-features/spec2_conditional_layer{args.layer}.json', report)
            print(json.dumps({'event':'complete', 'layer':args.layer, 'seconds':report['conditional']['seconds']}), flush=True)


if __name__ == '__main__':
    main()
