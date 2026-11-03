"""FIT/VALID and sealed SCORE stages; no work is performed at import."""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from research_20261103_local_affine_stage0_core import *


class InsufficientFIT(RuntimeError):
    pass


def dataset(folder, layer, split_ids):
    folder = Path(folder)
    report = json.loads((folder / 'capture_report.json').read_text())
    assert report['status'] == 'complete' and report['native_GGUF_trajectory']
    assert report['jobs'] == len(split_ids)
    rows = [json.loads(s) for s in (folder / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == report['rows']
    assert all(r['row'] == i and r['offset'] == i * 2048 * 4 for i, r in enumerate(rows))
    assert (folder / 'inputs.f32').stat().st_size == len(rows) * 2048 * 4
    raw = np.memmap(folder / 'inputs.f32', np.float32, 'r', shape=(len(rows), 2048))
    selected = [r for r in rows if r['layer'] == layer]
    assert set(r['job'] for r in selected) == set(split_ids)
    counts = defaultdict(lambda: [0, 0])
    for r in selected:
        assert r['phase'] in ['prefill', 'decode']
        counts[r['job']][r['phase'] == 'decode'] += 1
    for job, (prefill, dec) in counts.items():
        js = json.loads((folder / (job + '.json')).read_text())
        assert prefill == len(js['retained_token_ids']) and dec == 8
        assert len(js['generated_token_ids']) == 8
        own = [r for r in selected if r['job'] == job]
        assert sorted(r['token_in_batch'] for r in own if r['phase'] == 'prefill') == list(range(prefill))
        assert all(r['step'] == -1 for r in own if r['phase'] == 'prefill')
        assert sorted(r['step'] for r in own if r['phase'] == 'decode') == list(range(8))
        assert all(r['token_in_batch'] == 0 for r in own if r['phase'] == 'decode')
    assert len({(r['job'], r['phase'], r['step'], r['token_in_batch']) for r in selected}) == len(selected)
    ix = np.array([r['row'] for r in selected])
    return {'x': np.ascontiguousarray(raw[ix]), 'ids': np.array([r['experts'] for r in selected], np.int32),
            'alpha': np.array([r['alpha'] for r in selected], np.float32),
            'prompt': np.array([split_ids.index(r['job']) for r in selected], np.int32),
            'phase': np.array([r['phase'] == 'decode' for r in selected], np.int8), 'jobs': split_ids}


def route(ds, expert):
    natural = np.any(ds['ids'] == expert, axis=1)
    alpha = np.sum(np.where(ds['ids'] == expert, ds['alpha'], 0), axis=1, dtype=np.float64)
    return natural, alpha


def select_experts(fit_folder, layers, fit_ids):
    out = []
    for layer in layers:
        ds = dataset(fit_folder, layer, fit_ids)
        counts = np.bincount(ds['ids'].ravel(), minlength=64)
        order = sorted([e for e in range(64) if counts[e] > 0], key=lambda e: (-counts[e], e))
        if len(order)<2:
            raise InsufficientFIT(json.dumps({'layer':layer,'reason':'fewer_than_two_nonzero_experts','counts':counts.tolist()}))
        chosen = [order[0], order[(len(order) - 1) // 2]]
        if len(set(chosen))!=2:
            raise InsufficientFIT(json.dumps({'layer':layer,'reason':'prespecified_first_and_median_not_distinct','counts':counts.tolist()}))
        for e in chosen:
            natural, _ = route(ds, e)
            out.append({'layer': layer, 'expert': e, 'FIT_natural_rows': int(natural.sum()),
                        'FIT_natural_distinct_prompts': int(np.unique(ds['prompt'][natural]).size),
                        'FIT_counts_all64': counts.tolist(), 'ranked_nonzero_experts': order})
    return out


def tree_archive(folder, tree, basis, sigma, num, x):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {'basis': basis, 'sigma': sigma}
    meta = {}
    for k, node in tree.items():
        arrays[f'mean{k}'] = node['mean']
        meta[str(k)] = {a: b for a, b in node.items() if a not in ['mean', 'normal']}
        if node['normal'] is not None:
            arrays[f'normal{k}'] = node['normal']
    occupancy = {}
    for C in [1, 4, 16, 32]:
        ids = select_tree(num, tree, x, C)
        occupancy[str(C)] = {'actual_runtime_counts': np.bincount(ids[ids >= 0], minlength=C).tolist(),
                             'unsupported_rows': int(np.sum(ids < 0)), 'balanced_count_guarantee': False}
    np.savez(folder / 'tree_projection.npz', **arrays)
    save(folder / 'tree.json', {'nodes': meta, 'runtime_FIT_occupancy': occupancy,
                              'unsupervised_FIT_prompts': 120, 'learning80_uses_full120_tree_P': True})


def tree_load(folder):
    a = np.load(Path(folder) / 'tree_projection.npz', allow_pickle=False)
    m = json.loads((Path(folder) / 'tree.json').read_text())
    tree = {}
    for key, node in m['nodes'].items():
        k = int(key)
        tree[k] = {**node, 'mean': a[f'mean{k}'], 'normal': a[f'normal{k}'] if f'normal{k}' in a else None}
    return tree, a['basis'], a['sigma'], m


def load_weights(folder, expected_I):
    folder = Path(folder)
    info = json.loads((folder / 'weights.json').read_text())
    expected = {'gate': (2048, expected_I), 'up': (2048, expected_I), 'down': (expected_I, 2048)}
    result = {}
    for role, (C, R) in expected.items():
        assert (info['projections'][role]['C'], info['projections'][role]['R']) == (C, R)
        w = np.fromfile(folder / (role + '.i8'), np.int8)
        s = np.fromfile(folder / (role + '.scales.f32'), np.float32)
        assert w.size == C * R and s.size == R and np.isfinite(s).all() and (s > 0).all()
        result[role] = (w.reshape(R, C), s)
    return result


def query_targets(num, ds, weights, path, operator, split):
    path = Path(path)
    assert not path.exists(), 'preserve prior targets'
    begin = time.monotonic()
    y = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(ds['x']), 2048))
    before = num.dot_audits
    for start in range(0, len(y), 256):
        y[start:start + 256] = teacher(num, ds['x'][start:start + 256], weights)
    y.flush()
    natural, _ = route(ds, operator['expert'])
    save(path.with_suffix('.query.json'), {'operator': operator, 'split': split, 'queried_all_layer_inputs': True,
         'rows': len(y), 'natural_rows': int(natural.sum()), 'offroute_rows': int((~natural).sum()),
         'distinct_prompts': len(ds['jobs']), 'natural_distinct_prompts': int(np.unique(ds['prompt'][natural]).size),
         'target_bytes': y.nbytes, 'seconds': time.monotonic() - begin,
         'DGEMM_I64_exact_checks': num.dot_audits - before, 'sha256': sha(path),
         'by_phase':{p:{'rows':int(np.sum(ds['phase']==ph)),'natural_rows':int(np.sum(natural&(ds['phase']==ph))),
              'natural_distinct_prompts':int(np.unique(ds['prompt'][natural&(ds['phase']==ph)]).size)} for p,ph in [('prefill',0),('decode',1)]},
         'independent_inputs_not_increased_by_offroute_queries': True})
    return y


def fit_operator(num, opfolder, operator, train, valid, y, yv, tree, basis, sigma, intermediate):
    fit_started=time.monotonic()
    opfolder = Path(opfolder)
    opfolder.mkdir()
    bank = opfolder / 'models'
    bank.mkdir()
    natural_fit, _ = route(train, operator['expert'])
    natural_val, _ = route(valid, operator['expert'])
    points, grids, inconclusive = [], [], []
    design_cache = {}
    def get_design(C, rank, fmt):
        key = (C, rank, fmt)
        if key not in design_cache:
            design_cache[key] = (design(num, train['x'], tree, C, basis, sigma, rank, fmt),
                                 design(num, valid['x'], tree, C, basis, sigma, rank, fmt))
        return design_cache[key]
    def register(model, params, dv, pred, diagnostic, cutoff, val_loss, reg_info):
        covered = np.ones(len(pred), bool) if params['guard'] == 100 else (np.zeros(len(pred), bool) if cutoff is None else dv['distance'] <= cutoff)
        unsupported = np.isin(dv['leaf'], model['unsupported'])
        covered &= ~unsupported
        path, model_hash = archive_model(bank, model)
        coverage = float(covered[natural_val].mean()) if natural_val.any() else 0.
        resources = cost(model, params['rank'], params['C'], 2048, natural_val, covered, dv['leaf'], intermediate, params['format'])
        if params['family'] == 'R0':
            resources['host_projection_macs'] = 0
        record = {**operator, **params, 'model_path': path, 'model_hash': model_hash,
                  'id': point_identity(model, {**operator, **params}), 'cutoff': cutoff, 'VALID_loss': val_loss,
                  'VALID_natural_rows': int(natural_val.sum()), 'VALID_covered_rows': int(np.sum(natural_val & covered)),
                  'VALID_coverage': coverage, 'VALID_unsupported_rows': int(np.sum(natural_val & unsupported)),
                  'FP32_fit_VALID_diagnostic_loss': diagnostic, 'resources': resources, 'ridge_diagnostics': reg_info}
        record['FP32_diagnostic_definition'] = 'F32 map coefficients, same exact-real A8 design, F64 BLAS dot then F32 cast and independently calibrated F32 bias; VALID diagnostic only'
        record['unsupported_leaves'] = list(model['unsupported'])
        record['FIT_sum_weights'] = float(w.sum())
        record['FIT_Kish_N'] = float(w.sum() ** 2 / (w @ w))
        record['FIT_distinct_prompts'] = int(np.unique(train['prompt'][fit_ix]).size)
        record['FIT_leaf_counts'] = [{'leaf':c,'rows':int(np.sum(lf==c)),
             'sum_weights':float(w[lf==c].sum()),'Kish_N':float(w[lf==c].sum()**2/(w[lf==c]@w[lf==c])) if np.any(lf==c) else 0.,
             'distinct_prompts':int(np.unique(train['prompt'][fit_ix][lf==c]).size)} for c in range(params['C'])]
        if all(h=='zero' for h in model['map_hashes']):
            const = model['bias'] if params['family']=='R0' else np.stack([num.add_bias(np.zeros((1,model['bias'].shape[1]),np.float32),b)[0] for b in model['bias']])
            record['constant_output_hash'] = array_hash(const)
        else:
            record['constant_output_hash'] = None
        points.append(record)
        if record['VALID_unsupported_rows']:
            inconclusive.append({'point': record['id'], 'reason': 'unsupported_original_fallback_region', 'rows': record['VALID_unsupported_rows']})
        return record
    def fp32_diagnostic(As, d, dv, weights, fit_y, covered, pooled=False):
        pred = np.empty_like(yv)
        for c in range(len(As) if not pooled else C):
            A = As[0 if pooled else c].astype(np.float32).astype(np.float64)
            ix = np.flatnonzero(d['leaf'] == c)
            vi = np.flatnonzero(dv['leaf'] == c)
            if not len(ix):
                pred[vi] = yv[vi]
                continue
            fitpred = (d['v'][ix] @ A).astype(np.float32)
            b = num.bias(fit_y[ix], fitpred, weights[ix])
            pred[vi] = num.add_bias((dv['v'][vi] @ A).astype(np.float32), b)
        return loss(pred, yv, natural_val, covered)
    for C in [1, 4, 16, 32]:
        check = select_tree(num, tree, train['x'], C)
        if np.any(check < 0) or np.bincount(check, minlength=C).min() == 0:
            inconclusive.append({'C': C, 'reason': 'actual_runtime_tree_not_full_nonempty'})
            continue
        for nfit in [80, 120]:
            fit_ix = np.flatnonzero(train['prompt'] < nfit)
            for weighting in ['unweighted', 'route1_offroute_quarter']:
                w = np.ones(len(fit_ix)) if weighting == 'unweighted' else np.where(natural_fit[fit_ix], 1., .25)
                # R0 uses the same C-leaf intercept class, without a map or P.
                df, dv = get_design(C, 128, 'normal')
                lf = df['leaf'][fit_ix]
                bias = regional_mean(num, y[fit_ix], lf, w, C)
                r0 = {'qmaps': np.zeros((C, 2048, 1), np.int8), 'scales': np.ones((C, 2048), np.float32),
                      'bias': bias, 'map_hashes': ['zero'] * C,
                      'unsupported': [c for c in range(C) if not np.any(lf == c)]}
                p0 = bias[dv['leaf']]
                for guard in GUARDS:
                    cutoff = quantile_threshold(dv['distance'], natural_val, guard)
                    cover = np.ones(len(yv), bool) if guard == 100 else (np.zeros(len(yv), bool) if cutoff is None else dv['distance'] <= cutoff)
                    cover &= ~np.isin(dv['leaf'], r0['unsupported'])
                    params = {'C': C, 'rank': 0, 'format': 'none', 'weight': weighting, 'nfit': nfit,
                              'guard': guard, 'family': 'R0', 'endpoint': 'control', 'eta': 'infinity', 'lambda': 'infinity'}
                    register(r0, params, dv, p0, None, cutoff, loss(p0, yv, natural_val, cover), {})
                families = [(128, fmt, 'R1') for fmt in ['normal', 'balanced']]
                families += [(512, fmt, 'R2') for fmt in ['normal', 'balanced']]
                families += [(2048, 'identity', 'R2')]
                if C in [16, 32]:
                    families += [(64, fmt, 'R1') for fmt in ['normal', 'balanced']]
                for rank, fmt, family in families:
                    df, dv = get_design(C, rank, fmt)
                    d = {k: (a[fit_ix] if k != 'P' and isinstance(a, np.ndarray) else a) for k, a in df.items()}
                    pcov, pcross, localstats = statistics(d['v'], y[fit_ix], d['leaf'], w, C, train['prompt'][fit_ix])
                    pfactor = eigen(pcov)
                    localfactors = [eigen(st['cov']) for st in localstats] if family == 'R1' else None
                    pool = []
                    for eta in GRID:
                        A, info = solve(pcov, pcross, eta, factor=pfactor)
                        model = lower(num, [A], d, y[fit_ix], w, C, pooled=True)
                        pred = predict(num, model, dv)
                        pool.append({'eta': eta, 'A': A, 'info': info, 'model': model, 'pred': pred})
                    localcache = {}
                    for guard in GUARDS:
                        cutoff = quantile_threshold(dv['distance'], natural_val, guard)
                        cover = np.ones(len(yv), bool) if guard == 100 else (np.zeros(len(yv), bool) if cutoff is None else dv['distance'] <= cutoff)
                        for p in pool:
                            supported = cover & ~np.isin(dv['leaf'], p['model']['unsupported'])
                            p.setdefault('losses', {})[guard] = loss(p['pred'], yv, natural_val, supported)
                        eligible = [p for p in pool if p['losses'][guard] is not None]
                        if not eligible:
                            inconclusive.append({'C': C, 'rank': rank, 'format': fmt, 'guard': guard,
                                                 'reason': 'VALID_no_natural_hit_or_zero_energy', 'nfit': nfit, 'weight': weighting})
                            continue
                        prior = min(eligible, key=lambda p: (p['losses'][guard], GRID.index(p['eta'])))
                        common = {'C': C, 'rank': rank, 'format': fmt, 'weight': weighting, 'nfit': nfit, 'guard': guard}
                        params = {**common, 'family': family, 'endpoint': 'control', 'eta': prior['eta'], 'lambda': 'infinity'}
                        fpdiag = fp32_diagnostic([prior['A']], d, dv, w, y[fit_ix], cover, pooled=True)
                        register(prior['model'], params, dv, prior['pred'], fpdiag, cutoff, prior['losses'][guard], {**prior['info'],'prequantization_total_df_per_output':prior['info']['df']+sum(s['supported'] for s in localstats)})
                        grids.append({**common, 'family': family, 'pooled_eta_VALID_losses': [(p['eta'], p['losses'][guard]) for p in pool]})
                        if family != 'R1':
                            continue
                        locals_ = []
                        for lam in GRID:
                            cachekey = (prior['eta'], lam)
                            if cachekey in localcache:
                                cached = localcache[cachekey]
                                supported = cover & ~np.isin(dv['leaf'], cached['model']['unsupported'])
                                locals_.append({**cached, 'loss': loss(cached['pred'], yv, natural_val, supported)})
                                continue
                            As, infos = [], []
                            for st, factor in zip(localstats, localfactors):
                                A, info = solve(st['cov'], st['cross'], lam, prior['A'], factor=factor)
                                As.append(A)
                                infos.append({**info, **{k: st[k] for k in ['sum_w', 'kish', 'prompts', 'supported']}})
                            model = lower(num, As, d, y[fit_ix], w, C)
                            pred = predict(num, model, dv)
                            supported = cover & ~np.isin(dv['leaf'], model['unsupported'])
                            value = loss(pred, yv, natural_val, supported)
                            point = {'lambda': lam, 'model': model, 'pred': pred, 'loss': value, 'info': infos, 'slopes': As}
                            localcache[cachekey] = point
                            locals_.append(point)
                        available = [p for p in locals_ if p['loss'] is not None]
                        selected = min(available, key=lambda p: (p['loss'], GRID.index(p['lambda'])))
                        finite = min([p for p in available if p['lambda'] != 'infinity'], key=lambda p: (p['loss'], GRID.index(p['lambda'])))
                        for endpoint, point in [('selected', selected), ('best_finite', finite)]:
                            params = {**common, 'family': 'local', 'endpoint': endpoint, 'eta': prior['eta'], 'lambda': point['lambda']}
                            diag = fp32_diagnostic(point['slopes'], d, dv, w, y[fit_ix], cover)
                            full_df=local_smoother_df(pfactor,prior['eta'],localstats,localfactors,point['lambda'])
                            register(point['model'], params, dv, point['pred'], diag, cutoff, point['loss'], {'pooled': prior['info'], 'leaves': point['info'],
                                'prequantization_total_df_per_output':full_df,'prequantization_total_df_all_outputs':full_df*2048})
                        grids.append({**common, 'family': 'local', 'pooled_eta': prior['eta'],
                                      'lambda_VALID_losses': [(p['lambda'], p['loss']) for p in locals_]})
                    del pool
    energy=float(np.sum(yv[natural_val].astype(np.float64)**2))
    canonical={**operator,'id':hashlib.sha256(json.dumps([operator['model'],operator['layer'],operator['expert'],'canonical_teacher']).encode()).hexdigest(),
        'family':'canonical','endpoint':'control','C':1,'rank':0,'format':'canonical_W8A8','weight':'not_applicable','nfit':120,
        'guard':100,'cutoff':None,'eta':None,'lambda':None,'model_path':None,'model_hash':'canonical_teacher',
        'VALID_loss':0. if energy>0 and natural_val.any() else None,'VALID_natural_rows':int(natural_val.sum()),
        'resources':{'active_coefficient_macs':3*2048*intermediate,'host_projection_macs':0,'retained_storage_bytes':0,
                     'original_fallback_fixed_storage_shared_with_all_families':True,'original_projection_epochs':2},
        'constant_output_hash':None,'unsupported_leaves':[]}
    points.append(canonical)
    save(opfolder / 'points.json', points)
    save(opfolder / 'VALID_grids.json', grids)
    save(opfolder / 'inconclusive_configurations.json', inconclusive)
    save(opfolder/'fit_report.json',{'operator':operator,'seconds':time.monotonic()-fit_started,
        'points':len(points),'inconclusive_configurations':len(inconclusive),'all_intermediate_grid_VALID_losses_retained':True,
        'CPU_fit_walltime_not_B1_deployment_latency':True})
    return points, inconclusive


def learning_curve(points, path):
    groups=defaultdict(dict)
    keys=['model','layer','expert','C','rank','format','weight','guard','family','endpoint']
    for p in points:
        if p['family']!='canonical':groups[tuple(p[k] for k in keys)][p['nfit']]=p
    rows=[]
    for key,pair in groups.items():
        a,b=pair.get(80),pair.get(120)
        rows.append({**dict(zip(keys,key)),'VALID_loss_80':a['VALID_loss'] if a else None,
            'VALID_loss_120':b['VALID_loss'] if b else None,
            'VALID_loss_120_minus_80':b['VALID_loss']-a['VALID_loss'] if a and b and a['VALID_loss'] is not None and b['VALID_loss'] is not None else None,
            '80_eta_lambda':[a['eta'],a['lambda']] if a else None,'120_eta_lambda':[b['eta'],b['lambda']] if b else None})
    save(path,{'scope':'VALID-only learning curve; same full120 unsupervised tree/P; primary N remains120; no automatic plateau/failure threshold','rows':rows})


def freeze_comparisons(points, out, score_ids, inconclusive):
    comparisons = []
    keys = ['model', 'layer', 'expert', 'C', 'weight', 'nfit', 'guard']
    for candidate in points:
        if candidate['family'] != 'local' or candidate['lambda'] == 'infinity':
            continue
        controls = [p for p in points if all(p[k] == candidate[k] for k in keys)
                    and (p['family'] == 'R0' or (p['family'] == 'R1' and p['rank'] == candidate['rank'] and p['format'] == candidate['format']))]
        eligible = []
        for p in controls:
            a, b = p['resources'], candidate['resources']
            if all(a[k] <= b[k] for k in ['active_coefficient_macs', 'host_projection_macs', 'retained_storage_bytes']):
                eligible.append(p)
        if not eligible:
            inconclusive.append({'candidate': candidate['id'], 'reason': 'no_VALID_matched_control_with_no_greater_cost'})
            continue
        control = min(eligible, key=lambda p: (p['VALID_loss'] if p['VALID_loss'] is not None else float('inf'), p['id']))
        for phase in ['prefill', 'decode']:
            cid = candidate['id'] + ':' + phase
            rid = control['id'] + ':' + phase
            comparisons.append({'id': cid + '__minus__' + rid, 'candidate': cid, 'control': rid,
                                'operator': [candidate['model'], candidate['layer'], candidate['expert']],
                                'phase': phase, 'nfit': candidate['nfit'], 'format': candidate['format'],
                                'scope': 'natural_route_operator', 'control_matched_only_on_VALID': True})
    # The complete composite comparison list is formed without SCORE data.
    groups = defaultdict(list)
    byid = {p['id']: p for p in points}
    for c in comparisons:
        p = byid[c['candidate'].rsplit(':', 1)[0]]
        key = tuple(p[k] for k in ['C', 'rank', 'format', 'weight', 'nfit', 'guard', 'endpoint']) + (c['phase'],)
        groups[key].append(c)
    composite = []
    for key, group in groups.items():
        for model in ['deepseek', 'olmoe', 'both']:
            rows = [c for c in group if model == 'both' or c['operator'][0] == model]
            expected = 8 if model == 'both' else 4
            if len(rows) != expected:
                inconclusive.append({'composite': list(key) + [model], 'reason': 'incomplete_operator_comparison_family'})
                continue
            for mode in ['equal_operator', 'pooled_energy']:
                identity = hashlib.sha256(json.dumps([key, model, mode], sort_keys=True).encode()).hexdigest()
                composite.append({'id': identity, 'candidate': identity + ':candidate', 'control': identity + ':control',
                                  'components': rows, 'aggregation': mode, 'model_scope': model, 'scope': 'composite_natural_route',
                                  'phase': key[-1], 'nfit': key[4]})
    freeze = {'status': 'frozen_before_SCORE_capture_or_targets', 'score_prompt_ids': score_ids,
              'points': points, 'comparisons': comparisons + composite, 'inconclusive_configurations': inconclusive,
              'bootstrap': {'repeats': 10000, 'seed': SEED, 'same40IDs': True,
                            'formula': 's_j=SD_boot(Delta_bj), c=q95(max_j abs((Delta_bj-Delta_hat_j)/s_j)); bands Delta_hat_j+-c*s_j',
                            'undefined': 'any no-hit/zero-energy/nonliteral zero-SD draw makes family inference inconclusive; no draw removal'},
              'selection': 'every format reported separately; matched control among same-format same-rank pooled R1 and same-leaf R0; all points including R2 and lambda-infinity reported; no automatic science decision'}
    save(out, freeze)
    return freeze


def score_point(pred, y, ds, natural, alpha, covered):
    per_prompt = {}
    residual = pred.astype(np.float64) - y.astype(np.float64)
    square = np.sum(residual * residual, axis=1)
    energy = np.sum(y.astype(np.float64) ** 2, axis=1)
    absolute = np.sum(np.abs(residual), axis=1)
    dot = np.sum(pred.astype(np.float64) * y, axis=1)
    pn = np.sqrt(np.sum(pred.astype(np.float64) ** 2, axis=1) * energy)
    for phase, ph in [('prefill', 0), ('decode', 1)]:
        values = {}
        for scope, use in [('natural', natural), ('all_x', np.ones(len(y), bool))]:
            sse, den, hits, coverage = [], [], [], []
            abs_error, cos_sum, cos_n, alpha_sse, alpha_energy = [], [], [], [], []
            for p in range(40):
                mask = (ds['prompt'] == p) & (ds['phase'] == ph) & use
                cov = mask & covered
                fallback = mask & ~covered
                sse.append(float(square[cov].sum()))
                den.append(float(energy[mask].sum()))
                hits.append(int(mask.sum())); coverage.append(int(cov.sum()))
                abs_error.append(float(absolute[cov].sum()))
                good = cov & (pn > 0)
                # Exact fallback predictions are teacher, hence cosine1 on
                # nonzero-energy fallback rows. Undefined zero-energy cosine
                # is counted, never silently included as perfect agreement.
                cos_sum.append(float(np.sum(dot[good] / pn[good])) + int(np.sum(fallback & (energy > 0))))
                cos_n.append(int(good.sum() + np.sum(fallback & (energy > 0))))
                alpha_sse.append(float(np.sum(square[cov] * alpha[cov] ** 2)))
                alpha_energy.append(float(np.sum(energy[mask] * alpha[mask] ** 2)))
            values[scope] = {'sse': sse, 'energy': den, 'hits': hits, 'covered_hits': coverage,
                             'absolute_error': abs_error, 'cosine_sum': cos_sum, 'cosine_defined_rows': cos_n,
                             'route_alpha_squared_sse': alpha_sse, 'route_alpha_squared_energy': alpha_energy,
                             'covered_sse': sse, 'covered_energy': [float(energy[(ds['prompt'] == p) & (ds['phase'] == ph) & use & covered].sum()) for p in range(40)]}
        per_prompt[phase] = values
    return per_prompt


def score_stage(num, freeze, datasets, targets, trees, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    statistics_ = {}
    literal = set()
    details = {}
    pred_cache = {}
    design_cache = {}
    for point in freeze['points']:
        op = (point['model'], point['layer'], point['expert'])
        ds, y = datasets[op[:2]], targets[op]
        natural, alpha = route(ds, point['expert'])
        if point['family']=='canonical':
            metric=score_point(y,y,ds,natural,alpha,np.ones(len(y),bool))
            costs={phase:{'natural_rows':int(np.sum(natural&(ds['phase']==ph))),
                'total_active_coefficient_macs':int(np.sum(natural&(ds['phase']==ph)))*3*2048*point['intermediate'],
                'active_coefficient_macs':3*2048*point['intermediate'] if np.any(natural&(ds['phase']==ph)) else None,
                'host_projection_macs':0,'original_projection_epochs':2,'physical_latency_claim':False}
                for phase,ph in [('prefill',0),('decode',1)]}
            details[point['id']]={'point':point,'metrics':metric,'SCORE_costs_by_phase':costs}
            for phase in ['prefill','decode']:statistics_[point['id']+':'+phase]=metric[phase]['natural']
            continue
        dkey = op[:2] + (point['C'], point['rank'] or 128, point['format'] if point['format'] != 'none' else 'normal')
        if dkey not in design_cache:
            tree, P, sigma, _ = trees[op[:2]]
            design_cache[dkey] = design(num, ds['x'], tree, point['C'], P, sigma, dkey[-2], dkey[-1])
        d = design_cache[dkey]
        model = load_model(point['model_path'])
        pkey = (op, point['model_hash'], dkey)
        if pkey not in pred_cache:
            pred_cache = {}  # bounded to current map/design, not a hidden cache benefit
            pred_cache[pkey] = model['bias'][d['leaf']] if point['family'] == 'R0' else predict(num, model, d)
        pred = pred_cache[pkey]
        covered = np.ones(len(y), bool) if point['guard'] == 100 else (np.zeros(len(y), bool) if point['cutoff'] is None else d['distance'] <= point['cutoff'])
        covered &= ~np.isin(d['leaf'], model['unsupported'])
        metric = score_point(pred, y, ds, natural, alpha, covered)
        details[point['id']] = {'point': point, 'metrics': metric, 'unsupported_SCORE_natural_rows': int(np.sum(natural & np.isin(d['leaf'], model['unsupported'])))}
        details[point['id']]['SCORE_costs_by_phase'] = {phase: cost(model, point['rank'], point['C'], 2048,
            natural & (ds['phase'] == ph), covered, d['leaf'], point['intermediate'], point['format'])
            for phase, ph in [('prefill', 0), ('decode', 1)]}
        for phase in ['prefill', 'decode']:
            statistics_[point['id'] + ':' + phase] = metric[phase]['natural']
    pointbyid = {p['id']: p for p in freeze['points']}
    for c in freeze['comparisons']:
        if 'components' not in c:
            a = pointbyid[c['candidate'].rsplit(':', 1)[0]]
            b = pointbyid[c['control'].rsplit(':', 1)[0]]
            # Function ties require identical lowered matrices/scales/biases,
            # basis/selector/guard semantics, not merely coincident SCORE errors.
            same_guard=all(a[k]==b[k] for k in ['C','guard','cutoff','unsupported_leaves'])
            same_constant=a['constant_output_hash'] is not None and a['constant_output_hash']==b['constant_output_hash']
            if same_guard and (same_constant or (a['model_hash'] == b['model_hash'] and all(a[k] == b[k] for k in ['rank', 'format']))):
                literal.add(c['id'])
            continue
        for endpoint in ['candidate', 'control']:
            parts = [statistics_[p[endpoint]] for p in c['components']]
            numers = np.stack([p['sse'] for p in parts])
            denoms = np.stack([p['energy'] for p in parts])
            if c['aggregation'] == 'pooled_energy':
                numers, denoms = numers.sum(axis=0), denoms.sum(axis=0)
            statistics_[c[endpoint]] = {'sse': numers.tolist(), 'energy': denoms.tolist()}
        if all(p['id'] in literal for p in c['components']):
            literal.add(c['id'])
    save(output / 'prompt_metrics.json', details)
    inferred = bootstrap(freeze['comparisons'], statistics_, freeze['score_prompt_ids'], literal)
    save(output / 'paired_bootstrap.json', inferred)
    save(output / 'report.json', {'status': 'complete_pending_Root_critic_scientific_review',
         'SCORE_used_for_selection': False, 'comparisons': len(freeze['comparisons']), 'points': len(freeze['points']),
         'family_inconclusive': inferred['family_inconclusive'], 'local_operator_error_not_whole_model_quality': True})
