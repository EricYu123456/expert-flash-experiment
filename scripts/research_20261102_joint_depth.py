#!/usr/bin/env python3
"""Train-only work/depth tradeoff for unchanged, complete-row GEMVs.

A G4 GU matrix needs 24 waves if its eight jobs each take three waves.
Allowing 25 waves permits 16 jobs taking at most two waves.  Combining this
with twelve one-wave down jobs can balance each expert's joined chip work at
two or three waves.  These are placement cost screens, not timing claims.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
DEST = BASE / 'pipeline-joint-depth'
SOURCE = BASE / 'geometry-study/granite4tiny.frozen.json'
EXPECTED = 'a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit(selected, weights, initial, objective, seed):
    rng = np.random.default_rng(seed)
    # Per expert, five extra-work chips avoid the four-chip omitted group.
    profiles = np.array(initial, dtype=np.int64)
    e_count, chips = profiles.shape
    assert chips == 17  # omit label, followed by sixteen extra-mask bits
    mask = profiles[:, 1:]
    omits = profiles[:, 0]
    omitted = (np.arange(16)[None, :] // 4 == omits[:, None]).astype(np.int64)
    assert np.all(mask.sum(axis=1) == 5) and not np.any(mask & omitted)
    extra = mask[selected].sum(axis=1)
    missing = omitted[selected].sum(axis=1)
    k = selected.shape[1]
    adjacency = [np.flatnonzero((selected == e).any(axis=1)) for e in range(e_count)]

    def costs(ext, omit):
        joined = 2*k + ext.max(axis=1)
        phase = 2*k + (ext+omit).max(axis=1)-omit.min(axis=1)
        # Lexicographic tie-break; a full route-wave always dominates ties.
        primary, secondary = (joined, phase) if objective == 'joined' else (phase, joined)
        return primary*10000 + secondary

    values = costs(extra, missing)
    score = int(values @ weights)
    best_score, best = score, profiles.copy()
    proposals = 10000
    total_weight = int(weights.sum())
    for step in range(proposals):
        e = int(rng.integers(e_count)); old = profiles[e].copy(); new = old.copy()
        if rng.random() < .72:
            a = int(rng.choice(np.flatnonzero(old[1:])))
            free = np.flatnonzero((old[1:] == 0) & (np.arange(16)//4 != old[0]))
            b = int(rng.choice(free)); new[1+a] = 0; new[1+b] = 1
        else:
            g = int(rng.integers(3)); g += g >= old[0]
            new[0] = g
            new[1+4*g:1+4*(g+1)] = 0
            needed = 5-int(new[1:].sum())
            free = np.flatnonzero((new[1:] == 0) & (np.arange(16)//4 != g))
            if needed:
                new[1+rng.choice(free, size=needed, replace=False)] = 1
        ii = adjacency[e]
        dmask = new[1:] - old[1:]
        domit = ((np.arange(16)//4 == new[0]).astype(np.int64) -
                 (np.arange(16)//4 == old[0]).astype(np.int64))
        ne = extra[ii] + dmask
        no = missing[ii] + domit
        nv = costs(ne, no)
        delta = int((nv-values[ii]) @ weights[ii])
        temperature = .03 * (.00015/.03)**(step/(proposals-1))
        if delta <= 0 or rng.random() < math.exp(-delta/(total_weight*10000*temperature)):
            profiles[e] = new; extra[ii] = ne; missing[ii] = no; values[ii] = nv; score += delta
            if score < best_score:
                best_score, best = score, profiles.copy()
    # Exhaustive one-bit exchanges with the omit group held fixed.
    profiles = best.copy(); mask = profiles[:, 1:]
    omitted = (np.arange(16)[None, :]//4 == profiles[:, 0, None]).astype(np.int64)
    extra = mask[selected].sum(axis=1); missing = omitted[selected].sum(axis=1)
    values = costs(extra, missing); score = int(values @ weights)
    moves = 0; converged = False
    for sweep in range(32):
        changed = 0
        for e in rng.permutation(e_count):
            old = profiles[e].copy(); ii = adjacency[e]; choice = None
            for a in np.flatnonzero(old[1:]):
                for b in np.flatnonzero((old[1:] == 0) & (np.arange(16)//4 != old[0])):
                    ne = extra[ii].copy(); ne[:, a] -= 1; ne[:, b] += 1
                    nv = costs(ne, missing[ii]); delta = int((nv-values[ii]) @ weights[ii])
                    if delta < 0 and (choice is None or delta < choice[0]):
                        choice = delta, a, b, ne, nv
            if choice:
                delta, a, b, ne, nv = choice
                profiles[e, 1+a] = 0; profiles[e, 1+b] = 1
                extra[ii] = ne; values[ii] = nv; score += delta; changed += 1
        moves += changed
        if not changed:
            converged = True; break
    assert np.all(profiles[:, 1:].sum(axis=1) == 5)
    assert not np.any(profiles[:, 1:] & (np.arange(16)[None, :]//4 == profiles[:, 0, None]))
    joined = 2*k + extra.max(axis=1)
    phase = 2*k + (extra+missing).max(axis=1)-missing.min(axis=1)
    return dict(objective=objective, seed=seed, profiles=profiles.tolist(),
                mean_joined_work_bound_waves=float(joined @ weights/total_weight),
                mean_barrier_work_bound_waves=float(phase @ weights/total_weight),
                anneal_proposals=proposals, greedy_mask_sweeps=sweep+1,
                greedy_mask_moves=moves, mask_exchange_local_converged=converged,
                limitations='No global optimum certificate; greedy termination covers fixed-omit one-bit exchanges only.')


def rows_from_profiles(p):
    p = np.asarray(p, dtype=np.int64)
    extra = p[:, 1:]
    omitted = np.arange(16)[None, :]//4 == p[:, 0, None]
    high = extra.astype(bool) | omitted
    gu = np.full_like(extra, 40)
    for e in range(len(p)):
        chips = np.flatnonzero(high[e]); assert len(chips) == 9
        gu[e, chips] = 82; gu[e, chips[:6]] += 1
    down = np.where(omitted, 0, 128)
    assert np.all(gu.sum(axis=1) == 1024) and np.all(down.sum(axis=1) == 1536)
    gw = (gu*1536+65535)//65536; dw = (down*512+65535)//65536
    assert np.all(gw.sum(axis=1) == 25) and np.all(dw.sum(axis=1) == 12)
    assert np.array_equal(gw+dw, 2+extra)
    return gu.tolist(), down.tolist()


def main():
    assert sha(SOURCE) == EXPECTED
    p = json.loads(SOURCE.read_text()); routepath = Path(p['training_routes'])
    assert sha(routepath) == p['training_routes_sha256']
    with np.load(routepath) as f:
        routes = f['routes']; layers = f['layers'].tolist(); ids = f['prompt_ids'].tolist()
    assert routes.shape == (32,64,40,6) and all(x.startswith('train_') for x in ids)
    rng = np.random.default_rng(20261102)
    results = []
    for li, layer in enumerate(layers[:3]):
        selected, weights = np.unique(np.sort(routes[:,:,li].reshape(-1,6),axis=1),axis=0,return_counts=True)
        initial = np.zeros((64,17),dtype=np.int64)
        for e in range(64):
            g = e%4; initial[e,0] = g
            free = np.flatnonzero(np.arange(16)//4 != g)
            initial[e,1+rng.choice(free,size=5,replace=False)] = 1
        runs = []
        for objective in ('barrier','joined'):
            for si in range(3):
                result = fit(selected, weights, initial, objective, 20261102+100*li+si)
                runs.append(result)
                print(layer, objective, si, result['mean_joined_work_bound_waves'],
                      result['mean_barrier_work_bound_waves'], flush=True)
        chosen = {}
        for objective in ('barrier','joined'):
            key = 'mean_joined_work_bound_waves' if objective == 'joined' else 'mean_barrier_work_bound_waves'
            best = min((r for r in runs if r['objective'] == objective),key=lambda r:r[key])
            gu, down = rows_from_profiles(best['profiles'])
            chosen[objective] = {**best, 'gu_rows':gu, 'down_rows':down}
        results.append(dict(layer=layer, **chosen, runs=runs))
        DEST.mkdir(parents=True,exist_ok=True)
        (DEST/'partial.json').write_text(json.dumps(results,indent=2)+'\n')
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
               frozen_plan=str(SOURCE),frozen_plan_sha256=EXPECTED, training_routes=str(routepath),
               training_routes_sha256=sha(routepath), scope='Train32x64 first3 layers only; no heldout. Necessary work bounds, not schedules.',
               physical_contract='Complete dot products and one-copy weights. GU 25 waves/100 pages instead of 24 waves/96 pages; down remains12 waves/48pages. Extra bytes are row padding, not compressed weights.',
               numeric_scope='Same nominal raw-I8 cost replay caveat as prior studies. Relocation preserves each full-row native operation; no column partial sums.',
               idea='Work/depth Pareto tradeoff: GU adds one aggregate chip-wave to lower its per-job maximum from3 to2; joint omit/extra masks balance ready-stage load.',
               work_depth_certificate=dict(gu_rows=1024,columns=1536,wave_bytes=65536,
                   capacity_at_24waves_depth2=12*85,
                   capacity_at_24waves_depth2_all16active=8*85+8*42,
                   capacity_at_25waves_depth2=9*85+7*42,
                   three_wave_jobs=8,three_wave_total=24,two_wave_total=25,
                   explanation='With max2 waves/job and24total, twelve two-wave jobs maximize capacity at1020 rows<1024. If all16chips are active, eight two-wave and eight one-wave jobs give1016 rows. Our25wave construction permits1059 rows. At24total,8three-wave128-row jobs are exact.'),
               results=results,
               limitations=['Existing declustering expresses these atoms; no novelty is established by this screen.',
                   'Dispatch, host activation/combine, channels, PCIe and per-chip capacity remain pending separate replay.',
                   'Matched objective budgets and one-bit exchange checks do not prove global optimizer strength.'])
    assert out['work_depth_certificate']['capacity_at_24waves_depth2']==1020
    assert out['work_depth_certificate']['capacity_at_25waves_depth2']==1059
    (DEST/'screen.json').write_text(json.dumps(out,indent=2)+'\n')
    assert sha(SOURCE) == EXPECTED


if __name__ == '__main__':
    main()
