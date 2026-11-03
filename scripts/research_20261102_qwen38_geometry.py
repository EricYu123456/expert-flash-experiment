#!/usr/bin/env python3
"""Route-independent whole-row work/depth certificates for the new shape.

This is a geometry calculation under the existing nominal AiF service model.
It is not a new algorithm, routing measurement, or claim about NVFP4 hardware.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'results/research-2026-11-02/round2/qwen38-placement'
REV = '21a7c3ca68484a9987d62fd51b78130d9b9a11f8'
URL = 'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/'+REV+'/config.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def complete_rows(waves, rows, columns):
    w = np.asarray(waves, dtype=np.int64)
    lo = np.where(w > 0, (w-1)*65536//columns+1, 0)
    hi = w*65536//columns
    if not int(lo.sum()) <= rows <= int(hi.sum()):
        return None
    result = lo.copy(); remaining = rows-int(result.sum())
    while remaining:
        take = np.flatnonzero(result < hi)[:remaining]
        result[take] += 1; remaining -= len(take)
    assert int(result.sum()) == rows
    assert np.array_equal((result*columns+65535)//65536, w)
    return result.tolist()


def frontier(rows, columns):
    equal = np.full(16, rows//16)
    equal[:rows%16] += 1
    budget = int(((equal*columns+65535)//65536).sum())
    useful_floor = (rows*columns+65535)//65536
    results = []
    # Exact separable capacity DP; restrict sums only above a known feasible
    # budget. Any solution above it cannot improve total-wave count.
    for depth in range(1, budget+1):
        dp = np.full((17, budget+1), -1, dtype=np.int64)
        parent = np.full_like(dp, -1); dp[0, 0] = 0
        for n in range(1, 17):
            for w in range(1, min(depth, budget)+1):
                old = dp[n-1, :budget+1-w]
                capacity = old+w*65536//columns
                target = dp[n, w:]
                better = (old >= 0) & (capacity > target)
                target[better] = capacity[better]; parent[n, w:][better] = w
            feasible = np.flatnonzero(dp[n] >= rows)
            if not len(feasible):
                continue
            total = int(feasible[0]); remain = total; ws = []
            for nn in range(n, 0, -1):
                w = int(parent[nn, remain]); assert w > 0
                ws.append(w); remain -= w
            assert remain == 0
            ws = sorted(ws, reverse=True)+[0]*(16-n)
            rr = complete_rows(ws, rows, columns)
            # Minimal-wave capacity construction can have excessive lower ceil
            # endpoints for small shapes; those templates would be infeasible.
            if rr is not None:
                results.append(dict(total_waves=total, max_job_waves=max(ws),
                                    positive_jobs=n, rows=rr, waves=ws))
    bykey = {(r['total_waves'], r['max_job_waves'], r['positive_jobs']):r for r in results}
    nondominated = [r for key, r in bykey.items()
                    if not any(all(a <= b for a, b in zip(k, key)) and k != key for k in bykey)]
    # Retain the most evenly distributed *whole-wave* geometry for each total;
    # scalar depth/job Pareto dominance does not imply query-load dominance.
    balanced = []
    for total in range(useful_floor, budget+1):
        w = np.full(16, total//16); w[:total%16] += 1
        rr = complete_rows(w, rows, columns)
        if rr is not None:
            balanced.append(dict(total_waves=total, max_job_waves=int(w.max()),
                                 positive_jobs=int((w > 0).sum()), rows=rr, waves=w.tolist()))
            break
    assert balanced
    width_controls = []
    for g in range(1, 17):
        r = np.full(g, rows//g); r[:rows%g] += 1
        w = (r*columns+65535)//65536
        width_controls.append(dict(width=g, total_waves=int(w.sum()), max_job_waves=int(w.max()),
                                    rows=r.tolist(), waves=w.tolist()))
    return dict(rows=rows, columns=columns, useful_bytes=rows*columns,
                pooled_useful_work_floor=useful_floor,
                uniform16=dict(rows=equal.tolist(), waves=((equal*columns+65535)//65536).tolist(),
                               total_waves=budget),
                exact_capacity_dp_frontier=nondominated, balanced_whole_wave=balanced[0],
                equal_row_all_width_controls=width_controls,
                proof='For n positive jobs, depth cap d and total waves s, DP maximizes sum floor(w_i*65536/C). A matrix with R whole rows requires at least R capacity; materialized rows additionally satisfy every ceil interval. Certificates retain the parent allocation.')


def main():
    DEST.mkdir(parents=True, exist_ok=True); config = DEST/'official_config.json'
    if not config.exists():
        response = requests.get(URL, timeout=30); response.raise_for_status()
        config.write_bytes(response.content)
    original = json.loads(config.read_text()); c = original['text_config']
    assert (c['hidden_size'], c['moe_intermediate_size'], c['num_experts'], c['num_experts_per_tok'], c['num_hidden_layers']) == (2560, 640, 512, 10, 48)
    assert c['shared_expert_intermediate_size'] == 640
    phases = [dict(name='gate_up', **frontier(1280, 2560)), dict(name='down', **frontier(2560, 640))]
    assert phases[0]['balanced_whole_wave']['total_waves'] == 51
    assert phases[1]['balanced_whole_wave']['total_waves'] == 26
    for p in phases:
        assert max(p['balanced_whole_wave']['waves']) == max(p['uniform16']['waves'])
    output = dict(status='complete route-independent geometry certificates',
                  created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
                  official_config=dict(url=URL, revision=REV, sha256=sha(config)), phases=phases,
                  scope='Whole output rows, one copy,16 chips,16KiB pages,4-plane64KiB waves, ordinary reads. A raw-I8 geometry calculation only.',
                  limits=['No true route costs or speedups have been computed here.',
                          'The public routing checkpoint is NVFP4; these byte counts do not describe its original deployed format.',
                          'Useful NAND work can improve while commands, channels, dependencies and feasible schedules determine actual modeled service.',
                          'These templates are applications of existing geometry/declustering ideas, not a novelty claim.'])
    path = DEST/'geometry_certificates.json'; path.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(path=str(path), sha256=sha(path), phases=[dict(name=p['name'], uniform_work=p['uniform16']['total_waves'],
                      balanced_work=p['balanced_whole_wave']['total_waves'], useful_floor=p['pooled_useful_work_floor']) for p in phases]), indent=2))


if __name__ == '__main__':
    main()
