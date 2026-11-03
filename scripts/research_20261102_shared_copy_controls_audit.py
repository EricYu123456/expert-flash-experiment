#!/usr/bin/env python3
"""Independent copied-owner, scalar-bound and resource-trace verification.

Does not import the experiment's allocator, selector or scheduler. The trace
checker comes from the separately implemented earlier geometry audit.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from research_20261102_shared_more_audit import audit_trace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
DEST = BASE / 'shared-copy-controls'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def service(rows, columns):
    r = np.asarray(rows, dtype=np.int64)
    return ((r*columns+65535)//65536)*28000 + np.where(r > 0, 250+(columns+1)//2+2*r, 0)


def best_shared(occupied, allowed):
    values = []
    for p in range(8):
        costs = []
        for a in allowed:
            add = service(np.asarray([128*a, 128*(2-a)]), 1536)
            costs.append(np.maximum(occupied[:, p]+add[0], occupied[:, p+8]+add[1]))
        values.append(np.min(costs, axis=0))
    return np.max(values, axis=0)


def main():
    path = DEST/'screen.json'; screen = json.loads(path.read_text())
    frozen = json.loads((BASE/'geometry-study/granite4tiny.frozen.json').read_text())
    joint = json.loads((BASE/'pipeline-joint/joint_work_screen.json').read_text())
    oracle = json.loads((BASE/'shared-oracle/screen.json').read_text())
    assert all(sha(p) == h for p, h in screen['inputs_sha256'].items())
    assert sha(ROOT/'scripts/research_20261102_shared_copy_controls.py') == screen['source_sha256']
    assert sha(frozen['training_routes']) == screen['training_routes_sha256']
    with np.load(frozen['training_routes']) as f:
        routes = f['routes'][:32, :64].copy(); layers = f['layers'].tolist()
        assert all(s.startswith('train_') for s in f['prompt_ids'].tolist())
    phases = {(p['layer'], p['name']):p for p in frozen['phases']}
    checks = []; costs = {}; scalar_count = node_count = job_count = 0
    frozen_hashes = {str(path):sha(path), **screen['inputs_sha256']}
    for record in screen['records']:
        label = record['label']; layer = record['layer']; mapping = record['mapping']
        selected = routes[:, :, layers.index(layer)].reshape(-1, 6)
        assert all(len(set(s)) == 6 for s in selected.tolist())
        gu = np.asarray(phases[layer, 'gate_up']['rows'])
        down = np.asarray(phases[layer, 'down']['rows'])
        if mapping == 'joint':
            colors = np.asarray(next(x for x in joint['results'] if x['layer'] == layer)['joint_best']['labels'])
            down = np.where(np.arange(16)[None, :]//4 == colors[:, None], 0, 128)
        elif mapping == 'opposite':
            down = np.where(gu > 0, 0, 192)
        else:
            assert mapping == 'source'
        assert np.all(gu.sum(axis=1) == 1024) and np.all(down.sum(axis=1) == 1536)
        assert np.all(((gu == 0) | (gu == 128)))
        plan = next(x for x in oracle['layers'] if x['layer'] == layer)
        if record['policy'] == 'hybrid':
            copied = [plan['optimal_hybrid']['routed_expert']]
            allowed = (0, 1) if plan['optimal_hybrid']['copy_original_shared_half'] == 'A' else (1, 2)
        else:
            copied = plan[record['policy']]; allowed = (1,)
        assert copied == record['copied_experts'] and list(allowed) == record['shared_allowed_a']
        extra_tiles = 8*len(copied) + (8 if record['policy'] == 'hybrid' else 0)
        assert extra_tiles == 16
        sd = np.full(16, 64); sd[record['side']:record['side']+8] = 128
        npz = DEST/(label+'.npz'); decpath = DEST/(label+'.decisions.json'); tracepath = DEST/(label+'.trace.json')
        assert sha(npz) == record['npz_sha256'] and sha(decpath) == record['decisions_sha256']
        assert sha(tracepath) == record['trace_sha256']
        frozen_hashes.update({str(p):sha(p) for p in (npz, decpath, tracepath)})
        decisions = json.loads(decpath.read_text()); assert len(decisions) == 2048
        rr = gu[selected].copy(); sg = np.zeros((2048, 16), dtype=np.int64)
        for i, decision in enumerate(decisions):
            flipped = decision['flipped_logical_experts']; a = decision['shared_choices']
            assert len(flipped) == len(set(flipped)) and all(e in copied and e in selected[i] for e in flipped)
            assert len(a) == 8 and all(v in allowed for v in a)
            for ei, e in enumerate(selected[i]):
                if e in flipped:
                    rr[i, ei] = np.r_[rr[i, ei, 8:], rr[i, ei, :8]]
            sg[i, :8] = np.asarray(a)*128; sg[i, 8:] = (2-np.asarray(a))*128
        assert np.all(rr.sum(axis=-1) == 1024) and np.all(sg.sum(axis=1) == 2048)
        occupied = service(rr, 1536).sum(axis=1) + service(down[selected], 512).sum(axis=1) + service(sd, 1024)
        bound = (occupied+service(sg, 1536)).max(axis=1)+2000
        # Enumerate all same-copy whole-expert choices independently, followed by
        # every legal shared pair split. No original selector code is imported.
        optimum = np.full(2048, np.iinfo(np.int64).max, dtype=np.int64)
        for bits in itertools.product((False, True), repeat=len(copied)):
            candidate = gu[selected].copy()
            for e, flag in zip(copied, bits):
                if flag:
                    mask = selected == e
                    candidate[mask] = np.r_[gu[e, 8:], gu[e, :8]]
            used = service(candidate, 1536).sum(axis=1) + service(down[selected], 512).sum(axis=1) + service(sd, 1024)
            optimum = np.minimum(optimum, best_shared(used, allowed)+2000)
        assert np.array_equal(bound, optimum)
        with np.load(npz) as f:
            actual = f['cost'].copy()
            assert np.array_equal(bound, f['native_chip_bound']) and np.all(actual >= bound)
        assert record['mean_ns'] == float(actual.mean())
        assert record['p99_ns'] == float(np.percentile(actual, 99))
        trace = json.loads(tracepath.read_text()); first = decisions[0]['shared_choices']
        ranges = []
        for p, a in enumerate(first):
            if a == 1:
                ranges += [(p, 256*p, 128), (p+8, 256*p+128, 128)]
            else:
                ranges += [(p if a == 2 else p+8, 256*p, 256)]
        trace['shared_gu_ranges'] = ranges; trace['replica'] = record['policy'] == 'hybrid'
        ns, js = audit_trace(trace, rr[0], down[selected[0]], sd, sg[0], actual[0])
        node_count += ns; job_count += js; scalar_count += len(actual)
        key = (mapping, record['policy'], record['side'])
        costs.setdefault(key, []).append(actual)
        checks.append(dict(label=label, queries=len(actual), optimal_fixed_copy_bound=True,
                           extra_tiles=extra_tiles, trace_nodes=ns))
    for total in screen['totals']:
        values = np.stack(costs[total['mapping'], total['policy'], total['side']]).sum(axis=0)
        assert float(values.mean()) == total['mean_ns']
        assert float(np.percentile(values, 99)) == total['p99_ns']
    expanded = json.loads((BASE/'shared-pipeline-more/screen.json').read_text())
    new = min((r for r in expanded['totals'] if r['replica']), key=lambda r:r['mean_ns'])
    old = min(screen['totals'], key=lambda r:r['mean_ns'])
    assert all(sha(p) == h for p, h in frozen_hashes.items())
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), screen_sha256=sha(path), scalar_queries=scalar_count,
                  independently_checked_trace_nodes=node_count, trace_jobs=job_count,
                  records=checks, best_same_capacity_control=old, best_paired_copy=new,
                  additional_mean_gain_pct=100*(1-new['mean_ns']/old['mean_ns']),
                  verified=['All fixed native copy inventories add16 full GU128x1536 tiles=3MiB/layer.',
                            'Every selected full row has an allowed original or copied owner; hybrid shared moves only the stored copied half.',
                            'All110592 scalar bounds independently reconstructed, including every legal routed whole-copy/shared split combination.',
                            '54 full resource/dependency/chip-ownership traces checked by independent earlier auditor.',
                            'Aggregate mean/P99 recomputed from the same per-query three-layer sums.'],
                  limits=['Copy inventories came from the GU NAND-only oracle, not a native full-DAG optimum over all possible replica locations.',
                          'No native numeric ABI or device measurement follows from an address/timing audit.',
                          'Gains remain train-only first3layer and are not publication novelty evidence.'])
    (DEST/'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','verified')}, indent=2))


if __name__ == '__main__':
    main()
