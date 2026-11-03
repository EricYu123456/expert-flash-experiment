#!/usr/bin/env python3
"""Independent post-run integrity/arithmetic checks; does not select data/bounds."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path('/home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round01-stage0')
PROTO = ROOT.parent / 'gate_zero_stage0_protocol.frozen.json'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def f32(v):
    return struct.unpack('<f', struct.pack('<f', v))[0]


def val(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]


def bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def fnv(v):
    h = 14695981039346656037
    for byte in v:
        h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(h)


def main():
    # Freeze this verifier's identity before opening empirical output files.
    audit_hash = sha(Path(__file__))
    protocol = json.loads(PROTO.read_text())
    assert sha(PROTO) == '39f9fed28de41de740bbd659a00360c7294bd61ee81c9832028b76598ce5e65d'
    ready = json.loads((ROOT / 'latest_ready.json').read_text())
    freeze = json.loads(Path(ready['receipt']).read_text())
    for p, h in freeze['file_sha256'].items():
        assert sha(p) == h, p
    run = json.loads((ROOT / 'run_receipt.json').read_text())
    assert run['status'] == 'complete'
    result = {'status': 'passed', 'verifier_sha256': audit_hash,
              'scope': 'Independent post-run integrity/exact integer/sample F32 checks; no new inference, thresholds, or data selection', 'models': {}}
    source_jobs = protocol['selection']['jobs']
    jobs = {j['id']: j['selection_index'] for j in source_jobs}
    names = ['global', 'block128', 'conditional', 'primary_rowmin', 'deflation16', 'all_rowmin']
    for model in protocol['models']:
        name = model['name']
        md = ROOT / name
        layers = 27 if name == 'deepseek-v2-lite' else 16
        reports = {j: json.loads((md / (j + '.json')).read_text()) for j in jobs}
        for j, r in reports.items():
            assert r['selection_index'] == jobs[j] and r['decoded_steps'] == 8
            assert len(r['generated_token_ids']) == 8
            assert r['retained_token_ids'] == r['full_token_ids'][:128]
            assert len(r['retained_token_ids']) == min(128, r['full_prompt_tokens'])
        inpbytes = (md / 'common_decode_inputs.i8').read_bytes()
        seen = set()
        energy = [[0] * 2048 for _ in range(layers)]
        with open(md / 'common_decode_inputs.jsonl') as f:
            for line in f:
                r = json.loads(line)
                k = (r['job'], r['step'], r['layer'])
                assert k not in seen and 0 <= r['step'] < 8 and 0 <= r['layer'] < layers
                seen.add(k)
                assert r['count'] == 2048 and math.isfinite(val(r['sx_bits'])) and val(r['sx_bits']) > 0
                b = inpbytes[r['offset']:r['offset'] + r['count']]
                assert len(b) == 2048 and fnv(b) == r['qx_fnv1a64']
                if jobs[r['job']] < 4:
                    for i, x in enumerate(struct.unpack('<2048b', b)):
                        energy[r['layer']][i] += x * x
        assert len(seen) == layers * 8 * 8 and len(inpbytes) == len(seen) * 2048
        coords = [json.loads(s) for s in (md / 'deflation_coordinates.frozen.jsonl').read_text().splitlines()]
        assert len(coords) == layers
        for r in coords:
            expected = sorted(range(2048), key=lambda j: (-energy[r['layer']][j], j))[:16]
            assert r['unique_decode_inputs'] == 32 and r['coordinates'] == expected
            assert r['selected_energy'] == [energy[r['layer']][j] for j in expected]
        sample_checks = 0
        with open(md / 'audit_samples.jsonl') as f:
            for line in f:
                r = json.loads(line)
                V, G, C, X, d = (r[k] for k in ['V', 'G', 'C', 'X2', 'gate_dot'])
                B = r['bounds']
                global_b = math.isqrt(V * X)
                assert B[0] == global_b
                cond = global_b
                if G:
                    assert G * V >= C * C and G * X >= d * d
                    s = math.isqrt((G * V - C * C) * (G * X - d * d))
                    lo = -(-(C * d - s) // G)
                    hi = (C * d + s) // G
                    assert lo <= hi
                    cond = min(global_b, max(abs(lo), abs(hi)))
                assert B[2] == cond and B[3] == min(B[:3]) and B[5] == min(B[3:5])
                assert all(abs(r['up_dot']) <= b <= global_b for b in B)
                up = f32(f32(r['up_dot']) * f32(val(r['sw_bits']) * val(r['sx_bits'])))
                assert bits(up) == r['up_bits']
                assert bits(f32(val(r['h_bits']) * up)) == r['a_bits']
                sample_checks += 1
        expected_samples = (1 + 26 * 7) * 8 * 8 if layers == 27 else 16 * 8 * 8 * 8
        assert sample_checks == expected_samples
        accumulated = {}
        dispatch_checks = 0
        with open(md / 'decode_dispatch.csv') as f:
            for row in csv.DictReader(f):
                key = tuple(row[k] for k in ['job', 'layer', 'class', 'bits', 'witness', 'bound'])
                meta = ['job', 'selection_index', 'step', 'layer', 'class', 'expert', 'bits', 'witness', 'bound', 'L_bits', 'amax_bits', 'scale_bits']
                v = {k: int(a) for k, a in row.items() if k not in meta}
                assert row['bound'] in names and int(row['selection_index']) == jobs[row['job']]
                assert 0 <= int(row['step']) < 8
                qmax = {'8': 127, '6': 31, '4': 7}[row['bits']]
                amax = val(int(row['amax_bits']))
                assert bits(1.0 if amax == 0 else f32(amax / qmax)) == int(row['scale_bits'])
                L = val(int(row['L_bits']))
                assert 0 <= L <= amax and math.isfinite(L)
                if row['witness'] == '0':
                    assert int(row['L_bits']) == int(row['amax_bits']) and v['seed_rows'] == 0
                else:
                    assert v['seed_rows'] == 8 and v['cand16'] <= v['total16'] - 1 and v['cand4'] <= v['total4'] - 4
                assert v['up_bytes'] == v['rows'] * 2048 and v['invariant_rows'] == v['rows']
                assert v['extra_rows'] <= v['candidate_rows'] <= v['true_zero_rows'] <= v['rows']
                assert v['candidate_rows'] - v['extra_rows'] <= v['h0_rows']
                for unit, nrows in [('4', 2), ('16', 8)]:
                    assert v['cand' + unit] == v['base' + unit] + v['added' + unit] - v['lost' + unit]
                    assert v['total' + unit] == v['rows'] // nrows
                acc = accumulated.setdefault(key, {k: 0 for k in v})
                for k, value in v.items():
                    acc[k] += value
                dispatch_checks += 1
        with open(md / 'aggregates.csv') as f:
            for row in csv.DictReader(f):
                assert row['phase'] == 'decode'
                key = tuple(row[k] for k in ['job', 'layer', 'class', 'bits', 'witness', 'bound'])
                acc = accumulated.pop(key)
                assert all(int(row[k]) == value for k, value in acc.items())
        assert not accumulated
        coverage = [json.loads(s) for s in (md / 'coverage.jsonl').read_text().splitlines()]
        counts = {}
        for r in coverage:
            assert r['job'] in jobs and 0 <= r['layer'] < layers
            tokens = len(reports[r['job']]['retained_token_ids']) if r['phase'] == 'prefill' else 1
            assert r['tokens'] == tokens
            key = (r['job'], r['phase'], r['layer'], r['class'], r['role'])
            counts[key] = counts.get(key, 0) + 1
        branches = 53 if layers == 27 else layers
        assert len(counts) == 8 * 2 * branches * 3
        assert all(v == (1 if k[1] == 'prefill' else 8) for k, v in counts.items())
        result['models'][name] = {'unique_common_inputs_checked': len(seen), 'fit_coordinates_independently_recomputed': layers * 16,
                                  'arithmetic_F32_audit_sample_rows': sample_checks, 'dispatch_policy_rows_reaggregated': dispatch_checks,
                                  'coverage_entries_checked': len(coverage), 'coverage_keys_checked': len(counts)}
    (ROOT / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
