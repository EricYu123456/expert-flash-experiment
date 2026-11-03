#!/usr/bin/env python3
"""Freeze/build, then (after human/source review) run the bounded Stage0."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

REPO = Path('/home/ericyu/expert_flash_experiment')
OUT = REPO / 'results/research-2026-11-03-no-cr/round01-stage0'
PROTOCOL = OUT.parent / 'gate_zero_stage0_protocol.frozen.json'
EXPECTED = '39f9fed28de41de740bbd659a00360c7294bd61ee81c9832028b76598ce5e65d'
CPP = REPO / 'scripts/research_20261103_gate_zero_stage0.cpp'
SELF = Path(__file__).resolve()
UPSTREAM = Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c')
BUILD = Path('/home/nfs_client/ericyu/expert_flash_experiment/build/llama-route-trace/llama')
LIBS = [BUILD / p for p in ['src/libllama.a', 'ggml/src/libggml.a', 'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a']]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def protocol():
    assert sha(PROTOCOL) == EXPECTED, 'protocol identity changed'
    return json.loads(PROTOCOL.read_text())


def stat_identity(p):
    s = Path(p).stat()
    return {'path': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
            'device': s.st_dev, 'inode': s.st_ino}


def prepare():
    pr = protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    freezes = OUT / 'freezes'
    freezes.mkdir(exist_ok=True)
    rev = 1 + max([int(p.name.removeprefix('rev')) for p in freezes.glob('rev*')] or [0])
    rd = freezes / f'rev{rev:03d}'
    rd.mkdir()
    binary = rd / 'collector'
    cmd = ['c++', '-O3', '-march=native', '-fno-fast-math', '-ffp-contract=off',
           '-std=c++17', '-fopenmp', '-pthread', '-I' + str(UPSTREAM / 'include'),
           '-I' + str(UPSTREAM / 'ggml/include'), str(CPP), '-o', str(binary),
           str(LIBS[0]), str(LIBS[1]), '-ldl', str(LIBS[2]), str(LIBS[3]), '-lm']
    save(rd / 'build_command.json', cmd)
    built = subprocess.run(cmd, capture_output=True, text=True)
    (rd / 'build.stdout').write_text(built.stdout)
    (rd / 'build.stderr').write_text(built.stderr)
    if built.returncode:
        raise RuntimeError(f'build tooling failure; retained {rd}: {built.stderr[-3000:]}')
    frozen_files = [CPP, SELF, PROTOCOL, *LIBS,
                    UPSTREAM / 'ggml/src/ggml-cpu/vec.cpp',
                    UPSTREAM / 'ggml/src/ggml-cpu/vec.h',
                    UPSTREAM / 'src/llama-graph.cpp',
                    UPSTREAM / 'src/models/deepseek2.cpp',
                    UPSTREAM / 'src/models/olmoe.cpp', binary, rd / 'build_command.json']
    receipt = {'status': 'source_build_frozen_before_candidate_data_read',
               'frozen_at_utc': now(), 'revision': rev,
               'protocol_sha256': EXPECTED, 'build_command': cmd,
               'compiler': subprocess.check_output(['c++', '--version'], text=True),
               'cpu_isa': subprocess.check_output(['lscpu'], text=True),
               'uname': platform.uname()._asdict(),
               'file_sha256': {str(p): sha(p) for p in frozen_files},
               'model_stat_identities': [stat_identity(m['gguf']) for m in pr['models']],
               'model_contents_not_read': True,
               'review_required_before_run': True}
    for p in [CPP, SELF, PROTOCOL]:
        shutil.copy2(p, rd / p.name)
    save(rd / 'freeze_receipt.json', receipt)
    # The immutable source/build receipt precedes every synthetic execution.
    test = subprocess.run([str(binary), '--self-test'], capture_output=True, text=True)
    (rd / 'synthetic.stdout').write_text(test.stdout)
    (rd / 'synthetic.stderr').write_text(test.stderr)
    save(rd / 'synthetic_receipt.json', {'at_utc': now(), 'returncode': test.returncode,
                                      'binary_sha256': sha(binary),
                                      'stdout_sha256': sha(rd / 'synthetic.stdout')})
    if test.returncode:
        raise RuntimeError(f'synthetic tooling failure; retained {rd}: {test.stderr}')
    save(OUT / 'latest_ready.json', {'revision': rev, 'receipt': str(rd / 'freeze_receipt.json'),
                                  'binary': str(binary), 'synthetic_passed': True})
    print(json.dumps({'ready_for_source_review': str(rd), 'source_sha256': sha(CPP),
                      'runner_sha256': sha(SELF), 'binary_sha256': sha(binary),
                      'synthetic_stdout': test.stdout}, indent=2), flush=True)


def run(review):
    assert review, 'source-review approval reference must be supplied'
    pr = protocol()
    latest = json.loads((OUT / 'latest_ready.json').read_text())
    receipt = json.loads(Path(latest['receipt']).read_text())
    for p, h in receipt['file_sha256'].items():
        assert sha(p) == h, f'source/build changed after freeze: {p}'
    assert not (OUT / 'run_receipt.json').exists(), 'preserve prior run; use a fresh explicitly named attempt'
    jobs = OUT / 'jobs.frozen.tsv'
    jobs.write_text(''.join(j['path'] + '\t' + j['id'] + '\n' for j in pr['selection']['jobs']))
    identities = []
    for index, m in enumerate(pr['models']):
        p = Path(m['gguf'])
        print(f'{now()} stream SHA256 {m["name"]}', flush=True)
        before = stat_identity(p)
        assert before == receipt['model_stat_identities'][index], 'model identity changed since source freeze'
        identity = {**before, 'sha256': sha(p), 'hashed_at_utc': now()}
        assert stat_identity(p) == before, 'model mutated during hash'
        identities.append(identity)
    prompt_ids = [{**stat_identity(j['path']), 'sha256': sha(j['path']), 'id': j['id']}
                  for j in pr['selection']['jobs']]
    record = {'status': 'running', 'started_at_utc': now(), 'review_approval': review,
              'source_revision': latest['revision'], 'model_identities': identities,
              'prompt_identities': prompt_ids, 'jobs_tsv_sha256': sha(jobs),
              'threads_per_process': 8, 'max_concurrent_models': 2, 'processes': []}
    save(OUT / 'run_receipt.json', record)
    env = dict(os.environ, OMP_NUM_THREADS='8', OMP_THREAD_LIMIT='8',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    children = []
    for m in pr['models']:
        md = OUT / m['name']
        md.mkdir()
        cmd = [latest['binary'], m['name'], m['gguf'], str(jobs), str(md)]
        log = open(md / 'collector.log', 'w')
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        children.append((m, child, log))
        record['processes'].append({'model': m['name'], 'pid': child.pid, 'argv': cmd})
    save(OUT / 'run_receipt.json', record)
    while any(c.poll() is None for _, c, _ in children):
        time.sleep(10)
    for m, c, log in children:
        log.close()
        record['processes'][pr['models'].index(m)]['returncode'] = c.returncode
    record['finished_at_utc'] = now()
    record['status'] = 'complete' if all(c.returncode == 0 for _, c, _ in children) else 'invalid_tooling_or_numerical_run'
    save(OUT / 'run_receipt.json', record)
    assert record['status'] == 'complete', 'run invalid; retain artifacts and repair in same round'
    summarize()


def summarize():
    pr = protocol()
    outputs = {}
    for model in pr['models']:
        md = OUT / model['name']
        report = json.loads((md / 'report.json').read_text())
        assert report['status'] == 'complete' and report['product_bit_mismatches'] == 0
        groups = {}
        with open(md / 'aggregates.csv') as f:
            for row in csv.DictReader(f):
                index = next(j['selection_index'] for j in pr['selection']['jobs'] if j['id'] == row['job'])
                split = 'fit_first4' if index < 4 else 'last4'
                for label, klass in [('all8', 'pooled'), (split, 'pooled'), ('all8', row['class']), (split, row['class'])]:
                    key = (row['phase'], label, klass, int(row['bits']), int(row['witness']), row['bound'])
                    group = groups.setdefault(key, {'totals': {}, 'prompts': {}})
                    vals = {k: int(v) for k, v in row.items() if k not in ['job', 'phase', 'layer', 'class', 'bits', 'witness', 'bound']}
                    for target in [group['totals'], group['prompts'].setdefault(row['job'], {})]:
                        for k, v in vals.items():
                            target[k] = target.get(k, 0) + v
        tables = []
        for key, group in sorted(groups.items()):
            phase, split, klass, qbits, witness, bound = key
            t = group['totals']
            # Both frozen models have H=2048 in all Up branches. Byte pooling
            # still uses each dispatch's physical I*H denominator.
            extra_bytes = t['extra_rows'] * 2048
            net_bytes = (t['candidate_rows'] - t['h0_rows']) * 2048
            strong = 3 * t['rows'] - t['true_zero_rows'] - t['h0_rows']
            row = {'phase': phase, 'split': split, 'class': klass, 'bits': qbits,
                   'witness': ['true_amax_oracle', 'fixed_first_up_page'][witness],
                   'bound': bound, **t, 'extra_up_weight_bytes': extra_bytes,
                   'net_up_weight_bytes_vs_h0': net_bytes,
                   'gross_extra_up_byte_fraction': extra_bytes / t['up_bytes'],
                   'net_up_byte_fraction_vs_h0': net_bytes / t['up_bytes'],
                   'prompt_mean_gross_extra_up_byte_fraction': sum(v['extra_rows'] * 2048 / v['up_bytes'] for v in group['prompts'].values()) / len(group['prompts']),
                   'gate_zero_fraction': t['h0_rows'] / t['rows'],
                   'exact_post_A_down_zero_fraction': t['true_zero_rows'] / t['rows'],
                   'ideal_incremental_coefficient_work_fraction_vs_gate0_sparseDown': (t['candidate_rows'] - t['h0_rows']) / strong,
                   'ideal_dense3_coefficient_work_fraction': (t['candidate_rows'] - t['h0_rows']) / (3 * t['rows'])}
            tables.append(row)
        primary = next(t for t in tables if (t['phase'], t['split'], t['class'], t['bits'], t['witness'], t['bound']) == ('decode', 'all8', 'pooled', 8, 'true_amax_oracle', 'primary_rowmin'))
        secondary = next(t for t in tables if (t['phase'], t['split'], t['class'], t['bits'], t['witness'], t['bound']) == ('decode', 'last4', 'pooled', 8, 'true_amax_oracle', 'all_rowmin'))
        with open(md / 'prefill_diagnostics.csv') as f:
            prefill = list(csv.DictReader(f))
        outputs[model['name']] = {'report': report, 'primary_decode': primary,
                                 'secondary_last4': secondary, 'tables': tables,
                                 'prefill_diagnostics': prefill,
                                 'metadata_scope': 'Report metadata bytes count the union of decode-touched Up expert matrices (cross cache nonempty), not full-model deployment capacity. Cached W8 bytes and weight-cache ledger include all prefill-touched projections. Per-cache norm arrays for Gate/Down are implementation overhead, not required candidate metadata. Deflation bytes are a logical selected16-I8 plus U64 residual representation estimate; current host evaluator reads selected weights from full Up cache and recomputes residual exactly. Common per-layer input energy is fitted once per unique decode input; conditional raw gate dot is an optimistic extra ABI capability.',
                                 'prespecified_negative_numeric_conditions': all(t['gross_extra_up_byte_fraction'] <= .01 and t['added16'] == 0 for t in [primary, secondary])}
        save(md / 'summary.json', outputs[model['name']])
    result = {'status': 'complete_pending_root_critic_scientific_review', 'summarized_at_utc': now(),
              'protocol_sha256': EXPECTED, 'models': outputs,
              'both_models_prespecified_negative_numeric_conditions': all(m['prespecified_negative_numeric_conditions'] for m in outputs.values()),
              'scope': 'TRAIN8 cap128 decode8 canonical CPU W8A8 logical opportunity only; no native AiF speed or heldout quality claim'}
    save(OUT / 'summary.json', result)
    lines = ['# Round1 Stage0 completed numerical results', '', result['scope'], '',
             '| Model | Primary extra Up bytes | Primary added16KiB | Last4 + deflation extra Up bytes | Last4 added16KiB |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, m in outputs.items():
        a, b = m['primary_decode'], m['secondary_last4']
        lines.append(f'| {name} | {100*a["gross_extra_up_byte_fraction"]:.8f}% | {a["added16"]} | {100*b["gross_extra_up_byte_fraction"]:.8f}% | {b["added16"]} |')
    lines += ['', 'The table uses free true-amax witnesses and pooled physical Up weight bytes. Primary is per-row minimum of global/block128/conditional bounds; last4 also includes fit16 deflation. A6/A4, paid seed, class and prompt-mean tables remain in summary.json; prefill diagnostics are in prefill_diagnostics.csv and summary.json. Prefill certificates were not collected; all prefill FFNs were canonicalized and checked for finite outputs and exact full-I product bits. Root and critic make the scientific round decision.', '',
              'All certificates are checked against each quantizer’s own q bits and unchanged amax/scale bits. Coverage and finite checks must complete. 16KiB/4KiB pages are independently aligned original-order logical opportunity diagnostics; 4KiB is hypothetical. Equal-projection coefficient work is not device latency.', '']
    lines += ['Metadata in report.json covers the union of decode-touched Up matrices, not full-model deployment capacity. Deflation storage is a logical selected16-I8 plus U64 residual estimate; host code reads the selected coordinates from full cached Up weights. Weight-cache storage includes Gate/Up/Down touched during prefill. See metadata_scope in summary.json for the cost boundary.', '']
    (OUT / 'result.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--source-review', default='')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'run':
        run(args.source_review)
    else:
        summarize()
