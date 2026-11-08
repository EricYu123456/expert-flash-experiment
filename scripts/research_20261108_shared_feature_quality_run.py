#!/usr/bin/env python3
"""Freeze and run live W8A8 shared-readout quality arms, using CPU only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/ericyu/expert_flash_experiment')
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')
MODEL = Path('/home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-v2-lite-chat-q8_0.gguf')
CPU_GROUPS = [list(range(a, a + 8)) for a in (0, 8, 16, 32, 40, 48)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run():
    out = DATA / 'quality'
    out.mkdir(exist_ok=True)
    if (out / 'protocol.frozen.json').exists():
        raise RuntimeError('Quality execution already frozen; preserve it')
    registries = {'native': '-'}
    identities = {}
    corpus = json.loads((DATA / 'corpus/manifest.json').read_text())
    for row in corpus['records']:
        path = Path(row['path'])
        actual = sha(path)
        assert actual == row['prompt_sha256'], f'Changed corpus text: {path}'
        identities[str(path)] = actual
    identities[str(MODEL)] = sha(MODEL)
    for basis, arm in [('shared_z', 'shared_z_all6'), ('raw_x', 'raw_x_all6')]:
        lines = []
        for layer in (1, 13, 26):
            fit = DATA / f'fits-refined/layer{layer}'
            selection = fit / 'selection.json'
            identities[str(selection)] = sha(selection)
            for line in (fit / 'registry.tsv').read_text().splitlines():
                cols = line.split('\t')
                if cols[2] != basis:
                    continue
                assert int(cols[0]) == layer
                lines.append(line)
                for suffix in ('.i8', '.scales.f32', '.bias.f32'):
                    path = Path(cols[4] + suffix)
                    identities[str(path)] = sha(path)
        assert len(lines) == 6
        registry = out / f'{arm}.registry.tsv'
        registry.write_text('\n'.join(lines) + '\n')
        registries[arm] = str(registry)
        identities[str(registry)] = sha(registry)
    single = out / 'shared_z_layer1.registry.tsv'
    single.write_text('\n'.join(line for line in Path(registries['shared_z_all6']).read_text().splitlines()
                                if line.split('\t')[0] == '1') + '\n')
    registries['shared_z_layer1_valid'] = str(single)
    identities[str(single)] = sha(single)
    binary = DATA / 'build/shared-feature-quality'
    for path in (binary, Path(__file__), ROOT / 'scripts/research_20261108_shared_feature_quality.cpp',
                 ROOT / 'scripts/research_20261108_shared_feature_capture.cpp', DATA / 'corpus/manifest.json'):
        identities[str(path)] = sha(path)
    # Single-layer VALID diagnostic is fixed before reading any of its outputs.
    # Both TEST approximations replace the same six predeclared experts.
    pending = [('shared_z_layer1_valid', 'validation', s, 0) for s in range(4)]
    pending += [(arm, 'test', s, 8) for arm in ('native', 'shared_z_all6', 'raw_x_all6') for s in range(4)]
    for _, split, shard, _ in pending:
        path = DATA / f'corpus/{split}.shard{shard}.jobs.tsv'
        identities[str(path)] = sha(path)
    protocol = dict(at_utc=now(), inputs_sha256=identities, pending=pending, registries=registries,
                    gpu_layers=0, CUDA_VISIBLE_DEVICES='', max_tokens=16, threads_per_worker=8,
                    cpu_groups=CPU_GROUPS, test_prompts_per_arm=1000, valid_single_layer_prompts=200,
                    quality_margin_relative_subset_ppl=.02,
                    quality_scope='15 actual next-token targets per prompt; fresh KV, live downstream routing; 8 greedy continuation tokens with EOG stop',
                    native_controls='Quality runner native prefill checked against capture runner; actual live shared_z memcmp in every replaced layer/call',
                    exposure_scope='Six fixed experts across three of 26 MoE layers; not an all-expert compression result',
                    CPU_wall_time_scope='Functional harness recomputes original FFN and shared GU; not compressed runtime or AiF speedup')
    save(out / 'protocol.frozen.json', protocol)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    active, finished = {}, []
    start, last_report, failed = time.monotonic(), 0, False
    while pending or active:
        free = [i for i in range(len(CPU_GROUPS)) if i not in active]
        while free and pending and not failed:
            slot = free.pop(0)
            arm, split, shard, steps = pending.pop(0)
            folder = out / arm / f'shard{shard}'
            folder.mkdir(parents=True, exist_ok=False)
            cmd = ['taskset', '-c', ','.join(map(str, CPU_GROUPS[slot])), str(binary), str(MODEL),
                   str(DATA / f'corpus/{split}.shard{shard}.jobs.tsv'), str(folder / 'data'),
                   '16', '8', str(steps), registries[arm], 'w8']
            save(folder / 'command.json', dict(argv=cmd, at_utc=now(), environment={k: env[k] for k in
                 ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}))
            stdout, stderr = (folder / 'stdout.log').open('w'), (folder / 'stderr.log').open('w')
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            active[slot] = dict(proc=proc, arm=arm, split=split, shard=shard, folder=folder,
                                stdout=stdout, stderr=stderr, start=time.monotonic())
            print(json.dumps(dict(event='launch', arm=arm, shard=shard, pid=proc.pid, slot=slot)), flush=True)
        for slot, w in list(active.items()):
            code = w['proc'].poll()
            if code is None:
                continue
            w['stdout'].close()
            w['stderr'].close()
            result = dict(arm=w['arm'], shard=w['shard'], exit_code=code,
                          wall_seconds=time.monotonic() - w['start'], at_utc=now())
            if code == 0:
                completion = json.loads((w['folder'] / 'data/completion.json').read_text())
                assert completion['status'] == 'complete' and completion['gpu_layers'] == 0
                assert completion['jobs'] == (50 if w['split'] == 'validation' else 250)
                result['completion'] = completion
            else:
                failed = True
                pending.clear()
            save(w['folder'] / 'execution.json', result)
            finished.append(result)
            del active[slot]
            print(json.dumps(dict(event='complete', **result)), flush=True)
        if time.monotonic() - last_report >= 30:
            current = []
            for slot, w in active.items():
                path = w['folder'] / 'stdout.log'
                completed = sum(line.startswith('QUALITY_COMPLETE ') for line in path.read_text().splitlines())
                current.append(dict(slot=slot, arm=w['arm'], shard=w['shard'], completed_prompts=completed))
            snapshot = dict(at_utc=now(), elapsed_seconds=time.monotonic() - start,
                            finished=finished, active=current, pending=len(pending))
            save(out / 'progress.json', snapshot)
            print(json.dumps(dict(event='progress', **snapshot)), flush=True)
            last_report = time.monotonic()
        if active:
            time.sleep(2)
    result = dict(status='failed' if failed else 'complete', at_utc=now(),
                  wall_seconds=time.monotonic() - start, batches=finished)
    save(out / 'execution.json', result)
    if not failed:
        for path, expected in identities.items():
            assert sha(path) == expected, f'Changed quality input: {path}'
    print(json.dumps(result), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
