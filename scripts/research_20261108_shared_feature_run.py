#!/usr/bin/env python3
"""Run the immutable 600/200/1000 shared-feature capture on four CPU workers."""
import argparse
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
BINARY = DATA / 'build/shared-feature-capture'
CPU_GROUPS = [list(range(0, 8)), list(range(8, 16)), list(range(32, 40)), list(range(40, 48))]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run():
    out = DATA / 'captures'
    out.mkdir(exist_ok=True)
    if (out / 'started.json').exists():
        raise RuntimeError('An execution already exists; inspect and preserve it before another run')
    corpus = json.loads((DATA / 'corpus/manifest.json').read_text())
    protocol = ROOT / 'docs/research_2026_11_08_shared_features_protocol.md'
    pending = [(split, shard) for split in ('train', 'validation', 'test') for shard in range(4)]
    inputs = [protocol, Path(__file__), ROOT / 'scripts/research_20261108_shared_feature_capture.cpp',
              BINARY, DATA / 'corpus/manifest.json']
    inputs += [DATA / f'corpus/{s}.shard{b}.jobs.tsv' for s, b in pending]
    stat = MODEL.stat()
    freeze = dict(at_utc=now(), source_hashes={str(p): sha(p) for p in inputs},
                  model=str(MODEL), model_stat=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                  counts=corpus['counts'], max_tokens=16, threads_per_worker=8,
                  cpu_groups=CPU_GROUPS, queue=pending, gpu_layers=0,
                  CUDA_VISIBLE_DEVICES='', layers={'1': [14, 40], '13': [10, 56], '26': [0, 1]})
    save(out / 'protocol.frozen.json', freeze)
    save(out / 'started.json', dict(at_utc=now(), pid=os.getpid(), protocol_sha256=sha(out / 'protocol.frozen.json')))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', GGML_CUDA_DISABLE_GRAPHS='1')
    active, finished = {}, []
    start = time.monotonic()
    last_report = 0
    failed = False
    while pending or active:
        free = [i for i in range(4) if i not in active]
        while free and pending and not failed:
            slot = free.pop(0)
            split, shard = pending.pop(0)
            folder = out / split / f'shard{shard}'
            folder.mkdir(parents=True, exist_ok=False)
            jobs = DATA / f'corpus/{split}.shard{shard}.jobs.tsv'
            cmd = ['taskset', '-c', ','.join(map(str, CPU_GROUPS[slot])), str(BINARY), str(MODEL),
                   str(jobs), str(folder), '16', '8', 'w8', '1:14,40;13:10,56;26:0,1']
            save(folder / 'command.json', dict(argv=cmd, at_utc=now(), environment={k: env[k] for k in
                 ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}))
            stdout, stderr = (folder / 'stdout.log').open('w'), (folder / 'stderr.log').open('w')
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            active[slot] = dict(proc=proc, split=split, shard=shard, folder=folder,
                                stdout=stdout, stderr=stderr, start=time.monotonic())
            print(json.dumps(dict(event='launch', split=split, shard=shard, pid=proc.pid, slot=slot)), flush=True)
        for slot, worker in list(active.items()):
            code = worker['proc'].poll()
            if code is None:
                continue
            worker['stdout'].close()
            worker['stderr'].close()
            result = dict(split=worker['split'], shard=worker['shard'], exit_code=code,
                          wall_seconds=time.monotonic() - worker['start'], at_utc=now())
            if code == 0:
                manifest = json.loads((worker['folder'] / 'capture_manifest.json').read_text())
                expected = corpus['counts'][worker['split']] // 4
                assert manifest['status'] == 'complete' and manifest['jobs'] == expected
                assert manifest['gpu_layers'] == 0 and manifest['mode'] == 'w8'
                assert all(v['rows'] == expected * 16 for v in manifest['layers'].values())
                result['manifest_sha256'] = sha(worker['folder'] / 'capture_manifest.json')
            else:
                failed = True
                pending.clear()
            save(worker['folder'] / 'execution.json', result)
            finished.append(result)
            print(json.dumps(dict(event='complete', **result)), flush=True)
            del active[slot]
        if time.monotonic() - last_report >= 30:
            progress = []
            for slot, w in active.items():
                path = w['folder'] / 'stdout.log'
                text = path.read_text() if path.exists() else ''
                completed = sum(line.startswith('COMPLETE ') for line in text.splitlines())
                progress.append(dict(slot=slot, split=w['split'], shard=w['shard'], completed_prompts=completed))
            snapshot = dict(at_utc=now(), elapsed_seconds=time.monotonic() - start,
                            finished=finished, active=progress, pending=len(pending))
            save(out / 'progress.json', snapshot)
            print(json.dumps(dict(event='progress', **snapshot)), flush=True)
            last_report = time.monotonic()
        if active:
            time.sleep(2)
    result = dict(status='failed' if failed else 'complete', at_utc=now(),
                  wall_seconds=time.monotonic() - start, batches=finished)
    save(out / 'execution.json', result)
    if not failed:
        for path, expected in freeze['source_hashes'].items():
            assert sha(path) == expected, f'Changed source during capture: {path}'
        assert MODEL.stat().st_size == stat.st_size and MODEL.stat().st_mtime_ns == stat.st_mtime_ns
    print(json.dumps(result), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
