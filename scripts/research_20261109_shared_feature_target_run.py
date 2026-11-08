#!/usr/bin/env python3
"""Run the six TRAIN/VALID layer target jobs on explicitly assigned CPU groups."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/ericyu/expert_flash_experiment')
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')
SCRIPT = ROOT / 'scripts/research_20261109_shared_feature_targets.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    temp = Path(str(path) + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def ready(split):
    for s in range(4):
        folder = DATA / f'captures64/{split}/shard{s}'
        p = folder / 'capture_manifest.json'
        if not p.exists():
            return False
        try:
            m = json.loads(p.read_text())
            for l in ('1', '13', '26'):
                for key in ('x', 'expert_outputs', 'selected_experts'):
                    rec = m['layers'][l]['arrays'][key]
                    count = 4
                    for v in rec['shape']:
                        count *= v
                    if (folder / rec['path']).stat().st_size != count:
                        return False
        except (ValueError, FileNotFoundError):
            return False
    return True


def main():
    out = DATA / 'targetgen'
    out.mkdir(exist_ok=True)
    assert not (out / 'protocol.frozen.json').exists(), 'Preserve existing target generation'
    group_file = out / 'cpu_groups.json'
    if not group_file.exists():
        save(group_file, [[*range(24, 32)], [*range(56, 64)]])
    freeze = out / 'executed-freeze'
    freeze.mkdir()
    sources = [Path(__file__).resolve(), SCRIPT, ROOT / 'scripts/research_20261109_shared_feature_targets.cpp',
               ROOT / 'scripts/research_20261109_shared_feature_numeric.cpp',
               ROOT / 'scripts/research_20261109_shared_feature_transform.hpp',
               ROOT / 'scripts/research_20261108_shared_feature_capture.cpp',
               DATA / 'build/shared-feature-numeric.so', DATA / 'build/target-export',
               DATA / 'build/target_build.json', DATA / 'build/pinned_swiglu.cpp']
    identities = {}
    for p in sources:
        identities[str(p)] = sha(p)
        shutil.copy2(p, freeze / p.name)
    queue = [(split, l) for split in ('train', 'validation') for l in (1, 13, 26)]
    save(out / 'protocol.frozen.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
         inputs_sha256=identities, jobs=queue, threads_per_job=8, CUDA_VISIBLE_DEVICES='',
         prototype_rule='Each layer TRAIN native routing frequency top8, ID ascending tie-break; same TRAIN selection for VALID.',
         exact_contract='Canonical row W8/vector A8; integer-exact FP64 CPU BLAS, C++ F32 rescale and byte-exact pinned SwiGLU.'))
    active, completed = {}, []
    started = time.monotonic()
    last = 0
    failed = False
    while queue or active:
        groups = [tuple(x) for x in json.loads(group_file.read_text())]
        assert all(len(g) == 8 for g in groups) and len(set(sum((list(g) for g in groups), []))) == len(groups) * 8
        free = [g for g in groups if g not in active]
        while free and not failed:
            choices = [j for j in queue if ready(j[0])]
            if not choices:
                break
            job = choices[0]
            queue.remove(job)
            split, layer = job
            cpus = free.pop(0)
            task = out / f'{split}.layer{layer}'
            task.mkdir()
            weights = DATA / f'weights/layer{layer}'
            destination = DATA / f'targets64/{split}/layer{layer}'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE',
                       OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
            export_cmd = ['taskset', '-c', ','.join(map(str, cpus)), 'python3', str(SCRIPT), 'export',
                          '--layer', str(layer), '--experts', 'all', '--output', str(weights), '--threads', '8']
            # Exports occur before target jobs on that layer, at most 64 experts.
            subprocess.run(export_cmd, check=True, env=env, stdout=(task / 'export.stdout.log').open('w'), stderr=(task / 'export.stderr.log').open('w'))
            cmd = ['taskset', '-c', ','.join(map(str, cpus)), 'python3', str(SCRIPT), 'compute',
                   '--captures', str(DATA / f'captures64/{split}'), '--layer', str(layer), '--experts', 'all',
                   '--weights', str(weights), '--output', str(destination), '--threads', '8', '--batch', '2048',
                   '--save-hidden-experts', 'train-top8', '--prototype-train-captures', str(DATA / 'captures64/train')]
            save(task / 'command.json', dict(export_argv=export_cmd, argv=cmd, cpu_group=cpus, CUDA_VISIBLE_DEVICES=''))
            stdout, stderr = (task / 'stdout.log').open('w'), (task / 'stderr.log').open('w')
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
            active[cpus] = dict(proc=proc, split=split, layer=layer, stdout=stdout, stderr=stderr,
                                folder=task, started=time.monotonic(), destination=destination)
            print(json.dumps(dict(event='launch', split=split, layer=layer, cpus=cpus, pid=proc.pid)), flush=True)
        for cpus, info in list(active.items()):
            code = info['proc'].poll()
            if code is None:
                continue
            info['stdout'].close()
            info['stderr'].close()
            item = dict(split=info['split'], layer=info['layer'], exit_code=code,
                        seconds=time.monotonic() - info['started'])
            if code == 0:
                manifest = json.loads((info['destination'] / 'targets.json').read_text())
                assert manifest['status'] == 'complete' and manifest['expert_ids'] == list(range(64))
                assert len(manifest['prototype_hidden']) == 8
                fixed = {1: [14, 40], 13: [10, 56], 26: [0, 1]}[info['layer']]
                for e in fixed:
                    assert manifest['targets'][str(e)]['all_native_capture_fixed_target_floats_bitchecked'] == manifest['rows'] * 2048
                item.update(rows=manifest['rows'], prototype_experts=list(manifest['prototype_hidden']),
                            target_manifest=str(info['destination'] / 'targets.json'))
            else:
                failed = True
                queue.clear()
            save(info['folder'] / 'execution.json', item)
            completed.append(item)
            del active[cpus]
            print(json.dumps(dict(event='complete', **item)), flush=True)
        if time.monotonic() - last >= 30:
            snapshot = dict(elapsed_seconds=time.monotonic() - started, completed=completed,
                            active=[dict(split=v['split'], layer=v['layer'], experts_done=len(list(v['destination'].glob('expert*.json'))), cpus=k) for k, v in active.items()], pending=queue)
            save(out / 'progress.json', snapshot)
            print(json.dumps(dict(event='progress', **snapshot)), flush=True)
            last = time.monotonic()
        if queue or active:
            time.sleep(2)
    for p, identity in identities.items():
        assert sha(p) == identity, 'Frozen target code changed: ' + p
    result = dict(status='failed' if failed else 'complete', seconds=time.monotonic() - started, jobs=completed)
    save(out / 'execution.json', result)
    print(json.dumps(result), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
