#!/usr/bin/env python3
"""Preserve partial mmap attempts and resume TRAIN with verified stream I/O."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from research_20261109_shared_feature_targets import DATA, ROOT, sha, dump


def main():
    out = DATA / 'targetgen-stream'
    out.mkdir()
    stream = ROOT / 'scripts/research_20261109_shared_feature_targets_stream.py'
    freeze = out / 'executed-freeze'
    freeze.mkdir()
    paths = [Path(__file__).resolve(), stream, ROOT / 'scripts/research_20261109_shared_feature_targets.py',
             ROOT / 'scripts/research_20261109_shared_feature_numeric.cpp',
             ROOT / 'scripts/research_20261109_shared_feature_transform.hpp', DATA / 'build/shared-feature-numeric.so']
    identities = {}
    for p in paths:
        identities[str(p)] = sha(p)
        shutil.copy2(p, freeze / p.name)
    old = json.loads((DATA / 'targets64/train/layer1/expert14.json').read_text())
    probe = json.loads((DATA / 'stream-probe/layer1/expert14.json').read_text())
    assert old['target_sha256'] == probe['target_sha256']
    assert probe['all_native_capture_fixed_target_floats_bitchecked'] == probe['rows'] * 2048
    ps = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    stopped = []
    for line in ps.splitlines():
        cols = line.strip().split(None, 1)
        if len(cols) != 2:
            continue
        pid, command = int(cols[0]), cols[1]
        if 'research_20261109_shared_feature_targets.py compute' in command and str(DATA / 'captures64/train') in command:
            stopped.append(dict(pid=pid, command=command))
    assert len(stopped) == 3, stopped
    dump(DATA / 'targetgen/superseded.json', dict(status='superseded_partial',
         at_utc=datetime.now(timezone.utc).isoformat(), reason='NFS output mmap page-fault I/O; same canonical arithmetic resumes using streaming writes',
         TRAIN_stopped=stopped, VALID_policy='Original VALID jobs continue to completion',
         probe_same_target_sha256=probe['target_sha256'], old_probe_seconds=old['seconds'], stream_probe_seconds=probe['seconds']))
    for p in stopped:
        os.kill(p['pid'], signal.SIGTERM)
    for _ in range(120):
        alive = []
        for p in stopped:
            try:
                os.kill(p['pid'], 0)
                alive.append(p['pid'])
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(.5)
    assert not alive, 'Do not move files while an old writer remains alive'
    existing, preserved = {}, []
    for layer in (1, 13, 26):
        folder = DATA / f'targets64/train/layer{layer}'
        archive = DATA / f'targetgen/attempt001-partial/train.layer{layer}'
        archive.mkdir(parents=True)
        existing[str(layer)] = {}
        for e in range(64):
            receipt = folder / f'expert{e}.json'
            if receipt.exists():
                r = json.loads(receipt.read_text())
                existing[str(layer)][str(e)] = dict(target_sha256=r['target_sha256'], receipt_sha256=sha(receipt))
                continue
            for name in (f'expert{e}.f32', f'expert{e}.f32.partial', f'expert{e}.json.partial',
                         f'prototype{e}.hidden.f32', f'prototype{e}.hidden.f32.partial'):
                p = folder / name
                if p.exists():
                    item = dict(source=str(p), archived=str(archive / name), bytes=p.stat().st_size)
                    p.rename(archive / name)
                    preserved.append(item)
    dump(out / 'protocol.frozen.json', dict(status='executing', inputs_sha256=identities,
         original_complete_TRAIN_receipts=existing, preserved_incomplete_attempt001_files=preserved,
         same_canonical_numeric_binary_sha256=sha(DATA / 'build/shared-feature-numeric.so'),
         CPU_groups=[[*range(a, a + 8)] for a in (0, 8, 16)], prototype_rule='Unchanged TRAIN-only frequency top8'))
    active = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
    for layer, first_cpu in zip((1, 13, 26), (0, 8, 16)):
        folder = out / f'train.layer{layer}'
        folder.mkdir()
        cmd = ['taskset', '-c', f'{first_cpu}-{first_cpu + 7}', 'python3', str(stream),
               '--captures', str(DATA / 'captures64/train'), '--layer', str(layer),
               '--weights', str(DATA / f'weights/layer{layer}'), '--output', str(DATA / f'targets64/train/layer{layer}'),
               '--threads', '8', '--batch', '2048', '--save-hidden-experts', 'train-top8',
               '--prototype-train-captures', str(DATA / 'captures64/train')]
        dump(folder / 'command.json', dict(argv=cmd, CUDA_VISIBLE_DEVICES=''))
        stdout, stderr = (folder / 'stdout.log').open('w'), (folder / 'stderr.log').open('w')
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
        active[layer] = dict(proc=proc, stdout=stdout, stderr=stderr, folder=folder)
        print(json.dumps(dict(event='stream_launch', layer=layer, pid=proc.pid)), flush=True)
    completed, last = [], 0
    while active:
        for layer, row in list(active.items()):
            code = row['proc'].poll()
            if code is None:
                continue
            row['stdout'].close()
            row['stderr'].close()
            dump(row['folder'] / 'execution.json', dict(layer=layer, exit_code=code))
            assert code == 0, 'Streaming target generation failed'
            completed.append(layer)
            del active[layer]
        if time.monotonic() - last > 30:
            print(json.dumps(dict(event='stream_progress', completed_layers=completed,
                  experts={str(l): len(list((DATA / f'targets64/train/layer{l}').glob('expert*.json'))) for l in (1, 13, 26)})), flush=True)
            last = time.monotonic()
        if active:
            time.sleep(2)
    for p, identity in identities.items():
        assert sha(p) == identity, 'Frozen stream revision changed: ' + p
    combined = dict(status='complete', original_mmap_attempt_status='superseded_partial',
                    numeric_binary_sha256=sha(DATA / 'build/shared-feature-numeric.so'), expert_provenance={}, manifests={})
    for split in ('train', 'validation'):
        for layer in (1, 13, 26):
            path = DATA / f'targets64/{split}/layer{layer}/targets.json'
            while not path.exists():
                time.sleep(2)
            m = json.loads(path.read_text())
            assert m['status'] == 'complete' and m['expert_ids'] == list(range(64)) and len(m['prototype_hidden']) == 8
            combined['manifests'][f'{split}/layer{layer}'] = dict(path=str(path), sha256=sha(path), rows=m['rows'])
            fixed = {1: [14, 40], 13: [10, 56], 26: [0, 1]}[layer]
            for e, r in m['targets'].items():
                old_revision = split == 'validation' or e in existing[str(layer)]
                if split == 'train' and old_revision:
                    assert r['target_sha256'] == existing[str(layer)][e]['target_sha256']
                assert r['numeric_binary_sha256'] == combined['numeric_binary_sha256']
                if int(e) in fixed:
                    assert r['all_native_capture_fixed_target_floats_bitchecked'] == m['rows'] * 2048
                combined['expert_provenance'][f'{split}/layer{layer}/expert{e}'] = dict(
                    revision='mmap-attempt001' if old_revision else 'stream-attempt002', target_sha256=r['target_sha256'])
    dump(out / 'execution.json', combined)
    print(json.dumps(dict(event='combined_complete', manifests=combined['manifests'])), flush=True)


if __name__ == '__main__':
    main()
