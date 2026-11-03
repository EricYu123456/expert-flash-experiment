#!/usr/bin/env python3
"""Fetch one official IBM GGUF with pinned metadata and LFS hash verification."""
import hashlib
import json
from pathlib import Path
import time

import requests

HERE = Path(__file__).resolve().parent
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment')
MODEL = DATA / 'models/granite4tiny-q4_k_m.gguf'
OFFICIAL = 'ibm-granite/granite-4.0-h-tiny'
GGUF_REPO = OFFICIAL + '-GGUF'
FILENAME = 'granite-4.0-h-tiny-Q4_K_M.gguf'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    evidence = HERE / 'granite4tiny-source'
    evidence.mkdir(parents=True, exist_ok=True)
    records = {}
    for repo in (OFFICIAL, GGUF_REPO):
        response = requests.get('https://huggingface.co/api/models/' + repo,
            params={'blobs': 'true'}, timeout=30)
        response.raise_for_status()
        info = response.json()
        revision = info['sha']
        (evidence / (repo.split('/')[-1] + '.api.json')).write_text(json.dumps(info, indent=2) + '\n')
        assert info['cardData']['license'] == 'apache-2.0'
        small = {}
        for entry in info['siblings']:
            name = entry['rfilename']
            if name not in ('config.json', 'README.md', 'LICENSE', 'LICENSE.txt'):
                continue
            url = 'https://huggingface.co/' + repo + '/resolve/' + revision + '/' + name
            body = requests.get(url, timeout=30)
            body.raise_for_status()
            path = evidence / (repo.split('/')[-1] + '.' + name)
            path.write_bytes(body.content)
            small[name] = dict(url=url, path=str(path), sha256=sha(path))
        records[repo] = dict(revision=revision, license='apache-2.0', small_files=small)
        if repo == GGUF_REPO:
            item = next(entry for entry in info['siblings'] if entry['rfilename'] == FILENAME)
            assert item['size'] == item['lfs']['size']
            records[repo]['file'] = item
    assert records[GGUF_REPO]['small_files']['README.md']
    expected = records[GGUF_REPO]['file']['lfs']['sha256']
    url = 'https://huggingface.co/' + GGUF_REPO + '/resolve/' + records[GGUF_REPO]['revision'] + '/' + FILENAME
    size = records[GGUF_REPO]['file']['size']
    if MODEL.exists():
        assert MODEL.stat().st_size == size and sha(MODEL) == expected
    else:
        partial = MODEL.with_suffix(MODEL.suffix + '.partial')
        assert not partial.exists(), 'An incomplete file exists; inspect before resuming.'
        response = requests.get(url, stream=True, timeout=(30, 120))
        response.raise_for_status()
        digest = hashlib.sha256()
        written = 0
        started = last = time.time()
        with partial.open('xb') as out:
            for block in response.iter_content(8 << 20):
                out.write(block)
                digest.update(block)
                written += len(block)
                if time.time() - last >= 15:
                    print(f'{written / 1e9:.3f}/{size / 1e9:.3f} GB, {written / (time.time() - started) / 1e6:.1f} MB/s', flush=True)
                    last = time.time()
        assert written == size and digest.hexdigest() == expected
        partial.replace(MODEL)
    result = dict(model='granite4tiny', source_records=records,
        model_path=str(MODEL), model_bytes=size, sha256=expected, download_url=url,
        status='official_IBM_GGUF_pinned_LFS_hash_verified',
        scope='A released fixed Q4_K_M checkpoint is used to collect real routing; no quantization method or precision change is proposed.',
        script_sha256=sha(Path(__file__)))
    (evidence / 'source-proof.json').write_text(json.dumps(result, indent=2) + '\n')
    (DATA / 'models/granite4tiny-source-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
