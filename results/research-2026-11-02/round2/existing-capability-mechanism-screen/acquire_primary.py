#!/usr/bin/env python3
"""Bounded, read-only primary-source acquisition; no models or device access."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

targets = {
    'slim.v1.html': 'https://arxiv.org/html/2507.09201v1',
    'slim.v1.pdf': 'https://arxiv.org/pdf/2507.09201v1',
    'nasic.v1.html': 'https://arxiv.org/html/2605.23294v1',
    'gao.publications.html': 'https://people.iiis.tsinghua.edu.cn/~gaomy/publications.html',
    'gao.publications.bib': 'https://people.iiis.tsinghua.edu.cn/~gaomy/files/my_publications/publications.bib',
}
protocol = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Primary evidence and small arithmetic only. No source/plans edits, inference, timing run, device operation, or new hardware.',
    'producer_sha256': sha(__file__),
    'targets': targets,
    'request_timeout_seconds': 20,
    'parallel_workers': 4,
    'maximum_source_bytes': 15000000,
}
(HERE/'protocol.frozen.json').write_text(json.dumps(protocol, indent=2)+'\n')

def fetch(item):
    name, url = item
    ans = {'url': url, 'checked_utc': datetime.now(timezone.utc).isoformat()}
    try:
        r = requests.get(url, timeout=20)
        ans.update(status=r.status_code, final_url=r.url, content_type=r.headers.get('Content-Type'), bytes=len(r.content))
        if r.status_code == 200 and len(r.content) < protocol['maximum_source_bytes']:
            p=HERE/'primary'/name
            p.write_bytes(r.content)
            ans.update(path=str(p.relative_to(HERE)), sha256=sha(p))
    except requests.RequestException as e:
        ans['error'] = str(e)
    return name, ans

with ThreadPoolExecutor(max_workers=4) as pool:
    out = dict(pool.map(fetch, targets.items()))
(HERE/'acquisition.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k: {j:v for j,v in x.items() if j in ['status','bytes','error','sha256']} for k,x in out.items()}, indent=2))
