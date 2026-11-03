import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/home/ericyu/expert_flash_experiment')
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


audit = {}
for model in ['granite4tiny', 'deepseek']:
    result_path = OUT / model / 'result.json'
    d = json.loads(result_path.read_text())
    z = np.load(OUT / model / 'arrays.npz')
    rr = z['arbitrary_rows_mip__rows']
    warm = d['records']['arbitrary_rows_mip']['warm_family']
    D, I, E, K, L, SI = d['shape']
    assert np.array_equal(rr, z[warm + '__rows'])
    assert np.all(rr[:E, 0].sum(1) == 2 * I)
    assert np.all(rr[:E, 1].sum(1) == D)
    assert np.array_equal(z['routes'], np.load(d['training']['path'])['routes'][:, :, 0].reshape(-1, K))
    jobs = z['arbitrary_rows_mip__jobs']
    assert np.all(jobs[:, 3] == (jobs[:, 1].astype('i8') * jobs[:, 2] + 16383) // 16384)
    assert np.all(jobs[:, 4] == (jobs[:, 3] + 3) // 4)
    assert jobs[:, 4].max() <= 192
    chip_useful = [int(sum(int(r) * int(c) for chip, r, c, pages, waves in jobs if chip == j)) for j in range(16)]
    hist = {}
    for phase in range(2):
        before = z['whole_rows__rows'][:E, phase]
        after = rr[:E, phase]
        hist[str(phase)] = dict(
            row_multiset_perexpert_unchanged=all(sorted(a) == sorted(b) for a, b in zip(before, after)),
            before_unique_rowvectors=len(np.unique(before, axis=0)),
            after_unique_rowvectors=len(np.unique(after, axis=0)),
            assigned_histogram_changed=sorted(map(tuple, before)) != sorted(map(tuple, after)))
    audit[model] = dict(result_sha256=sha(result_path), arrays_sha256=sha(OUT/model/'arrays.npz'),
                        MIP_rows_identical_to_warm_incumbent=True, all512_train_routes_exact=True,
                        complete_onecopy_rows=True, all_oneblock=True, max_job_waves=int(jobs[:,4].max()),
                        perchip_useful_bytes=chip_useful, geometry=hist,
                        solve_status=d['records']['arbitrary_rows_mip']['fit']['status'],
                        case_wall_seconds=d['elapsed_seconds'])
(OUT/'independent_checks.json').write_text(json.dumps(audit,indent=2)+'\n')
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
source=ROOT/'scripts/research_20261103_arbitrary_rows_headroom.py'
shutil.copyfile(source,snap/source.name)
files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*'))
       if p.is_file() and p.name!='completion.json']
(OUT/'completion.json').write_text(json.dumps(dict(
    status='sealed_bounded_inconclusive_no_new_geometry_incumbent',
    utc=datetime.now(timezone.utc).isoformat(),source=str(source),source_sha256=sha(source),
    no_heldout_used=True,files=files),indent=2)+'\n')
print('completion_sha256',sha(OUT/'completion.json'))
