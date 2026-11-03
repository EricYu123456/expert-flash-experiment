#!/usr/bin/env python3
"""Exhaustive static cover/capacity check without loading any routes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round2/qwen38-placement/diverse-mask-control'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    construction=json.loads((OUT/'construction.frozen.json').read_text());reports=[]
    for cert in construction['certificates']:
        path=OUT/f"L{cert['layer']}.seed{cert['seed']}.construction.npz"
        assert sha(path)==cert['npz_sha256']
        with np.load(path) as a:
            rows=a['rows'];gu=a['gu_high'];down=a['down_high'];gp=a['gu_plus'];dp=a['down_plus']
        assert np.all(~(gu&down)) and len(np.unique(gu,axis=0))==512
        assert np.all(gu.sum(axis=1)==3) and np.all(gu.sum(axis=0)==96)
        assert np.all(down.sum(axis=1)==10) and np.all(down.sum(axis=0)==320)
        assert np.all(gp<=~gu) and np.all(gp.sum(axis=1)==2) and np.all(gp.sum(axis=0)==64)
        assert np.all(dp<=down) and np.all(dp.sum(axis=1)==4) and np.all(dp.sum(axis=0)==128)
        assert np.array_equal(rows,np.stack([np.where(gu,101,75)+gp,np.where(down,198,96)+dp]))
        for phase,c in enumerate((2560,640)):
            prefix=np.cumsum(np.c_[np.zeros(512,dtype=int),rows[phase]],axis=1)
            assert np.all(np.diff(prefix,axis=1)>0) and np.all(prefix[:,-1]==(1280,2560)[phase])
            assert np.all(rows[phase].sum(axis=0)==(40960,81920)[phase])
            pages=(rows[phase].astype(np.int64)*c+16383)//16384
            waves=(rows[phase].astype(np.int64)*c+65535)//65536
            assert np.all(pages.sum(axis=1)==(204,104)[phase])
            assert np.all(waves.sum(axis=1)==(51,26)[phase])
            assert np.all(pages.sum(axis=0)==(6528,3328)[phase])
            assert np.all(waves.sum(axis=0)==(1632,832)[phase])
        reports.append(dict(layer=cert['layer'],seed=cert['seed'],complete_nonoverlapping_positive_fullrow_intervals=1024,
            distinct_gu_masks=len(np.unique(gu,axis=0)),distinct_down_masks=len(np.unique(down,axis=0)),
            distinct_joined_low_masks=len(np.unique(~(gu|down),axis=0)),
            each_expert_gu_high3_down_high10_disjoint=True,
            per_chip_gu_rowtypes=cert['inventory']['gate_up']['rowtype_counts_per_chip'],
            per_chip_down_rowtypes=cert['inventory']['down']['rowtype_counts_per_chip']))
    out=dict(status='PASS exhaustive static row-interval audit',audit_source_sha256=sha(__file__),
        construction_sha256=sha(OUT/'construction.frozen.json'),
        construction_generator_source_sha256=sha(Path(__file__).with_name('research_20261102_qwen38_diverse_mask_control.py')),
        matrices_checked=12288,all_complete_nonoverlapping_positive_fullrow_intervals=True,
        gu_perchip=dict(rows=40960,high101=96,low76=64,low75=352,useful_bytes=104857600,pages=6528,waves=1632),
        down_perchip=dict(rows=81920,high199=128,high198=192,low96=192,useful_bytes=52428800,pages=3328,waves=832),
        records=reports,
        scope='All12routeblind seed/layer constructions; prefix rowranges tile each fullmatrix exactly. No route file opened by this static auditor. Per-job blockaddresses and numeric hardware behavior remain outside this proof.')
    path=OUT/'static_coverage_audit.json';path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path),matrices_checked=12288),indent=2))


if __name__=='__main__':main()
