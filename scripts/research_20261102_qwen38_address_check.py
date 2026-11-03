#!/usr/bin/env python3
"""Separate address/cell-coverage checker; does not import the lowerer."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/research-2026-11-02/round2/qwen38-placement/physical-addresses'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary=json.loads((DEST/'summary.json').read_text());records=[]
    for record in summary['records']:
        path=Path(record['address_file']);assert sha(path)==record['address_sha256']
        with np.load(path) as f:
            a=f['allocations'].copy();fields=f['fields'].tolist();blocks=f['blocks_per_chip'].copy()
        def col(n):return a[:,fields.index(n)].astype(np.int64)
        layer,expert,phase,chip=(col(n) for n in ('layer','expert','phase','chip'))
        rows,columns,pages,waves=(col(n) for n in ('rows','columns','pages','waves'))
        block,wl=col('block'),col('wordline')
        assert np.all((layer>=0)&(layer<48)) and np.all((expert>=0)&(expert<513))
        assert np.isin(phase,[0,1]).all() and np.all((chip>=0)&(chip<16))
        assert np.all((block>=0)&(block<blocks[chip])) and np.all((wl>=0)&(wl+waves<=192))
        assert np.array_equal(pages,(rows*columns+16383)//16384)
        assert np.array_equal(waves,(pages+3)//4)
        nblocks=int(blocks.max());flat=(chip*nblocks+block)*192+wl
        ids=np.concatenate([flat[waves>i]+i for i in range(int(waves.max()))])
        used=np.bincount(ids,minlength=16*nblocks*192);assert used.max()==1
        # A WL slot here reserves all four same-address planes. Disjoint WL
        # bitmap entries imply disjoint physical LSB plane pages as well.
        assert int(used.sum())==int(waves.sum())
        matrix=(layer*513+expert)*2+phase
        # Lowered rows are contiguous by matrix. Detect all boundaries from
        # identifiers; do not assume a fixed job count or shared geometry.
        assert np.all(matrix[1:]>=matrix[:-1])
        starts=np.r_[0,np.flatnonzero(matrix[1:]!=matrix[:-1])+1];ends=np.r_[starts[1:],len(a)]
        assert len(starts)==48*513*2 and np.array_equal(matrix[starts],np.arange(48*513*2))
        certificates=set()
        for lo,hi in zip(starts,ends):
            s=a[lo:hi];assert len(np.unique(s[:,fields.index('chip')]))==len(s)
            p=int(s[0,fields.index('phase')]);rr,cc=(1280,2560)[p],(2560,640)[p]
            rect=tuple(sorted(tuple(int(v) for v in row) for row in s[:,[fields.index(n) for n in ('row_start','rows','column_start','columns')]]))
            if (p,rect) in certificates:continue
            # Integrate each horizontal cell row independently. This check is
            # different from the lowerer's 2-D rectangle-counter sweep.
            for r in range(rr):
                intervals=sorted((c,c+m) for r0,n,c,m in rect if r0<=r<r0+n)
                cursor=0
                for c,end in intervals:
                    assert c==cursor and c<end<=cc;cursor=end
                assert cursor==cc
            assert all(0<=r<r+n<=rr and 0<=c<c+m<=cc for r,n,c,m in rect)
            certificates.add((p,rect))
        useful=int((rows*columns).sum());assert useful==48*513*3*2560*640==record['unique_weight_bytes']
        assert int(pages.sum())*16384==record['programmed_page_bytes']
        assert int(waves.sum())*65536==record['reserved_wave_bytes']
        assert int(blocks.sum())*192*65536==record['allocated_block_bytes']
        records.append(dict(case=record['case'],matrices=len(starts),jobs=len(a),reserved_wordlines=int(used.sum()),distinct_geometry_checks=len(certificates),address_sha256=sha(path)))
        print(record['case'],'PASS',len(a),flush=True)
    report=dict(status='PASS separate bitmap address and row-interval coefficient checker',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),lowering_summary_sha256=sha(DEST/'summary.json'),records=records,
                scope='Independent checking algorithm on virtual address certificates; not real flash programming, native output ABI or timing calibration.')
    (DEST/'address_check.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
