#!/usr/bin/env python3
"""Lower frozen logical views to virtual chip/block/WL/plane addresses.

This is an address certificate, not a programmed flash device or weight image.
Every native job occupies one block and retains its replayed command count.
"""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np

from research_20261102_qwen38_fit import sha
from research_20261102_qwen38_replay import BASE,inventories

SOURCE=BASE/'qwen38-placement/native-full48'
DEST=BASE/'qwen38-placement/physical-addresses'
FIELDS=('layer','expert','phase','chip','row_start','rows','column_start','columns','pages','waves','block','wordline')
EXPECTED='9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505'


def coverage(rectangles,rows,columns):
    rb=sorted({0,rows}|{r for r,n,c,m in rectangles}|{r+n for r,n,c,m in rectangles})
    cb=sorted({0,columns}|{c for r,n,c,m in rectangles}|{c+m for r,n,c,m in rectangles})
    assert rb[0]==0 and rb[-1]==rows and cb[0]==0 and cb[-1]==columns
    rc={r:i for i,r in enumerate(rb)};cc={c:i for i,c in enumerate(cb)}
    a=np.zeros((len(rb)-1,len(cb)-1),dtype=np.int8)
    for r,n,c,m in rectangles:
        assert n>0 and m>0
        a[rc[r]:rc[r+n],cc[c]:cc[c+m]]+=1
    assert np.all(a==1)


def allocate(a):
    block_counts=[]
    for chip in range(16):
        ids=np.flatnonzero(a[:,3]==chip)
        ids=ids[np.argsort(-a[ids,9],kind='stable')]
        # Standard decreasing best-fit baseline. It adds no copies or encoding.
        buckets=[[] for _ in range(193)];free=[]
        for i in ids:
            n=int(a[i,9]);assert 0<n<=192
            found=next((r for r in range(n,193) if buckets[r]),None)
            if found is None:
                b=len(free);free.append(192);found=192
            else:b=buckets[found].pop()
            a[i,10]=b;a[i,11]=192-found
            free[b]-=n
            if free[b]:buckets[free[b]].append(b)
        block_counts.append(len(free))
    order=np.lexsort((a[:,11],a[:,10],a[:,3]));s=a[order]
    same=(s[1:,3]==s[:-1,3])&(s[1:,10]==s[:-1,10])
    assert np.all(s[1:,11][same]>=s[:-1,11][same]+s[:-1,9][same])
    assert np.all(a[:,11]>=0) and np.all(a[:,11]+a[:,9]<=192)
    assert np.all((a[:,8]+3)//4==a[:,9])
    return block_counts


def main():
    DEST.mkdir(parents=True,exist_ok=True);freeze=SOURCE/'plans.frozen.json';assert sha(freeze)==EXPECTED
    f=json.loads(freeze.read_text());inv=inventories();records=[];geometries=set()
    for case in sorted({r['case'] for r in f['records']}):
        plans=sorted((r for r in f['records'] if r['case']==case),key=lambda r:r['layer']);data=[]
        for record in plans:
            path=Path(record['plan']);assert sha(path)==record['plan_sha256'];plan=json.loads(path.read_text())
            for expert in range(513):
                for phase,name in enumerate(('gate_up','down')):
                    r=plan['plans'][name] if expert<512 else plan['fixed_shared'][name]
                    candidate=inv[name][r['label']];color=r['labels'][expert] if expert<512 else r['color']
                    jobs=candidate['variants'][color]
                    logical=tuple(sorted((j['row_start'],j['rows'],j['column_start'],j['columns']) for j in jobs))
                    key=(phase,logical)
                    if key not in geometries:
                        coverage(logical,(1280,2560)[phase],(2560,640)[phase]);geometries.add(key)
                    assert len({j['chip'] for j in jobs})==len(jobs)
                    for j in jobs:
                        useful=j['rows']*j['columns'];pages=(useful+16383)//16384;waves=(pages+3)//4
                        data.append((record['layer'],expert,phase,j['chip'],j['row_start'],j['rows'],j['column_start'],j['columns'],pages,waves,-1,-1))
        a=np.asarray(data,dtype=np.int32);del data;counts=allocate(a)
        useful=int((a[:,5].astype(np.int64)*a[:,7]).sum());pages=int(a[:,8].astype(np.int64).sum());waves=int(a[:,9].astype(np.int64).sum())
        assert useful==48*513*3*2560*640
        path=DEST/(case+'.addresses.npz');np.savez_compressed(path,allocations=a,fields=np.asarray(FIELDS),blocks_per_chip=np.asarray(counts,dtype=np.int32))
        by_chip=[]
        for c in range(16):
            s=a[a[:,3]==c];by_chip.append(dict(chip=c,jobs=len(s),useful_bytes=int((s[:,5].astype(np.int64)*s[:,7]).sum()),programmed_page_bytes=int(s[:,8].astype(np.int64).sum())*16384,
                                             reserved_wave_bytes=int(s[:,9].astype(np.int64).sum())*65536,allocated_block_bytes=counts[c]*192*65536,blocks=counts[c]))
        records.append(dict(case=case,address_file=str(path),address_sha256=sha(path),matrices=48*513*2,jobs=len(a),unique_weight_bytes=useful,
                            programmed_page_bytes=pages*16384,reserved_wave_bytes=waves*65536,allocated_block_bytes=sum(counts)*192*65536,blocks_per_chip=counts,per_chip=by_chip))
        print(case,len(a),'LSB block GiB',sum(counts)*192*65536/2**30,flush=True)
    assert sha(freeze)==EXPECTED
    report=dict(status='PASS virtual physical address lowering of all240 frozen views',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),freeze_sha256=EXPECTED,
                fields=FIELDS,records=records,distinct_geometry_certificates=len(geometries),
                address_rule='Within each chip, block is a four-plane group. Native job starts at the recorded LSB wordline. Coefficient byte i uses plane=(i//16384)%4, wordline=start+i//65536, page_byte=i%16384. All four planes share block/WL addresses.',
                checks=['Every matrix exact coefficient coverage from rectangle-cell sweep, no overlap or gaps; each chip has at most one job per projection.',
                        'Every job fits within one192-WL block; allocated wave intervals are disjoint in each chip/block. No extra matrix commands or block crossings.',
                        'Useful weights, programmed16KiB pages, reserved64KiB waves, and allocated12MiB LSB blocks reported separately. No weights duplicated.'],
                limits=['Virtual addresses only; no real NAND FTL programming, weight bytes downloaded, PE numerical execution, or timing measurement.',
                        'The model uses nominalrawI8 weight bytes. These are not original NVFP4 checkpoint addresses or storage sizes.',
                        'Only the LSB IFP address space is counted. CSB/MSB usable data and metadata are not represented as LSB weight capacity.',
                        'Standard best-fit packing is a lowering baseline, not a new placement algorithm or compression method.'])
    (DEST/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
