#!/usr/bin/env python3
"""Same-budget strong control: all64routed experts on three private groups.

Uses the unchanged frozen replay driver; only a new frozen menu/protocol.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import full_ffn_addendum as driver

HERE=Path(__file__).resolve().parent
ORIGINAL=driver.DEST
DEST=HERE/'two-block-addendum'


def address_certificate(menus):
    out=[]
    for name,maps in menus.items():
        for layer in range(3):
            chips=[]
            for chip in range(16):
                fill=[128,0];jobs=[]
                for e in range(65):
                    rows=int(maps[layer,e,chip])
                    if not rows:continue
                    C=1024 if e==64 else 512;pages=rows*C//16384
                    assert rows*C%16384==0
                    waves=(pages+3)//4
                    block=next((b for b in range(2) if fill[b]+waves<=192),None)
                    assert block is not None
                    jobs.append(dict(expert=e,rows=rows,columns=C,pages=pages,waves=waves,
                                     block=block,wl_start=fill[block],commands=1,cold_starts=1))
                    fill[block]+=waves
                chips.append(dict(chip=chip,block_WL_fill=fill,Down_jobs=jobs))
            out.append(dict(menu=name,layer=layer,uses_second_block=any(x['block_WL_fill'][1]>0 for x in chips),chips=chips))
    return dict(status='PASS',GU_reservation='Conservative block0 first128WL on allfourplanes, sufficient for everyGU comparator. Entire Down job goes into one contiguous region in block0 orblock1.',
                allocated_budget_blocks_per_chip=2,claims='No job crosses a block, so second allocated block does not add a command orcold read to any existing private GEMV. Onecopy coefficients and complete row coverage unchanged.',records=out)


def freeze():
    DEST.mkdir(exist_ok=True);assert not (DEST/'protocol.frozen.json').exists()
    original=driver.f.read(ORIGINAL/'protocol.frozen.json')
    with np.load(ORIGINAL/'menus.frozen.npz') as z:menus={k:z[k].copy() for k in z.files}
    cold=original['cold_expert_by_layer']
    for kind in ['learned','static']:
        for destination in range(3):
            m=menus[f'dedicated_shared4_{kind}'].copy()
            for layer,e in enumerate(cold):
                m[layer,e]=0;m[layer,e,4*destination:4*destination+4]=384
            menus[f'all64_shared4_{kind}_to{destination}']=m
    for m in menus.values():assert np.all(m.sum(axis=2)==1536)
    np.savez_compressed(DEST/'menus.frozen.npz',**menus)
    cert=address_certificate(menus);driver.f.dump(DEST/'physical_capacity_audit.json',cert)
    protocol=dict(original)
    protocol['inputs_sha256']=dict(original['inputs_sha256'])
    for p in [Path(__file__),DEST/'menus.frozen.npz',ORIGINAL/'protocol.frozen.json']:
        protocol['inputs_sha256'][str(p)]=driver.f.sha(p)
    protocol.update(status='new_frozen_two_block_capacity_control',frozen_utc=driver.f.utc(),menus=list(menus),
        parent_protocol_sha256=driver.f.sha(ORIGINAL/'protocol.frozen.json'),
        capacity='Common2blocks/chip primarybudget for allGU/allmenus. Entire12page privatejob may move toblock1, noextra command/cold.63+1menus additionallyfitoneblock; this is not used to exclude64routed variants.',
        extra_control='For each already-frozen learned/static63+1map, move the rareexpert intact to each of3privategroups. No new row/group fitting. All64privateexperts then distributed22/21/21; shared4group hasno routedexpert.',
        selection='Same parent selection rule over10shared menus for all3GUfamilies. Screen64indices/layer input1 GF/DF, requireallPB/128B FIFOvalid. Freeze menu before full512 input1/4 GF/DF, retaining each GU originalincumbent.')
    driver.f.dump(DEST/'protocol.frozen.json',protocol)
    print(json.dumps(dict(status='frozen',menus=list(menus))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','screen','run']);args=ap.parse_args()
    if args.mode=='freeze':freeze()
    else:
        driver.DEST=DEST
        driver.execute(args.mode)
