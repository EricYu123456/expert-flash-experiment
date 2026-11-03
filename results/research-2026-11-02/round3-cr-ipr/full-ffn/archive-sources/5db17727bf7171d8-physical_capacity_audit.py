#!/usr/bin/env python3
"""Full one-copy address certificates: common two-block budget for every plan.

Block0 contains all GU weights; block1 contains private Down jobs, each with
its own ordinary cold start. Replacing local per-job timing-template addresses
with these unique static addresses changes no phase/wave/dimension count.
"""
from pathlib import Path
from collections import Counter
import gzip,json,sys,hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'scripts/research_20261102_cr_ipr_full_ffn.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261102_cr_ipr_full_ffn as f
strong,strong_inputs=f.load_strong()
with np.load(f.BASE/'baselines/frozen_TRAIN_packed_expert_order.npz') as z:orders=z['slot_to_expert']
with np.load(OUT/'g8-addendum/group_maps.frozen.npz') as z:g8=z['learned']


def gu_pages(kind,layer):
    pages=[];rows=np.zeros((65,16),dtype=np.int64)
    if kind in ['legacy_strong_whole_rows','separate_GU']:
        cursor=np.zeros(16,dtype=int)
        for e in range(65):
            for c in range(16):
                r=int(strong[layer,e,0,c]) if kind=='legacy_strong_whole_rows' else (128 if e==64 else 64)
                rows[e,c]=r
                n=(r*1536+16383)//16384
                for j in range(n):pages.append((c,j%4,0,int(cursor[c])+j//4,e,0,j))
                cursor[c]+=(n+3)//4
        assert max(cursor)<=192
    elif kind.startswith('g8_'):
        groups=np.arange(64)%2 if kind=='g8_identity' else g8[layer]
        assert int(groups.sum())==32
        for group in [0,1]:
            ids=np.flatnonzero(groups==group)
            for position,e in enumerate(ids):
                for c in range(group*8,group*8+8):
                    rows[e,c]=128
                    for j in range(12):pages.append((c,j%4,0,int(position)*3+j//4,int(e),0,j))
        rows[64]=128
        for c in range(16):
            for j in range(12):pages.append((c,j%4,0,96+j//4,64,0,j))
    else:
        rows[:64]=64;rows[64]=128
        inverse=np.argsort(orders[layer]) if kind=='sameWL_matching' else np.arange(64)
        for e in range(65):
            for j in range(12 if e==64 else 6):
                if kind=='candidate':
                    plane=([3,2]*6)[j] if e==64 else [0,1,2,0,1,3][j]
                    wl=64+j//2 if e==64 else 2*e+j//3 if plane<2 else e
                elif kind=='IPR_phase_aligned_balanced_shared':
                    plane=j%4
                    wl=(128 if plane<2 else 64)+j//4 if e==64 else (2*e+j//4 if plane<2 else e)
                else:
                    address=384+j if e==64 else int(inverse[e])*6+j
                    plane,wl=address%4,address//4
                for c in range(16):pages.append((c,plane,0,wl,e,0,j))
    assert np.all(rows[:64].sum(axis=1)==1024) and rows[64].sum()==2048
    return pages,rows


def down_pages(kind,layer):
    rows=np.zeros((65,16),dtype=np.int64)
    for e in range(65):
        if kind=='old_strong':rows[e]=strong[layer,e,1]
        elif e==64 or kind=='g16_equal':rows[e]=96
        else:rows[e,(np.arange(12)+e)%16]=128
    assert np.all(rows.sum(axis=1)==1536)
    cursor=np.zeros(16,dtype=int);pages=[]
    for e in range(65):
        columns=1024 if e==64 else 512
        for c,r in enumerate(rows[e]):
            n=(int(r)*columns+16383)//16384
            for j in range(n):pages.append((c,j%4,1,int(cursor[c])+j//4,e,1,j))
            cursor[c]+=(n+3)//4
    assert max(cursor)<=192
    return pages,rows


results=[];all_addresses=0;ledgers=[]
for layer in range(3):
    for gu in ['candidate','sameWL_matching','IPR_standard_RR_row_safe','IPR_phase_aligned_balanced_shared','separate_GU','legacy_strong_whole_rows','g8_identity','g8_learned']:
        aa,rr=gu_pages(gu,layer)
        for down in ['g16_equal','g12_cyclic','old_strong']:
            bb,dd=down_pages(down,layer);addresses=aa+bb
            occupied=set();counts=np.zeros((16,2,4),dtype=int)
            for c,p,b,w,e,phase,j in addresses:
                key=(c,p,b,w);assert key not in occupied and 0<=w<192 and b==phase
                occupied.add(key);counts[c,b,p]+=1
            useful=int(rr.sum()*1536+sum(int(dd[e].sum())*(1024 if e==64 else 512) for e in range(65)))
            assert useful==155713536
            page_bytes=len(addresses)*16384
            assert page_bytes>=useful and int(counts.max())<=192
            array=np.asarray(addresses,dtype=np.int32)
            name=f'L{layer}.{gu}.{down}'
            results.append(dict(name=name,layer=layer,GU=gu,Down=down,
                                useful_coefficient_bytes=useful,programmed_page_bytes=page_bytes,
                                coefficient_padding_bytes=page_bytes-useful,
                                allocated_blocks_per_chip=[2]*16,
                                allocated_lsb_bytes=2*16*4*192*16384,
                                actual_pages_per_chip_block_plane=counts.tolist(),
                                address_sha256=hashlib.sha256(array.tobytes()).hexdigest()))
            ledgers.append(dict(name=name,columns=['chip','plane','block','WL','expert','phase','local_page'],addresses=addresses))
            all_addresses+=len(addresses)
with gzip.open(OUT/'full_model_unique_addresses.json.gz','wt') as file:json.dump(ledgers,file,separators=(',',':'))
result=dict(status='pass',scope='Three G4 layers separately; all64routed plus shared weights; no replicas.',
            source_sha256=f.sha(__file__),primary_source_sha256=f.sha(ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py'),
            count=len(results),physical_page_addresses_checked=all_addresses,
            common_capacity='Two four-plane192WL blocks/chip for every GU/Down control. GU block0, private Down block1. All one-copy useful coefficient bytes equal155713536/layer; physical padding reported explicitly.',
            claim='Unique immutable addresses replace private-job local timing templates. All private jobs begin cold, retain full input, pages and source0123 ordering; address offset changes do not change modeled timing. This is a virtual address/capacity certificate, not programmed NAND.',
            strong_sources=strong_inputs,plans=results,ledger_sha256=f.sha(OUT/'full_model_unique_addresses.json.gz'))
f.dump(OUT/'physical_capacity_audit.json',result)
print(json.dumps(dict(status='pass',plans=len(results),pages_checked=all_addresses,common_blocks_per_chip=2)))
