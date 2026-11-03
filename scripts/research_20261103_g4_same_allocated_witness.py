#!/usr/bin/env python3
"""Static same-allocated addresses/row identities; no native evaluator imported."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
JOINT = BASE/'g4-joint-prefix-replica-screen'
OUT = JOINT/'same-allocated-witness'
GU = BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix'
DOWN = BASE/'g4-down-paired-merge-screen'
INC = BASE/'compact-coupled-solver/continuation/native/0107.npz'
SELF = Path(__file__).resolve()
PAGE, WAVE, BLOCK = 16384, 65536, 192


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def ceil(n, d):
    return (int(n)+int(d)-1)//int(d)


def load(p):
    with np.load(p) as z:
        return {k:z[k].copy() for k in z.files}


def freeze():
    OUT.mkdir(exist_ok=False)
    inputs = [INC, GU/'rowid_ledgers.json', GU/'contract_arrays.npz', DOWN/'arrays.npz',
              JOINT/'protocol.frozen.json', JOINT/'work_arrays.npz',
              JOINT/'native-v1/protocol.frozen.json', JOINT/'native-v1/arrays.npz',
              JOINT/'native-v1/row_owners.npz',
              JOINT/'independent-contract/physical_rows_and_addresses.json.gz',
              JOINT/'independent-contract/completion.json']
    inputs += [JOINT/f'native-v1/q{q}.joint.npz' for q in (0,255,511)]
    producer = ROOT/'scripts/research_20261103_g4_joint_prefix_replica_screen.py'
    native_producer = ROOT/'scripts/research_20261103_g4_joint_prefix_replica_native.py'
    inputs += [producer,native_producer]
    p = json.loads((JOINT/'protocol.frozen.json').read_text())
    n = json.loads((JOINT/'native-v1/protocol.frozen.json').read_text())
    assert sha(producer)==p['source_sha256']
    assert sha(native_producer)==n['source_sha256']
    shutil.copy2(SELF, OUT/'executed_source.py')
    dump(OUT/'protocol.frozen.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SELF), producer_source_sha256=sha(producer),
        native_producer_source_sha256=sha(native_producer),
        producer_protocol_sha256=sha(JOINT/'protocol.frozen.json'),
        inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],
        physical='Block0 on every chip holds original B GU/shared and maximal Down14 core|append images. Block1 holds full A1 GU backups or full A1+A2 complementary GU backups. Every image is wave-aligned and single-block.',
        comparison='A1 retains exactly chosen_u/chosen_v and original query/expert/phase/chip job order. Every requested descriptor and useful rowID is checked; only address and discarded fringe contents may differ.',
        coverage='All64experts GU1024/Down1536 rows and shared e64 GU2048/Down1536, original full input dimensions. No omitted shared projection.',
        limitations='No native timing, device operation, coefficient quantization/numeric ABI proof, new selector, all-layer globalpack or heldout read. Address-invariant ordinary model and existing row-separable/reset contract retained.'))


def original_ids(counts):
    starts=np.r_[0,np.cumsum(counts)]
    return {c:list(range(int(starts[c]),int(starts[c+1]))) for c in range(16)}


def make_images(rows, cols, entries, edges, include_A2):
    images=[]
    def add(kind,e,p,c,ids,block):
        if not ids:return
        nc=int(cols[e,p]); nr=len(ids); pages=ceil(nr*nc,PAGE); waves=ceil(pages,4)
        images.append(dict(image_id=len(images),kind=kind,expert=e,phase=p,chip=c,
                           rows=nr,columns=nc,row_ids=list(map(int,ids)),block=block,
                           programmed_pages=pages,reserved_waves=waves))
    for e in range(65):
        gu_orig=original_ids(rows[e,0]); down_orig=original_ids(rows[e,1])
        if e<64:
            assert gu_orig=={int(c):v for c,v in entries[e]['B_original_rows'].items()}
        down_img={c:v.copy() for c,v in down_orig.items()}
        if e<64:
            for src,dst in edges[e]:down_img[int(dst)]+=down_orig[int(src)]
        for p,data in [(0,gu_orig),(1,down_img)]:
            for c in range(16):add('B_GU' if p==0 else 'Down14' if e<64 else 'shared_Down',e,p,c,data[c],0)
    for e,entry in enumerate(entries):
        src=set(map(int,entry['sources']))
        a1={c:list(entry['resident_prefix_image'][str(c)]) for c in src}
        assert all(len(x)==128 for x in a1.values())
        assert Counter(r for x in a1.values() for r in x)==Counter(range(1024))
        for c in sorted(src):add('A1_GU',e,0,c,a1[c],1)
        if include_A2:
            dst=sorted(set(range(16))-src); original=original_ids(rows[e,0])
            donors=[r for c in sorted(src) for r in original[c]]; at=0; a2={}
            for c in dst:
                n=128-len(original[c]);a2[c]=original[c]+donors[at:at+n];at+=n
            assert at==len(donors)
            assert Counter(r for x in a2.values() for r in x)==Counter(range(1024))
            for c in dst:add('A2_GU',e,0,c,a2[c],1)
    cursor=np.zeros((16,2),np.int32)
    for x in images:
        c,b=x['chip'],x['block'];start=int(cursor[c,b]);stop=start+x['reserved_waves']
        assert start>=0 and stop<=BLOCK
        x.update(wordline=start,end_wordline=stop,first_plane=0)
        cursor[c,b]=stop
    assert np.all(cursor[:,0]<=189)
    if include_A2:assert np.all(cursor[:,1]==192)
    else:assert np.all(cursor[:,1]<=117)
    # Physical intervals and programmed pages never intersect across images.
    for c in range(16):
        for b in range(2):
            xx=sorted((x for x in images if x['chip']==c and x['block']==b),key=lambda x:x['wordline'])
            assert all(x['end_wordline']<=y['wordline'] for x,y in zip(xx,xx[1:]))
    totals=dict(images=len(images),per_chip_block_waves=cursor.tolist(),
                per_chip_reserved_waves=cursor.sum(1).tolist(),
                coefficient_bytes=sum(x['rows']*x['columns'] for x in images),
                programmed_pages=sum(x['programmed_pages'] for x in images),
                reserved_waves=int(cursor.sum()),allocated_blocks=32,
                allocated_LSB_bytes=32*BLOCK*WAVE,
                shared_images=[{k:v for k,v in x.items() if k!='row_ids'} for x in images if x['expert']==64])
    for e in range(65):
        for p in (0,1):
            selected=[x for x in images if x['expert']==e and x['phase']==p]
            counts=Counter(r for x in selected for r in x['row_ids'])
            assert set(counts)==set(range(int(rows[e,p].sum())))
            if e==64:assert set(counts.values())=={1}
            elif p==0:assert set(counts.values())=={3 if include_A2 else 2}
            else:assert sum(v==2 for v in counts.values())==768 and sum(v==1 for v in counts.values())==768
    with gzip.open(OUT/('A1_A2_physical.json.gz' if include_A2 else 'A1_physical.json.gz'),'wt') as f:
        json.dump(dict(inventory=totals,images=images,
                       padding='Zero extend each maximal image through its final programmed page. Reserved wave pages beyond programmed pages are not requested. Row IDs denote immutable original coefficient rows and host scale aliases.'),f)
    return images,totals


def run():
    proto=json.loads((OUT/'protocol.frozen.json').read_text())
    assert sha(SELF)==proto['source_sha256']==sha(OUT/'executed_source.py')
    for x in proto['inputs']:assert sha(x['path'])==x['sha256']
    assert not (OUT/'audit.json').exists()
    b,w,native,owners=load(INC),load(JOINT/'work_arrays.npz'),load(JOINT/'native-v1/arrays.npz'),load(JOINT/'native-v1/row_owners.npz')
    entries=json.loads((GU/'rowid_ledgers.json').read_text())[3]['experts']
    rows,cols,routes=b['rows'],b['input_columns'],w['routes']
    assert np.array_equal(rows,w['rows_B']) and np.array_equal(routes,native['routes']) and np.array_equal(routes,owners['routes'])
    assert rows[:64,0].sum(1).tolist()==[1024]*64 and rows[:64,1].sum(1).tolist()==[1536]*64
    assert rows[64].sum(1).tolist()==[2048,1536]
    assert np.array_equal(cols[:64],np.tile([1536,512],(64,1))) and cols[64].tolist()==[1536,1024]
    A1,inv1=make_images(rows,cols,entries,w['Down_edges'],False)
    A12,inv12=make_images(rows,cols,entries,w['Down_edges'],True)
    assert inv1['reserved_waves']==4360 and inv12['reserved_waves']==5896
    assert inv1['coefficient_bytes']==int(268.5*2**20) and inv12['coefficient_bytes']==int(364.5*2**20)
    lookup={(x['kind'],x['expert'],x['phase'],x['chip']):x for x in A1}
    with gzip.open(JOINT/'independent-contract/physical_rows_and_addresses.json.gz','rt') as f:joint=json.load(f)
    joint_addr={(x['expert'],x['phase'],x['chip']):x for x in joint['jobs']}
    joint_img={}
    for x in joint['row_images']:
        e=x['expert']
        for c,r in x['GU_physical_original_row_ids'].items():joint_img[e,0,int(c)]=r
        for c,r in enumerate(x['Down_physical_original_row_ids']):joint_img[e,1,c]=r
    for p in (0,1):
        for c,r in original_ids(rows[64,p]).items():joint_img[64,p,c]=r
    job_records=[];addresses=[];joint_addresses=[];indices=[];query_offsets=[];checks=[];fringe_differences=0
    for q,route in enumerate(routes):
        start=len(job_records);selected=np.r_[route,64];local=rows[selected].copy()
        for ordinal,e0 in enumerate(route):
            e=int(e0);entry=entries[e];u=w['chosen_u'][q,ordinal];v=w['chosen_v'][q,ordinal]
            for i,c in enumerate(entry['sources']):local[ordinal,0,c]=128 if u[i] else rows[e,0,c]
            for i,c in enumerate(entry['low_destinations']):local[ordinal,0,c]=0 if u[i] else 40
            hd=entry['high_destination'];local[ordinal,0,hd]=int(rows[e,0,hd]) if not u[7] else entry['high_prefix_rows'] if not u.all() else 0
            for i,(src,dst) in enumerate(w['Down_edges'][e]):
                if v[i]:local[ordinal,1,src]=0;local[ordinal,1,dst]=256
        assert np.array_equal(local,native['compact_rows'][q]) and np.array_equal(local[6],rows[64])
        local_jobs=[];local_offsets=np.zeros((7,2,2),np.int32)
        for ordinal,e0 in enumerate(selected):
            e=int(e0)
            for p in (0,1):
                local_offsets[ordinal,p,0]=len(local_jobs)
                computed_ids={}
                for c,n0 in enumerate(local[ordinal,p]):
                    nr=int(n0)
                    if not nr:continue
                    kind='B_GU' if p==0 else 'Down14' if e<64 else 'shared_Down'
                    if p==0 and e<64 and c in entries[e]['sources'] and nr==128:kind='A1_GU'
                    x=lookup[kind,e,p,c];nc=int(cols[e,p]);pages=ceil(nr*nc,PAGE);waves=ceil(pages,4)
                    assert nr<=x['rows'] and pages<=x['programmed_pages'] and x['wordline']+waves<=192
                    assert x['row_ids'][:nr]==joint_img[e,p,c][:nr]
                    emit=pages*PAGE//nc
                    if kind=='B_GU' and e<64 and c in entries[e]['sources'] and nr<128:
                        assert emit>nr;fringe_differences+=emit-nr
                    for pos,rid in enumerate(x['row_ids'][:nr]):computed_ids.setdefault(rid,(c,pos))
                    old=joint_addr[e,p,c]
                    descriptor=[c,nr,nc,pages,waves];local_jobs.append(descriptor)
                    job_records.append([q,ordinal,e,p]+descriptor)
                    addresses.append([c,x['block'],x['wordline'],waves,pages,0])
                    joint_addresses.append([c,old['block'],old['wordline'],waves,pages,0])
                    indices.append(x['image_id'])
                local_offsets[ordinal,p,1]=len(local_jobs)
                expected_R=int(rows[e,p].sum())
                assert set(computed_ids)==set(range(expected_R))
                if e<64:
                    expected=owners['GU' if p==0 else 'Down'][q,ordinal]
                    for rid,(c,pos) in enumerate(expected):
                        xx=lookup[('A1_GU' if p==0 and int(c) in entries[e]['sources'] and local[ordinal,0,c]==128 else 'B_GU' if p==0 else 'Down14'),e,p,int(c)]
                        assert int(pos)<local[ordinal,p,c] and xx['row_ids'][int(pos)]==rid
            # GU duplicates are explicitly computed but host consumes one immutable logical row each.
        stop=len(job_records);query_offsets.append([start,stop])
        if q in (0,255,511):
            original=load(JOINT/f'native-v1/q{q}.joint.npz')
            assert np.array_equal(local_jobs,original['jobs']) and np.array_equal(local_offsets,original['offsets'])
            assert np.array_equal(cols[selected],original['input_columns'])
            assert np.array_equal(local.sum(2),original['total_rows'])
            assert np.array_equal(np.asarray(joint_addresses[start:stop]),np.column_stack([original['jobs'][:,0],original['aliases'][:,1:3],original['jobs'][:,4],original['jobs'][:,3],np.zeros(len(local_jobs),np.int32)]))
            dump(OUT/f'q{q}.address_relabel.json',dict(query=q,selected=selected.tolist(),
                jobs=local_jobs,offsets=local_offsets.tolist(),control_addresses=addresses[start:stop],
                producer_joint_addresses=joint_addresses[start:stop],control_image_ids=indices[start:stop],
                descriptor_order_exact=True,useful_row_IDs_exact=True,no_native_execution=True))
        checks.append(dict(query=q,jobs=len(local_jobs),routed_GU_unique_rows=6*1024,
                           routed_Down_unique_rows=6*1536,shared_GU_unique_rows=2048,shared_Down_unique_rows=1536))
    np.savez_compressed(OUT/'all_query_address_relabels.npz',jobs=np.asarray(job_records,np.int32),
                        query_offsets=np.asarray(query_offsets,np.int32),control_addresses=np.asarray(addresses,np.int32),
                        producer_joint_addresses=np.asarray(joint_addresses,np.int32),image_ids=np.asarray(indices,np.int32))
    dump(OUT/'audit.json',dict(status='PASS symbolic single-layer addresses,rowIDs and service-descriptor equivalence',
        protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(SELF),
        producer_source_sha256=proto['producer_source_sha256'],native_producer_source_sha256=proto['native_producer_source_sha256'],
        controls=dict(A1=inv1,A1_A2=inv12),query_checks=checks,total_queries=512,total_jobs=len(job_records),
        all_65_experts_both_projections_covered=True,shared_row_totals=[2048,1536],shared_input_columns=[1536,1024],
        copied_fringe_complete_outputs_whose_values_may_differ=fringe_differences,
        service_proof='Every query keeps the exact ordered chip/rows/fullC/pages/waves descriptor list, original input domains,fullGU/fullA8 barriers,Down bits and host logical-row consumption. It changes only physical block/WL bases. Ordinary timing accepts no physical addresses and has address-invariant service in the declared model; no native run was performed.',
        valid_row_scope='All requested useful prefix row IDs and original host scale associations match. Discarded short-source fringe may change from copied coefficients to zero padding; those outputs are still computed/drained/billed. No claim of bitwise equality for all emitted fringe or a verified chip output/reset ABI.',
        capacity_scope='Both controls occupy32LSBblocks,2/chip,matching joint. They exceed joint useful/programmed/wave vector budgets. Block0 repacks Down cores with their append packets; originalB addresses are not held fixed. No all-layer optimal allocated-capacity conclusion.',
        no_new_selector=True,no_native_or_numeric_device_execution=True))
    for x in proto['inputs']:assert sha(x['path'])==x['sha256']
    dump(OUT/'completion.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SELF),inputs_unchanged=True,
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(dict(status='PASS',queries=512,jobs=len(job_records),audit_sha256=sha(OUT/'audit.json')),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.action=='freeze' else run()
