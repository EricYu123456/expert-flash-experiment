#!/usr/bin/env python3
"""Independent physical and descriptor check of sealed A1/A1+A2 controls."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/research-2026-11-02/round2/g4-joint-prefix-replica-screen'
WIT = SOURCE / 'same-allocated-witness'
OUT = SOURCE / 'same-allocated-independent'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2)+'\n')


def readgz(p):
    with gzip.open(p,'rt') as f:return json.load(f)


def main():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__).resolve(),WIT/'protocol.frozen.json',WIT/'completion.json',WIT/'audit.json',
           WIT/'A1_physical.json.gz',WIT/'A1_A2_physical.json.gz',WIT/'all_query_address_relabels.npz',
           SOURCE/'work_arrays.npz',SOURCE/'native-v1/arrays.npz',
           SOURCE/'independent-contract/physical_rows_and_addresses.json.gz']
    protocol=dict(scope='Read-only all images/intervals and all512query job descriptors. No native or new fitting.',
                  inputs=[dict(path=str(p),sha256=sha(p)) for p in files])
    dump(OUT/'protocol.frozen.json',protocol)
    witness_proto=json.loads((WIT/'protocol.frozen.json').read_text())
    for x in witness_proto['inputs']:assert sha(x['path'])==x['sha256']
    completion=json.loads((WIT/'completion.json').read_text())
    for p,h in completion['files'].items():assert sha(WIT/p)==h
    joint=readgz(SOURCE/'independent-contract/physical_rows_and_addresses.json.gz')
    joint_img={}
    for e in joint['row_images']:
        for c,ids in e['GU_physical_original_row_ids'].items():joint_img[e['expert'],0,int(c)]=ids
        for c,ids in enumerate(e['Down_physical_original_row_ids']):joint_img[e['expert'],1,c]=ids
    with np.load(SOURCE/'work_arrays.npz') as z:w={k:z[k].copy() for k in z.files}
    with np.load(SOURCE/'native-v1/arrays.npz') as z:n={k:z[k].copy() for k in z.files}
    for phase in (0,1):
        counts=w['rows_B'][64,phase];start=np.r_[0,np.cumsum(counts)]
        for c in range(16):joint_img[64,phase,c]=list(range(int(start[c]),int(start[c+1])))
    inventories=[];A1=None
    for name,expectedMiB,expectedWaves in [('A1',268.5,4360),('A1_A2',364.5,5896)]:
        data=readgz(WIT/(name+'_physical.json.gz'));images=data['images'];inv=data['inventory']
        assert all(x['columns']==(1536 if x['phase']==0 else 512 if x['expert']<64 else 1024) for x in images)
        for x in images:
            assert len(x['row_ids'])==x['rows']
            assert x['programmed_pages']==(x['rows']*x['columns']+16383)//16384
            assert x['reserved_waves']==(x['programmed_pages']+3)//4
            assert x['first_plane']==0 and x['end_wordline']==x['wordline']+x['reserved_waves']<=192
        perchip=np.zeros((16,2),np.int64)
        for c in range(16):
            for b in range(2):
                jobs=sorted((x for x in images if x['chip']==c and x['block']==b),key=lambda x:x['wordline'])
                assert jobs and all(x['end_wordline']<=y['wordline'] for x,y in zip(jobs,jobs[1:]))
                perchip[c,b]=sum(x['reserved_waves'] for x in jobs)
        assert np.all(perchip[:,0]<=189)
        assert np.all(perchip[:,1]==192) if name=='A1_A2' else np.all(perchip[:,1]<=117)
        assert int(perchip.sum())==expectedWaves==inv['reserved_waves']
        coefficient=sum(x['rows']*x['columns'] for x in images)
        assert coefficient==inv['coefficient_bytes']==int(expectedMiB*1048576)
        for e in range(65):
            for phase,R in ((0,1024 if e<64 else 2048),(1,1536)):
                selected=[x for x in images if x['expert']==e and x['phase']==phase]
                multiplicity=Counter(r for x in selected for r in x['row_ids'])
                assert set(multiplicity)==set(range(R))
                if e==64:assert set(multiplicity.values())=={1}
                elif phase==0:assert set(multiplicity.values())==({2} if name=='A1' else {3})
                else:assert sum(v==2 for v in multiplicity.values())==768 and sum(v==1 for v in multiplicity.values())==768
        inventories.append(dict(name=name,coefficient_bytes=coefficient,programmed_pages=sum(x['programmed_pages'] for x in images),
                                reserved_waves=int(perchip.sum()),allocated_LSB_bytes=384*1048576,
                                per_chip_block_waves=perchip.tolist(),shared_GU_rows=2048,shared_Down_rows=1536,
                                shared_input_columns=[1536,1024]))
        if name=='A1':A1=images
    with np.load(WIT/'all_query_address_relabels.npz') as z:r={k:z[k].copy() for k in z.files}
    assert len(r['query_offsets'])==512 and len(r['jobs'])==75594
    prefix_ids=0
    for q,(begin,end) in enumerate(r['query_offsets']):
        allrows=r['jobs'][begin:end];control_addr=r['control_addresses'][begin:end]
        chipwork=np.zeros(16,np.int64);counters=np.zeros((2,7),np.int64)
        computed_ids={}
        selected=np.r_[w['routes'][q],64]
        for offset,record in enumerate(allrows):
            qq,ordinal,e,phase,c,nr,nc,pages,waves=map(int,record)
            assert qq==q and e==selected[ordinal]
            image=A1[int(r['image_ids'][begin+offset])]
            assert (image['expert'],image['phase'],image['chip'])==(e,phase,c)
            assert list(control_addr[offset])==[c,image['block'],image['wordline'],waves,pages,0]
            assert pages<=image['programmed_pages'] and waves<=image['reserved_waves']
            assert image['wordline']+waves<=192
            assert image['row_ids'][:nr]==joint_img[e,phase,c][:nr]
            assert nr==n['compact_rows'][q,ordinal,phase,c]
            emitted=pages*16384//nc
            chipwork[c]+=250+(nc+1)//2+waves*28000+2*emitted
            counters[phase]+=[1,nr*nc,pages,waves,nc,emitted*4,(nc+1)//2+2*emitted]
            computed_ids.setdefault((ordinal,phase),[]).extend(image['row_ids'][:nr]);prefix_ids+=nr
        assert chipwork.max()==w['chosen_work'][q]==n['actual_work'][q]
        assert np.array_equal(counters,n['counters'][q])
        for ordinal,e in enumerate(selected):
            for phase in (0,1):
                ids=computed_ids[ordinal,phase];R=1024 if phase==0 and e<64 else 2048 if phase==0 else 1536
                assert set(ids)==set(range(R))
                if phase==1:assert len(ids)==R
        if q in (0,255,511):
            with np.load(SOURCE/f'native-v1/q{q}.joint.npz') as z:
                assert np.array_equal(allrows[:,4:],z['jobs'])
                assert np.array_equal(z['input_columns'],np.array([[1536,512]]*6+[[1536,1024]]))
    for x in protocol['inputs']+witness_proto['inputs']:assert sha(x['path'])==x['sha256']
    report=dict(status='PASS sameallocated A1/A1+A2 independent physical/service witness',protocol_sha256=sha(OUT/'protocol.frozen.json'),
                witness_protocol_sha256=sha(WIT/'protocol.frozen.json'),inventories=inventories,
                all_query_count=512,all_job_descriptor_prefix_checks=75594,computed_prefix_row_ID_checks=prefix_ids,
                shared_both_projections_once=True,Down24MiB_core_sharing=True,
                descriptor_and_service_equivalence=True,
                scope='Sameallocated32blocks(2/chip), not sameuseful/programmed/wave budgets. Only immutable useful prefix IDs/charged service equal; shortsource fringe values may differ and aredropped. No new native/selector fit or device ABI proof.')
    dump(OUT/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
