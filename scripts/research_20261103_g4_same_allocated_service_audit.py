#!/usr/bin/env python3
"""Independent service-isomorphism check for separatelystored fullGU control.

No scheduler/native/selector fitting or replay; seals an address relabel proof.
"""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
JOINT=BASE/'g4-joint-prefix-replica-screen';WITNESS=JOINT/'same-allocated-witness'
OUT=JOINT/'same-allocated-service-equivalence'
GU=BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}


def freeze():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),WITNESS/'protocol.frozen.json',WITNESS/'completion.json',WITNESS/'audit.json',
           WITNESS/'A1_physical.json.gz',WITNESS/'A1_A2_physical.json.gz',WITNESS/'all_query_address_relabels.npz',
           JOINT/'work_arrays.npz',JOINT/'native-v1/arrays.npz',JOINT/'native-v1/row_owners.npz',
           JOINT/'independent/completion.json',GU/'rowid_ledgers.json']
    files += [JOINT/f'native-v1/q{q}.joint.npz' for q in (0,255,511)]
    dump(OUT/'protocol.frozen.json',dict(status='frozen_before_independent_equivalence_audit',source_sha256=sha(__file__),
         inputs=[dict(path=str(p),sha256=sha(p)) for p in files],
         scope='Checkall75594orderedjobs,512queryowner maps andtwo32-blockwitnessimages. No newnative run/fit/selector.',
         theorem='Address-invariantordinaryservice,identicalorderedjobs/inputs/dependencies/resources/outputs costs andsameaccepted logicalrow/scale IDs implyisomorphicnativeDAG andsameT. Discardedfringe values neednotmatch, butbytesstillcharged.',
         boundary='SameL0allocatedLSBblocks only; useful/programmed/wavebudgets differ. Noall-layerglobalpacking dominanceor device ABI claim.'))
    print(sha(OUT/'protocol.frozen.json'))


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for item in p['inputs']:assert sha(item['path'])==item['sha256']
    assert not (OUT/'completion.json').exists()
    w,n=load(JOINT/'work_arrays.npz'),load(JOINT/'native-v1/arrays.npz')
    owner=load(JOINT/'native-v1/row_owners.npz');maps=load(WITNESS/'all_query_address_relabels.npz')
    ledger=json.loads((GU/'rowid_ledgers.json').read_text())[3]['experts'];rows=w['rows_B']
    physical={}
    for tag in ('A1','A1_A2'):
        with gzip.open(WITNESS/(tag+'_physical.json.gz'),'rt') as f:physical[tag]=json.load(f)
        images=physical[tag]['images'];occupied=defaultdict(list);core=defaultdict(list)
        pages=waves=coeff=0
        for image in images:
            e,phase,c,nr,nc=[int(image[k]) for k in ('expert','phase','chip','rows','columns')]
            pg=(nr*nc+16383)//16384;wa=(pg+3)//4
            assert pg==image['programmed_pages'] and wa==image['reserved_waves']
            assert nr==len(image['row_ids']) and image['first_plane']==0
            begin=int(image['wordline']);end=int(image['end_wordline'])
            assert end==begin+wa<=192
            occupied[c,int(image['block'])].append((begin,end))
            pages+=pg;waves+=wa;coeff+=nr*nc
            if image['kind'] in ('A1_GU','A2_GU'):core[image['kind'],e]+=image['row_ids']
        for bands in occupied.values():
            bands.sort();assert all(a[1]<=b[0] for a,b in zip(bands,bands[1:]))
        assert len(occupied)==32 and sorted(occupied)==[(c,b) for c in range(16) for b in range(2)]
        for e in range(64):
            assert sorted(core['A1_GU',e])==list(range(1024))
            if tag=='A1_A2':assert sorted(core['A2_GU',e])==list(range(1024))
        inventory=physical[tag]['inventory']
        assert pages==inventory['programmed_pages'] and waves==inventory['reserved_waves'] and coeff==inventory['coefficient_bytes']
        assert inventory['allocated_blocks']==32 and inventory['allocated_LSB_bytes']==402653184
    control_images={int(i['image_id']):i for i in physical['A1']['images']}
    # Rebuild the entire ordered jobarray from the independentlyaudited dynamic
    # computed rowcounts, rather than trusting the witness's descriptor list.
    expected=[];boundaries=[]
    for q,route in enumerate(w['routes']):
        start=len(expected)
        for ordinal,e in enumerate(np.r_[route,64]):
            for phase in range(2):
                nc=1536 if phase==0 else 1024 if e==64 else 512
                for c,nr in enumerate(n['compact_rows'][q,ordinal,phase]):
                    nr=int(nr)
                    if nr:
                        pg=(nr*nc+16383)//16384;wa=(pg+3)//4
                        expected.append([q,ordinal,int(e),phase,c,nr,nc,pg,wa])
        boundaries.append([start,len(expected)])
    assert np.array_equal(expected,maps['jobs']) and np.array_equal(boundaries,maps['query_offsets'])
    assert len(expected)==75594
    byquery=defaultdict(dict);fringe_may_differ=0
    for record,addr,oldaddr,iid in zip(maps['jobs'],maps['control_addresses'],maps['producer_joint_addresses'],maps['image_ids']):
        q,ordinal,e,phase,c,nr,nc,pg,wa=map(int,record);image=control_images[int(iid)]
        assert [image[k] for k in ('expert','phase','chip','columns')]==[e,phase,c,nc]
        assert nr<=image['rows'] and pg<=image['programmed_pages'] and wa<=image['reserved_waves']
        assert list(addr)==[c,image['block'],image['wordline'],wa,pg,0]
        assert oldaddr[0]==c and oldaddr[3]==wa and oldaddr[4]==pg and oldaddr[5]==0
        if phase==0 and e<64:
            candidate=ledger[e]['resident_prefix_image'][str(c)]
            expected_kind='A1_GU' if c in ledger[e]['sources'] and nr==128 else 'B_GU'
            assert image['kind']==expected_kind and image['row_ids'][:nr]==candidate[:nr]
            if c in ledger[e]['sources'] and nr<128:fringe_may_differ+=pg*16384//nc-nr
        elif phase==1 and e<64:
            starts=np.r_[0,np.cumsum(rows[e,1])]
            candidate=list(range(int(starts[c]),int(starts[c+1])))
            for src,dest in w['Down_edges'][e]:
                if c==dest:candidate+=list(range(int(starts[src]),int(starts[src+1])))
            assert image['kind']=='Down14' and image['row_ids'][:nr]==candidate[:nr]
        else:
            starts=np.r_[0,np.cumsum(rows[e,phase])]
            assert image['row_ids'][:nr]==list(range(int(starts[c]),int(starts[c])+nr))
        byquery[int(q)][int(ordinal),phase,c]=image
    # Same consumed IDs, even though shortsource fringe has zeroes in control
    # and appended donor coefficients in the compact prefix image.
    for q in range(512):
        for ordinal in range(6):
            for phase,key,size in ((0,'GU',1024),(1,'Down',1536)):
                result=np.full(size,-1,np.int16)
                cc,pp=owner[key][q,ordinal].T
                for chip in np.unique(cc):
                    take=cc==chip;image=byquery[q][ordinal,phase,int(chip)]
                    result[take]=np.asarray(image['row_ids'],np.int16)[pp[take]]
                assert np.array_equal(result,np.arange(size))
    for q in (0,255,511):
        plan=load(JOINT/f'native-v1/q{q}.joint.npz');start,stop=boundaries[q]
        assert np.array_equal(plan['jobs'],maps['jobs'][start:stop,4:])
    report=dict(status='PASS_sameallocated_service_equivalence_finite_no_go',protocol_sha256=sha(OUT/'protocol.frozen.json'),
        source_sha256=sha(__file__),ordered_jobs_exact=75594,queries=512,GU_unique_owner_maps=3072,Down_unique_owner_maps=3072,
        same_full_input_domains=True,same_singlecontext_resources=True,same_fullGU_A8_dependencies=True,
        same_accepted_row_and_scale_IDs=True,discarded_fringe_outputs_may_differ=fringe_may_differ,
        identical_native_costs_in_declared_model='Byexactordereddescriptor/DAGisomorphism,notnewnativeexecution.',
        joint_allocated_blocks=32,A1control_allocated_blocks=32,A1_A2control_allocated_blocks=32,
        joint_useful_bytes=216517632,joint_pages=13344,joint_waves=3336,
        A1control_useful_bytes=physical['A1']['inventory']['coefficient_bytes'],A1control_pages=physical['A1']['inventory']['programmed_pages'],
        A1control_waves=physical['A1']['inventory']['reserved_waves'],
        conclusion='CompactprefixcandidatehasnolatencyorallocatedL0capacityadvantageoveranexplicitfullGUbackupcontrol. Itstillreducesuseful/programmed/reservedwavebytes;differentbudgetquestionsmustbeseparate.',
        limits=['Address-invariant28usordinaryservice assumed;notmeasureddevice/rowfunction/ABI proof.',
                'L0allocated32-blockbudget only,notglobalall-layerpackingoruseful-byte/perchipwave-budget dominance.',
                'A2isunusedinservice-equivalence,butfitsasadditionalfullGUcontrolatthesameallocatedblockcount.',
                'No newselector/fit/nativeCP/heldout/alllayerexecuted. Studygoalremainsactive;thiscandidateattributionbranchisclosed.'])
    dump(OUT/'audit.json',report)
    text='''The joint prefix-replica branch is closed as a finite attribution no-go under the L0 allocated-LSB-block budget. An explicit original B + identical Down14 + separately stored complete GU A1 control fits exactly the same32blocks (2/chip), and another complementary complete GU A2 also fits without increasing that allocation.

Independent checking reproduced all75,594 ordered runtime job descriptors and all512 GU/Down consumed-row maps. Long GU source reads use an A1 full-image band; short GU source and offsource completion reads use the original B band; Down14 views remain identical. Each job keeps its chip, fullinputC, rows, pages, waves, output bytes, ordinal and dependencies. The original fullGU/A8 barrier and singlecontext ownership therefore induce the same event DAG and timing in the declared address-invariant model. Only block/WL bases and discarded shortsource fringe values change; allfringe bytes are still drained/charged. No native rerun was needed to prove this structural equivalence.

The compact candidate uses216,517,632 useful bytes/13,344pages/3,336waves; the complete-A1 control uses281,542,656 useful bytes/17,440pages/4,360waves. Both allocate384MiB of LSBblocks, versus B192MiB. Prefix sharing saves programmed/useful bytes but does not establish a latency or allocated-capacity advantage for this L0 regime. This does not prove all-layer/global packing or a strict useful-byte budget equivalent.

The previously measured modeled joint mean2.152% gain over B is reproducible by the full-image control with the sameselector and software cost. Relative to the stronger Down-only branch it is only0.208%mean, P99 worsens, and the largest perquery regression is29.118us. No production selector, new fit, nativeCP, heldout or all-layer expansion followed. The overall research goal remains open.
'''
    (OUT/'report.md').write_text(text)
    files=[OUT/'protocol.frozen.json',OUT/'audit.json',OUT/'report.md',Path(__file__),
           JOINT/'independent/completion.json',JOINT/'independent-contract/completion.json',WITNESS/'completion.json',JOINT/'cp32/protocol.frozen.json']
    dump(OUT/'completion.json',dict(status='sealed_finite_attribution_no_go',files=[dict(path=str(p),sha256=sha(p)) for p in files]))
    print(json.dumps(dict(status=report['status'],completion_sha256=sha(OUT/'completion.json')),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
