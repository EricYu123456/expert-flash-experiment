#!/usr/bin/env python3
"""Independent two-phase availability, physical bytes and fixed-source GU audit."""
from collections import Counter
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr'
OUT=BASE/'gu-phase-replica-independent'
FIX=BASE/'numeric/checkpoint'
P,H,E,CH,RR,SR=16384,1536,64,16,1024,2048


def check(ok,msg):
    if not ok: raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def write(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def assign(route):
    a=sorted(e for e in route if e<3)
    b=sorted(e for e in route if 3<=e<6)
    d=sorted(e for e in route if e>=6)
    take=3-len(a)
    a=sorted(a+d[:take]);b=sorted(b+d[take:])
    check(len(a)==len(b)==3,'phase assignment unavailable')
    return a,b


def count_proof():
    classes=[]
    for a,b in itertools.product(range(4),repeat=2):
        c=6-a-b
        ways=math.comb(3,a)*math.comb(3,b)*math.comb(58,c)
        route=list(range(a))+list(range(3,3+b))+list(range(6,6+c))
        A,B=assign(route)
        check(len(A)==len(B)==3 and set(A+B)==set(route),'count-class witness')
        classes.append(dict(phase0_only=a,phase2_only=b,dual=c,top6_count=ways,route=route,A=A,B=B))
    total=sum(x['top6_count'] for x in classes)
    check(total==math.comb(64,6),'all top6 count identity')
    # Independent small explicit Hall check: replace58 interchangeable duals
    # with six. Exhaust all C(12,6) labeled routes and every server subset.
    small=0
    for route in itertools.combinations(range(12),6):
        allowed={e:({0} if e<3 else {2} if e<6 else {0,2}) for e in route}
        for size in range(1,7):
            for ss in itertools.combinations(route,size):
                neighbors=set().union(*(allowed[e] for e in ss))
                check(size<=3*len(neighbors),'capacitated Hall violation')
        assign(route);small+=1
    return dict(classes=classes,all_top6_count=total,small_labeled_top6_cases=small,
                theorem='Known two-server capacity3 combinatorial batch-code special case; no new theorem claim.',
                proof='For a phase0-only and b phase2-only experts in top6, a,b<=3. Assign3-a of6-a-b dual experts to phase0 and3-b to phase2. These nonnegative counts sum to the dual count. Capacitated Hall also holds: a subset restricted to one server has at most3 experts; a subset touching both has at most6.',
                lower_bound=dict(extra_complete_copies=58,total_complete_copies=122,
                    proof='At most3 experts may lack phase0: otherwise choose4 lacking it plus2 other experts, leaving fewer than3 eligible for phase0. The same holds for phase2. Thus at most6 experts can have only one phase, so at least58 must have both.',
                    scope='All64 experts resident, only two full dense6page phase choices, route-independent copies, and everytop6 must assign exactly3 experts to each phase. Not optimal over arbitrary page placement, more phases, fragments, overfetch or other stream constructions.'),
                insufficient57_witness=dict(phase0_only=[0,1,2,3],phase2_only=[4,5,6],dual=list(range(7,64)),
                    route=[0,1,2,3,7,8],reason='Four selected experts forced onto the phase0 server of capacity3.'))


def layout_and_pack():
    coords=np.full((780,2),-1,dtype='<i4')
    counters=[0]*4
    logical=[]
    for e in range(64):
        for phase in ([0] if e<3 else [2] if e<6 else [0,2]):
            for lp in range(6):
                plane=(phase+lp)%4;wl=counters[plane];counters[plane]+=1
                coords[e*12+(6 if phase==2 else 0)+lp]=plane,wl
                logical.append((e,phase,lp,plane,wl))
    check(counters==[183]*4,'routed phase-copy WL counts')
    for lp in range(12):
        plane=lp%4;wl=counters[plane];counters[plane]+=1
        coords[768+lp]=plane,wl;logical.append((64,0,lp,plane,wl))
    check(counters==[186]*4,'GU plane WL counts')
    coords.tofile(OUT/'coords.i32')
    w=np.memmap(FIX/'canonical.w8',dtype=np.int8,mode='r',shape=(64*RR+SR,H))
    canonical=w.reshape(-1)
    addresses=set();ownerbytes=0;ledger=[]
    for chip in range(16):
        bank=np.memmap(OUT/f'chip{chip:02d}.bin',dtype=np.int8,mode='w+',shape=(4*192*P,))
        bank[:]=0
        for e,phase,lp,plane,wl in logical:
            source=((e*RR+chip*64) if e<64 else (64*RR+chip*128))*H+lp*P
            key=(chip,0,plane,wl);check(key not in addresses,'physical GU collision');addresses.add(key)
            off=(wl*4+plane)*P;bank[off:off+P]=canonical[source:source+P]
            check(np.array_equal(bank[off:off+P],canonical[source:source+P]),'physical copy byte identity')
            ownerbytes+=P
            ledger.append(dict(chip=chip,expert=e,phase=phase,page_index=lp,plane=plane,block=0,wl=wl,canonical_byte=int(source),bank_byte=off,bytes=P))
        bank.flush();del bank
        fresh=np.memmap(OUT/f'chip{chip:02d}.bin',dtype=np.int8,mode='r')
        for e,phase,lp,plane,wl in logical:
            source=((e*RR+chip*64) if e<64 else (64*RR+chip*128))*H+lp*P
            check(np.array_equal(fresh[(wl*4+plane)*P:(wl*4+plane+1)*P],canonical[source:source+P]),'fresh resident readback')
        del fresh
    # Independent complete Down replica extents; coefficient values are not
    # programmed here, only addresses and byte capacity are certified.
    for chip in range(12):
        for wl in range(192):
            for plane in range(4):
                key=(chip,1,plane,wl);check(key not in addresses,'routed Down collision');addresses.add(key)
    for chip in range(12,16):
        for wl in range(186,192):
            for plane in range(4):
                key=(chip,0,plane,wl);check(key not in addresses,'shared Down collision');addresses.add(key)
    useful=len(addresses)*P
    check(ownerbytes==186*(1<<20),'GU payload')
    check(useful==int(331.5*(1<<20)),'full useful capacity')
    write(OUT/'page_ownership.json',ledger)
    return dict(GU_copies=122,extra_GU_copies=58,GU_pages=len(logical)*16,GU_payload_bytes=ownerbytes,
                extra_GU_payload_bytes=87*(1<<20),full_GU_plus_tripleDown_bytes=useful,
                full_unique_pages=len(addresses),GU_WL_per_plane=[186]*4,
                sharedDown_WL_in_block0=[186,191],routedDown_block1_WL=[0,191],
                max_blocks_per_chip=2,actual_nonempty_chipblocks=28,common_budget_chipblocks=32,
                all_programmed_GU_bytes_readback_exact=True,
                Down_certificate='Symbolic complete384row C512 bands and384row C1024 shared bands; Down coefficient arithmetic not rerun.',
                banks=[dict(path=str(OUT/f'chip{c:02d}.bin'),sha256=sha(OUT/f'chip{c:02d}.bin')) for c in range(16)])


def create_cases(proof):
    base_routes=np.fromfile(FIX/'routes.i32',dtype='<i4').reshape(-1,6)
    base_x=np.fromfile(FIX/'inputs.a8',dtype=np.int8).reshape(-1,H)
    base_sx=np.fromfile(FIX/'inputs.scales.f32',dtype='<f4')
    cases=[dict(kind='existing_realW8_fixture',route=r.tolist(),input_fixture=i) for i,r in enumerate(base_routes)]
    for i,c in enumerate(proof['classes']):
        for rev in [False,True]:
            cases.append(dict(kind='count_class_route',route=list(reversed(c['route'])) if rev else c['route'],
                              input_fixture=(i+int(rev))%len(base_routes),class_index=i))
    # Exercise every available expert/phase copy numerically as well as the
    # complete resident byte readback. Three fixed-only experts force all
    # three selected duals onto the opposite phase/server.
    dual=list(range(6,64))
    for k in range(0,len(dual),3):
        chunk=[dual[(k+j)%len(dual)] for j in range(3)]
        for fixed in [[0,1,2],[3,4,5]]:
            cases.append(dict(kind='all_resident_phase_numeric_coverage',route=fixed+chunk,
                              input_fixture=(k+fixed[0])%len(base_routes)))
    covered={(e,phase) for c in cases for phase,group in zip([0,2],assign(c['route'])) for e in group}
    expected={(e,phase) for e in range(64) for phase in ([0] if e<3 else [2] if e<6 else [0,2])}
    check(covered==expected,'every resident expert-phase numerical coverage')
    np.asarray([x['route'] for x in cases],dtype='<i4').tofile(OUT/'routes.i32')
    np.asarray([base_x[x['input_fixture']] for x in cases],dtype=np.int8).tofile(OUT/'inputs.a8')
    np.asarray([base_sx[x['input_fixture']] for x in cases],dtype='<f4').tofile(OUT/'inputs.scales.f32')
    write(OUT/'cases.json',cases)
    return cases


def g8_control_pack():
    """Ordinary whole-GU group replicas, physically packed from canonical rows."""
    directory=OUT/'ordinary-g8-control';directory.mkdir(exist_ok=True)
    for name in ['routes.i32','inputs.a8','inputs.scales.f32']:
        (directory/name).write_bytes((OUT/name).read_bytes())
    w=np.memmap(FIX/'canonical.w8',dtype=np.int8,mode='r',shape=(64*RR+SR,H)).reshape(-1)
    records=[];byte_count=0;logical_rows=Counter()
    for chip in range(16):
        resident=list(range(3)) if chip<8 else list(range(3,6))
        resident+=list(range(6,64))
        check(len(resident)==61,'g8 resident E count')
        bank=np.memmap(directory/f'chip{chip:02d}.bin',dtype=np.int8,mode='w+',shape=(4*192*P,));bank[:]=0
        seen=set()
        for pos,e in enumerate(resident+[64]):
            row_start=(chip%8)*128 if e<64 else chip*128
            for lp in range(12):
                plane=lp%4;wl=pos*3+lp//4
                check((plane,wl) not in seen and wl<186,'ordinary sameWL collision');seen.add((plane,wl))
                source=(e*RR+row_start if e<64 else 64*RR+row_start)*H+lp*P
                off=(wl*4+plane)*P;bank[off:off+P]=w[source:source+P]
                check(np.array_equal(bank[off:off+P],w[source:source+P]),'ordinary physical W8 bytes')
                byte_count+=P
                records.append(dict(chip=chip,group=chip//8,expert=e,page_index=lp,plane=plane,block=0,wl=wl,
                                    canonical_byte=int(source),bank_byte=off,bytes=P))
            for row in range(row_start,row_start+128):logical_rows[e,row]+=1
        bank.flush();del bank
        fresh=np.memmap(directory/f'chip{chip:02d}.bin',dtype=np.int8,mode='r')
        for p in records[-744:]:
            check(np.array_equal(fresh[p['bank_byte']:p['bank_byte']+P],w[p['canonical_byte']:p['canonical_byte']+P]),'ordinary fresh byte readback')
        del fresh
    for e in range(65):
        for row in range(1024 if e<64 else 2048):
            check(logical_rows[e,row]==(2 if 6<=e<64 else 1),'ordinary expert-row replica coverage')
    check(byte_count==186*(1<<20),'ordinary equal GU bytes')
    write(directory/'page_ownership.json',records)
    return dict(kind='Standard ordinary whole-GU group replication/CBC control, not a new construction claim.',
                group_chips=[list(range(8)),list(range(8,16))],resident_experts_per_group=61,
                routed_rows_per_participating_chip=128,shared_rows_per_chip=128,
                GU_payload_bytes=byte_count,extra_GU_payload_bytes=87*(1<<20),WL_per_plane=186,
                fixed_source_0123=True,every_read_wave_same_WL=True,IPR_required=False,
                coefficient_values_and_scales='Identical full expert tensors; row positions use localchip%8*128, unlike the16chip phase variant chip*64.',
                gate_up_scatter='Ordinary group localchips0..3 emit gate rows0..511 and localchips4..7 up rows512..1023. Phase variant globalchips0..7 emit gate rows0..511 and chips8..15 up rows512..1023. Shared uses globalchip*128 in both.',
                banks=[dict(path=str(directory/f'chip{c:02d}.bin'),sha256=sha(directory/f'chip{c:02d}.bin')) for c in range(16)])


def numpy_verify(cases,directory=OUT):
    w=np.memmap(FIX/'canonical.w8',dtype=np.int8,mode='r',shape=(64*RR+SR,H))
    sw=np.fromfile(FIX/'canonical.scales.f32',dtype='<f4')
    x=np.fromfile(OUT/'inputs.a8',dtype=np.int8).reshape(-1,H)
    sx=np.fromfile(OUT/'inputs.scales.f32',dtype='<f4')
    checked=0
    with (directory/'run.outputs.bin').open('rb') as f:
        for i,c in enumerate(cases):
            ix=np.concatenate([np.arange(e*RR,(e+1)*RR) for e in c['route']]+[np.arange(64*RR,64*RR+SR)])
            ref=w[ix].astype(np.int32)@x[i].astype(np.int32)
            fp=ref.astype(np.float32)*(sw[ix]*sx[i])
            first=None
            for mode in range(2):
                got=np.frombuffer(f.read(8192*8),dtype='<i8');y=np.frombuffer(f.read(8192*4),dtype='<f4')
                check(np.array_equal(got,ref),'independent NumPy integer mismatch')
                check(np.array_equal(y.view(np.uint32),fp.view(np.uint32)),'independent NumPy scale mismatch')
                if mode==0:first=(got.copy(),y.copy())
                else:check(np.array_equal(got,first[0]) and np.array_equal(y,first[1]),'dummy changes numeric result')
                checked+=8192
        check(not f.read(1),'extra output bytes')
    return dict(rows_checked=checked,all_integer_exact=True,all_F32_bits_exact=True,dummy_and_no_dummy_identical=True)


def builder_bridge():
    """Read frozen builder artifacts; reuse earlier independent page auditor."""
    directory=BASE/'gu-phase-replica'
    proto=json.loads((directory/'protocol.frozen.json').read_text())
    for p,h in proto['inputs_sha256'].items():check(sha(p)==h,'phase builder freeze changed')
    ledger=json.loads((directory/'resident_ledger.json').read_text())
    coords=np.fromfile(OUT/'coords.i32',dtype='<i4').reshape(780,2)
    allpages=[p for choices in ledger['copies'].values() for copy in choices.values() for p in copy]+ledger['shared_pages']
    check(len(allpages)==744,'builder resident page count')
    reverse={}
    for p in allpages:
        e,lp=p['expert'],p['page_index'];phase=p['copy_phase']
        ix=768+lp if e==64 else e*12+(6 if phase==2 else 0)+lp
        check(coords[ix].tolist()==[p['plane'],p['wl']] and p['block']==0,'builder vs independent physical address')
        check(p['source_byte_offset']==lp*P and p['source_row_bytes']==H and p['bytes']==P,'builder source logical bytes')
        reverse[p['plane'],p['wl']]=p
    independent=ROOT/'scripts/research_20261102_cr_ipr_pipeline_audit.py'
    spec=importlib.util.spec_from_file_location('prior_independent_page_auditor',independent)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    records=[]
    for path in [directory/name for name in ['trace.plain.copy100.json','trace.plain.copy500.json','trace.split.copy500.json']]:
        saved=json.loads(path.read_text());b=saved['builder'];t=saved['trace']
        A=b['assignment']['phase0_experts'];B=b['assignment']['phase2_experts'];route=A+B
        check(assign(route)==(A,B),'builder assignment rule')
        expected=[(e,lp) for pair in zip(A,B) for e in pair for lp in range(6)]+[(64,lp) for lp in range(12)]
        useful=[];dummy=[]
        for wave in b['waves']:
            check(wave['source_order']==[0,1,2,3],'builder fixed source order')
            used={p['page_id'] for p in wave['pages']}
            for p in sorted(wave['read_pages'],key=lambda p:p['plane']):
                physical=reverse[p['plane'],p['wl']]
                check(all(p[k]==physical[k] for k in ['expert','copy_phase','page_index','source_byte_offset','bytes']),'stream resident source binding')
                if p['page_id'] in used:useful.append((p['expert'],p['page_index']))
                else:
                    check(p['expert'] not in route and p['expert']!=64,'unselected true resident dummy')
                    dummy.append(p)
        check(useful==expected,'builder complete page/owner order')
        check(not({p['page_id'] for p in dummy}&{e['page_id'] for e in t['events'] if e['stage']=='ecc'}),'dummy reached ECC')
        audit=checker.audit_trace(b['waves'],t)
        records.append(dict(path=str(path),sha256=sha(path),**audit))
    group_proto=json.loads((directory/'group.protocol.frozen.json').read_text())
    for p,h in group_proto['inputs_sha256'].items():check(sha(p)==h,'ordinary group builder freeze changed')
    group_ledger=json.loads((directory/'group.resident_ledger.json').read_text())
    group_records=[]
    for group in [0,1]:
        resident=group_ledger[str(group)]
        pages=[p for copy in resident['copies'].values() for p in copy]+resident['shared_pages']
        check(len(pages)==744,'ordinary builder resident pages')
        reverse={}
        for p in pages:
            e,lp=p['expert'],p['page_index']
            pos=61 if e==64 else e%3 if e<6 else e-3
            check(p['block']==0 and p['plane']==lp%4 and p['wl']==pos*3+lp//4,'ordinary builder sameWL coordinates')
            check(p['source_byte_offset']==lp*P and p['source_row_bytes']==H and p['bytes']==P,'ordinary builder logical source bytes')
            reverse[p['plane'],p['wl']]=p
        for path in [directory/f'group{group}.trace.{name}.json' for name in ['plain.copy100','plain.copy500','split.copy500','whole_dummy.copy500']]:
            saved=json.loads(path.read_text());b=saved['builder'];t=saved['trace']
            A=b['assignment']['phase0_experts'];B=b['assignment']['phase2_experts'];route=A+B
            expected=[(e,lp) for e in (A if group==0 else B) for lp in range(12)]+[(64,lp) for lp in range(12)]
            useful=[];dummy=[];same_wl=True
            for wave in b['waves']:
                check(wave['source_order']==[0,1,2,3],'ordinary builder fixed source order')
                same_wl &= len({p['wl'] for p in wave['read_pages']})==1
                used={p['page_id'] for p in wave['pages']}
                for p in sorted(wave['read_pages'],key=lambda p:p['plane']):
                    physical=reverse[p['plane'],p['wl']]
                    check(all(p[k]==physical[k] for k in ['expert','page_index','source_byte_offset','bytes']),'ordinary stream resident source binding')
                    if p['page_id'] in used:useful.append((p['expert'],p['page_index']))
                    else:
                        check(p['expert'] not in route and p['expert']!=64,'ordinary real unselected dummy')
                        dummy.append(p)
            check(useful==expected,'ordinary complete page/row order')
            if 'split' not in path.name:check(same_wl,'ordinary plain control requires IPR')
            check(not({p['page_id'] for p in dummy}&{e['page_id'] for e in t['events'] if e['stage']=='ecc'}),'ordinary dummy reached ECC')
            audit=whole_dummy_trace_audit(b['waves'],t,checker) if 'whole_dummy' in path.name else checker.audit_trace(b['waves'],t)
            group_records.append(dict(path=str(path),sha256=sha(path),same_WL_every_wave=same_wl,**audit))
    verification=json.loads((directory/'verification.protocol.frozen.json').read_text())
    for p,h in verification['inputs_sha256'].items():check(sha(p)==h,'whole-dummy reference freeze changed')
    return dict(status='pass',phase_builder_sha256=sha(directory/'phase_builder.py'),
                protocol_sha256=sha(directory/'protocol.frozen.json'),resident_ledger_sha256=sha(directory/'resident_ledger.json'),
                resident_pages_bridged=744,saved_traces=records,reused_independent_page_checker_sha256=sha(independent),
                ordinary_group_builder_sha256=sha(directory/'group_builder.py'),ordinary_group_ledger_sha256=sha(directory/'group.resident_ledger.json'),
                ordinary_group_protocol_sha256=sha(directory/'group.protocol.frozen.json'),ordinary_group_pages_bridged=1488,
                ordinary_group_saved_traces=group_records,
                whole_dummy_source_sha256=sha(directory/'all_dummy_reference.py'),verification_protocol_sha256=sha(directory/'verification.protocol.frozen.json'),
                scope='Read-only bridge of three phase and eight ordinaryg8 saved traces. The g8 split policy uses IPR; the primary plain/whole-dummy ordinaryg8 control does not. Whole-dummy timing is checked against this audit\'s independent recurrence; no target simulator or builder imported.')


def finite_reference(dummy_boundary,copy_ns):
    """Separate fixed0123 two-latch recurrence, including an all-dummy wave.

    Array read-latch release is copy completion; per-plane cache remains
    occupied through ECC end. Dropped dummy pages release at array readiness.
    """
    read_release=[0]*4;cache_release=[0]*4;ecc_free=0;violations=[];events=[]
    useful_wave=0
    for wi in range(12+(dummy_boundary is not None)):
        start=0 if wi==0 else 28000+(wi-1)*9700
        ready=28000+wi*9700
        overwrite=start+(6040 if wi else 0)
        dummy=wi==dummy_boundary
        for plane in range(4):
            if read_release[plane]>overwrite:violations.append([wi,plane,read_release[plane]-overwrite])
            if dummy:
                read_release[plane]=ready
                events.append(dict(wave=wi,plane=plane,drop=True,ready=ready,overwrite=overwrite))
            else:
                begin=max(ready,cache_release[plane]);done=begin+copy_ns
                ecc_start=max(ecc_free,done);ecc_free=ecc_start+2560
                read_release[plane]=done;cache_release[plane]=ecc_free
                events.append(dict(wave=wi,plane=plane,drop=False,ready=ready,overwrite=overwrite,copy_start=begin,copy_end=done,ecc_start=ecc_start,ecc_end=ecc_free))
        if not dummy:useful_wave+=1
    return dict(dummy_boundary=dummy_boundary,cache_copy_ns=copy_ns,makespan_ns=max(ready,ecc_free),
                timing_feasible=not violations,deadline_violations=violations,useful_pages=4*useful_wave,
                physical_pages=len(events),ECC_pages=sum(not e['drop'] for e in events),events=events)


def whole_dummy_trace_audit(waves,trace,checker):
    """Check new saved whole-dummy event records with this file's recurrence."""
    empty=[i for i,w in enumerate(waves) if not w['pages']]
    check(empty==[10] and len(waves)==13,'whole-dummy boundary/geometry')
    ref=finite_reference(10,500)
    bykey={(e['wave'],e['plane']):e for e in ref['events']}
    copy_events={(e['wave'],e['plane']):e for e in trace['events'] if e['stage']=='cache_copy'}
    ecc_events={(e['wave'],e['plane']):e for e in trace['events'] if e['stage']=='ecc'}
    check(len(copy_events)==len(ecc_events)==48,'whole-dummy copy/ECC count')
    expected_ids=set();dummy_ids=set()
    for wi,wave in enumerate(waves):
        a=trace['array_timeline'][wi]
        start=0 if wi==0 else 28000+(wi-1)*9700
        check([a[k] for k in ['start_ns','end_ns','ready_ns','overwrite_ns']]==[start,28000+wi*9700,28000+wi*9700,start+(6040 if wi else 0)],'whole-dummy array exact times')
        for p in wave['read_pages']:
            key=(wi,p['plane']);e=bykey[key]
            if wi==10:
                check(key not in copy_events and key not in ecc_events,'whole-dummy enters copy/ECC')
                dummy_ids.add(p['page_id'])
            else:
                c=copy_events[key];z=ecc_events[key]
                check((c['start_ns'],c['end_ns'])==(e['copy_start'],e['copy_end']),'whole-dummy exact cache copy times')
                check((z['start_ns'],z['end_ns'])==(e['ecc_start'],e['ecc_end']),'whole-dummy exact ECC times')
                check(c['page_id']==z['page_id']==p['page_id'] and c['bytes']==z['bytes']==P,'whole-dummy event source binding')
                expected_ids.add(p['page_id'])
    check(not(dummy_ids&expected_ids),'dummy emitted logical page')
    check(trace['physical_read_pages']==52 and trace['ecc_pages']==trace['pages']==48,'whole-dummy page conservation')
    check(trace['physical_read_bytes']==52*P and trace['ecc_bytes']==trace['bytes']==48*P,'whole-dummy byte conservation')
    check(trace['cold_waves']==1 and trace['cr_waves']==12,'whole-dummy cold/CR count')
    check(trace['timing_feasible']==ref['timing_feasible'] and not trace['violations'],'whole-dummy finite deadline flag')
    check(trace['makespan_ns']==trace['hypothetical_makespan_ns']==ref['makespan_ns'],'whole-dummy makespan')
    read_by_plane=[[] for _ in range(4)];cache_by_plane=[[] for _ in range(4)]
    for r in trace['read_intervals']:
        e=bykey[r['wave'],r['plane']]
        release=e['ready'] if e['drop'] else e['copy_end']
        check((r['start_ns'],r['ready_ns'],r['end_ns'])==(e['overwrite'],e['ready'],release),'whole-dummy read-latch lifetime')
        read_by_plane[r['plane']].append((r['start_ns'],r['end_ns']))
    for c in trace['cache_intervals']:
        e=bykey[c['wave'],c['plane']]
        check(not e['drop'] and (c['start_ns'],c['end_ns'])==(e['copy_start'],e['ecc_end']),'whole-dummy cache lifetime')
        cache_by_plane[c['plane']].append((c['start_ns'],c['end_ns']))
    check(len(trace['read_intervals'])==52 and len(trace['cache_intervals'])==48,'whole-dummy lifetime inventory')
    peaks=[checker.peak(read_by_plane[p]+cache_by_plane[p]) for p in range(4)]
    check(all(checker.no_overlap(x) for x in cache_by_plane) and max(peaks)<=2,'whole-dummy two-slot capacity')
    check(checker.no_overlap([(e['start_ns'],e['end_ns']) for e in ecc_events.values()]),'whole-dummy exclusive ECC')
    rows=checker.row_check([p for w in waves for p in sorted(w['pages'],key=lambda p:p['plane'])])
    check(rows['ok'] and sum(rows['owner_bytes'].values())==48*P,'whole-dummy row ownership')
    return dict(pages=48,physical_pages=52,ecc_pages=48,bytes=48*P,cold_waves=1,cr_waves=12,
                peak_slots_per_plane=peaks,row_stream_ok=True,hypothetical_makespan_ns=ref['makespan_ns'],
                timing_feasible=True,independent_exact_copy_and_ECC_times=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    paths=[Path(__file__).resolve(),Path(__file__).with_suffix('.cpp'),FIX/'canonical.w8',FIX/'canonical.scales.f32',
           FIX/'inputs.a8',FIX/'inputs.scales.f32',FIX/'routes.i32',FIX/'provenance.json',BASE/'numeric/report.json']
    hashes={str(p):sha(p) for p in paths}
    write(OUT/'protocol.frozen.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=hashes,
          fixed_available_phases=dict(phase0_only=[0,1,2],phase2_only=[3,4,5],dual=list(range(6,64))),
          scope='Independent known-count proof; realW8 fixture byte interpreter; no heldout, no new quantization or fullFFN campaign.'))
    proof=count_proof();storage=layout_and_pack();cases=create_cases(proof);ordinary_storage=g8_control_pack()
    cmd=['g++','-O3','-std=c++17','-fopenmp-simd','-ffp-contract=off',str(Path(__file__).with_suffix('.cpp')),'-o',str(OUT/'interpreter')]
    build=subprocess.run(cmd,text=True,capture_output=True);write(OUT/'build_command.json',dict(command=cmd,returncode=build.returncode,stdout=build.stdout,stderr=build.stderr));check(build.returncode==0,'C++ compile')
    run_cmd=[str(OUT/'interpreter'),str(OUT),str(FIX),str(OUT/'run')]
    run=subprocess.run(run_cmd,text=True,capture_output=True);write(OUT/'run_command.json',dict(command=run_cmd,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr));check(run.returncode==0,'byte interpreter')
    numeric=json.loads((OUT/'run.json').read_text());numpy=numpy_verify(cases)
    ordinary_dir=OUT/'ordinary-g8-control'
    ordinary_cmd=[str(OUT/'interpreter'),str(ordinary_dir),str(FIX),str(ordinary_dir/'run'),'g8']
    ordinary_run=subprocess.run(ordinary_cmd,text=True,capture_output=True)
    write(ordinary_dir/'run_command.json',dict(command=ordinary_cmd,returncode=ordinary_run.returncode,stdout=ordinary_run.stdout,stderr=ordinary_run.stderr))
    check(ordinary_run.returncode==0,'ordinary g8 byte interpreter')
    ordinary_numeric=json.loads((ordinary_dir/'run.json').read_text());ordinary_numpy=numpy_verify(cases,ordinary_dir)
    check(sha(ordinary_dir/'run.outputs.bin')==sha(OUT/'run.outputs.bin'),'samebytes standard control outputs differ')
    bridge=builder_bridge()
    controls=[]
    for fault in ['phase','decode_dummy','scale']:
        res=subprocess.run(run_cmd[:-1]+[str(OUT/('mutation-'+fault)),fault],text=True,capture_output=True)
        check(res.returncode!=0,'undetected mutation '+fault)
        controls.append(dict(fault=fault,returncode=res.returncode,diagnostic=res.stderr.strip()))
    timing=[finite_reference(None,100),finite_reference(None,500)]+[finite_reference(b,500) for b in range(13)]
    write(OUT/'independent_timing.json',timing)
    selected=next(x for x in timing if x['dummy_boundary']==10 and x['cache_copy_ns']==500)
    check(selected['timing_feasible'] and selected['ECC_pages']==48 and selected['physical_pages']==52,'whole dummy boundary10')
    check(hashes=={p:sha(p) for p in hashes},'source inputs changed during audit')
    report=dict(status='pass_with_samebytes_ordinary_g8_control',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                inputs_sha256=hashes,proof=proof,storage=storage,cases=len(cases),numeric=numeric,numpy=numpy,
                numerical_resident_copy_coverage=dict(available_expert_phase_or_group_instances=122,all_exercised=True,
                                                     all64_experts_full_logical_rows=True,all16_chips_participate=True),
                accumulator_contract=dict(software_reference='I64 exact sum; one carried row accumulator, not multiple dot contexts',
                                          conservative_signed_W8_A8_H1536_abs_bound=H*128*128,existing_I32_sufficient=True),
                ordinary_g8_control=dict(storage=ordinary_storage,numeric=ordinary_numeric,numpy=ordinary_numpy,
                    phase_and_ordinary_scattered_output_files_bit_identical=True,outputs_sha256=sha(ordinary_dir/'run.outputs.bin'),
                    timing='Same48usefulpages and12fullwaves; copy100150980ns/no-drop, copy500whole-dummy13waves155680ns. SameWL reads need noIPR.'),
                builder_bridge=bridge,
                mutation_controls=controls,timing_summary=[{k:v for k,v in x.items() if k!='events'} for x in timing],
                elapsed_seconds=time.monotonic()-began,actual_commands=[cmd,run_cmd,ordinary_cmd],
                limits=['Known two-server cap3 CBC/count special case, not new theorem or solver.',
                    '58copy minimum is only under two full dense6page phase choices with everytop6 split3/3; not arbitrary placement optimum.',
                    'Weights/scales are existing reconstructed real-checkpoint W8 per-row tensors; fixture inputs are explicit synthetic vectors, not captured real activations. No requantization, task quality, nativeABI or GGUF equivalence claim.',
                    'All16chips execute one complete common1536B A8 input and one scalar row accumulator through page boundaries. Dummy drop preserves the live partial row/input index and produces no ECC or output.',
                    'Software usesI64 as an exact reference; the conservative1536*128*128=25165824 sum bound fitsI32, so this audit assumes no wider native accumulator or additional dot context.',
                    'Physical replicas add87MiB GU coefficients perlayer. Total331.5MiB includes standard tripleDown plus sharedg4, within common2blocks/chip; scale/scatter host metadata assumed resident.',
                    'IPR sameblock independent page addressing remains a premise. Whole-dummy preECC dropping remains conditional; copy100 uses12waves and no dummy.',
                    'Timing is a separate modeled recurrence, not measured hardware. Cold28us firstavailability and implicit6.4GB/s PE/MAC sustain remain assumptions; firstphase/discharge split unknown.',
                    'The same87MiB ordinaryg8 wholeGU group-replica control has the same CBC availability guarantee and identical scattered outputs/timing, while using sameWL reads with noIPR. Thus this witness supports no phase-specific performance or novelty advantage over that standard control.',
                    'This script builds its own layouts and streams. The saved builder bridge reuses the prior independent page checker; no target simulator or builder is imported.'])
    write(OUT/'report.json',report)
    (OUT/'report.md').write_text(
        f"Independent phase-replica construction passes all {proof['all_top6_count']:,} top6 combinations by16 count classes and a capacitated Hall proof;924 labeled reduced-universe routes are explicitly checked. At least58extra fullcopies are necessary within the fixed two-phase/3experts-per-phase dense6page model. This is a known two-server capacity3 CBC special case.\n\n"
        f"Real-checkpoint W8 bytes and existing per-row scales were packed into unique phase-copy addresses and read back byte-exact. {len(cases)} cases, both12wave/no-dummy and13wave/whole-dummy streams, all16chips: {numeric['row_comparisons']:,} exact I64 and F32-bit comparisons. Every122available expert/phase copy is numerically exercised. Independent NumPy agrees. No changed quantization; inputs remain the earlier explicit synthetic fixtures, so this is not task-quality evidence.\n\n"
        f"GU uses186WL/plane and186MiB useful coefficients (+87MiB). SharedDown g4 fits block0 WL186..191; routedDown triples fill block1 on12chips. Total331.5MiB useful/28nonempty chipblocks fits common32blocks. Full input is1536B; output scatter maps back to original route slots and canonical row scales. Only one live row accumulator is used; dummy reads do not advance it or enter ECC.\n\n"
        f"Independent finite-buffer recurrence:12waves copy100 {timing[0]['makespan_ns']}ns feasible={timing[0]['timing_feasible']};12waves copy500 feasible={timing[1]['timing_feasible']}; whole-dummy after10usefulwaves copy500 {selected['makespan_ns']}ns feasible={selected['timing_feasible']}. This full-wave spacer is distinct from the earlier3+1dummy split policy.\n\n"
        f"Actual samebytes standard control: two8chipgroups, each61wholeGU experts×12pages plus shared12pages,186WL/plane and the same+87MiB. Localchip128row shards reconstruct each full1024row expert; scales and output scatter are exact. Its {ordinary_numeric['row_comparisons']:,} I64/F32-bit comparisons pass and complete output files are bit-identical to the phase variant. All12readwaves are sameWL and fixed0123, so this control needs noIPR. Both have150980ns copy100/no-drop or155680ns copy500/whole-dummy modeled timing. This removes phase-specific gain/novelty for the tested witness; ordinaryCBC/group replication is standard.\n\n"
        'Frozen builder bridge passes744phase and1488ordinary-group resident page coordinates and all11saved traces. Nine nonempty traces reuse the earlier independent page checker; two new whole-dummy g8 traces match this file\'s independent recurrence at every copy/ECC timestamp, read/cache lifetime and two-slot count. Plain500 is correctly infeasible. The ordinaryg8 saved split500 trace requiresIPR; it is distinct from the primary sameWL whole-dummy control.\n\n'
        + '\n'.join(report['limits'])+'\n\nReproduce: `python '+str(Path(__file__).resolve())+'`\n')
    print(json.dumps(dict(status=report['status'],cases=len(cases),numeric=numeric,ordinary_numeric=ordinary_numeric,timing_at10={k:v for k,v in selected.items() if k!='events'},elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
