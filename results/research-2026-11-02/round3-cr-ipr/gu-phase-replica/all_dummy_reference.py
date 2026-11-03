#!/usr/bin/env python3
"""Restricted fixed-source finite-PB recurrence admitting all-dummy waves.

All four planes physically read each synchronous wave in a single block.
Useful pages use whole-page read->cache copy and serial fixed0123 ECC.
Dummy pages discard at arrayready using conditional pre-ECC drop. No cache
slot is cleared by dummy discard, no timing gap or hidden FIFO is inserted.
"""
from collections import Counter

def simulate_fullplane(waves,cache_copy_ns=500,row_bytes=1536):
    assert waves and 0<=cache_copy_ns and row_bytes>0
    cache_free=[0]*4;read_free=[0]*4;read_old=[None]*4;ecc_free=0
    physical=set();seen=set();events=[];arrays=[];done=[];violations=[];errors=[]
    owner_pages=Counter();owner_bytes=Counter();previous=None
    read_intervals=[];cache_intervals=[];dummy_count=0;all_dummy=0
    for wi,w in enumerate(waves):
        rp=w.get('read_pages',w['pages']);useful={p['page_id'] for p in w['pages']}
        assert {p['plane'] for p in rp}==set(range(4)) and len(rp)==4
        assert {p['block'] for p in rp}=={0}
        assert useful<={p['page_id'] for p in rp}
        for p in rp:
            assert p['bytes']==16384 and 0<=p['wl']<192
            address=(p['plane'],p['block'],p['wl'])
            assert address not in physical;physical.add(address)
            assert p['page_id'] not in seen;seen.add(p['page_id'])
        start=0 if wi==0 else 28000+(wi-1)*9700
        ready=28000+wi*9700;overwrite=0 if wi==0 else start+6040
        arrays.append(dict(wave=wi,start_ns=start,end_ns=ready,ready_ns=ready,overwrite_ns=overwrite,kind='cold' if wi==0 else 'cr',physical_pages=4,useful_pages=len(useful)))
        events.append(dict(stage='array_cold' if wi==0 else 'array_cr',wave=wi,start_ns=start,end_ns=ready))
        if not useful:all_dummy+=1
        for plane in range(4):
            if read_free[plane]>overwrite:
                violations.append(dict(kind='read_latch_overwrite',wave=wi,plane=plane,old_page=read_old[plane],release_ns=read_free[plane],overwrite_ns=overwrite,excess_ns=read_free[plane]-overwrite))
        for p in sorted(rp,key=lambda p:p['plane']):
            plane=p['plane'];read_old[plane]=p['page_id']
            common={k:p[k] for k in ('page_id','owner','page_index','plane','block','wl','bytes')}
            common.update(wave=wi,useful=p['page_id'] in useful)
            if p['page_id'] not in useful:
                dummy_count+=1;read_free[plane]=ready
                events.append(dict(stage='drop_before_ecc',start_ns=ready,end_ns=ready,**common))
                read_intervals.append(dict(start_ns=overwrite,ready_ns=ready,end_ns=ready,**common));continue
            if previous is not None and previous!=p['owner'] and owner_bytes[previous]%row_bytes:
                errors.append(dict(kind='owner_switch_inside_row',owner=previous,next_owner=p['owner']))
            if p['page_index']!=owner_pages[p['owner']]:errors.append(dict(kind='per_owner_page_order',owner=p['owner']))
            owner_pages[p['owner']]+=1;owner_bytes[p['owner']]+=p['bytes'];previous=p['owner']
            copy_start=max(ready,cache_free[plane]);copy_end=copy_start+cache_copy_ns
            ecc_start=max(ecc_free,copy_end);ecc_end=ecc_start+2560
            read_free[plane]=copy_end;cache_free[plane]=ecc_end;ecc_free=ecc_end
            events += [dict(stage='cache_copy',start_ns=copy_start,end_ns=copy_end,**common),dict(stage='ecc',start_ns=ecc_start,end_ns=ecc_end,source='cache',**common)]
            read_intervals.append(dict(start_ns=overwrite,ready_ns=ready,end_ns=copy_end,**common))
            cache_intervals.append(dict(start_ns=copy_start,end_ns=ecc_end,**common))
            done.append(dict(start_ns=ecc_start,end_ns=ecc_end,**common))
    for owner,n in owner_bytes.items():
        if n%row_bytes:errors.append(dict(kind='final_incomplete_row',owner=owner,bytes=n))
    for plane in range(4):
        intervals=[v for v in cache_intervals if v['plane']==plane]
        assert all(a['end_ns']<=b['start_ns'] for a,b in zip(intervals,intervals[1:]))
    assert all(a['end_ns']<=b['start_ns'] for a,b in zip(done,done[1:]))
    functional=not errors;timing=not violations;finish=max(arrays[-1]['end_ns'],ecc_free)
    return dict(model='restricted_fullplane_fixed0123_all_dummy_extension',feasible=functional and timing,
        functional_ok=functional,timing_feasible=timing,makespan_ns=finish if functional and timing else None,
        hypothetical_makespan_ns=finish,waves=len(waves),pages=len(done),bytes=len(done)*16384,
        physical_read_pages=4*len(waves),physical_read_bytes=4*len(waves)*16384,
        ecc_pages=len(done),ecc_bytes=len(done)*16384,dummy_pages=dummy_count,all_dummy_waves=all_dummy,
        cold_waves=1,cr_waves=len(waves)-1,issue_stall_ns=0,page_completions=done,ecc_page_completions=done,
        array_timeline=arrays,events=events,read_intervals=read_intervals,cache_intervals=cache_intervals,
        read_latch_release_ns=read_free,violations=violations,
        stream_validation=dict(ok=functional,errors=errors,owner_pages=dict(owner_pages),owner_bytes=dict(owner_bytes),owner_rows={o:n//row_bytes for o,n in owner_bytes.items()}),
        assumptions='Defaultconservativefirstready28us,finaldischarge0;strict9.7us fullplaneCR;6040ns evaloverwrite;perplanecopyoverlapsrecycle;wholepagecachethroughECCend;preECCdropconditional;twootherlatchesinput. Nohardwarevalidation.')
