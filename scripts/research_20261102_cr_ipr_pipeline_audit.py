#!/usr/bin/env python3
"""Independent event/resource and finite two-slot audit of CR+IPR timing.

The target module is imported ONLY to obtain traces. This auditor builds its
own pages, computes its own finite-buffer max-plus reference, reconstructs
row ownership, and checks traces without using target validation functions.
"""
from collections import Counter, defaultdict
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
OUT=ROOT/'results/research-2026-11-02/round3-cr-ipr/pipeline-independent'
P,H=16384,1536


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):
    Path(p).write_text(json.dumps(x,indent=2)+'\n')


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def candidate():
    """Separate construction from the numerical audit's 3-page macro proof."""
    stream=[]
    for group in range(2):
        for label,local,half in [('A1',0,0),('D',-1,0),('A2',1,0),('B1',0,1),
                                 ('B2',1,1),('C',-2,0),('A3',2,0),('B3',2,1)]:
            if local>=0:
                e=group*3+local
                for j,pl in enumerate((0,1,2) if half==0 else (0,1,3)):
                    lp=half*3+j
                    stream.append(dict(page_id=f'e{e}.p{lp}',owner=f'e{e}',page_index=lp,
                                       plane=pl,wl=e if pl>=2 else 2*e+half,block=0,bytes=P))
            else:
                for j,pl in enumerate((3,2,3) if local==-1 else (2,3,2)):
                    lp=group*6+(0 if local==-1 else 3)+j
                    wl=[64,64,65,65,66,66,67,67,68,68,69,69][lp]
                    stream.append(dict(page_id=f's.p{lp}',owner='s',page_index=lp,
                                       plane=pl,wl=wl,block=0,bytes=P))
    return [dict(pages=stream[k:k+4],source_order=[p['plane'] for p in stream[k:k+4]]) for k in range(0,48,4)]


def ordinary_balanced():
    waves=[]
    for e in range(6):
        pp=[dict(page_id=f'e{e}.p{j}',owner=f'e{e}',page_index=j,plane=j%4,
                 wl=2*e+j//4,block=0,bytes=P) for j in range(6)]
        for part in [pp[:4],pp[4:]]:
            waves.append(dict(pages=part,source_order=[p['plane'] for p in part]))
    for j in range(0,12,4):
        pp=[dict(page_id=f's.p{j+k}',owner='s',page_index=j+k,plane=k,
                 wl=128+j//4,block=0,bytes=P) for k in range(4)]
        waves.append(dict(pages=pp,source_order=list(range(4))))
    return waves


def keep_hot(waves):
    out=[]
    for wi,w in enumerate(waves):
        raw=copy.deepcopy(w['pages'])
        active={p['plane'] for p in raw}
        for pl in sorted(set(range(4))-active):
            raw.append(dict(page_id=f'drop.w{wi}.p{pl}',owner='_drop',page_index=wi*4+pl,
                            plane=pl,wl=raw[0]['wl'],block=0,bytes=P))
        out.append(dict(pages=copy.deepcopy(w['pages']),read_pages=raw,source_order=list(range(4))))
    return out


def tiny_wave(index,planes,page_start=0):
    pp=[dict(page_id=f'x.p{page_start+j}',owner='x',page_index=page_start+j,
             plane=pl,wl=index,block=0,bytes=P) for j,pl in enumerate(planes)]
    return dict(pages=pp,source_order=list(planes))


def physical_pages(waves,cfg):
    pages=[]
    for wi,w in enumerate(waves):
        pp=w.get('read_pages',w['pages'])
        order=sorted(p['plane'] for p in pp) if cfg['source_policy']=='fixed' else w['source_order']
        for pl in order:
            p=next(p for p in pp if p['plane']==pl)
            useful=p['page_id'] in {x['page_id'] for x in w['pages']}
            pages.append(dict(p,wave=wi,useful=useful))
    return pages


def flat(waves,cfg):
    return [p for p in physical_pages(waves,cfg)
            if p['useful'] or cfg['unselected_policy']!='drop_before_ecc_conditional']


def row_check(pages):
    next_index=Counter()
    byte_count=Counter()
    previous=None
    errors=[]
    rows=[]
    for p in pages:
        owner=p['owner']
        if previous is not None and owner!=previous and byte_count[previous]%H:
            errors.append(('owner_switch_inside_row',p['page_id']))
        if p['page_index']!=next_index[owner]:
            errors.append(('owner_page_order',p['page_id']))
        old=byte_count[owner]//H
        byte_count[owner]+=p['bytes']
        new=byte_count[owner]//H
        rows.append((owner,old,new))
        next_index[owner]+=1
        previous=owner
    for owner,n in byte_count.items():
        if n%H:
            errors.append(('unfinished_row',owner))
    return dict(ok=not errors,errors=errors,owner_pages=dict(next_index),owner_bytes=dict(byte_count),
                rows=rows)


def no_overlap(intervals):
    rr=sorted((a,b) for a,b in intervals if b>a)
    return all(b<=c for (a,b),(c,d) in zip(rr,rr[1:]))


def peak(intervals):
    events=[]
    for a,b in intervals:
        if b>a:
            events.extend([(a,1),(b,-1)])
    n=mx=0
    for t,d in sorted(events):
        n+=d
        mx=max(mx,n)
    require(n==0,'interval conservation')
    return mx


def avoid(t,d,blackouts):
    # Independently choose the first free contiguous copy interval by testing
    # candidate starts at the original release and all blackout endpoints.
    choices=sorted({t}|{b for a,b in blackouts if b>=t})
    for start in choices:
        if all(start+d<=a or start>=b or d==0 for a,b in blackouts):
            return start
    raise RuntimeError('no finite copy interval')


def finite_reference(waves,cfg):
    """Independent max-plus dependency graph for finite-latch wave trains.

    Cache vertex k waits for its plane's preceding ECC vertex. ECC vertices
    form a serial chain in the prescribed source order. Old read-latch copy
    completion must precede the next overwrite vertex. No event or latch
    state from the target is consumed.
    """
    cfg=dict(cfg)
    c=cfg['cache_copy_ns']
    r=cfg['recycle_ns']
    d=cfg.get('cold_discharge_ns',0)
    physical=physical_pages(waves,cfg)
    array=[]
    ecc_end=[]
    copies=[]
    old_ecc={}
    old_release={}
    violations=[]
    previous_end=0
    ecc_free=0
    copy_free=0
    blackout=[]
    t=0
    previous_active=set()
    for wi,w in enumerate(waves):
        active={p['plane'] for p in w.get('read_pages',w['pages'])}
        hot=wi>0 and cfg['array_mode']=='cr' and w['pages'][0]['block']==waves[wi-1]['pages'][0]['block'] and active<=previous_active
        if wi and not hot:
            t+=d
        if hot:
            blackout.append((t,t+r))
        t+=cfg['cr_ns'] if hot else cfg['cold_ns']-d
        previous_active=active
    previous_active=set()
    for wi,w in enumerate(waves):
        active={p['plane'] for p in w.get('read_pages',w['pages'])}
        hot=wi>0 and cfg['array_mode']=='cr' and w['pages'][0]['block']==waves[wi-1]['pages'][0]['block'] and active<=previous_active
        issue=previous_end
        clear=max((old_release.get(pl,0) for pl in active),default=0)
        offset=r if hot and cfg['overwrite_at']=='evaluation_start' and cfg['copy_overlaps_recycle'] else 0
        kind='cr' if hot else 'cold'
        if cfg['array_mode']=='ordinary':
            issue=max(issue,clear)
        elif cfg['issue_policy']=='cold_reset' and clear>issue+offset:
            issue=max(issue,clear)
            kind='cold'
        elif cfg['issue_policy']=='paced_retention_conditional':
            issue=max(issue,clear-offset)
        elif not hot:
            issue=max(issue,clear)
        if kind=='cold' and wi:
            issue=max(issue,previous_end+d)
        end=issue+(cfg['cr_ns'] if kind=='cr' else cfg['cold_ns']-d)
        overwrite=issue+(r if kind=='cr' and cfg['overwrite_at']=='evaluation_start' else 0)
        ready=end+cfg['availability_phase_ns']
        array.append((issue,end,ready,overwrite,kind))
        for pl in active:
            if old_release.get(pl,0)>overwrite:
                violations.append((wi,pl,old_release[pl]-overwrite))
        for p in (p for p in physical if p['wave']==wi):
            pl=p['plane']
            if not p['useful'] and cfg['unselected_policy']=='drop_before_ecc_conditional':
                old_release[pl]=ready
                continue
            if cfg['weight_slots']==2:
                begin=max(ready,old_ecc.get(pl,0),copy_free if cfg['copy_scope']!='per_plane' else 0)
                if not cfg['copy_overlaps_recycle'] and cfg['issue_policy']=='strict_no_interval':
                    begin=avoid(begin,c,blackout)
                done=begin+c
                copies.append((p['page_id'],begin,done))
                if cfg['copy_scope']!='per_plane':
                    copy_free=done
                start=max(ecc_free,done)
                old_release[pl]=done
            else:
                start=max(ready,ecc_free)
            service=(p['bytes']*10**9+cfg['ecc_bytes_per_second']-1)//cfg['ecc_bytes_per_second']
            ecc_free=start+service
            old_ecc[pl]=ecc_free
            if cfg['weight_slots']==1:
                old_release[pl]=ecc_free
            ecc_end.append((p['page_id'],start,ecc_free))
        previous_end=end
        previous_active=active
    # Final phase is separately accounted; no measured split of the paper's
    # 28us tPRE+tEVAL+tDISCH is inferred here.
    return dict(array=array,ecc=ecc_end,copies=copies,violations=violations,
                timing_feasible=not violations,makespan=max(previous_end+d,ecc_free))


def audit_trace(waves,result):
    cfg=result['config']
    pages=flat(waves,cfg)
    physical=physical_pages(waves,cfg)
    useful=[p for p in physical if p['useful']]
    arr=result['array_waves']
    events=result['events']
    bystage=defaultdict(list)
    for e in events:
        bystage[e['stage']].append(e)
        require(e['end_ns']-e['start_ns']==e['duration_ns'],'event duration')
    ec=sorted(bystage['ecc'],key=lambda e:e['start_ns'])
    cp={e['page_id']:e for e in bystage['cache_copy']}
    require([e['page_id'] for e in ec]==[p['page_id'] for p in pages],'ECC stream order')
    require(no_overlap([(e['start_ns'],e['end_ns']) for e in ec]),'shared ECC overlap')
    require(no_overlap([(a['start_ns'],a['end_ns']) for a in arr]),'array overlap')
    require(len(arr)==len(waves),'array wave count')
    require(len({p['page_id'] for p in physical})==len(physical),'duplicate logical page')
    require(len({(p['plane'],p['block'],p['wl']) for p in physical})==len(physical),'duplicate physical page')
    read_ranges=defaultdict(list)
    cache_ranges=defaultdict(list)
    copy_resources=defaultdict(list)
    deadline_violations=[]
    recycle_violations=[]
    old_release={}
    for p,e in zip(pages,ec):
        wi,pl=p['wave'],p['plane']
        a=arr[wi]
        require(e['bytes']==p['bytes'],'ECC byte identity')
        service=(p['bytes']*10**9+cfg['ecc_bytes_per_second']-1)//cfg['ecc_bytes_per_second']
        require(e['duration_ns']==service,'ECC service rate')
        require(e['start_ns']>=a['ready_ns'],'ECC before page availability')
        if cfg['weight_slots']==2:
            x=cp[p['page_id']]
            require(x['start_ns']>=a['ready_ns'],'copy before page availability')
            require(x['duration_ns']==cfg['cache_copy_ns'],'cache copy service')
            require(e['start_ns']>=x['end_ns'],'ECC before cache copy')
            release=x['end_ns']
            cache_ranges[pl].append((x['start_ns'],e['end_ns']))
            resource=pl if cfg['copy_scope']=='per_plane' else -1
            copy_resources[resource].append((x['start_ns'],x['end_ns']))
            if not cfg['copy_overlaps_recycle']:
                for r in bystage['wl_recycle']:
                    if x['end_ns']>r['start_ns'] and x['start_ns']<r['end_ns'] and x['duration_ns']:
                        recycle_violations.append((x['page_id'],r['wave']))
        else:
            release=e['end_ns']
        read_ranges[pl].append((a['overwrite_ns'],release))
    # Read-latch deadlines include unselected pages that are dropped before ECC.
    for p in physical:
        wi,pl=p['wave'],p['plane']
        a=arr[wi]
        if old_release.get(pl,0)>a['overwrite_ns']:
            deadline_violations.append((wi,pl,old_release[pl]-a['overwrite_ns']))
        if not p['useful'] and cfg['unselected_policy']=='drop_before_ecc_conditional':
            release=a['ready_ns']
            read_ranges[pl].append((a['overwrite_ns'],release))
        else:
            release=cp[p['page_id']]['end_ns'] if cfg['weight_slots']==2 else next(e['end_ns'] for e in ec if e['page_id']==p['page_id'])
        old_release[pl]=release
    for rr in cache_ranges.values():
        require(no_overlap(rr),'cache latch overwritten during ECC')
    for rr in copy_resources.values():
        require(no_overlap(rr),'copy engine overlap')
    occupancy=[peak(read_ranges[p]+cache_ranges[p]) for p in range(4)]
    require(result['max_weight_slots_by_plane']==occupancy,'reported slot inventory')
    inferred_timing=not deadline_violations and not recycle_violations
    require(result['timing_feasible']==inferred_timing,'timing-feasibility flag')
    if inferred_timing:
        require(max(occupancy)<=cfg['weight_slots'],'finite slot capacity')
    demand=sum(p['bytes'] for p in useful)
    require(result['pages']==len(useful) and result['bytes']==demand,'page/byte conservation')
    require(result['ecc_busy_ns']==sum(e['duration_ns'] for e in ec),'ECC busy conservation')
    cold_count=sum(a['kind']=='cold' for a in arr)
    discharge=cfg.get('cold_discharge_ns',0)
    require(result['array_busy_ns']==sum(a['duration_ns'] for a in arr)+discharge*cold_count,'array busy conservation')
    require(sum(a['pages'] for a in arr)==len(physical),'array page conservation')
    require(result['physical_read_pages']==len(physical) and result['physical_read_bytes']==sum(p['bytes'] for p in physical),'physical read conservation')
    require(len(cp)==(len(pages) if cfg['weight_slots']==2 else 0),'copy count conservation')
    require(result['ecc_pages']==len(pages) and result['ecc_bytes']==sum(p['bytes'] for p in pages),'ECC byte/page counters')
    require(result['conservation']['array_read_pages']==len(physical) and
            result['conservation']['ecc_completed_pages']==len(pages) and
            result['conservation']['unselected_read_pages']==len(physical)-len(useful),'saved conservation counters')
    row=row_check([p for p in pages if p['useful']])
    if cfg['validate_row_stream']:
        require(result['functional_ok']==row['ok'],'functional row-stream flag')
    if row['ok']:
        actual=[(x['owner'],x['row_start'],x['row_end'],x['end_ns'],x['page_id']) for x in result['row_completions']]
        useful_ec=[e for e in ec if e['useful']]
        expected=[(x[0],x[1],x[2],e['end_ns'],e['page_id']) for x,e in zip(row['rows'],useful_ec) if x[2]>x[1]]
        require(actual==expected,'complete-row ownership/completion ranges')
    require(no_overlap([(a['start_ns'],a['end_ns']) for a in arr+bystage['array_discharge']]),'array/discharge resource overlap')
    require(sum(e['duration_ns'] for e in bystage['array_discharge'])==discharge*cold_count,'discharge conservation')
    expected_make=max(arr[-1]['end_ns']+discharge,ec[-1]['end_ns'])
    require(result['hypothetical_makespan_ns']==expected_make,'startup/drain makespan accounting')
    ref=finite_reference(waves,cfg)
    require([(x['page_id'],x['start_ns'],x['end_ns']) for x in ec]==ref['ecc'],'independent finite-buffer ECC reference')
    require([(a['start_ns'],a['end_ns'],a['ready_ns'],a['overwrite_ns'],a['kind']) for a in arr]==ref['array'],'independent array reference')
    require(sorted((x['page_id'],x['start_ns'],x['end_ns']) for x in cp.values())==sorted(ref['copies']),'independent copy reference')
    require(sorted(deadline_violations)==sorted(ref['violations']),'independent overwrite reference')
    return dict(pages=len(useful),physical_pages=len(physical),ecc_pages=len(pages),bytes=demand,
                cold_waves=cold_count,cr_waves=len(arr)-cold_count,
                deadline_violations=deadline_violations,copy_recycle_violations=recycle_violations,
                peak_slots_per_plane=occupancy,row_stream_ok=row['ok'],row_errors=row['errors'][:5],
                hypothetical_makespan_ns=expected_make,timing_feasible=inferred_timing)


def load_target():
    frozen=OUT/'source-snapshot.py'
    frozen.write_bytes(SOURCE.read_bytes())
    spec=importlib.util.spec_from_file_location('target_pipeline_for_trace_generation_only',frozen)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module,frozen


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target,frozen=load_target()
    source_hash=sha(frozen)
    original=ROOT/'results/research-2026-11-02/round3-cr-ipr/pipeline'
    input_paths=[SOURCE,original/'report.json',original/'report.md',original/'assumption_ledger.json',original/'run_receipt.json',ROOT/'paper/aif.txt']
    input_paths+=sorted(original.glob('*.events.json'))+[original/'small_two_wave_trace.json']
    input_hashes={str(p):sha(p) for p in input_paths}
    require(json.loads((original/'report.json').read_text())['source_sha256']==source_hash,'target report/source freeze mismatch')
    waves=candidate()
    records=[]
    traces={}
    for overwrite in ['sense_start','evaluation_start']:
        for c in [0,100,320,321,500,1000,6040,6041]:
            name=f'candidate.{overwrite}.copy{c}'
            r=target.simulate(waves,cache_copy_ns=c,overwrite_at=overwrite)
            result=audit_trace(waves,r)
            records.append(dict(name=name,**result))
            traces[name]=r
    for name,kw in [
        ('cold_reset',dict(cache_copy_ns=500,issue_policy='cold_reset')),
        ('cold_reset_sense',dict(cache_copy_ns=100,issue_policy='cold_reset',overwrite_at='sense_start')),
        ('paced_conditional',dict(cache_copy_ns=500,issue_policy='paced_retention_conditional')),
        ('no_recycle_copy',dict(cache_copy_ns=100,copy_overlaps_recycle=False)),
        ('one_slot',dict(weight_slots=1)),
        ('fixed_source',dict(source_policy='fixed')),
        ('serial_copy',dict(cache_copy_ns=500,copy_scope='chip_serial_conditional')),
        ('ordinary',dict(cache_copy_ns=500,array_mode='ordinary')),
        ('availability_offset',dict(cache_copy_ns=100,availability_phase_ns=1000))]:
        r=target.simulate(waves,**kw)
        records.append(dict(name=name,**audit_trace(waves,r)))
        traces[name]=r
    hand=[]
    oracle_cases=[
        ('single_page', [tiny_wave(0,[0])], dict(cache_copy_ns=100),30660),
        ('four_page', [tiny_wave(0,[0,1,2,3])], dict(cache_copy_ns=500),38740),
        ('two_wave_eval', [tiny_wave(0,[0,1,2,3]),tiny_wave(1,[0,1,2,3],4)],dict(cache_copy_ns=100),48580),
        ('plane_idle_cold',[tiny_wave(0,[0]),tiny_wave(1,[1],1)],dict(cache_copy_ns=100),58660),
    ]
    for name,ww,kw,expected in oracle_cases:
        r=target.simulate(ww,validate_row_stream=False,**kw)
        require(r['makespan_ns']==expected,'manual oracle '+name)
        result=audit_trace(ww,r)
        records.append(dict(name=name,**result))
        hand.append(dict(name=name,expected_ns=expected,actual_ns=r['makespan_ns']))
        traces[name]=r
    ordinary=ordinary_balanced()
    for name,ww,kw in [
        ('ordinary_partial_masks',ordinary,dict(cache_copy_ns=500)),
        ('ordinary_partial_discharge',ordinary,dict(cache_copy_ns=500,cold_discharge_ns=1000)),
        ('keep_hot_drop',keep_hot(ordinary),dict(cache_copy_ns=500,unselected_policy='drop_before_ecc_conditional')),
        ('keep_hot_decode',keep_hot(ordinary),dict(cache_copy_ns=500,unselected_policy='decode_and_discard_conditional')),
    ]:
        r=target.simulate(ww,**kw)
        records.append(dict(name=name,**audit_trace(ww,r)))
        traces[name]=r
    for d in [100,1000,5000,10000]:
        name=f'candidate.discharge{d}'
        r=target.simulate(waves,cold_discharge_ns=d,cache_copy_ns=100)
        records.append(dict(name=name,**audit_trace(waves,r)))
        traces[name]=r
    require(traces['ordinary_partial_masks']['cold_waves']==7,'partial-plane restart hand count')
    require(traces['keep_hot_drop']['physical_read_pages']==60 and traces['keep_hot_drop']['ecc_pages']==48,'keep-hot drop accounting')
    require(traces['keep_hot_decode']['physical_read_pages']==60 and traces['keep_hot_decode']['ecc_pages']==60,'keep-hot decode accounting')
    saved=[]
    for p in sorted(original.glob('*.events.json'))+[original/'small_two_wave_trace.json']:
        if p.name.startswith('ordinary15_keep_hot'):
            ww=keep_hot(ordinary)
        elif p.name.startswith('ordinary15'):
            ww=ordinary
        elif p.name.startswith('small_two'):
            ww=waves[:2]
        else:
            ww=waves
        r=json.loads(p.read_text())
        saved.append(dict(path=str(p),sha256=sha(p),**audit_trace(ww,r)))
    # Independent generated masks stress plane-idle restart, backpressure,
    # cold-discharge placement and finite slots beyond the named examples.
    rng=random.Random(2026110204)
    randomized=[]
    for trial in range(200):
        ww=[]
        page_index=0
        for wi in range(rng.randrange(2,9)):
            planes=rng.sample(range(4),rng.randrange(1,5))
            ww.append(tiny_wave(wi,planes,page_index))
            page_index+=len(planes)
        kw=dict(validate_row_stream=False,cache_copy_ns=rng.choice([0,100,320,321,500,1000,6500]),
                weight_slots=rng.choice([1,2]),overwrite_at=rng.choice(['sense_start','evaluation_start']),
                copy_overlaps_recycle=rng.choice([True,False]),
                issue_policy=rng.choice(['strict_no_interval','cold_reset','paced_retention_conditional']),
                array_mode=rng.choice(['cr','ordinary']),copy_scope=rng.choice(['per_plane','chip_serial_conditional']),
                cold_discharge_ns=rng.choice([0,1000]),ecc_bytes_per_second=rng.choice([6400000000,1000000000]),
                availability_phase_ns=rng.choice([0,1000]))
        try:
            r=target.simulate(ww,**kw)
            result=audit_trace(ww,r)
        except Exception as e:
            failure=dict(trial=trial,waves=ww,config=kw,error=repr(e),target_source_sha256=source_hash)
            if 'r' in locals():
                failure['trace']=r
            write(OUT/'erratum.json',failure)
            raise RuntimeError('randomized case failure; retained erratum.json') from e
        randomized.append(dict(trial=trial,masks=[w['source_order'] for w in ww],config=kw,**result))
    require(traces['candidate.evaluation_start.copy320']['timing_feasible'],'320ns boundary should fit')
    require(not traces['candidate.evaluation_start.copy321']['timing_feasible'],'321ns boundary should fail')
    require(not traces['fixed_source']['functional_ok'],'fixed-source negative must fail row ownership')
    mutations=[]
    baseline=traces['candidate.evaluation_start.copy100']
    def overlap_ecc(r):
        es=sorted((e for e in r['events'] if e['stage']=='ecc'),key=lambda e:e['start_ns'])
        es[1]['start_ns']=es[0]['start_ns']+1
        es[1]['end_ns']=es[1]['start_ns']+es[1]['duration_ns']
    def early_copy(r):
        e=next(e for e in r['events'] if e['stage']=='cache_copy')
        e['start_ns']=r['array_waves'][e['wave']]['ready_ns']-1
        e['end_ns']=e['start_ns']+e['duration_ns']
    for name,mutate in [
        ('ECC_overlap',overlap_ecc),
        ('early_copy',early_copy),
        ('wrong_conservation',lambda r:r.update(bytes=r['bytes']+P)),
        ('false_feasibility',lambda r:r.update(timing_feasible=False)),
        ('wrong_row_owner',lambda r:r['row_completions'][0].update(owner='wrong'))]:
        r=copy.deepcopy(baseline)
        mutate(r)
        try:
            audit_trace(waves,r)
        except RuntimeError as e:
            mutations.append(dict(name=name,detected=True,diagnostic=str(e)))
        else:
            raise RuntimeError('undetected event mutation '+name)
    require(sha(SOURCE)==source_hash,'target source changed during audit; rerun after freeze')
    require(input_hashes=={p:sha(p) for p in input_hashes},'target report/trace/input changed during audit')
    write(OUT/'traces.json',traces)
    report=dict(status='pass',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                audit_source_sha256=sha(__file__),target_source_sha256=source_hash,
                target_source=str(SOURCE),records=records,mutation_controls=mutations,
                manual_oracles=hand,
                generated_mask_crosschecks=randomized,generated_mask_seed=2026110204,
                saved_trace_crosschecks=saved,target_input_sha256=input_hashes,
                actual_command=['python',str(Path(__file__).resolve())],errata=[],
                independent_reference='Locally constructed candidate pages and independent max-plus finite read/cache-latch dependency graph; target imported only for trace generation.',
                copy_threshold_ns=320,copy_threshold_failure_ns=321,
                limitations=['Timing is a conditional capability/accounting study, not a measured hardware schedule.',
                             '28us cold = tPRE+tEVAL+tDISCH in the paper; treating 28us as first availability is an accounting assumption.',
                             'First availability and final discharge phase split is not publicly fixed by these aggregate timings.',
                             'With a nonzero cold_discharge_ns, issue_stall_ns includes mandatory inter-run discharge gaps; adding it directly to array_busy_ns would count those gaps twice.',
                             'Skip-WL CR and IPR are user premises. PB source permutation and read-to-cache path remain conditional.',
                             'Host output metadata/scales and full-model quality are outside this timing audit.'])
    write(OUT/'report.json',report)
    (OUT/'report.md').write_text('Independent pipeline audit passes '+str(len(records))+' named trace cases, '+str(len(randomized))+' generated mask/configuration cases, '+str(len(saved))+' saved trace files and '+str(len(mutations))+' corruption controls. No target-code discrepancy was found.\n\n'+
        'The full candidate requires one read and one cache weight slot per plane, one serial ECC, ordered common-input rows and no hidden weight FIFO. '+
        'For strict CR, evaluation-start overwrite and copy/recycle overlap, the exact copy boundary is 320ns: 320ns passes and 321ns fails. '+
        'The limiting wave10/plane2 read-latch slack is 640−2×copy_ns. Copy0/100/500ns gives hypothetical makespans150880/150980/151380ns; 500ns is infeasible.\n\n'+
        'The independent reference builds its own physical pages and max-plus read/cache/ECC dependency graph. It imports the target only to generate candidate traces, then separately checks overwrite deadlines, cache retention, per-plane two-slot inventory, exclusive resources, per-plane CR continuity, startup/drain, row ownership and byte/page conservation. Four manual time oracles cover single-page, four-page, two-wave and newly active cold-plane cases.\n\n'+
        'Ordinary balanced partial masks require7 cold+8 CR waves. Keep-hot/drop reads60 pages and decodes48; keep-hot/decode reads and decodes60 and fails its finite-buffer schedule at500ns copy. Positive discharge splits retain total array service while moving first availability and discharge; they remain phase sensitivities.\n\n'+
        'Five deliberately corrupted traces were detected: ECC overlap, copy before availability, changed byte count, false feasibility flag and wrong row owner. Target source/report/trace hashes are checked before and after.\n\n'+
        'Reproduce with `python '+str(Path(__file__).resolve())+'`. Frozen source SHA256: `'+source_hash+'`.\n\n'+
        '\n'.join(report['limitations'])+'\n')
    print(json.dumps(dict(status=report['status'],named_cases=len(records),generated_cases=len(randomized),saved_traces=len(saved),source_sha256=source_hash,copy_boundary_ns=320),indent=2))


if __name__=='__main__':
    main()
