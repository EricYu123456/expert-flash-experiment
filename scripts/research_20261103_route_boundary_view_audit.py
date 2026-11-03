#!/usr/bin/env python3
"""Independent complete-row prefix identity/physical ledger, not native hardware."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,itertools,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
DEST=BASE/'route-boundary-view-audit'
P=16384;PLANES=4;B=P*PLANES;WL=192
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def ceil(n,d):return (int(n)+d-1)//d
def compositions(R,M):
    if M==1:yield (R,);return
    for r in range(R+1):
        for tail in compositions(R-r,M-1):yield (r,)+tail

def construct(A,Bv):
    assert len(A)==len(Bv) and sum(A)==sum(Bv)
    source=[];cursor=0;common=[];packet=[]
    for c,(a,b) in enumerate(zip(A,Bv)):
        source.append(list(range(cursor,cursor+a)));cursor+=a
        n=min(a,b);common.append(source[-1][:n])
        if a>b:packet.extend(source[-1][n:])
    physical=[];cursor=0
    for c,(a,b) in enumerate(zip(A,Bv)):
        if a>=b:physical.append(source[c].copy())
        else:
            physical.append(common[c]+packet[cursor:cursor+b-a]);cursor+=b-a
    assert cursor==len(packet)
    assert sum(map(len,physical))==sum(A)+sum(abs(a-b) for a,b in zip(A,Bv))//2
    for view in (A,Bv):
        selected=[row for c,n in enumerate(view) for row in physical[c][:n]]
        assert Counter(selected)==Counter(range(sum(A)))
    # A row order is preserved verbatim; only logical B ownership is derived.
    assert all(physical[c][:n]==source[c] for c,n in enumerate(A))
    return physical

def compress(rows):
    bands=[]
    for row in rows:
        if bands and bands[-1][0]+bands[-1][1]==row:bands[-1][1]+=1
        else:bands.append([row,1])
    return bands

def ledger(A,Bv,C,detail=False):
    physical=construct(A,Bv);M=len(A);R=sum(A);delta=sum(abs(a-b) for a,b in zip(A,Bv))//2
    def stats(v):
        pages=[ceil(r*C,P) for r in v];waves=[ceil(p,PLANES) for p in pages]
        return dict(rows=list(v),commands=sum(r>0 for r in v),pages=pages,waves=waves,
                    total_pages=sum(pages),total_waves=sum(waves),useful_bytes=R*C,
                    all_page_complete_outputs=sum(p*P//C for p in pages))
    sa=stats(A);sb=stats(Bv);maxrows=list(map(max,zip(A,Bv)));envelope=stats(maxrows)
    envelope['useful_bytes']=sum(maxrows)*C
    assert max(envelope['waves'])<=WL
    slots=Counter(row for chip in physical for row in chip)
    assert len(slots)==R and sum(x==2 for x in slots.values())==delta and max(slots.values())<=2
    out=dict(rows=R,columns=C,A=sa,B=sb,delta_rows=delta,
             extra_coefficient_bytes=delta*C,extra_host_scale_bytes_if_per_copy=delta*4,
             union_coefficient_bytes=sum(maxrows)*C,
             extra_programmed_page_bytes=(envelope['total_pages']-sa['total_pages'])*P,
             extra_reserved_wave_bytes=(envelope['total_waves']-sa['total_waves'])*B,
             envelope=envelope,one_block_per_chip=True,exact_two_view_row_coverage=True,
             duplicated_logical_rows=[row for row,count in slots.items() if count==2],
             host_scale_note='Logical host rowScale vectors can be shared by rowID;delta*4 is optional duplicate per-copy metadata, not coefficient-page bytes.')
    if detail:
        chips=[]
        for c,rows in enumerate(physical):
            info=dict(chip=c,block=0,wordline_start=0,plane_start=0,
                      resident_row_id_bands=compress(rows),resident_rows=len(rows),
                      resident_pages=ceil(len(rows)*C,P),resident_waves=ceil(ceil(len(rows)*C,P),PLANES),views=[])
            for label,n in [('A',A[c]),('B',Bv[c])]:
                pages=ceil(n*C,P);readbytes=pages*P;emitted=readbytes//C
                valid_extra=rows[n:min(emitted,len(rows))]
                info['views'].append(dict(name=label,valid_rows=n,pages=pages,waves=ceil(pages,PLANES),
                    valid_logical_row_id_bands=compress(rows[:n]),
                    all_page_complete_outputs=emitted,drop_complete_output_rows=emitted-n,
                    drop_existing_append_row_ids=valid_extra,
                    drop_zero_padding_complete_rows=max(0,emitted-len(rows)),
                    trailing_partial_row_bytes=readbytes%C,
                    trailing_partial_row_is_stored_weight=emitted<len(rows) and readbytes%C>0,
                    physical_read_byte_range=[0,readbytes],logical_prefix_byte_range=[0,n*C]))
            chips.append(info)
        out['chips']=chips
    return out

def main():
    assert not DEST.exists(),'Never overwrite an executed audit.'
    DEST.mkdir(parents=True)
    files=[BASE/'ordinary-read-compiler/full-campaign-v4-deduplicated'/f'{m}.L{l}.cmd250/whole_rows.plan.json' for m in ('qwen38','granite4tiny') for l in range(3)]
    write(DEST/'protocol.frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),inputs_sha256={str(p):sha(p) for p in files+[ROOT/'paper/aif.txt']},
        main_variant='One source s with A[s]>A[s xor8] and differing wave counts. B exchanges only those two rowcounts;other14chips unchanged. No routes or heldout inputs are read.',
        secondary_control='Whole-vector xor8 is separately labeled and never conflated with the single-edge variant.',
        tests='Exhaust all nonnegative equal-sum A/B vectors,R0..7,M1..4;construct symbolic logical row IDs and verify both complete prefixes. Inspect actual three-layer frozen v4 geometry ledgers,not numerical W8v5winner claims.'))
    exhaustive=0
    for M in range(1,5):
        for R in range(8):
            values=list(compositions(R,M))
            for a in values:
                for b in values:construct(a,b);exhaustive+=1
    records=[];examples={};secondary=[]
    for p in files:
        doc=json.loads(p.read_text());model=doc['model'];layer=doc['layer'];logical=doc['logical']
        for expert in range(doc['shape'][2]):
            rows=[x for x in logical if x['expert']==expert and x['phase']==0]
            assert len(rows)==16 and len({x['chip'] for x in rows})==16
            A=[next(x['rows'] for x in rows if x['chip']==c) for c in range(16)]
            C=rows[0]['columns'];assert all(x['columns']==C for x in rows)
            for s in range(16):
                t=s^8
                if A[s]<=A[t] or ceil(A[s]*C,B)==ceil(A[t]*C,B):continue
                target=A.copy();target[s],target[t]=A[t],A[s]
                detail=(layer==0 and expert==0)
                x=ledger(A,target,C,detail)
                assert x['delta_rows']==A[s]-A[t]
                assert x['A']['commands']==x['B']['commands']==16
                assert x['A']['total_pages']==x['B']['total_pages']
                assert x['A']['total_waves']==x['B']['total_waves']
                records.append(dict(model=model,layer=layer,expert=expert,source=s,target=t,delta_rows=x['delta_rows'],
                    extra_coefficient_bytes=x['extra_coefficient_bytes'],extra_programmed_page_bytes=x['extra_programmed_page_bytes'],
                    extra_reserved_wave_bytes=x['extra_reserved_wave_bytes'],commands_per_view=x['A']['commands'],pages_per_view=x['A']['total_pages'],waves_per_view=x['A']['total_waves']))
                if detail:examples[f'{model}.s{s}.t{t}']=x
            if layer==0 and expert==0:
                x=ledger(A,[A[c^8] for c in range(16)],C,True)
                secondary.append(dict(model=model,scope='Full-vector xor8,not main variant',certificate=x))
    # Canonical shape witnesses also cover the Q38 101-vs76 case even when the
    # inspected fixed A template has no high/xor8-low76 pair.
    A=[101]*3+[76]*2+[75]*11
    for label,s,t in [('qwen38_101_75_singleedge',0,8),('qwen38_101_76_singleedge_samechannel_reordered',0,8)]:
        if '76_' in label:A=[101]*3+[75]*5+[76]*2+[75]*6
        target=A.copy();target[s],target[t]=A[t],A[s]
        examples[label]=ledger(A,target,2560,True)
    result=dict(status='PASS symbolic complete-row prefix construction and arithmetic',
        exhaustive_equal_sum_vector_pairs=exhaustive,actual_single_edge_certificates=len(records),
        source_files=len(files),models=['qwen38','granite4tiny'],layers=[0,1,2],
        fixed_logical_B_counterexample=dict(A_counts=[1,1],B_counts=[1,1],
            A_rowsets=[[0],[1]],B_rowsets=[[1],[0]],count_L1_lower_bound=0,
            actual_extra_rows_required_for_fixed_sets=2,
            conclusion='The count formula is tight only when B logical ownership is derived/flexible. Arbitrary fixed A/B row sets need sum union sizes,and equal-length differing prefixes cannot share one base prefix.'),
        records=records,examples=examples,secondary_full_xor8_control=secondary)
    write(DEST/'construction_and_ledger.json',result)
    write(DEST/'completion.construction.json',dict(status='complete',outputs_sha256={str(p):sha(p) for p in sorted(DEST.iterdir()) if p.is_file()},source_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='PASS',exhaustive_pairs=exhaustive,actual_single_edge_certificates=len(records),ledger_sha256=sha(DEST/'construction_and_ledger.json')),indent=2))

if __name__=='__main__':main()
