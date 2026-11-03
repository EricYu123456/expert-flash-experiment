#!/usr/bin/env python3
"""Independent parser/scalar check of the new context-work bounds."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'native-mechanism-next'
SRC=BASE/'ordinary-read-compiler/full-campaign-v3'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    a=np.load(OUT/'qwen38_shared_copy_oracles.npz'); counts=0; jobs_checked=0
    for layer in range(48):
        z=np.load(SRC/f'qwen38.L{layer}.cmd250/whole_rows.npz')
        jobs=z['jobs'].astype(np.int64); offsets=z['offsets']
        costs=np.zeros((513,16),dtype=np.int64); shared=[]
        assert np.all(jobs[:,3]==(jobs[:,1]*jobs[:,2]+16383)//16384)
        assert np.all(jobs[:,4]==(jobs[:,3]+3)//4)
        for expert in range(513):
            for phase in range(2):
                lo,hi=offsets[expert,phase]; part=jobs[lo:hi]
                assert part[:,1].sum()==[1280,2560][phase]
                for chip,rows,columns,pages,waves in part:
                    duration=28000*waves+250+(columns+1)//2+2*(pages*16384//columns)
                    costs[expert,chip]+=duration; jobs_checked+=1
                    if expert==512:shared.append((phase,int(chip),int(duration)))
        routed=costs[z['selection_routes']].sum(1)
        assert np.array_equal(a[f'L{layer}.onecopy_work'],2000+(routed+costs[512]).max(1))
        for name,phases in [('paired_shared_gu',{0}),('paired_shared_down',{1}),('paired_shared_both',{0,1})]:
            expected=[]
            for query in range(64):
                peak=0
                for chip in range(8):
                    fixed=[int(routed[query,chip]),int(routed[query,chip+8])]; movable=[]
                    for phase,owner,duration in shared:
                        if owner not in (chip,chip+8):continue
                        if phase in phases:movable.append(duration)
                        else:fixed[int(owner==chip+8)]+=duration
                    optimum=10**18
                    for bits in range(1<<len(movable)):
                        load=fixed.copy()
                        for j,duration in enumerate(movable):load[(bits>>j)&1]+=duration
                        optimum=min(optimum,max(load))
                    peak=max(peak,optimum)
                expected.append(2000+peak);counts+=1
            assert np.array_equal(a[f'L{layer}.{name}'],expected)
    for name in ['observed','onecopy_work','paired_shared_gu','paired_shared_down','paired_shared_both',
                 'shared_unlimited_fractional','all_jobs_unlimited_fractional']:
        assert np.array_equal(a['total.'+name],sum(a[f'L{layer}.'+name] for layer in range(48)))
    path=SRC/'summary.json'; d=json.loads(path.read_text())
    records=[x for x in d['records'] if x['model']=='qwen38' and x['command_ns']==250]
    assert len(records)==48
    assert all(x['selection_training_indices']==records[0]['selection_training_indices'] for x in records)
    assert all(x['training']['selected_indices']==records[0]['training']['selected_indices'] for x in records)
    quad=np.load(OUT/'qwen38_shared_quad_oracle.npz')
    assert np.array_equal(quad['observed'],sum(a[f'L{layer}.observed'] for layer in range(3)))
    protocol=BASE/'full-ffn-integer-reference/frozen_protocol.json'
    audit=dict(status='PASS independent NPZ-native-jobs parser and scalar finite enumeration',
        source_sha256=sha(__file__), shared_pair_query_policy_checks=counts,
        native_jobs=jobs_checked, queries=64, layers=48, all_layer_pairing_equal=True,
        summary_sha256=sha(path), training_selection_indices=records[0]['selection_training_indices'],
        training_source_receipt=records[0]['training'],
        independent_numeric_reference_protocol_sha256=sha(protocol), no_source_mutation=True,
        limits='Context-work lower bounds only; no dependency schedule or host selection time verified.')
    (OUT/'independent_bound_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['training_selection_indices','training_source_receipt']}))

if __name__=='__main__':main()
