#!/usr/bin/env python3
"""Independent train-route and address/capacity audit; never opens heldout."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from materialize_plan import WaveAllocator
from research_20261102_mixed_columns import DEST, FAMILIES, components, scalar
from run_corpus_traces import sha256


def main():
    screen=DEST/'screen.json';d=json.loads(screen.read_text());records=[]
    for m in d['models']:
        source=Path(m['source_plan']);assert sha256(source)==m['source_plan_sha256']
        base=json.loads(source.read_text());route=Path(m['training_routes']);assert sha256(route)==m['training_routes_sha256']
        with np.load(route) as f:
            routes=f['routes'][:32,:64];layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
        assert all(s.startswith('train_') for s in ids)
        for family in FAMILIES:
            alloc=[WaveAllocator() for _ in range(16)];cases=0;fragments=0
            replacements={(p['layer'],p['name']):p for p in m['phases']}
            for p in base['whole_model_rows']:
                key=(p['layer'],p['name']);C=p['columns']
                if key in replacements:
                    q=replacements[key];R=q['rows'];g=np.asarray(q['winners'][family]['g'],dtype=np.int32)
                    selected=routes[:,:,layers.index(q['layer'])].reshape(-1,base['active'])
                    vector=components(selected,g,R,C,q['common'])
                    for i,s in enumerate(selected):
                        for k,v in scalar(s,g,R,C,q['common']).items():assert vector[k][i]==v
                    cases+=len(selected)
                    assert R%8==C%3==0
                    for e in range(base['experts']):
                        bytes_e=pages_e=0
                        for c in range(16):
                            rowblock=c%8;width=(int(g[e]) if c<8 else 3-int(g[e]))*(C//3)
                            if width==0:continue
                            size=(R//8)*width;assert size%65536==0
                            pages,waves,runs,segments=alloc[c].allocate(size)
                            assert pages==size//16384 and waves==size//65536 and runs==1
                            assert len(segments)==1 and segments[0][1]+waves<=192
                            # For each rowblock, the first-half columns [0,gC/3)
                            # and second-half columns [gC/3,C) partition all columns.
                            assert 0<=rowblock<8 and 0<=g[e]<=3
                            bytes_e+=size;pages_e+=pages;fragments+=1
                        assert bytes_e==R*C and pages_e==R*C//16384
                else:
                    for row in p['rows']:
                        for c,r in enumerate(row):
                            if r:alloc[c].allocate(int(r)*C)
            chips=[a.summary(c) for c,a in enumerate(alloc)]
            useful=sum(c['useful_int8_bytes'] for c in chips);pages=sum(c['physical_pages'] for c in chips)
            original=base['updated_whole_model_inventory']
            assert useful==original['total_useful_bytes'] and pages==original['total_physical_pages']
            maximum=max(c['reserved_lsb_bytes'] for c in chips);assert maximum<20*(1<<30)
            records.append(dict(model=m['model'],family=family,all_training_route_ledger_checks=cases,
                replacement_rectangles=fragments,whole_model_useful_bytes=useful,whole_model_pages=pages,
                max_reserved_lsb_bytes_per_chip=maximum,physical_cap_bytes=20*(1<<30),same_total_bytes_pages=True,
                reserved_bytes_and_per_chip_inventory_not_required_equal=True,chips=chips))
        for p in m['phases']:
            K=base['active'];R=p['rows'];C=p['columns'];assert K%2==0
            oracle=28000*(3*K//2)+(2000+R//2)*K+2000+K*C//2
            p['optimistic_query_dependent_shape_oracle_ns']=oracle
            p['oracle_qualification']='Each query may choose its own expert shapes:K/2 whole matrices on each half, no partial outputs. Infeasible as a single static placement in general; only a lower bound.'
    result=dict(status='PASS under conditional FP32-output service contract',created_utc=datetime.now(timezone.utc).isoformat(),
        screen_sha256=sha256(screen),records=records,oracle_records=[dict(model=m['model'],layer=p['layer'],
            query_dependent_oracle_ns=p['optimistic_query_dependent_shape_oracle_ns'],
            best_fixed_mean_ns=p['winners'][p['best_fixed_family']]['cost']['mean_ns']) for m in d['models'] for p in m['phases']],
        method='All32x64 routes checked by independent literal chip/rectangle ledger. Whole-model WaveAllocator allocation checks disjoint sequential intervals, legal4-plane waves and192WLblock fits. Unchanged rows and disjoint column ranges prove one copy of every element.',
        limitations=['Native output/dequantization contract, host summation, actual PE throughput and physical timing remain unverified.',
            'Capacity assumes20GiB/chip and only routed weights; unrelated tensors/ECC/spares must fit separately.',
            'Same useful bytes/pages does not imply same per-chip inventory or reserved space. No heldout opened.'],
        source_sha256=sha256(Path(__file__)))
    (DEST/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    completion=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='complete; mixed-shape search adds zero beyond shared fixed-family controls; no expansion',
        artifacts={p.name:sha256(p) for p in [screen,DEST/'audit.json',Path(__file__),Path(__file__).with_name('research_20261102_mixed_columns.py'),Path(__file__).with_name('research_20261102_mixed_columns.cpp')]},
        model_results=[{k:m[k] for k in ['model','routed_ffn_screen_total_ns','whole_ffn_mixed_gain_beyond_fixed_pct','whole_ffn_mixed_gain_vs_legacy_excluding_host_sum_pct','aggregate_mean_host_sum_budget_vs_legacy_ns']} for m in d['models']])
    (DEST/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion,indent=2))


if __name__=='__main__':main()
