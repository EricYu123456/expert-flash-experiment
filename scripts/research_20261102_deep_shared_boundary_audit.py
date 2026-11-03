#!/usr/bin/env python3
"""Independent arithmetic/range audit of the Deep shared boundary screen."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round2/deep-shared-boundary'
FULL=OUT/'full'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def work(rows,columns):
    rows=np.asarray(rows,dtype=np.int64)
    w=(rows*columns+65535)//65536
    return 28000*w+np.where(rows>0,250+(columns+1)//2+2*rows,0)


def wave(rows,columns):return (np.asarray(rows,dtype=np.int64)*columns+65535)//65536


def chain(gu,down):
    g=work(gu,2048)+(np.asarray(gu)*4+7)//8
    d=work(down,2816)+(np.asarray(down)*4+7)//8
    return 2000+max(0,256-250)+g.max(axis=-1)+2000+max(0,352-250)+int(d.max())


def stats(x):
    return dict(mean=float(np.mean(x)),p99=float(np.percentile(x,99)),min=int(np.min(x)),max=int(np.max(x)))


def inventory(base,gu_stored,down):
    result={k:v.copy() for k,v in base.items()}
    for rows,c in [(gu_stored,2048),(down,2816)]:
        result['useful_bytes']+=np.asarray(rows)*c
        result['actual_pages']+=(np.asarray(rows)*c+16383)//16384
        result['reserved_waves']+=wave(rows,c)
    return result


def main():
    doc=json.loads((FULL/'oracle.json').read_text())
    assert sha(FULL/'source.snapshot.py')==doc['source_sha256']
    source=json.loads(Path(doc['source_plan']).read_text())
    assert sha(doc['source_plan'])==doc['source_plan_sha256']
    assert sha(doc['train_path'])==doc['train_sha256']
    with np.load(doc['train_path']) as f:
        train=f['routes'][:32,:64].copy();ids=f['prompt_ids'][:32].tolist();layers=f['layers'].tolist()
    assert all(p.startswith('train_') for p in ids)
    all_inventory={label:{k:np.zeros(16,dtype=np.int64) for k in ['useful_bytes','actual_pages','reserved_waves']}
        for label in ['onecopy','r0','r1','r2','r3','r11']}
    costs={k:[] for k in all_inventory};nands={k:[] for k in all_inventory}
    manifests=[];certs=[];checked=0
    corrected={}
    for record in doc['records']:
        layer=record['layer'];f=np.load(record['npz'])
        assert sha(record['npz'])==record['npz_sha256']
        selected=train[:,:,layers.index(layer)].reshape(-1,6)
        assert np.array_equal(f['selected'],selected)
        phases={p['name']:p for p in source['phases'] if p['layer']==layer}
        g=np.asarray(phases['gate_up']['chosen_rows']);d=np.asarray(phases['down']['chosen_rows'])
        dn=(work(g,2048)+work(d,1408))[selected].sum(axis=1)
        dw=(wave(g,2048)+wave(d,1408))[selected].sum(axis=1)
        assert np.array_equal(f['routed_native_chip_ns'],dn)
        assert np.array_equal(f['routed_nand_chip_waves'],dw)
        base={k:np.zeros(16,dtype=np.int64) for k in ['useful_bytes','actual_pages','reserved_waves']}
        for rows,c in [(g,2048),(d,1408)]:
            assert np.all(rows.sum(axis=1)==(2816 if c==2048 else 2048))
            base['useful_bytes']+=(rows*c).sum(axis=0)
            base['actual_pages']+=((rows*c+16383)//16384).sum(axis=0)
            base['reserved_waves']+=wave(rows,c).sum(axis=0)
        sg=f['onecopy_gu_rows'];sd=f['onecopy_down_rows']
        assert sg.shape==(16,) and sd.shape==(16,) and sg.sum()==5632 and sd.sum()==2048
        assert np.all(sg%32==0)
        oc=(dn+work(sg,2048)+work(sd,2816)).max(axis=1)+2000
        ch=np.full(len(oc),chain(sg,sd),dtype=np.int64)
        assert np.array_equal(oc,f['onecopy_native_chip_bound_ns'])
        assert np.all(ch==f['onecopy_chain_bound_ns']-500)
        assert np.array_equal(np.maximum(oc,ch),f['onecopy_bound_ns'])
        costs['onecopy'].append(np.maximum(oc,ch))
        nands['onecopy'].append((dw+wave(sg,2048)+wave(sd,2816)).max(axis=1))
        inv=inventory(base,sg,sd)
        for k,v in inv.items():all_inventory['onecopy'][k]+=v
        corrected[f'layer{layer}_onecopy_chain_bound_ns']=ch
        ranges=[];start=0
        for chip,count in enumerate(sg.tolist()):
            if count:ranges.append(dict(chip=chip,logical_start_row=start,rows=count,stored_offset_rows=0))
            start+=count
        assert start==5632
        current=dict(layer=layer,onecopy=dict(gu_rows=sg.tolist(),shared_down_rows=sd.tolist(),gu_ranges=ranges),replicas=[])
        for r in (0,1,2,3,11):
            label='r'+str(r);gr=f[label+'_gu_rows'];sr=f[label+'_down_rows'];splits=f[label+'_split_tiles']
            assert gr.shape==(2048,16) and splits.shape==(2048,8)
            assert np.all(gr.sum(axis=1)==5632) and sr.sum()==2048 and np.all(gr%32==0)
            assert np.all((splits>=11-r)&(splits<=11+r))
            assert np.array_equal(gr[:,:8]//32,splits)
            assert np.array_equal(gr[:,8:]//32,22-splits)
            choices=np.arange(11-r,12+r)
            legal_splits=[];ranges=[]
            pair_native=[];pair_nand=[]
            for p in range(8):
                # Independent full enumeration of available fixed copies.
                costs_a=work(32*choices,2048);costs_b=work(32*(22-choices),2048)
                values=np.maximum(dn[:,p,None]+work(sr[p],2816)+costs_a,
                    dn[:,p+8,None]+work(sr[p+8],2816)+costs_b)
                chosen=np.maximum(dn[:,p]+work(sr[p],2816)+work(gr[:,p],2048),
                    dn[:,p+8]+work(sr[p+8],2816)+work(gr[:,p+8],2048))
                assert np.array_equal(chosen,values.min(axis=1))
                pair_native.append(chosen)
                nv=np.maximum(dw[:,p,None]+wave(sr[p],2816)+choices,
                    dw[:,p+8,None]+wave(sr[p+8],2816)+22-choices)
                pair_nand.append(nv.min(axis=1))
                origin=704*p
                ranges.extend([dict(chip=p,logical_start_row=origin,rows=32*(11+r)),
                    dict(chip=p+8,logical_start_row=origin+32*(11-r),rows=32*(11+r))])
                for a in choices:
                    end=origin+int(a)*32
                    assert origin<=end<=origin+32*(11+r)
                    assert origin+32*(11-r)<=end<=origin+704
                    legal_splits.append(dict(pair=p,split_tiles=int(a),a_chip=p,b_chip=p+8,
                        a_logical_start_row=origin,a_rows=int(a)*32,a_stored_offset_waves=0,
                        b_logical_start_row=end,b_rows=(22-int(a))*32,b_stored_offset_waves=int(a)-(11-r)))
            native_bound=np.stack(pair_native,axis=1).max(axis=1)+2000
            nand_bound=np.stack(pair_nand,axis=1).max(axis=1)
            correct_chain=chain(gr,sr)
            assert np.array_equal(native_bound,f[label+'_native_chip_bound_ns'])
            assert np.array_equal(nand_bound,f[label+'_nand_bound_waves'])
            assert np.all(correct_chain==f[label+'_shared_chain_bound_ns']-500)
            assert np.array_equal(np.maximum(native_bound,correct_chain),f[label+'_occupancy_chain_bound_ns'])
            corrected[f'layer{layer}_{label}_chain_bound_ns']=correct_chain
            costs[label].append(np.maximum(native_bound,correct_chain));nands[label].append(nand_bound)
            inv=inventory(base,np.full(16,32*(11+r),dtype=np.int64),sr)
            for k,v in inv.items():all_inventory[label][k]+=v
            current['replicas'].append(dict(r=r,shared_down_rows=sr.tolist(),stored_gu_ranges=ranges,
                legal_split_manifest=legal_splits,perquery_split_npz_key=label+'_split_tiles'))
            checked+=len(selected)*8
        manifests.append(current)
        certs.extend(record['integer_wave_fit_certificates'])
    totals={k:stats(np.stack(v).sum(axis=0)) for k,v in costs.items()}
    for k in totals:
        totals[k]['mean_gain_vs_fitted_onecopy_pct']=100*(1-totals[k]['mean']/totals['onecopy']['mean'])
        totals[k]['p99_gain_vs_fitted_onecopy_pct']=100*(1-totals[k]['p99']/totals['onecopy']['p99'])
        totals[k]['nand_waves_sum26']=stats(np.stack(nands[k]).sum(axis=0))
    inv={}
    for label,table in all_inventory.items():
        inv[label]={k:v.tolist() for k,v in table.items()}
        inv[label].update(total_useful_bytes=int(table['useful_bytes'].sum()),
            total_actual_pages=int(table['actual_pages'].sum()),
            total_reserved_lsb_bytes=int(table['reserved_waves'].sum()*65536),
            min_chip_useful_bytes=int(table['useful_bytes'].min()),max_chip_useful_bytes=int(table['useful_bytes'].max()),
            max_chip_reserved_lsb_bytes=int(table['reserved_waves'].max()*65536))
    np.savez_compressed(OUT/'corrected_chain_bounds.npz',**corrected)
    (OUT/'executable_row_plans.json').write_text(json.dumps(dict(source_oracle_sha256=sha(FULL/'oracle.json'),
        plans=manifests,notes=['Fixed PPA installation is not implemented here. Contiguous full rows and selected offsets are fully specified.',
            'Start all stored ranges at whole4plane WL boundaries; each individual native job is below192WL. No partial dot products/extra contexts.',
            'Onecopy GU ranges use cumulative chip order. Copy ranges use fixed704row pairs with owners(p,p+8). Output gather maps logical rows.']),indent=2)+'\n')
    report=dict(status='PASS all26layers train32x64; independent full pair enumeration/range/accounting audit',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        original_oracle_sha256=sha(FULL/'oracle.json'),original_executed_source_sha256=sha(FULL/'source.snapshot.py'),
        corrected_main_source_sha256=sha(ROOT/'scripts/research_20261102_deep_shared_boundary_oracle.py'),
        checked_pair_queries=checked,shape_receipt=doc['shape_receipt'],
        chain_correction=dict(original_overcount_ns=500,reason='Each native command overlaps first250ns of its PCIe vector upload; twice for GU/down.',
            original_winner_chain_never_binding=True,changed_costs_or_splits_or_row_plans=False,
            corrected_npz=str(OUT/'corrected_chain_bounds.npz'),corrected_npz_sha256=sha(OUT/'corrected_chain_bounds.npz')),
        totals_ns=totals,inventory=inv,executable_row_plans=str(OUT/'executable_row_plans.json'),
        executable_row_plans_sha256=sha(OUT/'executable_row_plans.json'),
        static_nand_milp_certificate_summary={str(s):dict(
            mean_incumbent_waves=float(np.mean([c['incumbent_mean_waves'] for c in certs if c['total_shared_waves']==s])),
            mean_dual_lower_bound_waves=float(np.mean([c['dual_lower_bound_mean_waves'] for c in certs if c['total_shared_waves']==s])),
            mean_absolute_gap_waves=float(np.mean([c['absolute_gap_waves'] for c in certs if c['total_shared_waves']==s])),
            max_absolute_gap_waves=float(max(c['absolute_gap_waves'] for c in certs if c['total_shared_waves']==s))) for s in (264,265)},
        limitations=['No optimal full DAG schedules inferred from these necessary bounds.',
            'Onecopy native objective is strong constructive control, not a globally certified optimum; general NAND MILP duals are explicit.',
            'Exact scalar work/depth/jobs frontier alone cannot dominate a vector load profile. Balanced16active89wave and equal96wave references were retained.',
            'Per-chip storage needs are reported; no fabricated equal-per-chip quota or20GiB feasibility limit. Block fragmentation/firmware installation unmodeled.'])
    path=OUT/'audit.json';path.write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'audit_completion.json').write_text(json.dumps(dict(status='complete',audit_sha256=sha(path),
        row_plans_sha256=sha(OUT/'executable_row_plans.json'),source_sha256=sha(__file__)),indent=2)+'\n')
    print(json.dumps(dict(audit_sha256=sha(path),totals_ns=totals),indent=2))


if __name__=='__main__':main()
