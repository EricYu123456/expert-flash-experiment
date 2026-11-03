#!/usr/bin/env python3
"""Replay root's frozen joint-work mappings with the sealed native-job model.

No fitting, new contexts, partial-column operators or hidden input residency.
Zero-activation source cases reuse byte-identical sealed per-query schedules.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from materialize_plan import WaveAllocator
from research_20261102_granite4_pipeline import build, profiles, stats, SOURCE, EXPECTED, RESOURCE_NAMES
from run_corpus_traces import ROOT, sha256

DEST=ROOT/'results/research-2026-11-02/round2/pipeline-joint-replay'
JOINT=ROOT/'results/research-2026-11-02/round2/pipeline-joint/joint_work_screen.json'
SEALED=ROOT/'results/research-2026-11-02/round2/granite4-pipeline'


def rows_from_labels(labels,R):
    assert R%12==0
    rows=np.full((len(labels),16),R//12,dtype=np.int64)
    for e,g in enumerate(labels):rows[e,int(g)*4:int(g)*4+4]=0
    assert np.all(rows.sum(axis=1)==R)
    return rows


def inventory(doc,replacements):
    alloc=[WaveAllocator() for _ in range(16)]
    for p in doc['phases']:
        rows=replacements.get((p['layer'],p['name']),np.asarray(p['rows']))
        assert np.all(rows.sum(axis=1)==p['total_rows'])
        for e in range(doc['experts']):
            for c,r in enumerate(rows[e]):
                if r:alloc[c].allocate(int(r)*p['columns'])
    chips=[a.summary(c) for c,a in enumerate(alloc)]
    return dict(chips=chips,total_useful_bytes=sum(c['useful_int8_bytes'] for c in chips),
        total_physical_pages=sum(c['physical_pages'] for c in chips),
        total_stored_read_waves=sum(c['read_rounds'] for c in chips),
        maximum_required_lsb_bytes_per_chip=max(c['reserved_lsb_bytes'] for c in chips),
        capacity_interpretation='These are physical storage requirements for the routed weights, not a disclosed device capacity. No arbitrary per-chip cap or equal-inventory quota is imposed. Other tensors/ECC/spares require additional capacity.')


def paired(a,b):
    delta=a-b
    return dict(candidate_minus_reference_mean_ns=float(delta.mean()),mean_gain_pct=100*(1-float(a.mean())/float(b.mean())),
        candidate_faster_queries=int((delta<0).sum()),equal_queries=int((delta==0).sum()),candidate_slower_queries=int((delta>0).sum()),
        paired_delta_p05_ns=float(np.percentile(delta,5)),paired_delta_p50_ns=float(np.percentile(delta,50)),
        paired_delta_p95_ns=float(np.percentile(delta,95)),minimum_delta_ns=int(delta.min()),maximum_delta_ns=int(delta.max()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variants',nargs='+',default=['source','matched_phase_best','joint_best'],
        choices=['source','matched_phase_best','joint_best','g8_opposite'])
    ap.add_argument('--priorities',nargs='+',default=['down_first','gate_first'],choices=['down_first','gate_first'])
    ap.add_argument('--samples',type=int,default=0);ap.add_argument('--activation-ns',type=int,default=0)
    ap.add_argument('--combine-per-expert-ns',type=int,default=0);ap.add_argument('--output',type=Path,default=DEST/'zero')
    args=ap.parse_args();assert 'source' in args.variants
    args.output.mkdir(parents=True,exist_ok=True);seal=json.loads((SEALED/'seal.json').read_text())
    for rel,digest in seal['artifacts'].items():assert sha256(ROOT/rel)==digest,(rel,digest)
    assert sha256(SOURCE)==EXPECTED;doc=json.loads(SOURCE.read_text());j=json.loads(JOINT.read_text());joint_sha=sha256(JOINT)
    assert j['source_plan_sha256']==EXPECTED and j['training_routes_sha256']==doc['training_routes_sha256']
    route=Path(doc['training_routes']);assert sha256(route)==doc['training_routes_sha256']
    with np.load(route) as f:
        routes=f['routes'][:32,:64].copy();layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
    assert all(s.startswith('train_') for s in ids) and len(ids)==32
    indices=np.arange(2048) if args.samples==0 else np.unique(np.linspace(0,2047,args.samples,dtype=int))
    with np.load(SEALED/'schedule_costs.npz') as f:cached={k:f[k].copy() for k in f.files}
    np.testing.assert_array_equal(cached['indices'],np.arange(2048))
    replacements={v:{} for v in args.variants};profiles_by_layer={};layer_info=[]
    for q in j['results']:
        layer=q['layer'];ps=[p for p in doc['phases'] if p['layer']==layer];assert len(ps)==2
        gu=profiles(ps[0]['rows'],ps[0]['columns']);R=ps[1]['total_rows'];C=ps[1]['columns']
        home=(gu['rows'][:,0]>0).astype(int);np.testing.assert_array_equal(home,q['home_a'])
        original=rows_from_labels(q['original_labels'],R);np.testing.assert_array_equal(original,ps[1]['rows'])
        all_down={'source':original,'matched_phase_best':rows_from_labels(q['matched_phase_best']['labels'],R),
            'joint_best':rows_from_labels(q['joint_best']['labels'],R)}
        opposite=np.zeros((doc['experts'],16),dtype=np.int64)
        for e,h in enumerate(home):opposite[e,8:16] = R//8 if h else 0;opposite[e,0:8] = R//8 if not h else 0
        all_down['g8_opposite']=opposite
        selected=routes[:,:,layers.index(layer)].reshape(-1,doc['active']);gw=gu['waves'][selected].sum(axis=1)
        info=dict(layer=layer,variants={});profiles_by_layer[layer]={}
        for v in args.variants:
            down=profiles(all_down[v],C);profiles_by_layer[layer][v]=(gu,down);replacements[v][(layer,'down')]=all_down[v]
            dw=down['waves'][selected].sum(axis=1)
            info['variants'][v]=dict(down_group_counts=np.bincount(q['original_labels'] if v=='source' else q[v]['labels'],minlength=4).tolist() if v!='g8_opposite' else None,
                down_rows=all_down[v].tolist(),down_pages_per_expert=down['pages'].sum(axis=1).tolist(),
                down_stored_waves_per_expert=down['waves'].sum(axis=1).tolist(),
                total_gu_plus_down_stored_waves_per_expert=(gu['waves']+down['waves']).sum(axis=1).tolist(),
                all_train_serial_nand=stats((gw.max(axis=1)+dw.max(axis=1))*28000),
                all_train_joined_chip_nand_bound=stats((gw+dw).max(axis=1)*28000))
        layer_info.append(info)
    physical={v:inventory(doc,r) for v,r in replacements.items()}
    for v in args.variants:
        assert physical[v]['total_useful_bytes']==physical['source']['total_useful_bytes']
        assert physical[v]['total_physical_pages']==physical['source']['total_physical_pages']
    costs={};records=[];began=time.monotonic()
    for li,q in enumerate(j['results']):
        layer=q['layer'];selected=routes[:,:,layers.index(layer)].reshape(-1,doc['active']);result={}
        for v in args.variants:
            gu,down=profiles_by_layer[layer][v]
            for barrier in [True,False]:
                graph_cache={}
                for priority in args.priorities:
                    name=v+('__barrier' if barrier else '__ready')+'__'+priority
                    if v=='source' and args.activation_ns==args.combine_per_expert_ns==0:
                        source_name='frozen_g12'+('__barrier' if barrier else '__ready')+'__'+priority
                        values=cached[source_name][li,indices];origin='exact sealed per-query source cache'
                    else:
                        values=[];cache={}
                        for ix,idx in enumerate(indices):
                            s=selected[idx];key=tuple(s.tolist())
                            if key not in cache:
                                if key not in graph_cache:graph_cache[key]=build(s,gu,down,barrier,args.activation_ns,args.combine_per_expert_ns*doc['active'])
                                rr=graph_cache[key].run(priority=='down_first',trace=(ix==0))
                                cache[key]=rr['makespan_ns']
                                if ix==0:(args.output/f'layer{layer}.{name}.trace.json').write_text(json.dumps(dict(selected=s.tolist(),**rr),indent=2)+'\n')
                            values.append(cache[key])
                        values=np.asarray(values,dtype=np.int64);origin='sealed strict native-job Graph runtime audit'
                    costs.setdefault(name,[]).append(values);result[name]=dict(**stats(values),origin=origin)
                    print('layer',layer,name,float(values.mean()),'elapsed',round(time.monotonic()-began,1),flush=True)
        records.append(dict(layer=layer,schedules=result))
        (args.output/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
    totals={k:np.stack(v).sum(axis=0) for k,v in costs.items()};comparisons={}
    for barrier in [True,False]:
        for priority in args.priorities:
            suffix=('__barrier' if barrier else '__ready')+'__'+priority
            for v in args.variants:
                if v=='source':continue
                comparisons[v+'_vs_source'+suffix]=paired(totals[v+suffix],totals['source'+suffix])
            if 'joint_best' in args.variants and 'matched_phase_best' in args.variants:
                comparisons['joint_best_vs_matched_phase_best'+suffix]=paired(totals['joint_best'+suffix],totals['matched_phase_best'+suffix])
    for rel,digest in seal['artifacts'].items():assert sha256(ROOT/rel)==digest
    assert sha256(JOINT)==joint_sha and sha256(SOURCE)==EXPECTED
    npz=args.output/'schedule_costs.npz';np.savez_compressed(npz,indices=indices,**{k:np.stack(v) for k,v in costs.items()})
    out=dict(status='train-only replay; no new mapping fit or heldout',created_utc=datetime.now(timezone.utc).isoformat(),
        root_joint_source=str(JOINT),root_joint_source_sha256=joint_sha,frozen_plan_sha256=EXPECTED,
        source_training_routes=str(route),training_routes_sha256=sha256(route),prompt_ids=ids,training_slice='[:32,:64]',
        scheduled_flat_indices=indices.tolist(),arguments={**vars(args),'output':str(args.output)},
        physical_inventory=physical,all_train_geometry=layer_info,layer_schedules=records,
        total_schedules={k:stats(v) for k,v in totals.items()},paired_comparisons=comparisons,
        sealed_sources_verified_before_and_after=True,schedule_costs_sha256=sha256(npz),
        source_sha256=sha256(Path(__file__)),elapsed_seconds=time.monotonic()-began,
        qualifications=['All paired vectors use identical ordered route, priority, host/control/transfer fees and chip context ownership.',
            'Every native matrix reloads input and owns chip from command start to output channel drain. Down waits for all own-expert GU host outputs and activation.',
            'Necessary NAND bounds are stored separately from achieved feasible event schedules. No global optimal scheduler or measured end-to-end result.',
            'Activation sensitivity is per expert; final combine sensitivity is also per expert and summed into one final host combine.',
            'g8 opposite stores the same useful bytes/pages but down stored waves/expert increase12to16; combined GU/down36to40. No replication or arbitrary chip capacity quota.',
            'Service model inheritsA8 input/FP32 output byte costs and28us ordinary read. Native PE throughput and physical timing remain uncalibrated.'])
    target=args.output/'screen.json';target.write_text(json.dumps(out,indent=2)+'\n')
    (args.output/'completion.json').write_text(json.dumps(dict(status='complete',screen_sha256=sha256(target),source_sha256=sha256(Path(__file__)),
        schedule_costs_sha256=sha256(npz),paired_comparisons=comparisons),indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
