#!/usr/bin/env python3
"""Replay frozen GU wave rotations, with unchanged frozen g12 down mappings."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np

from research_20261102_granite4_pipeline import build,profiles,stats,SOURCE,EXPECTED
from research_20261102_pipeline_joint_replay import inventory,paired,SEALED
from run_corpus_traces import ROOT,sha256

DEST=ROOT/'results/research-2026-11-02/round2/pipeline-gu-screen'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=32)
    ap.add_argument('--activation-ns',type=int,default=0);ap.add_argument('--combine-per-expert-ns',type=int,default=0)
    ap.add_argument('--priorities',nargs='+',default=['down_first','gate_first'],choices=['down_first','gate_first'])
    ap.add_argument('--output',type=Path,default=DEST/'smoke');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert sha256(SOURCE)==EXPECTED;doc=json.loads(SOURCE.read_text());seal=json.loads((SEALED/'seal.json').read_text())
    for rel,h in seal['artifacts'].items():assert sha256(ROOT/rel)==h
    route=Path(doc['training_routes']);assert sha256(route)==doc['training_routes_sha256']
    with np.load(route) as f:routes=f['routes'][:32,:64].copy();ids=f['prompt_ids'][:32].tolist();layers=f['layers'].tolist()
    assert len(ids)==32 and all(s.startswith('train_') for s in ids)
    indices=np.arange(2048) if args.samples==0 else np.unique(np.linspace(0,2047,args.samples,dtype=int))
    with np.load(SEALED/'schedule_costs.npz') as f:cached={k:f[k].copy() for k in f.files}
    costs={};geometry=[];replacements={};began=time.monotonic();records=[]
    for li,layer in enumerate(layers[:3]):
        ps=[p for p in doc['phases'] if p['layer']==layer];s=routes[:,:,layers.index(layer)].reshape(-1,doc['active'])
        candidate=next(c for c in ps[0]['candidates'] if c['label']=='wave_rotations')
        new_rows=np.asarray(candidate['rows']);replacements[(layer,'gate_up')]=new_rows
        gu=profiles(new_rows,ps[0]['columns']);down=profiles(ps[1]['rows'],ps[1]['columns'])
        assert np.all(new_rows.sum(axis=1)==ps[0]['total_rows']) and np.all(gu['waves'].sum(axis=1)==25) and gu['waves'].max()==2
        gw=gu['waves'][s].sum(axis=1);dw=down['waves'][s].sum(axis=1)
        geometry.append(dict(layer=layer,rows=sorted(np.unique(new_rows).tolist()),gu_pages_per_expert=np.unique(gu['pages'].sum(axis=1)).tolist(),
            gu_waves_per_expert=25,down_waves_per_expert=12,combined_waves_per_expert=37,
            gu_commands_per_expert=16,source_gu_commands_per_expert=8,
            serial_nand=stats((gw.max(axis=1)+dw.max(axis=1))*28000),joined_nand_bound=stats((gw+dw).max(axis=1)*28000)))
        result={}
        for variant in ['source','gu_wave_rotations']:
            gp=profiles(ps[0]['rows'],ps[0]['columns']) if variant=='source' else gu
            for barrier in [True,False]:
                graphs={}
                for priority in args.priorities:
                    name=variant+('__barrier' if barrier else '__ready')+'__'+priority
                    if variant=='source' and args.activation_ns==args.combine_per_expert_ns==0:
                        ck='frozen_g12'+('__barrier' if barrier else '__ready')+'__'+priority;values=cached[ck][li,indices]
                    else:
                        cache={};values=[]
                        for ix,idx in enumerate(indices):
                            selected=s[idx];key=tuple(selected.tolist())
                            if key not in cache:
                                if key not in graphs:graphs[key]=build(selected,gp,down,barrier,args.activation_ns,args.combine_per_expert_ns*doc['active'])
                                rr=graphs[key].run(priority=='down_first',trace=ix==0);cache[key]=rr['makespan_ns']
                                if ix==0:(args.output/f'layer{layer}.{name}.trace.json').write_text(json.dumps(dict(selected=selected.tolist(),**rr),indent=2)+'\n')
                            values.append(cache[key])
                        values=np.asarray(values,dtype=np.int64)
                    costs.setdefault(name,[]).append(values);result[name]=stats(values)
                    print(layer,name,float(values.mean()),'elapsed',round(time.monotonic()-began,1),flush=True)
        records.append(dict(layer=layer,schedules=result))
    totals={k:np.stack(v).sum(axis=0) for k,v in costs.items()};comparisons={}
    for barrier in [True,False]:
        for priority in args.priorities:
            suffix=('__barrier' if barrier else '__ready')+'__'+priority
            comparisons['gu_wave_rotations_vs_source'+suffix]=paired(totals['gu_wave_rotations'+suffix],totals['source'+suffix])
    physical={'source':inventory(doc,{}),'gu_wave_rotations':inventory(doc,replacements)}
    assert physical['source']['total_useful_bytes']==physical['gu_wave_rotations']['total_useful_bytes']
    for rel,h in seal['artifacts'].items():assert sha256(ROOT/rel)==h
    assert sha256(SOURCE)==EXPECTED
    npz=args.output/'schedule_costs.npz';np.savez_compressed(npz,indices=indices,**{k:np.stack(v) for k,v in costs.items()})
    out=dict(status='train-only frozen-candidate replay; no new fit or heldout',created_utc=datetime.now(timezone.utc).isoformat(),
        source_plan=str(SOURCE),source_plan_sha256=EXPECTED,training_routes_sha256=sha256(route),prompt_ids=ids,training_slice='[:32,:64]',
        scheduled_indices=indices.tolist(),arguments={**vars(args),'output':str(args.output)},geometry=geometry,
        physical_inventory=physical,layer_schedules=records,total_schedules={k:stats(v) for k,v in totals.items()},paired_comparisons=comparisons,
        source_sha256=sha256(Path(__file__)),schedule_costs_sha256=sha256(npz),
        qualifications=['Same immutable dense whole-output-row matrices; native maximum job decreases3to2waves, GU per-expert jobs8to16.',
            'Useful payload unchanged; GU physical pages96to100 and waves24to25; completeFFNpages144to148 and waves36to37. No replication.',
            'Every job pays input reload and chip ownership until channel output drain; own-expert full GU host receipt precedes activation/down.',
            'Frozen source0costs reuse sealed exact per-query schedules; all new event schedules audited at runtime.',
            'Activation/combine and actual PE throughput are uncalibrated; this is a conditional routed-FFN service simulation, not measured full-model performance.'])
    path=args.output/'screen.json';path.write_text(json.dumps(out,indent=2)+'\n')
    (args.output/'completion.json').write_text(json.dumps(dict(status='complete',screen_sha256=sha256(path),source_sha256=sha256(Path(__file__)),paired_comparisons=comparisons),indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
