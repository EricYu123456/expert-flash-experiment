#!/usr/bin/env python3
"""Full train replay of corrected work/depth family and all five controls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np

from research_20261102_granite4_pipeline import build,profiles,stats,SOURCE,EXPECTED
from research_20261102_pipeline_joint_replay import inventory,paired,SEALED
from run_corpus_traces import ROOT,sha256

BASE=ROOT/'results/research-2026-11-02/round2'
INPUT=BASE/'pipeline-joint-depth/screen.json'
DEST=BASE/'pipeline-joint-depth-replay'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--one-us',action='store_true');args=ap.parse_args()
    output=DEST/('full-one-us' if args.one_us else 'full-zero');output.mkdir(parents=True,exist_ok=True)
    act=1000 if args.one_us else 0;combine=6000 if args.one_us else 0
    srcsha=sha256(INPUT);d=json.loads(INPUT.read_text());assert sha256(SOURCE)==EXPECTED
    doc=json.loads(SOURCE.read_text());seal=json.loads((SEALED/'seal.json').read_text())
    for rel,h in seal['artifacts'].items():assert sha256(ROOT/rel)==h
    cert=d['work_depth_certificate'];assert cert['capacity_at_24waves_depth2']==1020
    assert cert['capacity_at_24waves_depth2_all16active']==1016
    route=Path(d['training_routes']);assert sha256(route)==d['training_routes_sha256']
    with np.load(route) as f:routes=f['routes'][:32,:64].copy();layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
    assert len(ids)==32 and all(s.startswith('train_') for s in ids)
    tag='one-us' if args.one_us else 'zero';opp='opposite-one-us' if args.one_us else 'opposite-full';wave='one-us' if args.one_us else 'full'
    core=BASE/'pipeline-joint-replay'/tag/'schedule_costs.npz'
    baseline_paths={'source':(core,'source'),'matched_g12':(core,'matched_phase_best'),'joined_g12':(core,'joint_best'),
        'wave_rotations':(BASE/'pipeline-gu-screen'/wave/'schedule_costs.npz','gu_wave_rotations'),
        'g8_opposite':(BASE/'pipeline-joint-replay'/opp/'schedule_costs.npz','g8_opposite')}
    costs={};provenance={}
    for v,(path,prefix) in baseline_paths.items():
        with np.load(path) as f:
            np.testing.assert_array_equal(f['indices'],np.arange(2048))
            for barrier in ['barrier','ready']:
                suffix='__'+barrier+'__gate_first';costs[v+suffix]=f[prefix+suffix].copy()
                assert costs[v+suffix].shape==(3,2048)
        provenance[v]=dict(path=str(path),sha256=sha256(path))
    replacements={v:{} for v in ['phase_fit','joined_fit']};records=[];began=time.monotonic()
    for li,q in enumerate(d['results']):
        layer=q['layer'];selected=routes[:,:,layers.index(layer)].reshape(-1,doc['active']);result={}
        for v,k in [('phase_fit','barrier'),('joined_fit','joined')]:
            gurows=np.asarray(q[k]['gu_rows']);drows=np.asarray(q[k]['down_rows']);gu=profiles(gurows,1536);down=profiles(drows,512)
            assert np.all(gurows.sum(axis=1)==1024) and np.all(drows.sum(axis=1)==1536)
            assert np.all(gu['waves'].sum(axis=1)==25) and np.all(down['waves'].sum(axis=1)==12)
            assert np.all(((gu['waves']+down['waves'])==3).sum(axis=1)==5)
            replacements[v][(layer,'gate_up')]=gurows;replacements[v][(layer,'down')]=drows
            for barrier in [True,False]:
                name=v+('__barrier' if barrier else '__ready')+'__gate_first';values=[];cache={}
                for ix,ss in enumerate(selected):
                    key=tuple(ss.tolist())
                    if key not in cache:
                        graph=build(ss,gu,down,barrier,act,combine);rr=graph.run(False,trace=ix==0)
                        cache[key]=rr['makespan_ns']
                        if ix==0:(output/f'layer{layer}.{name}.trace.json').write_text(json.dumps(dict(selected=ss.tolist(),**rr),indent=2)+'\n')
                    values.append(cache[key])
                values=np.asarray(values,dtype=np.int64);costs.setdefault(name,[]).append(values);result[name]=stats(values)
                print(layer,name,float(values.mean()),'unique',len(cache),'elapsed',round(time.monotonic()-began,1),flush=True)
        records.append(dict(layer=layer,schedules=result));(output/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
    costs={k:np.asarray(v) for k,v in costs.items()};totals={k:v.sum(axis=0) for k,v in costs.items()};comparisons={}
    for suffix in ['__barrier__gate_first','__ready__gate_first']:
        bestmean=min(baseline_paths,key=lambda v:totals[v+suffix].mean())
        besttail=min(baseline_paths,key=lambda v:np.percentile(totals[v+suffix],99))
        for v in replacements:
            for ref in baseline_paths:comparisons[v+'_vs_'+ref+suffix]=paired(totals[v+suffix],totals[ref+suffix])
            comparisons[v+'_vs_best_existing'+suffix]=dict(best_mean_family=bestmean,best_p99_family=besttail,
                **paired(totals[v+suffix],totals[bestmean+suffix]),
                candidate_p99_ns=float(np.percentile(totals[v+suffix],99)),best_existing_p99_ns=float(np.percentile(totals[besttail+suffix],99)))
        comparisons['joined_fit_vs_matched_phase_fit'+suffix]=paired(totals['joined_fit'+suffix],totals['phase_fit'+suffix])
    physical={v:inventory(doc,r) for v,r in replacements.items()}
    for v,p in physical.items():assert p['total_useful_bytes']==6039797760 and p['total_physical_pages']==369408
    assert sha256(INPUT)==srcsha and sha256(SOURCE)==EXPECTED
    for rel,h in seal['artifacts'].items():assert sha256(ROOT/rel)==h
    npz=output/'schedule_costs.npz';np.savez_compressed(npz,indices=np.arange(2048),**costs)
    out=dict(status='complete first3layer full train32x64 same-scheduler replay; no heldout',created_utc=datetime.now(timezone.utc).isoformat(),
        root_source=str(INPUT),root_source_sha256=srcsha,work_depth_certificate=cert,frozen_plan_sha256=EXPECTED,
        training_routes_sha256=sha256(route),prompt_ids=ids,training_slice='[:32,:64]',indices=list(range(2048)),
        activation_per_expert_ns=act,combine_per_expert_ns=act,priority='gate_first',
        baseline_provenance=provenance,physical_inventory=physical,layer_schedules=records,total_schedules={k:stats(v) for k,v in totals.items()},
        paired_comparisons=comparisons,schedule_costs_sha256=sha256(npz),source_sha256=sha256(Path(__file__)),
        elapsed_seconds=time.monotonic()-began,
        limitations=['All seven families use identical ordered route vectors and sealed strict scheduler. Root candidate source and sealed sources verified unchanged.',
            'Eachnativejob pays inputreload and owns chip from commandstart through outputchannel drain. Down waits for own wholeexpertGU hostready and activation.',
            'GU25wave/100pages and down12wave/48pages; unchanged one-copy useful payload, extra padding/work fully counted.',
            'Source files remain nominal rawI8/A8/FP32 service model; native PE throughput and physical timing not calibrated.',
            'Only first3layers routedFFN. Always-active sharedFFN/SSM/attention are not included; do not infer full-model timing or add shared cost serially.'])
    target=output/'screen.json';target.write_text(json.dumps(out,indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps(dict(status='complete',screen_sha256=sha256(target),source_sha256=sha256(Path(__file__)),
        schedule_costs_sha256=sha256(npz)),indent=2)+'\n')
    print(json.dumps({k:v for k,v in comparisons.items() if 'best_existing' in k or 'matched_phase_fit' in k},indent=2))


if __name__=='__main__':main()
