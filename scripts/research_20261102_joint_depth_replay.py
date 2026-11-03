#!/usr/bin/env python3
"""Small train-only native-job replay of root's corrected joint-depth family."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np

from research_20261102_granite4_pipeline import build,profiles,stats,SOURCE,EXPECTED
from research_20261102_pipeline_joint_replay import inventory,paired
from run_corpus_traces import ROOT,sha256

BASE=ROOT/'results/research-2026-11-02/round2'
INPUT=BASE/'pipeline-joint-depth/screen.json'
DEST=BASE/'pipeline-joint-depth-replay'


def main():
    DEST.mkdir(parents=True,exist_ok=True);srcsha=sha256(INPUT);d=json.loads(INPUT.read_text());assert sha256(SOURCE)==EXPECTED
    doc=json.loads(SOURCE.read_text());assert d['frozen_plan_sha256']==EXPECTED
    cert=d['work_depth_certificate'];assert cert['capacity_at_24waves_depth2']==1020
    assert cert['capacity_at_24waves_depth2_all16active']==1016 and cert['capacity_at_25waves_depth2']==1059
    route=Path(d['training_routes']);assert sha256(route)==d['training_routes_sha256']
    with np.load(route) as f:routes=f['routes'][:32,:64].copy();layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
    assert len(ids)==32 and all(s.startswith('train_') for s in ids)
    indices=np.unique(np.linspace(0,2047,32,dtype=int));costs={};records=[];replacements={v:{} for v in ['phase_fit','joined_fit']}
    baseline_paths={'source':(BASE/'granite4-pipeline/schedule_costs.npz','frozen_g12'),
        'matched_g12':(BASE/'pipeline-joint-replay/zero/schedule_costs.npz','matched_phase_best'),
        'joined_g12':(BASE/'pipeline-joint-replay/zero/schedule_costs.npz','joint_best'),
        'wave_rotations':(BASE/'pipeline-gu-screen/smoke/schedule_costs.npz','gu_wave_rotations'),
        'g8_opposite':(BASE/'pipeline-joint-replay/opposite-smoke/schedule_costs.npz','g8_opposite')}
    provenance={}
    for v,(path,prefix) in baseline_paths.items():
        with np.load(path) as f:
            if len(f['indices'])==2048:
                np.testing.assert_array_equal(f['indices'],np.arange(2048));ix=indices
            else:np.testing.assert_array_equal(f['indices'],indices);ix=np.arange(32)
            for barrier in ['barrier','ready']:
                for priority in ['down_first','gate_first']:
                    suffix='__'+barrier+'__'+priority;costs[v+suffix]=f[prefix+suffix][:,ix].copy()
        provenance[v]=dict(path=str(path),sha256=sha256(path))
    began=time.monotonic()
    for li,q in enumerate(d['results']):
        layer=q['layer'];selected=routes[:,:,layers.index(layer)].reshape(-1,doc['active']);result={}
        for v,k in [('phase_fit','barrier'),('joined_fit','joined')]:
            g=np.asarray(q[k]['gu_rows']);dd=np.asarray(q[k]['down_rows']);gu=profiles(g,1536);down=profiles(dd,512)
            assert np.all(g.sum(axis=1)==1024) and np.all(dd.sum(axis=1)==1536)
            assert np.all(gu['waves'].sum(axis=1)==25) and np.all(down['waves'].sum(axis=1)==12)
            assert gu['waves'].max()==2 and down['waves'].max()==1
            joined=gu['waves']+down['waves'];assert np.all(((joined==2)|(joined==3))) and np.all((joined==3).sum(axis=1)==5)
            replacements[v][(layer,'gate_up')]=g;replacements[v][(layer,'down')]=dd
            for barrier in [True,False]:
                graphs={}
                for priority in ['down_first','gate_first']:
                    key=v+('__barrier' if barrier else '__ready')+'__'+priority;values=[]
                    for ix,idx in enumerate(indices):
                        ss=selected[idx];t=tuple(ss.tolist())
                        if t not in graphs:graphs[t]=build(ss,gu,down,barrier,0,0)
                        rr=graphs[t].run(priority=='down_first',trace=ix==0);values.append(rr['makespan_ns'])
                        if ix==0:(DEST/f'layer{layer}.{key}.trace.json').write_text(json.dumps(dict(selected=ss.tolist(),**rr),indent=2)+'\n')
                    costs.setdefault(key,[]).append(np.asarray(values));result[key]=stats(np.asarray(values))
                    print(layer,key,float(np.mean(values)),'elapsed',round(time.monotonic()-began,1),flush=True)
        records.append(dict(layer=layer,schedules=result))
    costs={k:np.asarray(v) for k,v in costs.items()};totals={k:v.sum(axis=0) for k,v in costs.items()};comparisons={}
    for suffix in ['__barrier__down_first','__barrier__gate_first','__ready__down_first','__ready__gate_first']:
        bs=min(baseline_paths,key=lambda v:totals[v+suffix].mean())
        for v in replacements:
            comparisons[v+'_vs_best_existing'+suffix]=dict(best_existing_family=bs,**paired(totals[v+suffix],totals[bs+suffix]))
        comparisons['joined_fit_vs_matched_phase_fit'+suffix]=paired(totals['joined_fit'+suffix],totals['phase_fit'+suffix])
    physical={v:inventory(doc,r) for v,r in replacements.items()}
    assert sha256(INPUT)==srcsha and sha256(SOURCE)==EXPECTED
    npz=DEST/'schedule_costs.npz';np.savez_compressed(npz,indices=indices,**costs)
    out=dict(status='32route train-only corrected-family smoke; no heldout/no new fit',created_utc=datetime.now(timezone.utc).isoformat(),
        root_source=str(INPUT),root_source_sha256=srcsha,root_work_depth_certificate=cert,frozen_plan_sha256=EXPECTED,
        training_routes_sha256=sha256(route),prompt_ids=ids,training_slice='[:32,:64]',scheduled_indices=indices.tolist(),
        baseline_provenance=provenance,physical_inventory=physical,layer_schedules=records,total_schedules={k:stats(v) for k,v in totals.items()},
        paired_comparisons=comparisons,schedule_costs_sha256=sha256(npz),source_sha256=sha256(Path(__file__)),
        decision='Smoke only; no globally optimal schedule or measured timing claim. Do not expand an unfavorable new family merely because its necessary bound improved.',
        qualifications=['Same sealed strict native-job scheduler, inputs fully reloaded, one context per chip, own-expert wholeGU host-ready dependency.',
            'GU25waves100pages, down12waves48pages. Same useful one-copy payload but more padding/work than GU24wave source.',
            'Comparisons preserve exactly the same32ordered routes as previous smoke; sample may differ from full2048.',
            'Activation/combine0 is optimistic and uncalibrated; resourceDAG predictions are not hardware end-to-end measurements.'])
    target=DEST/'screen.json';target.write_text(json.dumps(out,indent=2)+'\n')
    (DEST/'completion.json').write_text(json.dumps(dict(status='complete',screen_sha256=sha256(target),source_sha256=sha256(Path(__file__))),indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
