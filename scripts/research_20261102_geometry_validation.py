#!/usr/bin/env python3
"""Evaluate the entire frozen new-model policy; do not change its training fit."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from run_corpus_traces import ROOT, sha256
from research_20261102_projection_study import evaluate, bootstrap

DEST=ROOT/'results/research-2026-11-02/round2/geometry-study'
ROUTES=ROOT/'results/research-2026-11-02/round2/neuron-layout-prior/granite4tiny-routes'
FROZEN_SHA='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'


def marginal_rows(selected,experts=64):
    frequency=np.bincount(selected.reshape(-1),minlength=experts)
    count=np.zeros(4,dtype=int); weight=np.zeros(4,dtype=np.int64)
    colors=np.zeros(experts,dtype=int)
    for e in np.argsort(-frequency,kind='stable'):
        choices=[g for g in range(4) if count[g]<experts//4]
        g=min(choices,key=lambda x:(weight[x],count[x],x))
        colors[e]=g;count[g]+=1;weight[g]+=frequency[e]
    rows=np.full((experts,16),128,dtype=np.int32)
    for e,g in enumerate(colors):rows[e,g*4:g*4+4]=0
    return rows


def costs(selected,rows,p,command_ns=250,drain=False):
    v=evaluate(selected,rows,p['columns'],p['common'])
    pages=(np.asarray(rows,dtype=np.int64)*p['columns']+16383)//16384
    waves=(pages+3)//4
    commands=((waves+191)//192)[selected].sum(axis=(1,2))
    result=v['model_ns']+(command_ns-250)*commands
    if drain:
        serial=(waves[selected].sum(axis=1)*28000+
                pages[selected].sum(axis=1)*2560).max(axis=1)
        result=result+serial-v['flash_ns']
    return result


def evaluate_split(name,plan,marginal):
    path=ROUTES/name/'routes.npz'
    with np.load(path) as f:
        routes=f['routes'];ids=f['prompt_ids'].tolist()
        assert f['layers'].tolist()==plan['layers']
    assert not(set(ids)&set(plan['train_ids']))
    shape=routes.shape[:2]
    policies=['g16','phase_group','g16_down','chosen','g12_identity','g12_marginal']
    totals={k:np.zeros(shape,dtype=np.int64) for k in policies}
    down={k:np.zeros(shape,dtype=np.int64) for k in policies}
    sensitivity={str(c):{k:np.zeros(shape,dtype=np.int64) for k in ('phase_group','chosen')}
                 for c in (0,250,1000,5000)}
    drain={k:np.zeros(shape,dtype=np.int64) for k in ('phase_group','g16_down','chosen')}
    coverage=Counter()
    for p in plan['phases']:
        li=plan['layers'].index(p['layer'])
        selected=routes[:,:,li].reshape(-1,plan['active'])
        g16=np.full((plan['experts'],16),p['total_rows']//16,dtype=np.int32)
        group=np.asarray(p['group_rows'],dtype=np.int32)
        chosen=np.asarray(p['rows'],dtype=np.int32)
        identity=np.full_like(chosen,128)
        if p['name']=='down':
            for e in range(plan['experts']):identity[e,(e%4)*4:(e%4)*4+4]=0
            variants=dict(g16=g16,phase_group=group,g16_down=g16,chosen=chosen,
                          g12_identity=identity,g12_marginal=marginal[p['layer']])
            chipwaves=((chosen*p['columns']+65535)//65536)[selected].sum(axis=1)
            peak=chipwaves.max(axis=1)
            assert np.all((peak==5)|(peak==6))
            coverage['five_wave_events']+=int((peak==5).sum())
            coverage['down_events']+=len(peak)
        else:
            variants=dict(g16=g16,phase_group=group,g16_down=chosen,chosen=chosen,
                          g12_identity=chosen,g12_marginal=chosen)
        for label,rows in variants.items():
            v=costs(selected,rows,p).reshape(shape)
            totals[label]+=v
            if p['name']=='down':down[label]+=v
        for label,rows in [('phase_group',group),('chosen',chosen),
                           ('g16_down',g16 if p['name']=='down' else chosen)]:
            drain[label]+=costs(selected,rows,p,drain=True).reshape(shape)
            if label!='g16_down':
                for c in (0,250,1000,5000):
                    sensitivity[str(c)][label]+=costs(selected,rows,p,command_ns=c).reshape(shape)
    summary=dict(split=name,prompts=len(ids),decode_steps=routes.shape[1],routes_sha256=sha256(path),
        mean_ms={k:float(v.mean()/1e6) for k,v in totals.items()},
        down_mean_ms={k:float(v.mean()/1e6) for k,v in down.items()},
        chosen_gain_over_phase_group=1-float(totals['chosen'].mean()/totals['phase_group'].mean()),
        chosen_gain_over_fixed_g16_down=1-float(totals['chosen'].mean()/totals['g16_down'].mean()),
        coverage=dict(coverage),
        paired_prompt_bootstrap=bootstrap(totals['phase_group'],totals['chosen'],20261102),
        paired_prompt_bootstrap_fixed_g16_down=bootstrap(totals['g16_down'],totals['chosen'],20261102),
        conservative_drain=dict(mean_ms={k:float(v.mean()/1e6) for k,v in drain.items()},
            gain=1-float(drain['chosen'].mean()/drain['phase_group'].mean()),
            gain_over_fixed_g16_down=1-float(drain['chosen'].mean()/drain['g16_down'].mean())),
        command_sensitivity={c:dict(mean_ms={k:float(v.mean()/1e6) for k,v in pair.items()},
            gain=1-float(pair['chosen'].mean()/pair['phase_group'].mean())) for c,pair in sensitivity.items()})
    np.savez_compressed(DEST/(name+'.token_costs.npz'),**totals,prompt_ids=np.asarray(ids))
    return summary


def main():
    path=DEST/'granite4tiny.frozen.json'
    assert sha256(path)==FROZEN_SHA
    plan=json.loads(path.read_text())
    assert len(plan['phases'])==len(plan['layers'])*2==80
    for source,expected in plan['source_sha256'].items():
        assert sha256(ROOT/'scripts'/source)==expected
    with np.load(plan['training_routes']) as f:train=f['routes']
    assert sha256(Path(plan['training_routes']))==plan['training_routes_sha256']
    marginal={l:marginal_rows(train[:,:,li]) for li,l in enumerate(plan['layers'])}
    receipt=dict(purpose='train-only marginal-frequency baseline, generated before heldout cost calculation',
        original_plan_sha256=FROZEN_SHA,train_sha256=plan['training_routes_sha256'],
        rows={str(l):r.tolist() for l,r in marginal.items()},
        source_sha256=sha256(Path(__file__)))
    (DEST/'marginal_baseline_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    results=[evaluate_split(name,plan,marginal) for name in ('test16','external60')]
    assert sha256(path)==FROZEN_SHA
    doc=dict(frozen_plan_sha256=FROZEN_SHA,source_sha256=sha256(Path(__file__)),results=results,
        interpretation='New-model heldout confirmation, not full-model or device latency; '
            'model selected after geometry screening, public source families previously used for other models.',
        hardware='Same full-row dense GEMV on 16 chips, 4 same-address planes, 16KiB pages, 28us ordinary read; no CR.',
        control_cost_caveat='250ns chip/block and 2us host phase are uncalibrated model parameters.',
        novelty='Overlapping staggered striping and hypergraph declustering are prior art; novelty is not established.')
    (DEST/'heldout_summary.json').write_text(json.dumps(doc,indent=2)+'\n')
    for r in results:
        print(r['split'],r['mean_ms'],'gain',r['chosen_gain_over_phase_group'],
              'bootstrap',r['paired_prompt_bootstrap'],'drain',r['conservative_drain'],flush=True)


if __name__=='__main__':main()
