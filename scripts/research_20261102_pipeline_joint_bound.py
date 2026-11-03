#!/usr/bin/env python3
"""Train-only two-stage placement screen using a necessary chip-work bound.

No timing schedule is generated here. Summing per-chip GU and down work relaxes
the expert-ready and I/O constraints, so any improvement remains an oracle
screen pending the separate native job scheduler. Complete rows are retained.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
DEST = BASE / 'pipeline-joint'
SOURCE = BASE / 'geometry-study/granite4tiny.frozen.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(selected, weights, home_a, initial, objective, seed):
    rng = np.random.default_rng(seed)
    labels = np.array(initial, dtype=np.int64)
    k=selected.shape[1]
    gu_a=home_a[selected].sum(axis=1)*3
    gu=np.column_stack([gu_a,gu_a,3*k-gu_a,3*k-gu_a])
    counts=np.column_stack([(labels[selected]==g).sum(axis=1) for g in range(4)])
    adjacency=[np.flatnonzero((selected==e).any(axis=1)) for e in range(len(labels))]
    def event_cost(n,indices):
        return ((gu[indices]+k-n).max(axis=1) if objective=='joined' else
                k-n.min(axis=1))
    all_indices=np.arange(len(selected))
    values=event_cost(counts,all_indices)
    score=int(values@weights);best=score;bestlabels=labels.copy();bestcounts=counts.copy()
    proposals=6000
    for step in range(proposals):
        e=int(rng.integers(len(labels)));old=int(labels[e]);new=int(rng.integers(3));new+=new>=old
        indices=adjacency[e]
        candidate=counts[indices].copy();candidate[:,old]-=1;candidate[:,new]+=1
        costs=event_cost(candidate,indices)
        delta=int((costs-values[indices])@weights[indices])
        temperature=.03*(.0005/.03)**(step/(proposals-1))
        if delta<=0 or rng.random()<math.exp(-delta/(weights.sum()*temperature)):
            labels[e]=new;counts[indices]=candidate;values[indices]=costs;score+=delta
            if score<best:
                best=score;bestlabels=labels.copy();bestcounts=counts.copy()
    labels=bestlabels;counts=bestcounts;values=event_cost(counts,all_indices);score=best
    accepted=0;converged=False
    for sweep in range(32):
        changes=0
        for e in rng.permutation(len(labels)):
            old=int(labels[e]);indices=adjacency[e];choice=None
            for new in range(4):
                if new==old:continue
                candidate=counts[indices].copy();candidate[:,old]-=1;candidate[:,new]+=1
                costs=event_cost(candidate,indices)
                delta=int((costs-values[indices])@weights[indices])
                if delta<0 and (choice is None or delta<choice[0]):
                    choice=delta,new,candidate,costs
            if choice:
                delta,new,candidate,costs=choice
                labels[e]=new;counts[indices]=candidate;values[indices]=costs;score+=delta;changes+=1
        accepted+=changes
        if not changes:converged=True;break
    counts=np.column_stack([(labels[selected]==g).sum(axis=1) for g in range(4)])
    joined=(gu+k-counts).max(axis=1)
    phase=k-counts.min(axis=1)
    return dict(objective=objective,seed=seed,labels=labels.tolist(),
        group_inventory=np.bincount(labels,minlength=4).tolist(),
        mean_joined_waves=float(joined@weights/weights.sum()),
        mean_separate_down_waves=float(phase@weights/weights.sum()),
        greedy_passes=sweep+1,greedy_moves=accepted,converged=converged,
        anneal_proposals=proposals)


def main():
    p=json.loads(SOURCE.read_text());assert len(p['phases'])==80
    routepath=Path(p['training_routes']);assert sha(routepath)==p['training_routes_sha256']
    with np.load(routepath) as f:
        routes=f['routes'];layers=f['layers'].tolist();ids=f['prompt_ids'].tolist()
    assert routes.shape==(32,64,40,6) and all(x.startswith('train_') for x in ids)
    phases={(x['layer'],x['name']):x for x in p['phases']}
    results=[]
    for li,layer in enumerate(layers[:3]):
        gu=np.asarray(phases[layer,'gate_up']['rows'])
        down=np.asarray(phases[layer,'down']['rows'])
        assert np.all((gu>0).sum(axis=1)==8)
        assert np.all((down>0).sum(axis=1)==12)
        home_a=(gu[:,:8].sum(axis=1)>0).astype(np.int64)
        labels=np.argmin(down.reshape(64,4,4).sum(axis=2),axis=1)
        selected,weights=np.unique(np.sort(routes[:,:,li].reshape(-1,6),axis=1),axis=0,return_counts=True)
        runs=[]
        for objective in ('separate_down','joined'):
            for seed in (20261102,20261103,20261104):
                runs.append(fit(selected,weights,home_a,labels,objective,seed+li*100))
        phase_best=min((v for v in runs if v['objective']=='separate_down'),key=lambda v:(v['mean_separate_down_waves'],v['mean_joined_waves']))
        joint_best=min((v for v in runs if v['objective']=='joined'),key=lambda v:v['mean_joined_waves'])
        counts=np.column_stack([(labels[selected]==g).sum(axis=1) for g in range(4)])
        n=home_a[selected].sum(axis=1);gwork=np.column_stack([3*n,3*n,18-3*n,18-3*n])
        fixed_joined=(gwork+6-counts).max(axis=1)
        original_serial=3*np.maximum(n,6-n)+6-counts.min(axis=1)
        adaptive=np.where(n==3,14,3*np.maximum(n,6-n)+3)
        result=dict(layer=layer,home_a=home_a.tolist(),original_labels=labels.tolist(),
            source_separate_waves=float(original_serial@weights/weights.sum()),
            source_joined_bound_waves=float(fixed_joined@weights/weights.sum()),
            route_adaptive_omit_bound_waves=float(adaptive@weights/weights.sum()),
            matched_phase_best=phase_best,joint_best=joint_best,runs=runs)
        results.append(result)
        print(layer,'separate',result['source_separate_waves'],'fixedjoined',result['source_joined_bound_waves'],
              'phasefitjoined',phase_best['mean_joined_waves'],'jointfit',joint_best['mean_joined_waves'],
              'adaptive',result['route_adaptive_omit_bound_waves'],flush=True)
    DEST.mkdir(parents=True,exist_ok=True)
    doc=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        source_plan=str(SOURCE),source_plan_sha256=sha(SOURCE),training_routes_sha256=sha(routepath),
        protocol='Only first three G4 training layers, complete 32x64 route vectors; 3 seeds each '
                 'for phase-only and joined objectives; identical proposals and moves. No test or new inference.',
        scope='Necessary NAND-work bound only, not a realized latency. Existing GU g8 homes fixed; '
              'each down expert stores twelve full128x512 row tiles, omitting one4-chip group. '
              'One weight copy; group counts may vary, fixed quotas not imposed.',
        fixed_gu_sum_waves=24,fixed_down_sum_waves=12,
        physical_format='Same raw-I8 nominal timing assumption as previous layout studies; '
                        'unchanged complete per-row dot products independent of output dtype.',
        important='An independently fitted phase mapping is a baseline, not a method contribution. '
                  'Exact native-job schedules, command/input ownership, host activation and capacities '
                  'must be checked before claiming any pipeline gain. No global optimality claim.',
        results=results)
    (DEST/'joint_work_screen.json').write_text(json.dumps(doc,indent=2)+'\n')


if __name__=='__main__':main()
