#!/usr/bin/env python3
"""Training-only optimistic bound for dense same-input selected-WL unions.

No executable hardware plan is claimed. In particular C1536 cuts rows at WL
boundaries and lacks a documented native row-fringe/alignment operand.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

import networkx as nx
import numpy as np

from research_20261102_projection_study import equal_rows,evaluate
from run_corpus_traces import ROOT,sha256

DEST=ROOT/'results/research-2026-11-02/round2/selected-wl-union'
BASE=ROOT/'results/research-2026-11-02'
WL=65536


def load(model):
    if model=='granite4tiny':
        path=BASE/'round2/geometry-study/granite4tiny.frozen.json';doc=json.loads(path.read_text())
        route=Path(doc['training_routes']);phases=[dict(p,chosen_rows=p['rows']) for p in doc['phases']]
    else:
        path=BASE/'projection-study'/(model+'.frozen_plan.json');doc=json.loads(path.read_text())
        route=Path(doc['train_path']);phases=doc['phases']
    with np.load(route) as data:
        routes=data['routes'][:32,:64].copy();ids=data['prompt_ids'][:32].tolist();layers=data['layers'].tolist()
    assert len(ids)==32 and all(i.startswith('train_') for i in ids)
    refs={str(path):sha256(path),str(route):sha256(route)}
    alternatives={}
    strong=BASE/'round2/unbalanced-group'/(model+'.screen.strong.frozen.json')
    if strong.exists():
        refs[str(strong)]=sha256(strong)
        for p in json.loads(strong.read_text())['phases']:
            alternatives.setdefault((p['layer'],p['name']),[]).append(('strong_same_group',p['chosen_rows']))
    free=BASE/'round2/free-rows'/(model+'.frozen.json')
    if free.exists():
        refs[str(free)]=sha256(free)
        for p in json.loads(free.read_text())['phases']:
            alternatives.setdefault((p['layer'],p['phase']),[]).append(('frozen_free_rows',p['rows']))
    return doc,phases,routes,ids,layers,alternatives,refs


def main():
    DEST.mkdir(parents=True,exist_ok=True);models=[]
    for model in ['granite4tiny','deepseek','qwen3','granite']:
        doc,phases,routes,ids,layers,alternatives,refs=load(model)
        wanted=layers[:3 if model=='granite4tiny' else 2]
        E,K=doc['experts'],doc['active'];records=[];down_total=0.
        for layer in wanted:
            li=layers.index(layer);selected=np.sort(routes[:,:,li].reshape(-1,K),axis=1)
            for p in (p for p in phases if p['layer']==layer):
                options=[('main_frozen',p['chosen_rows'])]+alternatives.get((layer,p['name']),[])
                scored=[]
                for label,rows in options:
                    values=evaluate(selected,np.asarray(rows),p['columns'],p['common'])
                    scored.append((float(values['model_ns'].mean()),label,np.asarray(rows),values))
                base_cost,base_label,base_rows,values=min(scored,key=lambda x:x[0])
                if p['name']=='down':down_total+=base_cost;continue
                columns,total=p['columns'],p['total_rows'];assert total%16==0
                shard_bytes=total//16*columns;pages=(shard_bytes+16383)//16384
                geometry=[]
                for g in [16,8]:
                    nbytes=total//g*columns
                    geometry.append(dict(g=g,rows_per_shard=total//g,bytes_per_shard=nbytes,
                        pages_per_shard=(nbytes+16383)//16384,full_WL_aligned=nbytes%WL==0,
                        read_mechanism='identical to same-cost ordinary run-coalescing control' if nbytes%WL==0 else 'half-WL sharing; optimistic matching bound only'))
                record=dict(layer=layer,columns=columns,total_rows=total,reference_label=base_label,
                    reference_mean_ns=base_cost,reference_flash_mean_ns=float(values['flash_ns'].mean()),
                    reference_control_mean_ns=float(values['control_ns'].mean()),geometry=geometry,
                    block_rule='Every run crossing 192 WLs must split; relaxed away in the lower bound.',
                    native_full_WL_row_boundary_legal=WL%columns==0)
                if shard_bytes%WL==0:
                    record.update(status='no new read saving; g16 and g8 full-WL aligned',
                        incremental_read_saving_vs_same_cost_coalescing_control_ns=0.)
                else:
                    assert pages%4==2 and shard_bytes==pages*16384
                    co=np.zeros((E,E),dtype=np.int64)
                    for a in range(K):
                        for b in range(a+1,K):
                            np.add.at(co,(selected[:,a],selected[:,b]),1)
                            np.add.at(co,(selected[:,b],selected[:,a]),1)
                    graph=nx.Graph();graph.add_nodes_from(range(E))
                    graph.add_weighted_edges_from((a,b,int(co[a,b])) for a in range(E) for b in range(a+1,E))
                    matching=sorted(tuple(sorted(pair)) for pair in nx.max_weight_matching(graph,maxcardinality=True))
                    assert len(matching)==E//2 and len({e for pair in matching for e in pair})==E
                    expected_pairs=sum(int(co[a,b]) for a,b in matching)/len(selected)
                    expected_waves=K*((pages+3)//4)-expected_pairs
                    # Min one dense command per participating chip; ignore every
                    # inactive output, block boundary and MAC cost for this bound.
                    lower=expected_waves*28000+16*250+2000+columns+K*total*4/8
                    baseline_bytes=(base_rows[selected].sum(axis=1)*columns).max(axis=1)
                    nonflash=values['model_ns']-values['flash_ns']
                    sensitivity=[]
                    for mode in ['max_overlap','serial_read_compute']:
                        for compute_ns in [0,14000,28000,56000,112000]:
                            if mode=='max_overlap':
                                baseline=np.maximum(values['flash_ns'],baseline_bytes*compute_ns/WL)+nonflash
                                candidate=expected_waves*max(28000,compute_ns)+16*250+2000+columns+K*total*4/8
                            else:
                                baseline=values['model_ns']+baseline_bytes*compute_ns/WL
                                candidate=expected_waves*(28000+compute_ns)+16*250+2000+columns+K*total*4/8
                            sensitivity.append(dict(compute_ns_per_64KiB_MAC_work=compute_ns,overlap_model=mode,
                                reference_mean_ns=float(baseline.mean()),candidate_optimistic_lower_bound_ns=candidate,
                                saving_upper_bound_ns=max(0.,float(baseline.mean())-candidate)))
                    record.update(status='relaxed mathematical bound; not an executable layout',
                        max_weight_matching=matching,matching_total_coactivation_count=sum(int(co[a,b]) for a,b in matching),
                        matching_expected_selected_pairs=expected_pairs,
                        minimum_expected_critical_read_waves=expected_waves,
                        candidate_optimistic_lower_bound_ns=lower,
                        saving_upper_bound_ns=max(0.,base_cost-lower),MAC_sensitivity=sensitivity,
                        proof='For phase-block-aligned tight packing of equal 4q+2-page shards, every two adjacent experts share one WL and no other pair shares a WL. For each chip W=K*(q+1)-selected_pairs. Maximum-weight perfect matching minimizes its expected W exactly. Arbitrary chip-specific matchings cannot beat this mean because E[max_chip W] >= max_chip E[W] >= this optimum; identical optimal matchings attain the flash expectation. Control, inactive outputs, block splits and native row legality are relaxed favorably.',
                        other_page_offsets='Even page offsets give the same matching family. At odd offsets, alternating bad experts cost one extra WL while every adjacent expert pair can share one WL: expected W=K*(q+1)+sum_bad P(selected)-sum_path P(both). Split path edges into two matchings; optimal perfect matching weight M is at least half the path-edge weight sum. Each bad expert has at most two neighbors, so path weight sum<=2*sum_bad P(selected). Therefore M>=path_weight_sum-sum_bad P(selected), and an aligned matching has no larger expected read work than any odd-offset order.',
                        inactive_output_note='All covered WL rows must really compute and transfer. Ignoring their extra output makes this a favorable lower bound, never a measured/predicted runtime.',
                        native_limitation='C1536 requires WL start and length multiples of3 for fully row-aligned dense runs; the safe 3WL pair bundle reads3WL even when one expert is selected.' if WL%columns else 'C divides WL, so full-WL runs contain complete rows; native block/pagecount legality still belongs to the architecture audit.')
                records.append(record)
        total_ref=down_total+sum(p['reference_mean_ns'] for p in records)
        savings=sum(p.get('saving_upper_bound_ns',0) for p in records)
        sensitivity=[]
        for mode in ['max_overlap','serial_read_compute']:
            for compute in [0,14000,28000,56000,112000]:
                ref=down_total;gain=0.
                for p in records:
                    if 'MAC_sensitivity' in p:
                        item=next(s for s in p['MAC_sensitivity'] if s['overlap_model']==mode and s['compute_ns_per_64KiB_MAC_work']==compute)
                        ref+=item['reference_mean_ns'];gain+=item['saving_upper_bound_ns']
                    else:ref+=p['reference_mean_ns']
                sensitivity.append(dict(overlap_model=mode,compute_ns_per_64KiB_MAC_work=compute,
                    saving_upper_bound_ns=gain,whole_routed_FFN_saving_upper_bound_pct=100*gain/ref,
                    denominator_note='Down service kept at original no-extra-MAC baseline, which favors the candidate. Full-WL aligned GU contributes zero new read saving relative to identical coalescing controls.'))
        models.append(dict(model=model,layers=wanted,train_ids=ids,steps=64,input_sha256=refs,
            reference_routed_FFN_mean_ns=total_ref,reference_down_ns=down_total,GU=records,
            new_read_saving_upper_bound_ns=savings,whole_routed_FFN_saving_upper_bound_pct=100*savings/total_ref,
            MAC_sensitivity=sensitivity,
            decision='no-go for this read-packing branch: favorable whole-FFN bound below1%; do not expand ordering search or open heldout'))
    output=dict(status='complete training-only bound; no candidate hardware plan or method claim',models=models,
        frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha256(Path(__file__)),networkx_version=nx.__version__,
        hardware='16 chips,4planes,16KiB pages,64KiB WL wave,192WL/block,28us ordinary read,250ns command,2us host phase; same input GU only.',
        coalescing_control='When shard length is a whole WL, tight packing and selected-WL union are exactly ordinary contiguous-run command coalescing on the existing layout. Both candidate and baseline must receive the same learned ordering and run cost. No incremental packing benefit is credited.',
        MAC_warning='Synthetic sensitivity is specified in ns per64KiB of INT8 matrix MAC work, with max-overlap and serial variants. No PE throughput was measured or inferred from the6.4GB/s ECC/data path.',
        bound_scope='g16 and g8 equal-row geometry only, as requested initial screen. No bound for arbitrary mixed width/row templates or new native operands is claimed.',
        no_heldout_opened=True,no_main_plan_changes=True)
    path=DEST/'oracle_bound.json';path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({m['model']:dict(reference_ms=m['reference_routed_FFN_mean_ns']/1e6,upper_bound_saving_ns=m['new_read_saving_upper_bound_ns'],upper_bound_pct=m['whole_routed_FFN_saving_upper_bound_pct']) for m in models},indent=2))


if __name__=='__main__':main()
