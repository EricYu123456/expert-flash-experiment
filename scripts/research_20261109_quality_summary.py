#!/usr/bin/env python3
"""Summarize a completed frozen plan, using exactly matched native prefixes."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quality_summary_audit',HERE/'research_20261109_quality_analysis.py')
qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
DATA=Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')


def diagnostics(arm):
    rows=list(arm['summaries'].values())
    out=dict(unique_prompts=len(rows),scored_targets=sum(r['scored_tokens'] for r in rows),
             early_EOG_prompts=sum(r['stopped_eog'] for r in rows),
             generated_tokens=sum(r['actual_generated_length'] for r in rows),
             generation_length_range=[min(r['actual_generated_length'] for r in rows),
                                      max(r['actual_generated_length'] for r in rows)],
             causal_decode_inputs=sum(max(0,r['actual_generated_length']-1) for r in rows),
             decoded_last_unused_inputs=sum(not r['stopped_eog'] and r['decode_steps']>0 for r in rows))
    out['summary_counter_totals']={k:sum(r.get(k,0) for r in rows) for k in (
        'replaced_expert_token_events','native_shared_z_floats_bitchecked',
        'prototype_GU_row_pairs_computed','prototype_owner_selected_phi_floats_bitchecked',
        'prototype_cross_replaced_events','basis_A8_unique_rows','private_latent_A8_rows')}
    return out


def position_sensitivity(candidate,native):
    # Descriptive secondary partition, never an additional selection gate.
    ids=sorted(candidate['positions'])
    result={}
    for name,start,end in (('BOS',0,1),('nonBOS',1,None)):
        a=np.asarray([np.mean([r['nll_F64'] for r in candidate['positions'][i][start:end]]) for i in ids])
        b=np.asarray([np.mean([r['nll_F64'] for r in native['positions'][i][start:end]]) for i in ids])
        result[name]=dict(mean_delta_nll=float(np.mean(a-b)),relative_subset_PPL_change=math.expm1(float(np.mean(a-b))),
                          descriptive_only=True)
    return result


def run(args):
    plan_path=Path(args.plan);plan=qa.read(plan_path);folder=Path(plan['output'])
    execution=qa.read(folder/'execution.json')
    assert execution['status']=='complete' and execution['all_input_hashes_rechecked']
    assert plan['stage'] in ('validation','test')
    split='quality_validation' if plan['stage']=='validation' else 'test'
    records=[r for r in qa.read(DATA/'corpus103/manifest.json')['records'] if r['split']==split][:plan['prompts']]
    ids=[r['id'] for r in records]
    assert len(ids)==len({r['article_id'] for r in records})==plan['prompts']
    native_folder=Path(args.native or folder/'native')
    native=qa.load_arm(native_folder,expected_ids=ids)
    assert all(s['input_tokens']==plan['max_tokens'] and s['decode_steps']==plan['steps'] for s in native['summaries'].values())
    arms={a['name']:a for a in plan['arms'] if a['registry']!='-'}
    linear=None
    if args.linear:
        a=arms[args.linear];linear=qa.load_arm(folder/args.linear,a['registry'],ids)
    result=dict(stage=plan['stage'],plan=str(plan_path),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        execution=execution,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_reference=str(native_folder),native=diagnostics(native),arms={},
        TEST_used_for_selection=False if plan['stage']=='test' else None,
        inference_contract='Canonical CPU W8A8 FFNs, matched prefix/batch shape, fresh KV and live routing; subset PPL, not task accuracy')
    for name,arm in arms.items():
        value=linear if name==args.linear else qa.load_arm(folder/name,arm['registry'],ids)
        item=dict(registry=arm['registry'],vs_native=qa.compare(value,native),audit=diagnostics(value),
                  position_sensitivity=position_sensitivity(value,native))
        if linear is not None and name!=args.linear:
            item['vs_matched_linear_descriptive']=qa.compare(value,linear,compare_to_native=False)
        result['arms'][name]=item
        q=item['vs_native']
        print(json.dumps(dict(arm=name,relative_PPL=q['relative_subset_PPL_change'],CI=q['ci95_relative_subset_PPL_change'],verdict=q['verdict'])),flush=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--native');p.add_argument('--output',required=True);p.add_argument('--linear')
    run(p.parse_args())
