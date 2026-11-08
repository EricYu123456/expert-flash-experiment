#!/usr/bin/env python3
"""Frozen final TEST inference: one primary and descriptive secondary contrasts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('final_quality_helpers',HERE/'research_20261109_quality_summary.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
qa=helpers.qa


def run(args):
    plan_path=Path(args.plan);plan=qa.read(plan_path);folder=Path(plan['output'])
    assert plan['stage']=='test' and plan['prompts']==1000 and plan['steps']==16 and plan['max_tokens']==64
    execution=qa.read(folder/'execution.json')
    assert execution['status']=='complete' and execution['all_input_hashes_rechecked']
    selection=qa.read(plan['selection_receipt']);primary=selection['quality']['primary_arm']
    assert primary==plan['primary_arm']
    records=[r for r in qa.read(helpers.DATA/'corpus103/manifest.json')['records'] if r['split']=='test']
    ids=[r['id'] for r in records]
    assert len(ids)==len({r['article_id'] for r in records})==1000
    arms={a['name']:a for a in plan['arms']};assert arms['native']['registry']=='-'
    native=qa.load_arm(folder/'native',expected_ids=ids)
    linear_name=selection['quality']['matched_linear_arm'];dense_name=selection['quality']['dense_same_basis_arm']
    linear=qa.load_arm(folder/linear_name,arms[linear_name]['registry'],ids)
    dense=qa.load_arm(folder/dense_name,arms[dense_name]['registry'],ids)
    result=dict(stage='test',primary_arm=primary,plan=str(plan_path),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),execution=execution,
        selection_receipt=plan['selection_receipt'],TEST_results_used_for_selection=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native=helpers.diagnostics(native),arms={},
        primary_inference='One preregistered paired article bootstrap comparison, TF63, 10000 draws, seed20261109, upper95%CI<=log(1.02)',
        secondary_inference='Descriptive paired intervals; no secondary noninferiority verdict or familywise claim',
        limits='Experiment-heldout article split; not pretraining-unseen, standard concatenated PPL, task accuracy, long-context, measured flash IO or TPOT')
    for name,arm in arms.items():
        if name=='native':continue
        value=linear if name==linear_name else (dense if name==dense_name else qa.load_arm(folder/name,arm['registry'],ids))
        item=dict(registry=arm['registry'],primary=name==primary,
            vs_native=qa.compare(value,native,compare_to_native=name==primary),
            audit=helpers.diagnostics(value),position_sensitivity=helpers.position_sensitivity(value,native))
        if name==primary:
            item['vs_matched_linear_descriptive']=qa.compare(value,linear,compare_to_native=False)
            item['vs_dense_same_basis_descriptive']=qa.compare(value,dense,compare_to_native=False)
        result['arms'][name]=item
        q=item['vs_native']
        print(json.dumps(dict(arm=name,primary=name==primary,PPL_change=q['relative_subset_PPL_change'],CI=q['ci95_relative_subset_PPL_change'],verdict=q['verdict'])),flush=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
