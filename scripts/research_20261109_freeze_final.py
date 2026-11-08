#!/usr/bin/env python3
"""Freeze the five-arm final experiment before any fresh TEST execution."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


runner=module('final_freeze_runner',HERE/'research_20261109_quality_run.py')
cost=module('final_freeze_cost',HERE/'research_20261109_cost_audit.py')
DATA=runner.DATA


def freeze(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text());output=Path(plan['selection_receipt'])
    assert not output.exists() and not Path(plan['output']).exists()
    assert plan['stage']=='test' and plan['prompts']==1000 and plan['max_tokens']==64 and plan['steps']==16
    assert plan['threads_per_worker']==4 and len(plan['cpu_groups'])==16 and plan['shards_per_arm']==16
    names={a['name']:a['registry'] for a in plan['arms']}
    primary='prototype25_rank256_k12';linear='raw_x_select1408_eq_rank256_k12';dense='prototype25_dense_k12'
    assert plan['primary_arm']==primary
    assert set(names)=={'native',primary,linear,dense,'selected_alpha2_mean_k12'} and names['native']=='-'
    quality_path=Path(args.quality).resolve();quality=json.loads(quality_path.read_text())
    assert quality['stage']=='validation' and quality['execution']['status']=='complete'
    for name in (primary,linear):
        q=quality['arms'][name]['vs_native']
        assert q['prompts']==200 and q['ci95_relative_subset_PPL_change'][1]<=.02
        assert quality['arms'][name]['registry']==names[name]
    models={name:cost.qa.registry(path) for name,path in names.items() if path!='-'}
    keys=set(models[primary]);assert len(keys)==12 and all(l==1 and e!=25 for l,e in keys)
    assert all(set(m)==keys for m in models.values())
    assert all(m['factor_rank']==256 and m['prototypes']=={25} for m in models[primary].values())
    assert all(m['factor_rank']==256 and not m['prototypes'] for m in models[linear].values())
    assert all(m['factor_rank'] is None and m['prototypes']=={25} for m in models[dense].values())
    policies=[DATA/'native-cache-policy-qualityVALID64-nonBOS-v2/policy.frozen.json',
              DATA/'native-cache-policy-qualityVALID64-v1/policy.frozen.json']
    placements={}
    for label,nonlinear_heads in (('rank256',models[primary]),('dense',models[dense])):
        raw_path=(names[linear] if label=='rank256' else
                  str(DATA/'proto25-cohort12/registries/registry.raw_x_select1408_eq.k12.tsv'))
        raw_heads=cost.qa.registry(raw_path)
        inv=cost.inventory(nonlinear_heads,{(1,25)});raw=cost.inventory(raw_heads)
        assert inv['total_method_host_bytes']-raw['total_method_host_bytes']==cost.GU
        placements[label]=dict(nonlinear_inventory=inv,raw_inventory_without_forced_GU=raw,policies={})
        for path in policies:
            policy=json.loads(path.read_text());assert policy['status']=='frozen_before_TEST' and policy['TEST_opened'] is False
            ranking=policy['primary_ranking'];budget=inv['total_method_host_bytes']
            _,mixed,left=cost.mixed_cache(ranking,budget)
            _,raw_mixed,raw_left=cost.mixed_cache([r for r in ranking if (r['layer'],r['expert']) not in keys],cost.GU)
            order=[(r['layer'],r['expert']) for r in ranking]
            placements[label]['policies'][str(path)]=dict(
                same_budget_mixed_native=mixed,mixed_unused_bytes=left,
                same_budget_GU_cache=order[:budget//cost.GU],same_budget_full_expert_cache=order[:budget//cost.EXPERT],
                raw_readouts_plus_native_cache=raw_mixed,raw_unused_bytes=raw_left)
    paths={plan_path,Path(__file__).resolve(),HERE/'research_20261109_quality_run.py',
           Path(plan['binary']),quality_path,DATA/'corpus103/manifest.json',*policies}
    paths.update(Path(p).resolve() for p in plan['source_files'])
    for registry in names.values():paths.update(p.resolve() for p in runner.registry_files(registry))
    for raw_path in (DATA/'proto25-cohort12/registries/registry.raw_x_select1408_eq.k12.tsv',):
        paths.update(p.resolve() for p in runner.registry_files(str(raw_path)))
    expected_arm_files={str(p.resolve()) for registry in names.values() for p in runner.registry_files(registry)}
    files={str(p):runner.sha(p) for p in sorted(paths)}
    assert expected_arm_files<=set(files)
    snapshots=DATA/'final-source-freeze';snapshots.mkdir(exist_ok=False)
    copies={}
    for p in sorted(paths):
        if p.is_relative_to(runner.ROOT) and p.suffix in ('.py','.cpp','.hpp','.md'):
            dest=snapshots/p.name;assert not dest.exists();shutil.copyfile(p,dest)
            assert runner.sha(dest)==files[str(p)];copies[str(dest)]=files[str(p)]
    receipt=dict(status='frozen_before_new_TEST',at_utc=datetime.now(timezone.utc).isoformat(),
        TEST_results_used_for_selection=False,TEST_inference_not_started=True,files_sha256=files,
        source_snapshots_sha256=copies,all_five_arm_payloads_covered=True,model_identity=runner.model_identity(),
        quality=dict(primary_arm=primary,matched_linear_arm=linear,dense_same_basis_arm=dense,
                     prompts=1000,input_tokens=64,scored_targets_per_prompt=63,greedy_generation_max_tokens=16,
                     bootstrap_replicates=10000,bootstrap_seed=20261109,bootstrap_unit='article',
                     primary_margin_relative_PPL=.02,decision='upper95% paired-bootstrap CI <= log(1.02) pass; lower > margin fail; else inconclusive',
                     secondary='All other contrasts descriptive; no secondary margin verdict',
                     rank_selection='rank256: all12 and seed3 local SSE increases <=5%; rank128 fails seed3; no TEST-dependent rank/eligibility changes'),
        cost=dict(primary_scope='All26 native-reference nonBOS NLL input positions1..62',
                  primary_cache_policy=str(policies[0]),sensitivity_cache_policy=str(policies[1]),
                  secondary_scopes=['TF positions0..62','BOS','generation_causal','actual nonlinear live routing'],
                  budget='Incremental coefficient/scales/bias/transform/rank metadata; method-specific activation workspace is additional',
                  placements=placements,cache_optimality='Only full expert / GU pair / D whole-matrix units for byte-work objective'),
        not_claimed=['physical coefficient deletion','measured flash IO','NVMeVirt validation','end-to-end speedup','task accuracy','pretraining-unseen data'])
    runner.save(output,receipt)
    print(json.dumps(dict(receipt=str(output),SHA256=runner.sha(output),files=len(files),primary_arm=primary,
                          rank256_MiB=placements['rank256']['nonlinear_inventory']['MiB']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--quality',required=True)
    freeze(p.parse_args())
