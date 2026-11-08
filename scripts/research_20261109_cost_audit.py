#!/usr/bin/env python3
"""All26 routed-FFN inventory and frozen-VALID same-budget cache controls.

All coefficients include W8 row scales; readouts additionally include bias.
No result from this script is a measured flash transfer or speedup.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

import hashlib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('quality_analysis_1109', HERE/'research_20261109_quality_analysis.py')
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')
H, I = 2048, 1408
GU = 2*H*I + 8*I
DOWN = H*I + 4*H
EXPERT = GU + DOWN


def inventory(heads, hosted_GU=()):
    transforms = {m['transform'] for m in heads.values()}
    files = set();artifact_metadata=set()
    for m in heads.values():
        if m.get('factor_rank'):
            for p in m['factor_payload_files']:
                if p.endswith('.factor.json'):artifact_metadata.add(p)
                else:files.add(p)
        else:
            files.update(m['head']+suffix for suffix in ('.i8', '.scales.f32', '.bias.f32'))
    head_bytes = sum(Path(p).stat().st_size for p in files)
    transform_bytes = 0
    for prefix in transforms:
        for suffix in ('.sources.u8', '.indices.i32', '.center.f32', '.inv_scale.f32', '.prototypes.i32'):
            p = Path(prefix+suffix)
            if p.exists():
                transform_bytes += p.stat().st_size
    rank_metadata=4*sum(bool(m.get('factor_rank')) for m in heads.values())
    budget = head_bytes + transform_bytes + rank_metadata + len(hosted_GU)*GU
    return dict(readout_payload_bytes=head_bytes,transform_metadata_bytes=transform_bytes,
                runtime_factor_rank_metadata_bytes=rank_metadata,
                artifact_provenance_JSON_bytes_excluded=sum(Path(p).stat().st_size for p in artifact_metadata),
                artifact_JSON_is_frozen_but_not_runtime_coefficient_payload=True,
                prototype_GU_bytes=len(hosted_GU)*GU,resident_prototype_layer_experts=sorted(hosted_GU),
                total_method_host_bytes=budget,MiB=budget/(1<<20),
                budget_scope='Resident coefficient/scales/bias/transform metadata; excludes method-specific activation workspace, common model/KV/workspace',
                factorized_heads=sum(bool(m.get('factor_rank')) for m in heads.values()),
                readout_MAC_per_selected_expert={f'{l}:{e}':m.get('readout_coefficient_MAC',H*m['width']) for (l,e),m in heads.items()},
                activation_working_set_dimensions=dict(
                    prototype_F32_features_per_live_token=len(hosted_GU)*I,
                    unique_transformed_feature_widths={t:max(m['width'] for m in heads.values() if m['transform']==t) for t in transforms},
                    largest_private_latent_width=max((m.get('factor_rank') or 0 for m in heads.values()),default=0),
                    note='F32 features, A8 codes/scales, GU temporaries and private latent buffers also occupy host memory; this is a shape inventory, not a measured peak or total-host-memory budget'))


def mixed_cache(ranking,budget):
    # For fixed role counts, rearrangement assigns full/GU/D (in decreasing
    # bytes) to decreasing frequencies. Enumerate all full/GU counts and fill
    # remaining budget with D. This is optimal for these whole-matrix units;
    # it is not a flash-page/latency optimum.
    cumulative=[0]
    for r in ranking:cumulative.append(cumulative[-1]+r['event_count'])
    n=len(ranking);best=(-1,0,0,0,0)
    for full in range(min(n,budget//EXPERT)+1):
        remaining=budget-full*EXPERT
        for gu in range(min(n-full,remaining//GU)+1):
            down=min(n-full-gu,(remaining-gu*GU)//DOWN)
            score=(EXPERT*cumulative[full]+GU*(cumulative[full+gu]-cumulative[full])
                   +DOWN*(cumulative[full+gu+down]-cumulative[full+gu]))
            spent=full*EXPERT+gu*GU+down*DOWN
            if (score,-spent)>(best[0],-best[1]):best=score,spent,full,gu,down
    score,spent,full,gu,down=best;selected=[]
    for index,r in enumerate(ranking[:full+gu+down]):
        roles=('GU','D') if index<full else (('GU',) if index<full+gu else ('D',))
        selected.extend((r['layer'],r['expert'],role) for role in roles)
    return score,selected,budget-spent


def audit(registry_path, quality_folder, linear_registry=None, policy_path=None):
    heads = qa.registry(registry_path)
    used_prototypes = {(l, p) for (l, e), m in heads.items() for p in m['prototypes']}
    assert not set(heads).intersection(used_prototypes),'Hosted native owners must stay native'
    host=inventory(heads,used_prototypes);budget=host['total_method_host_bytes']
    policy_path=Path(policy_path or DATA/'native-cache-policy-qualityVALID64-v1/policy.frozen.json')
    policy=qa.read(policy_path)
    assert policy['status']=='frozen_before_TEST' and policy['TEST_opened'] is False
    ranking=policy['primary_ranking'];order=[(r['layer'],r['expert']) for r in ranking]
    assert len(order)==len(set(order))==26*64
    gu_cache=set(order[:budget//GU]);expert_cache=set(order[:budget//EXPERT])
    _,mixed,left=mixed_cache(ranking,budget)
    linear=None;linear_mixed=[]
    if linear_registry:
        other=qa.registry(linear_registry);assert set(other)==set(heads)
        assert all(not m['prototypes'] for m in other.values()),'Raw-x control cannot hide another GU bank'
        for m in other.values():
            assert set(qa.np.fromfile(m['transform']+'.sources.u8',dtype=qa.np.uint8))=={0},'Matched linear control must use raw x'
        linear=inventory(other,used_prototypes)
        linear['matched_eligibility_same_GU_bank_modeled_flash_coefficient_work_equal_to_method']=True
        linear['same_GU_bank_is_charged_to_both']=True
        linear['runtime_coefficient_budget_equal_to_method']=(linear['total_method_host_bytes']==budget)
        alone=inventory(other)
        spare=budget-alone['total_method_host_bytes']
        assert spare>=0,'Matched raw payload exceeds nonlinear coefficient budget'
        _,linear_mixed,linear_left=mixed_cache([r for r in ranking if (r['layer'],r['expert']) not in heads],spare)
        linear['without_forced_shared_GU_bank']=alone
        linear['optimized_native_cache_in_remaining_budget']=dict(
            available_bytes=spare,selected=linear_mixed,unused_bytes=linear_left,
            policy='Same frozen VALID ranking; exclude resident readout experts; full/GU/D role-count enumeration')
    scopes_all=('NLL_prefix63','nonBOS_NLL_prefix62','BOS',
        'last_prefix_for_first_generated','last_prefix_without_generation',
        'decode_causal','decode_last_unused','generation_causal')
    events={name:Counter() for name in scopes_all}
    host_mac={name:0 for name in events};total_mac={name:0 for name in events}
    contexts={name:0 for name in events}
    latent_contexts={name:0 for name in events}
    for path in Path(quality_folder).glob('**/*.routing.jsonl'):
        summary=qa.read(path.with_name(path.name.removesuffix('.routing.jsonl')+'.summary.json'))
        assert summary['scored_tokens']==63,'Cost audit is scoped to the final64-token experiment'
        for route in qa.lines(path):
            layer=route['layer'];ids=route['expert_ids'];keys={(layer,e) for e in ids}
            selected=[heads[k] for k in keys if k in heads]
            required={(layer,p) for m in selected for p in m['prototypes']}
            natural=keys.intersection(used_prototypes)
            bucket=qa.phase(route,summary)
            if bucket=='NLL_prefix63':
                scopes=('NLL_prefix63','BOS' if route['token_in_batch']==0 else 'nonBOS_NLL_prefix62')
            elif bucket=='last_prefix_for_first_generated':
                scopes=(bucket,'generation_causal') if summary['decode_steps'] else ('last_prefix_without_generation',)
            elif bucket=='decode_causal':scopes=(bucket,'generation_causal')
            else:
                assert bucket=='decode_last_unused'
                scopes=(bucket,)
            for scope in scopes:
                events[scope].update(keys)
                readout_mac=sum(m.get('readout_coefficient_MAC',H*m['width']) for m in selected)
                host_mac[scope]+=readout_mac+len(required|natural)*2*H*I
                total_mac[scope]+=(6-len(selected))*3*H*I+readout_mac+len(required-keys)*2*H*I
                contexts[scope]+=len({m['basis'] for m in selected})
                latent_contexts[scope]+=sum(bool(m.get('factor_rank')) for m in selected)
    results={}
    for scope,event in events.items():
        if not event:continue
        native=sum(event.values())*EXPERT
        def remaining(cache,unit):return native-sum(n*unit for key,n in event.items() if key in cache)
        candidate=sum(n*(0 if key in heads else EXPERT-(GU if key in used_prototypes else 0)) for key,n in event.items())
        mixed_saving=sum(event[(l,e)]*(GU if role=='GU' else DOWN) for l,e,role in mixed)
        mixed_host=sum(event[(l,e)]*((2 if role=='GU' else 1)*H*I) for l,e,role in mixed)
        linear_optimized_bytes=(sum(n*(0 if key in heads else EXPERT) for key,n in event.items())
            -sum(event[(l,e)]*(GU if role=='GU' else DOWN) for l,e,role in linear_mixed))
        results[scope]=dict(events=sum(event.values()),native_no_cache_bytes=native,
            candidate_resident_readout_bytes=candidate,candidate_saving_fraction=1-candidate/native,
            same_budget_hot_GU_cache_bytes=remaining(gu_cache,GU),
            same_budget_full_expert_cache_bytes=remaining(expert_cache,EXPERT),
            same_budget_mixed_native_cache_bytes=native-mixed_saving,
            same_budget_mixed_native_cache_saving_fraction=mixed_saving/native,
            same_budget_raw_readouts_plus_optimized_native_cache_bytes=linear_optimized_bytes if linear_registry else None,
            same_budget_raw_readouts_plus_optimized_native_cache_saving_fraction=(1-linear_optimized_bytes/native) if linear_registry else None,
            modeled_method_host_coefficient_MAC=host_mac[scope],modeled_mixed_cache_host_coefficient_MAC=mixed_host,
            modeled_method_total_routed_MAC_including_extra_GU=total_mac[scope],
            modeled_native_total_routed_MAC=sum(event.values())*3*H*I,
            additional_transformed_A8_context_rows=contexts[scope],
            additional_private_latent_A8_context_rows=latent_contexts[scope])
    return dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        host_inventory=host,actual_matched_linear_control_inventory=linear,
        units=dict(native_expert_W8_scales_bytes=EXPERT, native_GU_W8_scales_bytes=GU,
                   native_D_W8_scales_bytes=DOWN),
        modeled_routed_FFN_coefficient_work=results,
        cache_policies=dict(selection=policy['selection_source'],policy=str(policy_path),
                            policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                            hot_GU=sorted(gu_cache),full_expert=sorted(expert_cache),
                            mixed_GU_D=mixed,mixed_unused_bytes=left,
                            mixed_fragmentation_choice='Full/GU/D role-count enumeration; frozen VALID counts; whole-matrix units.'),
        scope='All26 routed layers; TF input positions0..62, nonBOS/BOS, last prefix and causal/unused decode separate. Supply native routing for fixed-reference counterfactual; actual arm routing separately.',
        not_measured=['flash bytes','NVMeVirt','memory bandwidth','TPOT','speedup'],
        caveats=['Native fallback retained; original model capacity is not removed.',
                 'Shared FFN, attention, LM head, KV and workspace are not in the routed-coefficient denominator.',
                 'Host B is incremental: common backbone/KV/workspace must be added to every deployment total.',
                 'Full native residency is legal when total host memory fits the complete canonical model.',
                 'Unused TRAIN top8 prototypes are excluded from minimal method deployment inventory.',
                 'Host-resident coefficients still require host compute and memory traffic.',
                 'VALID counts used to select cache policies make their own VALID cost optimistic; TEST is required.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',required=True)
    p.add_argument('--quality',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--linear-registry')
    p.add_argument('--cache-policy')
    a=p.parse_args()
    result=audit(a.registry,a.quality,a.linear_registry,a.cache_policy)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
