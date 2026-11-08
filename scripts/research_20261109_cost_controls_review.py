#!/usr/bin/env python3
"""Independent cost review and calibration-only all-layer cache comparisons."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from research_20261109_native_cache_policy import DATA, ROOT, GU, DOWN, FULL, cache_prefix, dump, sha
import research_20261109_quality_analysis as qa


def inventory(path):
    heads=qa.registry(path)
    owners={(l,p) for (l,e),m in heads.items() for p in m['prototypes']}
    assert not owners.intersection(heads)
    files={Path(m['head']+s).resolve() for m in heads.values() for s in ('.i8','.scales.f32','.bias.f32')}
    transforms={Path(m['transform']+s).resolve() for m in heads.values()
                for s in ('.sources.u8','.indices.i32','.center.f32','.inv_scale.f32','.prototypes.i32') if Path(m['transform']+s).exists()}
    for m in heads.values():
        assert Path(m['head']+'.i8').stat().st_size==2048*m['width']
        assert Path(m['head']+'.scales.f32').stat().st_size==8192
        assert Path(m['head']+'.bias.f32').stat().st_size==8192
    head_bytes=sum(p.stat().st_size for p in files)
    transform_bytes=sum(p.stat().st_size for p in transforms)
    return heads,owners,{'registry_path':str(path),'registry_sha256':sha(path),
                         'readout_bytes':head_bytes,'transform_metadata_bytes':transform_bytes,
                         'native_prototype_GU_bytes':len(owners)*GU,
                         'incremental_host_bytes':head_bytes+transform_bytes+len(owners)*GU}


def savings(events,heads,owners):
    assert not owners.intersection(heads)
    base=int(events.sum())*FULL
    saved=sum(int(events[l-1,e])*FULL for l,e in heads)
    saved+=sum(int(events[l-1,e])*GU for l,e in owners)
    return {'baseline_all26_routed_event_coefficient_bytes':base,
            'saved_event_coefficient_bytes':saved,'saved_fraction':saved/base}


def main():
    folder=DATA/'cost-controls-review'
    folder.mkdir(exist_ok=True)
    original=ROOT/'scripts/research_20261109_cost_audit.py'
    reviewed_sha=sha(original)
    (folder/'reviewed_cost_audit.py').write_bytes(original.read_bytes())
    root_policy=DATA/'native-cache-policy-qualityVALID64-v1'
    nonbos_policy=DATA/'native-cache-policy-qualityVALID64-nonBOS-v2'
    policies={'NLL63':json.loads((root_policy/'policy.frozen.json').read_text()),
              'nonBOS_NLL62':json.loads((nonbos_policy/'policy.frozen.json').read_text())}
    events={'NLL63':np.load(root_policy/'counts_nll63.i32.npy').sum(axis=0),
            'nonBOS_NLL62':np.load(nonbos_policy/'counts_nll_nonBOS62.i32.npy').sum(axis=0)}
    table=[]
    regs=DATA/'proto25-cohort12/registries'
    for cohort in (3,6,9,12):
        heads,owners,inv=inventory(regs/f'registry.prototype25.k{cohort}.tsv')
        fullraw,_,fullinv=inventory(regs/f'registry.raw_x_eq.k{cohort}.tsv')
        subset,_,subinv=inventory(regs/f'registry.raw_x_select1408_eq.k{cohort}.tsv')
        assert set(heads)==set(fullraw)==set(subset)
        record={'cohort':cohort,'method_inventory':inv,'raw2048_inventory_without_extra_GU':fullinv,
                'raw1408_inventory_without_extra_GU':subinv,
                'raw1408_same_GU_inventory_bytes':subinv['incremental_host_bytes']+len(owners)*GU,
                'scopes':{}}
        for scope,policy in policies.items():
            event=events[scope]
            unit_results={}
            base=int(event.sum())*FULL
            for kind,unit in (('GU',GU),('full_expert',FULL)):
                keys=cache_prefix(policy,inv['incremental_host_bytes'],kind)
                avoided=sum(int(event[l-1,e])*unit for l,e in keys)
                unit_results[kind]={'cached_layer_experts':sorted(keys),'cache_payload_bytes':len(keys)*unit,
                                    'unspent_budget_bytes':inv['incremental_host_bytes']-len(keys)*unit,
                                    'saved_event_coefficient_bytes':avoided,'saved_fraction':avoided/base}
            record['scopes'][scope]={'method_and_same_eligibility_same_GU_resident_raw':savings(event,heads,owners),
                                    'same_budget_all26_native_cache':unit_results,
                                    'warning':'Optimistic same-VALID selection/evaluation. Native-route counterfactual only; not fresh TEST, candidate AR exposure, flash IO or speed.'}
        table.append(record)
    metadata_path=Path('/home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-metadata.json')
    spec=importlib.util.spec_from_file_location('historical_static_cost_readonly',ROOT/'scripts/research_20261108_shared_feature_cost_audit.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    static=old.static_cost(json.loads(metadata_path.read_text()))
    weight=static['full_model_weight_accounting']
    issues=[
      {'priority':1,'lines':[44,56,59],'finding':'Ranking and denominator are restricted to 3 layers. Final single-layer cohorts require all26-layer native cache competitors and all26 routed denominator; shared FFN/attention/LMhead remain excluded from that arithmetic denominator.'},
      {'priority':1,'lines':[62,64],'finding':'With all private C resident and identical eligible experts/native GU pool, raw-x and nonlinear readout have exactly the same modeled flash coefficient work. Representation advantage must come from quality/coverage/host cost, not assigning raw C to flash.'},
      {'priority':1,'lines':[66,72],'finding':'Hard-coded fullraw2048 is not the actual same-width rawsubset1408 control. Read actual registry payload and transform sizes, and optionally fill remaining equal budget using the frozen native-cache ranking.'},
      {'priority':1,'lines':[29,62],'finding':'Assert prototype owners are disjoint from substituted heads. Native-owner overlap can offset extra GU only for identical native weights and an unchanged native owner path; a learned GU with the same owner ID is a distinct basis/cache.'},
      {'priority':2,'lines':[41,89],'finding':'Incremental C+GU budget is not a complete host memory budget. Specify common native shared FFN, nonFFN tensors, KV/workspace, page/allocator overhead and whether original fallback stays flash or host; full native residency is a legitimate upper-budget control.'},
      {'priority':2,'lines':[62,87],'finding':'Resident C has no flash coefficient reads, but still has host weight/activation traffic, H*r readout MACs, transform/A8 contexts and potentially serial GU→SwiGLU→A8→C latency. Do not infer speedup from coefficient access work.'},
      {'priority':2,'lines':[56,83],'finding':'Separate deterministic BOS from nonBOS; freeze rankings before TEST. NLL63/all64/decode exposures need separate causal scopes, and VALID-selected cache hit estimates on VALID are optimistic.'},
      {'priority':2,'lines':[49,50],'finding':'Pure GU/full-expert equal-unit prefixes are valid strong policies, not a global optimum over arbitrary mixed GU/D/full-expert caching and page granularity.'}]
    report={'status':'review_complete','reviewed_source_path':str(original),'reviewed_source_sha256':reviewed_sha,
            'review_source_sha256':sha(Path(__file__)),'findings':issues,'calibration_cost_table':table,
            'units_bytes':{'GU':GU,'D':DOWN,'full_expert':FULL},
            'full_model_weight_budget':{'metadata_sha256':sha(metadata_path),
                'modeled_native_all_weight_bytes':weight['baseline_modeled_weight_bytes']['bytes'],
                'modeled_native_all_weight_GiB':weight['baseline_modeled_weight_bytes']['bytes']/2**30,
                'source_GGUF_file_bytes':weight['actual_source_GGUF_file_bytes_retained_by_functional_harness'],
                'native_shared_GU_all26_bytes':26*(2*2048*2816+8*2816),
                'all26_routed_native_W8_scales_bank_bytes':26*64*FULL,
                'note':'If available host budget can hold all native model weights plus KV/workspace, native all-resident is exact and has zero routed flash coefficient work. The harness retains source GGUF plus verification caches; this inventory is not measured process RSS.'},
            'compute_formulas':{'native_routed_MAC_per_event':3*2048*1408,
                'native_prototype_GU_MAC_per_required_row':2*2048*1408,
                'C_MAC_per_replaced_event':'2048 * input_width',
                'extra_prototype_rows':'Per token/layer deduplicate required native owners; only required owners not naturally selected count as extra beyond native work.',
                'optimized_MAC_after':'native_MAC - replaced_events*3HI + sum(H*readout_width) + extra_native_proto_rows*2HI; not the functional harness wall time.',
                'width1408_single_replacement_no_owner_hit':'Extra GU 2HI + C HI = original expert 3HI; compute savings require same-row multihead reuse or native-owner overlap.',
                'A8_contexts':'Count unique (layer,basis_id) actually used on each token; distinct transformations are distinct contexts even when prototype phi is shared.'},
            'no_TEST_or_system_benchmark':True}
    dump(folder/'review.json',report)
    dump(ROOT/'results/research-2026-11-09-shared-features/cost_controls_review.json',report)
    lines=['成本審查針對 source SHA '+reviewed_sha+' 的 research_20261109_cost_audit.py；主 agent 後續修正時可逐項核對。GU／D／expert 的 W8+scale 單位正確，但現版 ranking 與分母只取三層，對最終單層小 cohort 的 native cache 強控制不足。', '',
           '已凍結兩份全 26 層 VALID-only cache 排名：NLL positions0..62 的 v1 與 nonBOS positions1..62 的 v2。它們可按任意非負 byte budget 取 GU／full-expert prefix；同份 VALID 估計成本偏樂觀，主要成本須用政策凍結後的 1000 TEST。', '',
           '| Cohort | 方法增量 host MiB | nonBOS 方法／matched raw 節省 | 同 budget native GU cache | 同 budget native full-expert cache |',
           '|---|---:|---:|---:|---:|']
    for r in table:
        s=r['scopes']['nonBOS_NLL62']
        lines.append(f"| {r['cohort']} | {r['method_inventory']['incremental_host_bytes']/2**20:.3f} | {100*s['method_and_same_eligibility_same_GU_resident_raw']['saved_fraction']:.4f}% | {100*s['same_budget_all26_native_cache']['GU']['saved_fraction']:.4f}% | {100*s['same_budget_all26_native_cache']['full_expert']['saved_fraction']:.4f}% |")
    lines.extend(['', '此表只列 native VALID 路由上的 uncached per-event 係數存取工作反事實，分母為全 26 層 routed FFN；不包含 shared FFN、attention、LMhead 工作，也不是 flash IO 或速度。所有 readouts 留 host 時，相同 eligibility／GU cache 的 raw 與 nonlinear readout 的這項工作完全相同；差別必須由品質、可替換覆蓋與 host 容量說明。', '',
                  'Host 常駐不代表計算免費。必須另列實際 required prototype rows、可抵扣的 native-owner overlap、額外 GU MAC、readout MAC、transform/A8 contexts 與 host traffic。width1408 時，一個 row 僅替換一個 expert 且 owner 未被選中，GU 2HI＋C HI 正好等於原 expert 3HI；沒有算術節省。learned GU 即使沿用 owner ID，也不能當作原生 self hit 抵扣。', '',
                  '原生全模型 modeled weights 約 14.718 GiB，尚需 KV／workspace 等容量。若 host budget 放得下，完整 native 常駐就是合法強控制，routed flash 係數工作為零。若採較小的增量 budget，需清楚列出共用 shared FFN／其他 weights 的常駐前提，以及 fallback 位在 flash 還是 host。', '',
                  'BOS router 重複只是一項觀察。208 KiB 的全層 F32 aggregate cache 是有條件的 payload 算術，尚未實作並驗證全層 bit-exact replacement，不能報成已驗證 exact cache 或實測收益。完整優先問題、actual rawsubset inventory 與數值表保存在 cost_controls_review.json。'])
    (ROOT/'results/research-2026-11-09-shared-features/cost_controls_review.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','reviewed_sha256':reviewed_sha,'nonBOS_table':[{'cohort':r['cohort'],'host_MiB':r['method_inventory']['incremental_host_bytes']/2**20,'fractions':r['scopes']['nonBOS_NLL62']} for r in table]}))


if __name__=='__main__':main()
