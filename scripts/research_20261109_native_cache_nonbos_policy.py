#!/usr/bin/env python3
"""Append a non-BOS native cache policy; never modify frozen v1 artifacts."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from research_20261109_native_cache_policy import DATA, ROOT, GU, DOWN, FULL, rank, dump, sha


def main():
    old=DATA/'native-cache-policy-qualityVALID64-v1'
    parent=old/'policy.frozen.json'
    policy=json.loads(parent.read_text())
    assert policy['status']=='frozen_before_TEST'
    out=DATA/'native-cache-policy-qualityVALID64-nonBOS-v2'
    out.mkdir()
    nll=np.load(old/'counts_nll63.i32.npy')
    all64=np.load(old/'counts_prefill64.i32.npy')
    bos=np.zeros_like(nll)
    sources=json.loads((old/'source_files.json').read_text())
    ids={}
    same={str(l):True for l in range(1,27)}
    for file in sources:
        p=file['source_file_index']
        raw=Path(file['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==file['sha256']
        seen=set()
        for line in raw.splitlines():
            r=json.loads(line)
            if r['phase']!='prefill' or r['token_in_batch']!=0:continue
            l=r['layer'];assert l not in seen;seen.add(l)
            for e in r['expert_ids']:bos[p,l-1,e]+=1
            if l in ids:same[str(l)] &= tuple(r['expert_ids'])==ids[l]
            else:ids[l]=tuple(r['expert_ids'])
        assert len(seen)==26
    counts=nll-bos
    assert np.all(counts>=0) and int(counts.sum())==200*62*26*6
    assert int(bos.sum())==200*26*6
    primary=rank(counts,all64)
    np.save(out/'counts_nll_nonBOS62.i32.npy',counts)
    np.save(out/'counts_BOS.i32.npy',bos)
    columns=('rank','layer','expert','event_count','prompt_article_incidence','cumulative_event_count')
    (out/'ranking_nll_nonBOS62.tsv').write_text('\t'.join(columns)+'\n'+'\n'.join('\t'.join(str(r[c]) for c in columns) for r in primary)+'\n')
    dump(out/'BOS_router_observation.json',{'route_order_same_for_all_200_prompts_by_layer':same,
            'first_prompt_BOS_expert_ids_by_layer':{str(l):list(v) for l,v in ids.items()},
            'scope':'Router-ID observation only; this is not an implementation or bitcheck of aggregate-output caching.'})
    result={'status':'frozen_before_TEST','selection_source':'Same native fresh quality-VALID200 ledger as v1; not TRAIN.',
            'parent_policy_path':str(parent),'parent_policy_sha256':sha(parent),
            'source_code_sha256':sha(Path(__file__)),'prompts':200,'articles':200,'layers':26,'experts_per_layer':64,
            'primary_scope':'Only NLL-causal non-BOS prefill positions1..62, 62 positions per prompt.',
            'primary_event_count':int(counts.sum()),'BOS_event_count_separate':int(bos.sum()),
            'primary_policy_rule':'Non-BOS NLL62 event frequency descending; layer/expert ID ascending ties. For B>=0 choose the first min(1664,floor(B/unit_bytes)) entries. Freeze before TEST; no TEST reordering.',
            'units_bytes':{'GU_W8_plus_row_scales':GU,'D_W8_plus_row_scales':DOWN,'full_expert_W8_plus_row_scales':FULL},
            'primary_ranking':primary,'source_route_sha256':{r['path']:r['sha256'] for r in sources},
            'data_sha256':{str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()},
            'same_VALID_selection_and_estimation_is_optimistic':True,'TEST_opened':False,
            'BOS_aggregate_cache_counterfactual':{'F32_vector_payload_bytes':26*2048*4,'KiB':26*2048*4/1024,
                   'condition':'Assume deterministic BOS at reset position0 and an implemented replacement of each layer routed aggregate that preserves exact arithmetic/context behavior.',
                   'implemented':False,'all26_layer_aggregate_bitchecked':False,
                   'claim':'208KiB payload arithmetic only; not a validated exact-cache or measured benefit.'}}
    dump(out/'policy.frozen.json',result)
    shutil.copy2(Path(__file__),out/'executed_source.py')
    report={k:v for k,v in result.items() if k not in ('primary_ranking','source_route_sha256','data_sha256')}
    report.update(policy_path=str(out/'policy.frozen.json'),policy_sha256=sha(out/'policy.frozen.json'),primary_top20=primary[:20])
    dump(ROOT/'results/research-2026-11-09-shared-features/native_cache_nonBOS_qualityVALID_policy.json',report)
    assert sha(parent)==result['parent_policy_sha256']
    print(json.dumps({'status':'frozen','path':report['policy_path'],'sha256':report['policy_sha256'],
                      'nonBOS_events':result['primary_event_count'],'BOS_events':result['BOS_event_count_separate']}))


if __name__=='__main__':main()
