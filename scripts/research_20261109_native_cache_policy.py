#!/usr/bin/env python3
"""Freeze all-layer hot-cache ranking from native quality-VALID only.

Primary policy: NLL-causal positions 0..62 of 64-token prefill, descending
expert-event frequency, then layer/expert ID. Given bytes B, select the first
floor(B/unit_bytes) keys. Counts on the selection corpus are optimistic
calibration statistics; primary cost must be evaluated on frozen-policy TEST.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from research_20261109_shared_feature_targets import DATA, ROOT, dump, sha

H,I,LAYERS,EXPERTS,K=2048,1408,26,64,6
GU=2*H*I+8*I
DOWN=H*I+4*H
FULL=GU+DOWN


def cache_prefix(policy,budget,kind='GU'):
    """Apply a pre-TEST ranking, without reading any evaluation event counts."""
    unit={'GU':GU,'full_expert':FULL}[kind]
    n=min(LAYERS*EXPERTS,int(budget)//unit)
    return {(r['layer'],r['expert']) for r in policy['primary_ranking'][:n]}


def rank(counts,full):
    total=counts.sum(axis=0)
    aux=full.sum(axis=0)
    order=sorted([(l,e) for l in range(1,27) for e in range(64)],key=lambda k:(-int(total[k[0]-1,k[1]]),k))
    result=[]
    cumulative=0
    for n,(l,e) in enumerate(order,1):
        count=int(total[l-1,e]);cumulative+=count
        result.append({'rank':n,'layer':l,'expert':e,'event_count':count,
                       'prompt_article_incidence':int(np.sum(counts[:,l-1,e]>0)),
                       'other_prefill_scope_event_count':int(aux[l-1,e]),
                       'cumulative_event_count':cumulative})
    return result


def main():
    source=DATA/'quality_valid_native'
    out=DATA/'native-cache-policy-qualityVALID64-v1'
    out.mkdir()
    corpus_path=DATA/'corpus103/manifest.json'
    corpus=json.loads(corpus_path.read_text())
    expected={r['id']:r for r in corpus['records'] if r['split']=='quality_validation'}
    assert len(expected)==200 and len({r['article_id'] for r in expected.values()})==200
    summaries=sorted(source.glob('**/*.summary.json'))
    assert len(summaries)==200
    files=[]
    count63=np.zeros((200,26,64),np.int32)
    count64=np.zeros_like(count63)
    records=0
    prompt_ids=[]
    seen_prompts=set()
    tokens=set()
    ledger=out/'route_provenance.jsonl'
    with ledger.open('w',buffering=1<<20) as provenance:
        for p,path in enumerate(summaries):
            summary=json.loads(path.read_text())
            prompt=summary['prompt_id']
            assert prompt in expected and prompt not in seen_prompts
            seen_prompts.add(prompt);prompt_ids.append(prompt)
            assert summary['input_tokens']==64 and summary['scored_tokens']==63 and summary['decode_steps']==0
            assert summary['replaced_expert_token_events']==0
            assert len(summary['token_ids'])==64
            assert tuple(summary['token_ids']) not in tokens
            tokens.add(tuple(summary['token_ids']))
            routefile=path.with_name(prompt+'.routing.jsonl')
            raw=routefile.read_bytes()
            identity=hashlib.sha256(raw).hexdigest()
            files.append({'source_file_index':p,'path':str(routefile),'sha256':identity,
                          'summary_path':str(path),'summary_sha256':sha(path),'prompt_id':prompt,
                          'article_id':expected[prompt]['article_id']})
            seen=set()
            for number,line in enumerate(raw.splitlines(keepends=True),1):
                r=json.loads(line)
                assert r['phase']=='prefill' and r['step']==-1
                t,l,ids=r['token_in_batch'],r['layer'],r['expert_ids']
                assert 0<=t<64 and 1<=l<=26 and len(ids)==len(set(ids))==6 and all(0<=e<64 for e in ids)
                assert (t,l) not in seen;seen.add((t,l))
                for e in ids:
                    count64[p,l-1,e]+=1
                    if t<63:count63[p,l-1,e]+=1
                item={'source_file_index':p,'source_file_sha256':identity,'source_line':number,
                      'source_record_sha256_including_newline':hashlib.sha256(line).hexdigest(),
                      'prompt_id':prompt,'article_id':expected[prompt]['article_id'],'route':r}
                provenance.write(json.dumps(item,separators=(',',':'))+'\n')
                records+=1
            assert len(seen)==64*26
    assert seen_prompts==set(expected) and records==200*64*26
    assert int(count63.sum())==200*63*26*6 and int(count64.sum())==200*64*26*6
    np.save(out/'counts_nll63.i32.npy',count63)
    np.save(out/'counts_prefill64.i32.npy',count64)
    dump(out/'source_files.json',files)
    dump(out/'prompt_order.json',prompt_ids)
    primary,secondary=rank(count63,count64),rank(count64,count63)
    columns=('rank','layer','expert','event_count','prompt_article_incidence','cumulative_event_count')
    for name,ranking in [('nll63',primary),('prefill64',secondary)]:
        (out/f'ranking_{name}.tsv').write_text('\t'.join(columns)+'\n'+'\n'.join('\t'.join(str(r[c]) for c in columns) for r in ranking)+'\n')
    policy={'status':'frozen_before_TEST','selection_source':'Native 200-article fresh quality-VALID64; NOT TRAIN.',
            'selection_source_folder':str(source),'source_corpus_manifest_sha256':sha(corpus_path),
            'source_code_sha256':sha(Path(__file__)),'prompts':200,'articles':200,'layers':26,'experts_per_layer':64,
            'primary_scope':'NLL causal prefill positions0..62; 63 positions per prompt, including BOS.',
            'auxiliary_scope':'All64 prefill positions; last prefix is not NLL-causal in AR0 runs.',
            'primary_event_count':int(count63.sum()),'auxiliary_event_count':int(count64.sum()),
            'primary_policy_rule':'Sort NLL63 event count descending, layer/expert ID ascending ties. For arbitrary incremental B choose the first min(1664,floor(B/unit_bytes)) entries; no TEST-driven reordering or scope choice.',
            'units_bytes':{'GU_W8_plus_row_scales':GU,'D_W8_plus_row_scales':DOWN,'full_expert_W8_plus_row_scales':FULL},
            'primary_ranking':primary,'auxiliary_prefill64_ranking':secondary,
            'provenance_files':{str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()},
            'calibration_caveat':'Ranking and apparent hit/byte savings on this same quality-VALID are optimistic. Compare frozen policies only on >=1000 fresh TEST prompts for the primary result.',
            'scope_limits':['No router alpha exists for all26 layers in these route logs, so ranking is event-frequency, not alpha-weighted.',
                            'GU-only and full-expert prefixes are legal uniform-size static policies, not a claim of globally optimal mixed-granularity cache allocation.',
                            'Per-event uncached coefficient-byte work is not physical flash traffic, latency, TPOT or a speedup.'],
            'TEST_opened':False}
    dump(out/'policy.frozen.json',policy)
    shutil.copy2(Path(__file__),out/'executed_source.py')
    report={k:v for k,v in policy.items() if k not in ('primary_ranking','auxiliary_prefill64_ranking','provenance_files')}
    report['policy_path']=str(out/'policy.frozen.json')
    report['policy_sha256']=sha(out/'policy.frozen.json')
    report['primary_top20']=primary[:20]
    report['source_route_files']=len(files)
    report['route_records_with_source_and_line_SHA']=records
    report['ranking_tsv_paths']=[str(out/f'ranking_{name}.tsv') for name in ('nll63','prefill64')]
    dump(ROOT/'results/research-2026-11-09-shared-features/native_cache_qualityVALID_policy.json',report)
    print(json.dumps({'status':'frozen','policy_path':report['policy_path'],'policy_sha256':report['policy_sha256'],
                      'events63':policy['primary_event_count'],'events64':policy['auxiliary_event_count']}))


if __name__=='__main__':main()
