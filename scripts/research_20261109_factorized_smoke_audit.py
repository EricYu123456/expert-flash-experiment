#!/usr/bin/env python3
"""Audit factor runtime identity, live counters and 64-token AR smoke."""
import json
from pathlib import Path
import shutil

import research_20261109_quality_analysis as qa
from research_20261109_shared_feature_targets import DATA, ROOT, dump, sha


def read(path):return json.loads(Path(path).read_text())


def identity(new,old):
    rows=[]
    for p in sorted(new.glob('*.summary.json')):
        a,b=read(p),read(old/p.name);key=a['prompt_id']
        apos=list(qa.lines(new/(key+'.positions.jsonl')))
        bpos=list(qa.lines(old/(key+'.positions.jsonl')))
        assert apos==bpos
        for field in ('token_ids','generated_token_ids','actual_generated_length','decoded_generated_inputs','stopped_eog','sum_nll_F64','mean_nll_F64'):
            assert a[field]==b[field],(key,field)
        def routes(folder):
            return {(r['phase'],r['step'],r['token_in_batch'],r['layer']):r['expert_ids'] for r in qa.lines(folder/(key+'.routing.jsonl'))}
        assert routes(new)==routes(old)
        rows.append({'prompt_id':key,'NLL_top1_token_records_identical':True,'generated_tokens_identical':True,
                     'all_native_route_IDs_identical':True,'scored_positions':len(apos),'generated_length':a['actual_generated_length']})
    assert len(rows)==2
    return rows


def main():
    root=DATA/'factorized-runtime-smoke'
    exports=DATA/'factorized-readouts/formal-export'
    checks={'native':identity(root/'native2',DATA/'benchmark/threads4_vs8/threads4/data'),
            'dense_legacy':identity(root/'dense2',DATA/'smoke/new_identity')}
    ids=[line.split('\t')[1] for line in (DATA/'smoke/jobs2.tsv').read_text().splitlines()]
    arms={}
    for name,registry in [('native2','-'),('dense2',str(DATA/'smoke/identity.registry.tsv')),
          ('prototype25_rank256_k12',str(exports/'registry.prototype25.rank256.k12.tsv')),
          ('rawsubset_rank256_k12',str(exports/'registry.raw_x_select1408_eq.rank256.k12.tsv'))]:
        arm=qa.load_arm(root/name,registry,ids)
        arms[name]={'causal_exposure':qa.aggregate(arm),'prompts':{}}
        for key,s in arm['summaries'].items():
            if 'rank256' in name:assert s['private_latent_A8_rows']==s['replaced_expert_token_events']
            arms[name]['prompts'][key]={k:s[k] for k in ('scored_tokens','actual_generated_length','decoded_generated_inputs','stopped_eog',
                'private_latent_A8_rows','factorized_latent_affine_values','factorized_V_coefficient_MAC','factorized_U_coefficient_MAC',
                'basis_A8_unique_rows','prototype_GU_row_pairs_computed','prototype_owner_selected_phi_floats_bitchecked',
                'native_shared_z_floats_bitchecked')}
    numeric=read(exports/'new_runtime_cpp_audit.json')
    assert numeric['status']=='passed' and numeric['model_count']==72
    assert sum(m['rank']==256 for m in numeric['models'])==24
    freeze=root/'executed-freeze';freeze.mkdir(exist_ok=True)
    sources=[ROOT/'scripts'/name for name in ('research_20261109_shared_feature_quality_factorized.cpp',
                'research_20261109_factorized_readout.hpp','research_20261109_factorized_numeric.cpp',
                'research_20261109_factorized_runtime_probe.py','research_20261109_factorized_smoke_audit.py',
                'research_20261109_quality_analysis.py')]
    sources += [DATA/'build/shared-feature-quality-factorized',DATA/'build/factorized-numeric-v1.so']
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,freeze/p.name)
    report={'status':'passed','source_and_binary_sha256':hashes,'identity_checks':checks,'arms':arms,
            'full_numeric_probe_path':str(exports/'new_runtime_cpp_audit.json'),
            'full_numeric_probe_sha256':sha(exports/'new_runtime_cpp_audit.json'),
            'full_numeric_models':72,'rank256_models':24,'full_numeric_output_floats_bitchecked':numeric['output_floats_bitchecked'],
            'EOG_scope':'Stop rule and length/decoded-input invariants audited; these natural TRAIN smoke generations did not stop at EOG, so this does not claim a separately forced EOG execution.',
            'quality_scope':'Two TRAIN prompts per smoke arm, 64-token prefix; factor/dense8-step AR, native16-step identity. No new TEST or statistical quality claim.'}
    dump(root/'audit.json',report)
    dump(ROOT/'results/research-2026-11-09-shared-features/factorized_runtime_smoke_audit.json',report)
    shutil.copy2(exports/'new_runtime_cpp_audit.json',ROOT/'results/research-2026-11-09-shared-features/factorized_runtime_cpp_full.json')
    print(json.dumps({'status':'passed','native_and_dense_identity_prompts':4,'factor_live_smoke_prompts':4,
                      'binary_sha256':hashes[str(DATA/'build/shared-feature-quality-factorized')],
                      'numeric_output_floats':numeric['output_floats_bitchecked']}))


if __name__=='__main__':main()
