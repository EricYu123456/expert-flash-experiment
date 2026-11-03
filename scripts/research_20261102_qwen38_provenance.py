#!/usr/bin/env python3
"""Seal provenance and current public-recorder lookup limits, without new traces."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-2026-11-02/round2/qwen38-trace-audit'
REV = 'fee2af4ea0da212d853d82943dad3897653066a2'
ENGINE = 'd8f683672709fd13a45e96651ae9a244ede396e9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(url):
    r = requests.get(url, timeout=30)
    return dict(url=url, status_code=r.status_code, response_bytes=len(r.content),
                response_sha256=hashlib.sha256(r.content).hexdigest(),
                short_response=r.text if len(r.content)<1000 else None)


def main():
    urls = [f'https://api.github.com/repos/FlashML-org/FreeToken/commits/{ENGINE}',
            f'https://raw.githubusercontent.com/FlashML-org/FreeToken/{ENGINE}/tools/expert_trace/read.py',
            'https://raw.githubusercontent.com/FlashML-org/FreeToken/expert-trace-study/tools/expert_trace/read.py']
    with ThreadPoolExecutor(max_workers=3) as pool:
        lookups = list(pool.map(check, urls))
    cfg = json.loads((OUT/'checkpoint.config.json').read_text())
    doc_receipts = json.loads((OUT/'doc_receipts.json').read_text())
    for receipt in doc_receipts:
        assert sha(receipt['path']) == receipt['sha256']
        receipt['reading_depth'] = 'entire file read'
    review = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        dataset=dict(name='aswinkumar99/qwen3.8-flash-next-expert-traces', revision=REV,
            url=f'https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces/tree/{REV}',
            license_in_dataset_card='CC-BY-4.0', attribution='aswinkumar99; exact fixed revision above',
            model_license_caveat='Dataset card calls source model Apache-2.0; RadixArk checkpoint card instead says license: other and points to Qwen source terms. Do not substitute the dataset assertion for model license.'),
        documents=doc_receipts,
        recorder=dict(engine='FreeToken 0.1.2 per dataset card',
            upstream='https://github.com/FlashML-org/FreeToken',
            header_commit=ENGINE, header_git_status_porcelain=None,
            advertised_branch='expert-trace-study',
            current_public_lookups=lookups,
            conclusion='Original recorder code was not retrieved from supplied public pointers at audit time. The independent binary parser and arithmetic spot-check do not establish original engine reproducibility.'),
        model=dict(checkpoint='RadixArk/Qwen3.8-Flash-Next-NVFP4',
            checkpoint_revision_checked='7b719225242aacd3dbd3f9407468c2ee9a9d2594',
            original_checkpoint_revision_present_in_trace_header=False,
            model_type=cfg['text_config']['model_type'],
            shapes={k: cfg['text_config'][k] for k in ['hidden_size','moe_intermediate_size','num_experts','num_experts_per_tok','num_hidden_layers','shared_expert_intermediate_size']},
            config_sha256=sha(OUT/'checkpoint.config.json'),
            router='Saved router weight matrices are float32 widened BF16; no bias per dataset. Independently replayed saved BF16 state and weights, then BF16-rounded GEMV logits.',
            precision='Routed experts NVFP4; gate/shared experts BF16. An I8 NAND study using these route IDs is a conditional geometry model, not original-checkpoint AiF execution.'),
        source_workload=dict(task='b01-cpp-ring-threadpool', source='diagnostic/smoke-batched-8way',
            task_kind='One systems C++ completion task; two rounds of eight prompt variants.',
            request_count=16, decode_rows=12784, prefill_rows=1920,
            prefix_cache='Shared prompt prefixes are cached; prefill rows are only executed uncached positions.',
            shared_expert='Always active shared expert is not represented by routed top-10 IDs.',
            source_cache='All 24576 routed experts resident, no eviction. hit_mask/slots are cold-start coverage, not deployment offload performance.',
            source_batch='Eight requests per source step; timing/cache fields are shared batch metadata.',
            split='All sixteen requests fixed as exploratory training before parsing routes. Other task binary payloads remain unopened.',
            client_tokens='Recorded decode forwards are not guaranteed identical to visible streamed chunks; runner live counters and final trace counts differ.'),
        independent_validation=dict(audit_path=str(OUT/'audit.json'), audit_sha256=sha(OUT/'audit.json'),
            parser='Independent fixed-width parser; full token-layer joins, ID ranges/uniqueness, top32 first10 and token history checked.',
            binary_spec_ambiguity='Actual extended step uses token_id byte28, previous IDs byte32, PLE byte44 and trailing four pad bytes; append-at32 reading contradicts token history.',
            spot_check='Four deterministic evenly spaced decode token rows x five layers (0,12,24,36,47), 20 router GEMVs. All ordered top10/top32 exact, centered saved top32 logits exact after BF16 rounding.',
            limits='Small arithmetic consistency audit, not independent model generation or every-layer-every-token arithmetic replay.'),
        source_storage=dict(main_trace_range='0..150809895 inclusive (all main token blocks, no complete states section)',
            main_trace_range_sha256='6e5b0919142ffa781333599802fd3f1ca0d3e9a37357787d88bc36b76aecb595',
            router_weights_sha256='c9914cf0d6021a7ad445a617ffd7fb2da3a28b57e3df48d69c929efcf4567c70',
            expected_full_trace_lfs_sha256='e97d6063aee6e4c7e620980193e850e58a6a100998dc0e024a0fcc0be6aabf95',
            full_trace_sha256_locally_verified=False,
            substantive_payload_bytes=461375877, initial_budget_bytes=10000000000,
            paths_and_schema=str(OUT/'schema.json'), schema_sha256=sha(OUT/'schema.json')))
    target = OUT/'provenance_review.json'
    target.write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps(dict(path=str(target), sha256=sha(target), public_recorder_status=[x['status_code'] for x in lookups]), indent=2))


if __name__ == '__main__':
    main()
