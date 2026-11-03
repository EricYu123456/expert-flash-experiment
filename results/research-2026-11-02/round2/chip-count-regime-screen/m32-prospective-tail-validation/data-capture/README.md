# Prospective blind G4 W8 capture

Capture is complete: **32 new prompts × 16 decode steps × 40 layers**, with six distinct routed experts at every recorded position. Four unchanged canonical-capture processes (eight threads each, eight prompts each) all exited zero in **68.013053 seconds** total inference wall, below the frozen 1,200-second budget. No prompt was replaced, retried, filtered by outputs or shortened after inference began.

No placement fitting, M32 cost replay, quality evaluation or three-plan paired numerical run was performed by this capture task. Other agents must wait for root's explicit release before reading the new routing archive.

## Prospective ordering

The three plans and analysis rules were frozen first under parent protocol SHA256 `911ecbdb77dcd18ccc880870f70a6a231c0a67730bfac7ac9144eea061e9b516`, completion SHA256 `36eb40d994a2163a9d066bd73777510f1b6bff1ede05bf894f686f551c7ef895`. All three plan file hashes were checked before acquisition/capture. The later stages are:

| Event | UTC |
|---|---|
| Data/capture protocol freeze | 2026-11-02 05:19:11.902971 |
| New payload acquisition began | 2026-11-02 05:20:38.896880 |
| Fixed prompt manifest completed | 2026-11-02 05:20:41.605215 |
| First canonical worker started | 2026-11-02 05:21:54.792428 |

The independent source check verifies strict parent-freeze → data-freeze → payload → capture time ordering. No new dataset payload was opened before the parent plan/analysis and data/capture protocols were frozen.

## Sources and fixed selection

There are 16 prompts from each of two previously unused project sources:

| Source | Pinned revision | Validation file | Bytes | SHA256 |
|---|---|---|---:|---|
| [PIQA author source](https://raw.githubusercontent.com/ybisk/ybisk.github.io/16c6be5562fdf2e56a6d5ee4f39a047494eb0617/piqa/data/valid.jsonl) | `16c6be5562fdf2e56a6d5ee4f39a047494eb0617` | `piqa/data/valid.jsonl` | 495,929 | `93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d` |
| [WinoGrande publisher dataset](https://huggingface.co/datasets/allenai/winogrande/tree/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande_xl) | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | `winogrande_xl/validation-00000-of-00001.parquet` | 85,928 | `f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1` |

PIQA's pre-acquisition metadata pinned Git blob SHA1 `da35972044f4a35e0053ff09a97ed7926ed3b8b4`, independently checked against the downloaded bytes; its SHA256 was then recorded above. HellaSwag's official GitHub metadata returned HTTP 451. It was replaced with WinoGrande before the data protocol; no HellaSwag payload or bypass mirror was fetched.

The fixed identity is each pinned file's zero-based row index. Selection sorts SHA256 of `seed NUL sourcekey NUL revision NUL split NUL sourceID`, with seed `M32-prospective-W8-v1-20261115`, and retains 16 per source under the frozen exact-duplicate exclusions. The old catalog contains **332 distinct IDs and prompt SHA256s**: every corpus-300 identity from existing capture validation metadata, plus the prior WikiText/ARC/OpenBookQA/BoolQ prompt manifests. No old heldout text, routing, outputs or answer sidecars were opened. Legacy initial trace basenames contain no prompt SHA in their replay metadata; this catalog is an exact ID/hash exclusion, not a guarantee of semantic deduplication or absence from model pretraining.

Prompts contain normalized goal/sentence, then two numbered candidate options. Correct labels are not appended. PIQA labels are a separate file that was never acquired; WinoGrande was read with only sentence/option1/option2 columns. The independent audit confirms all 96 original selected fields are nonempty strings, all selected IDs reproduce the frozen ranking, and all no-answer sample/prompt hashes match. It does not filter or replace any selected record.

The unchanged model vocabulary was used in a separate **vocab-only, no-inference** helper with exactly the collector's `add_special=true, parse_special=true` tokenizer flags. The frozen cap is 256 tokens including BOS, with a deterministic field-tail truncation rule specified before payload access. Actual prompts contain **16–162 tokens**; none required truncation. All final prompt bytes equal their full fixed no-answer rendering.

## Original numerical collector and complete coverage

The existing binary is `w8-format-training/build/granite4tiny-canonical-capture`, SHA256 `8f5171680fdab9c0284c11af11ac214998a2087314d2302e0af869f5a8476b80`. Its C++ source is unchanged, SHA256 `4e99c2a318e17e85c8c91fd72b41c45fe53f1528234c4679f291edeaa5859380`. The existing local GGUF is unchanged, SHA256 `5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd`; no checkpoint was downloaded.

The collector preserves the established full-row W8/full-vector A8/I64-dot/one-F32-rescale contract, including every routed and shared GU/Down projection in all 40 layers. Attention and router retain the original GGUF path. Its original 16-chip plan argument is retained only as required by that collector; none of the three new M32 plans is executed numerically. The new capture therefore does **not** extend the prior paired-placement numerical proof to those plans.

Coverage checks include all prefill positions and all 16 decode positions for all 40 layers, all 240 routed/shared projection tensors per prompt, unique top-6 IDs in range, exact positions/token IDs and no missing/duplicate records. There are **1,387 prefill tokens**, 512 complete decode positions, **1,438,986,240 finite full-FFN output values** and **54,591,488 finite continuation logits**. There are zero prompt validation failures. All **7,710** unique weight/cache entries visited in the new run match existing TRAIN metadata for canonical W8 bytes, row-scale hashes, source offsets/shapes and the original collector map; no new weight payload was opened for that metadata regression.

The output occupies about 273 MiB and local free space remains about 204 GiB, above the enforced 40 GiB floor. All raw/partial policies remain as frozen even though no failure occurred.

## Archive and integrity artifacts

- `granite4tiny/decode.routes.npz`: `routes int32[32,16,40,6]`, `prompt_ids/source_ids/source_keys[32]`, `layers int32[40]`, `completed bool[32]` (all true), `token_positions/token_ids int32[32,16]`, representation `fixed_full_FFN_W8A8_canonical_only`, split `prospective_blind`. Dataset item IDs and source grouping keys are separate fields.
- `granite4tiny/prefill.routes.npz`: separate prefill routes/offsets, excluded from the prospective decode analysis.
- `capture_summary.json`: complete status, all-40-layer finite/coverage guard, counts, per-batch reports and protocol/manifest/archive hashes. Compatibility aliases include `completed_prompts`, `all40layersfinite` and `decode_archive_sha256`.
- `independent-source-check/report.json`: strict raw selected-field types, deterministic IDs, exact no-answer sample/prompt hashes, exclusions and UTC ordering. `cache_metadata_regression.json` records the common-entry numerical-format metadata bridge.

Data/capture protocol SHA256: `b4f75b76c751b4afd870f1312b1522bb59b9adf7be8c980214ee7b4be00be3a1`.

Prompt manifest SHA256: `2e3f10d9d9505bf0ffee3af40e701c06be95a951c17f2785b02f10b8e24d533c`.

Decode archive SHA256: `d4f7cc54970ef1223668c92b68eea22f22c775be9f3293fa61203601df8535ca`.
