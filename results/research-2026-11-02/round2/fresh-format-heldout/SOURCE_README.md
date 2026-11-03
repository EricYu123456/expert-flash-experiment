# Fresh same-format text sources

The fixed [manifest](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/fresh-format-heldout/manifest.json) contains eight WikiText2 raw test chunks and eight ARC-Challenge validation questions. The [acquisition protocol](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/fresh-format-heldout/acquisition_protocol.frozen.json) was written before either target payload was downloaded; [selection_audit.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/fresh-format-heldout/selection_audit.json) checks source membership, deterministic ranks, rendering and hashes.

Selection uses seed20261103 through SHA256 ranking. WikiText prompts contain the first160 whitespace words of eligible nonheading rows. ARC prompts retain the complete question and four choices in source order, with no answer appended. Source records and the [answer sidecar](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/fresh-format-heldout/answers/arc_answers.json) are separate from prompt files. Item order is fixed: eight LM items followed by eight ARC items.

| Publisher source | Dataset revision | Split | Payload bytes |
|---|---|---|---:|
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3) | b08601e04326c79dfdd32d625aee71d232d685c3 | wikitext-2-raw-v1 test | 732,610 |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453) | 210d026faf9955653af8916fad021475a3f00453 | ARC-Challenge validation | 55,743 |

Both raw file SHA256 values match the pinned publisher LFS digests. Full source Parquet files, pinned dataset cards, selected original text records and all sixteen UTF8 prompts are retained. Dataset revisions are distinct from the model checkpoint revisions; the existing model's SHA256 and original placement/numeric protocol hashes are in the acquisition protocol.

Publisher attribution/license statements are retained in [source_provenance_notes.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/fresh-format-heldout/source_provenance_notes.json). WikiText's pinned card is internally inconsistent: YAML states CC BY-SA3.0 and GFDL while its prose links CC BY-SA4.0. ARC's YAML states CC BY-SA4.0. No single resolved WikiText version is invented.

The protocol also records an exposure erratum: publisher landing pages automatically showed a few non-target train examples during metadata verification. Neither target split was previewed before protocol freeze; those examples played no role in selection.

These are fresh workloads for this project, with no post-payload fitting or token/output-based replacement. The eight WikiText chunks cover seven articles. This does not establish model-pretraining holdout status or a full-benchmark result. ARC free generation is not benchmark accuracy. Teacher-forced quality and generation tests are delegated to the existing numeric-reference agent under separately frozen inference protocols.

