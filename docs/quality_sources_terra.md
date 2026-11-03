# PG-19 and HellaSwag provenance — metadata-only review

Reviewed at 2026-11-03 03:25 UTC. No book object, HellaSwag parquet file, model, candidate result, or evaluation was downloaded or inspected in this review. See the browser disclosure below about an embedded train example displayed by the dataset card. Listed IDs are a **proposed deterministic selection**, not authorization to execute or evidence that the parent has frozen it.

## PG-19 original sources

- [DeepMind repository, pinned commit b0e1f2925359688af19f8784c23c3deeda52d3ea](https://github.com/google-deepmind/pg19/tree/b0e1f2925359688af19f8784c23c3deeda52d3ea): README records 28,602 train, 50 validation, 100 test books, and presents full-dataset word-level perplexity. It declares Apache-2.0 in its dataset metadata; the texts derive from Project Gutenberg, so this declaration and any underlying text terms should not be silently conflated.
- [TensorFlow Datasets original PG-19 builder](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/pg19/pg19_dataset_builder.py): the source prefix is `gs://deepmind-gutenberg`, individual `<split>/<book_id>.txt` objects, `book_id` integer, text and title/date metadata; its generator applies `.strip()` to text, relevant to reproducibility. TensorFlow builder VERSION 0.1.1 is a builder version, not a Git commit and not a version ID for all cloud objects.
- Official [Cloud Storage JSON object listing API (validation)](https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?prefix=validation%2F&fields=items(name%2Csize%2Cgeneration%2Cmd5Hash%2Ccrc32c)%2CnextPageToken) and the same URL with `prefix=test%2F` returned 50 and 100 metadata entries, respectively, with no continuation. The object generation pins each book; `md5Hash` is a **base64 MD5**, not SHA256 ([Cloud Storage object resource](https://docs.cloud.google.com/storage/docs/json_api/v1/objects)). A later authorised downloader can obtain one `split/ID.txt` object via [Objects GET](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/get), URL-encoding the slash, with `alt=media&generation=<generation>` and verify the metadata MD5; fail closed if missing/mismatched and optionally compute and record SHA256 over the downloaded bytes. A pinned generation can cease to be available if the source deletes its version: do not silently fall back.
- The 16 proposed dev files total 5,037,895 bytes and 32 future held-out test files total 9,389,008 bytes. No bulk training download is needed. The 8-book option is the first 8 ranked validation IDs. Ranking here uses SHA256 of ASCII exactly `pg19-v1/<split>/<decimal_book_id>`, ascending raw digest, tie-break numeric ID. Selection uses object names and split alone, not book content, title or size. The lists below are ordered by this ranking and include full object generations and original MD5 as provenance. Actual tokenized lengths and availability of the specified 512-token windows are unverified; predeclare how invalid/short documents are handled rather than selecting based on results.
- The proposed context-256/target-256 at book midpoint is a *bounded-window subset metric*, not the official complete-corpus word PPL. Before any scoring, specify tokenizer model+revision, Unicode/line-ending/strip policy, BOS, whether midpoint is in bytes, characters, or model tokens, whether midpoint locates start or center of the 512-token window, short-book fail/replace rule, next-token target alignment, aggregation and requested token vs word NLL. Fix original-versus-compressed comparisons on identical IDs and token boundaries; compute prompt-level paired uncertainty. Validation can choose methods; official test books must not feed tuning or calibration. Pretraining exposure to these books is unknown.

## HellaSwag source and metric

- Author-hosted [Hugging Face dataset README](https://huggingface.co/datasets/Rowan/hellaswag/blob/218ec52e09a7e7462a5400043bb9a69a41d06b76/README.md) states 39,905 train, 10,042 validation, 10,003 test; `ind` int32, source ID, split, contexts, choices and label are fields. It labels dataset license MIT and links the original author repository license; the card does not spell out all underlying source corpus terms.
- Pinned dataset revision: `218ec52e09a7e7462a5400043bb9a69a41d06b76`. [HF file-tree metadata API](https://huggingface.co/api/datasets/Rowan/hellaswag/tree/218ec52e09a7e7462a5400043bb9a69a41d06b76?recursive=true&expand=true) exposes `data/validation-00000-of-00001.parquet`, 6,315,951 bytes; LFS content SHA256 `899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd` (distinct from Git blob oid). This is a single validation file, so downloading that one is small but includes all example text/labels; it has **not** been downloaded. I have not found a separate published list of validation `ind` values. Thus 256 actual IDs cannot responsibly be enumerated from metadata alone; do not infer IDs are 0–10041 or conflate `ind` with `source_id` or row offset. Once separate authorisation exists, an ID-column-only projection could obtain IDs without outputting contexts/labels, but would still access part of the dataset file; freeze row count, ID uniqueness, deterministic hash algorithm and ID list prior to scoring. Whether a specific remote column-only reader reads ancillary bytes must be verified rather than assumed.
- [EleutherAI task definition](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hellaswag/hellaswag.yaml), [preprocessor](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hellaswag/utils.py), [scoring](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/d6de81643928d653435c431bae19945d41d32520/lm_eval/api/task.py): zero-shot on validation is a **pinned harness protocol**, not an author-designated unique official zero-shot metric. It reports `acc` and `acc_norm`; `acc_norm` divides each continuation log-likelihood by Python string length of its preprocessed choice, not model token count or UTF-8 byte count, then chooses argmax. Pin task/harness commit, delimiter and model/tokenizer/revision; IDs should be selected before preprocessing because the processed docs do not preserve `ind`. A 256-item hash subset is a subset metric, not the full benchmark; paired per-item uncertainty is important.

## Proposed PG-19 metadata-ranked IDs

### validation: 16 objects

| ID | generation | bytes | GCS MD5 (base64) |
|---|---|---:|---|
| 56719 | 1568646527853903 | 261676 | cm3iVGaHhJ3RwMJLqwfp5g== |
| 356 | 1568646527901705 | 217960 | 2uGqmCQpr0nwp9Dj/Uw8/Q== |
| 16959 | 1568646527818253 | 188790 | fopsJAtB7WBEdWInBB9l+w== |
| 8545 | 1568646527845288 | 905381 | DGuEay1nqe/zb+xjCp5eAg== |
| 41074 | 1568646527832728 | 339379 | UZvT2wkxE0dLXOHorNmrwA== |
| 7510 | 1568646527843993 | 424105 | uhEHtBWU0iMZnkTCyzobyg== |
| 49091 | 1568646527835074 | 184256 | /zHeI6BmgE+zXxezcgykhA== |
| 54098 | 1568646527829271 | 153699 | QzOqXnoLNgmOFPcXMdZLGw== |
| 48089 | 1568646527817525 | 415583 | HOt3NfF4vATe0/adht5XTA== |
| 11155 | 1568646527851144 | 502104 | acMXPDyjgqpXkzaWd1P5wQ== |
| 25066 | 1568646527829755 | 554135 | S570jJwiw9ttROkh6K6Vng== |
| 27688 | 1568646527851162 | 143662 | OnUAoN5IQwIN7Vbk6CBsEA== |
| 38214 | 1568646527864461 | 564922 | W37F3AFFuvgPAOzxIYy2TA== |
| 34016 | 1568646527829379 | 56836 | w3DHNuoU9/mAKJEvl/k8ig== |
| 50355 | 1568646527837097 | 80516 | HaWuWVpmMhOE374X4SNGqQ== |
| 42306 | 1568646527865007 | 44891 | 2mA5Y24ssiMZoh1Mt0jwJQ== |

### test: 32 objects

| ID | generation | bytes | GCS MD5 (base64) |
|---|---|---:|---|
| 56410 | 1568646215300785 | 111994 | irWu+X6f4rpmx5q5p5bhKw== |
| 41607 | 1568646215121078 | 337917 | sWXAfJU21nHtb1P5valtIw== |
| 43536 | 1568646215466178 | 55719 | vyvZUy6veHnp4PUletyeBg== |
| 35246 | 1568646214040054 | 228141 | +6i6KE9e4GcXLkP7gVZlLw== |
| 37006 | 1568646214947694 | 156555 | y/6FguAj+ddtH500iUndnQ== |
| 54537 | 1568646215115044 | 200630 | marg1BoX503ye1aeeTrY1w== |
| 42655 | 1568646215322683 | 882218 | ajKtvSwCvl2Lgf+V8ap84w== |
| 31065 | 1568646213936243 | 276227 | xfn2Cod0QOqewsKJQPCmzQ== |
| 3340 | 1568646214725370 | 392865 | yDnuXZhtNXBw6gyMqZb3Rg== |
| 50287 | 1568646215339497 | 302777 | PWIjnWe7IAoB5rjY9N1bDw== |
| 28444 | 1568646215053698 | 335975 | zVORmqVG5NpODzj8MoYy0A== |
| 45881 | 1568646214949233 | 217753 | Tu/JHaClKJJjGg+VFKUw0A== |
| 37702 | 1568646215175760 | 892108 | 81KV35nDUUrOHrUp/lZiuA== |
| 22424 | 1568646214215868 | 444287 | GxIOmHKjOftITZ42fw3RBQ== |
| 26239 | 1568646214804082 | 354117 | 7J7pnhYTpYcA5Ek8RmeorA== |
| 44557 | 1568646215466153 | 244883 | exGDgFooF976+YYGvafHjA== |
| 35205 | 1568646214328320 | 364651 | R8we3D2PZ/YM7QUIflxbFQ== |
| 31974 | 1568646214288274 | 14880 | vfc8Y53yXSCuPC9uVUXrxg== |
| 47068 | 1568646215130359 | 351283 | r8kNySQ5zE73mY90Lin5jw== |
| 42081 | 1568646215284351 | 844569 | vL+by1xcREwnfNIB50Il9Q== |
| 48693 | 1568646215569648 | 410440 | HQqkD3dfGrfRbac8Urc7Yg== |
| 24553 | 1568646214202155 | 130627 | 4f6yqeI8XxmbWYoVKxVIcw== |
| 43845 | 1568646215071607 | 57163 | XFwKMaZW5yHaVY6Cd8ovjA== |
| 9315 | 1568646215505822 | 22463 | WUYQa4qTcXDotoXj+s2bbQ== |
| 30981 | 1568646214298769 | 153798 | 6F1ZN3tU9z84RrPUYsZ3MA== |
| 40700 | 1568646214899570 | 148045 | p7cUOPAhyTles1KmvQUGOg== |
| 33756 | 1568646214785863 | 322011 | u5IscJzP2Dq5TxCK8QgNtA== |
| 10321 | 1568646214501219 | 324979 | G4RkADyUavAaHmb1epQsyg== |
| 5956 | 1568646215477983 | 260972 | 4mNCpY33+RI/JSs6mD1AYQ== |
| 26618 | 1568646214276138 | 58957 | qEsFqvWvKZPxRES2am1iaQ== |
| 34467 | 1568646214303254 | 207793 | GO4xKK1EXzy7XtIGN+9goA== |
| 45888 | 1568646215340140 | 282211 | SzD7qBVcU0Pt6osWUSb7Uw== |


## Raw provenance archive

The complete, unfiltered GCS listing HTTP responses and headers, full 16/32 selected metadata as machine-readable JSON, retrieval time, exact ranking and SHA256 checksums are in [quality-pg19-terra](../results/research-2026-11-03-no-cr/round02-local-affine/prior-sources/quality-pg19-terra/README.md). These are metadata snapshots, not downloaded book files or completed evaluation. The ranking above was already performed as `pg19-v1/<split>/<decimal_book_id>` before the later salt suggestion, so it is preserved unchanged.

Browser disclosure: while finding split metadata on the author-hosted dataset card, the web tool automatically returned adjacent card text containing part of one embedded **train** example and its label. No HellaSwag dataset file or validation/test example was requested or downloaded, and the 256 validation IDs remain unknown. Subsequent HellaSwag accesses used the metadata API and task source only.
