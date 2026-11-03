第二批固定 workload 已取得，供 Granite4Tiny 與 DeepSeek 使用完全相同的 16 份 prompt bytes、每份 16 decode tokens。沒有模型執行、placement fitting、答案導向選樣或 output-based replacement。

凍結與下載次序（UTC）：

1. Root W8 plans anchor：2026-11-02 00:10:31.030633，SHA256 `00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00`。
2. Acquisition protocol：00:11:46.670775，SHA256 `393d3c7e6c59d50dd17714867dd3dd5ce6ff03baa1d5aa10bad1ce95c6456ce9`。
3. Target acquisition 開始：00:11:52.416756。取得前僅查 metadata／schema／publisher README，沒有讀 target Parquet、footer 或 viewer-rows endpoint。

`manifest.json` SHA256 為 `ccfabf3b3e1a4293f2d0354cdea289b7d94197fc6733253601ca97448dabcd8f`。其 `items` 有固定順序的 `{id,dataset,dataset_revision,dataset_config,split,kind,prompt_path,prompt_sha256,source_id,...}`。前八份是 OpenBookQA，後八份是 BoolQ；這一份 manifest 同時供兩個 models 使用。

| 官方 dataset | 固定 HF revision | Split | 原始列／eligible | 已選 | Payload bytes | 已選 prompt words |
|---|---|---|---:|---:|---:|---:|
| [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa/tree/388097ea7776314e93a529163e0fea805b8a6454) | `388097ea7776314e93a529163e0fea805b8a6454` | main/test | 500／500 | 8 | 55,535 | 16–62 |
| [google/boolq](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5) | `35b264d03638db9f4ce671b711558bf7ff0f80d5` | validation | 3,270／3,270 | 8 | 1,257,630 | 55–273 |

選樣為 seed `20261104` 的 SHA256 ID ranking，不依 tokenizer、answers 或模型輸出。OpenBookQA 用 publisher id；BoolQ HF schema 沒有 id，使用已固定 Parquet 的零起始列索引作 canonical source-row id，不冒稱 Google 原始 id。Eligibility 只要求完整非空結構，OpenBookQA 四個不同 labels／四個 choices；沒有長度或難度篩選。完整 stem／choices、完整 passage／question 都保留，無 128-token 截斷。

原始兩份 Parquet 在 `raw/`，原始 selected fields 在 `selected_source_records.json`；後者不含 answer fields。答案另外在 `answers/answers.json`，不附進 `prompts/`。自然文本或 choices 可能本來就包含正確答案文字，故檢查採 exact rendering，而不是錯誤地要求答案字串不得在原文出現。16-token free generation 不作完整 benchmark accuracy 主張。

兩份 payload 的實際 SHA256 已與 pinned HF tree 的 LFS SHA256 比對：OpenBookQA `cd5483e366daa230c1c87bbdc512d8b7229f14f6dd04d19fc8b1a3855aaaa8a3`；BoolQ `52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8`。Dataset revisions 與 model checkpoint hashes 分開記錄。

Frozen protocol 的 `bindings` 含 root anchor、v5/v6 campaign summaries／protocols／source-snapshot manifests、TRAIN manifest、兩份既有 numeric protocols，共十個 immutable files；取得後均重新核對。Model hashes 沿用已有 numeric protocols 的確定記錄，這個 acquisition 階段沒有下載或重新掃描多 GB checkpoint：

- G4 numeric protocol：`3163341827c25b210c42126ae7465c1fe8be9827aab93d314a89ef8a54975f40`；GGUF `5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd`。
- Deep numeric protocol：`84e100600d882eb809d85fa1bdc3e469b0fb9a885e5627c60479339078a84a7a`；GGUF `09d9abe9170d14386b0a54fe3969035f6a770956cfefa7d2fbaa12e0900a2ba4`。

License evidence 如實保留：BoolQ 的 [Google 官方 README](https://github.com/google-research-datasets/boolean-questions/blob/90af34107399cc7a446b373dc4ee35b8001da7c2/README.md) 與 HF card 都列 CC-BY-SA 3.0。OpenBookQA pinned HF card 仍列 `unknown`；[官方模型程式 repo](https://github.com/allenai/OpenBookQA/tree/b51971646e9371a61508d9953fc706645e194a71) 的 Apache 2.0 不能直接推成外部 dataset license。沒有用第三方整理表替代此證據。

Metadata exposure 已記在 protocol：publisher README 有示例，其 split membership 未查；AI2 homepage 曾 redirect 到可能帶自動 preview 的 HF landing。這些示例不參與 ranking／eligibility；取得前未開 target payload。另三份 GitHub metadata 初次保存時多出的最後一個 LF，已在 protocol freeze 前修正，每份現已與 pinned Gitblob SHA1 精確匹配，記錄在 metadata preparation erratum。

`independent_acquisition_check.py` 不 import 選樣產生器；它從兩份 raw Parquet 獨立重算全部 eligibility／ranking、16 份完整 rendering／hashes、答案 sidecar、來源列數與 freeze 時序，結果 PASS。`independent_acquisition_check.json` SHA256 為 `4077abb43fe382e92e51b194cfc11803f86d9335e7bc0497a3240d4c2788c2c0`。

數值驗證代理已收到 manifest path／SHA，可先凍結其 inference protocol 再跑兩模型測試。本資料只代表專案中未用過的來源格式，不宣稱未進入模型 pretraining，也不代表整個 benchmark 的抽樣品質。
