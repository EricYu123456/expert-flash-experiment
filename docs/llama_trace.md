# Real MoE routing traces

All models, source archives, build outputs and raw runs are in
`/home/nfs_client/ericyu/expert_flash_experiment`. Only small scripts and this
documentation are in the working directory. The existing sibling
`/home/ericyu/llama.cpp-aif` was read without modification; its uncommitted AiF
changes are not part of this experiment.

## Implementation

The standalone `scripts/llama_route_trace.cpp` harness links unmodified
llama.cpp commit `927dada6c9143ba4c940b72004d2698fa5e4e930`. Its public
`llama_context_params.cb_eval` callback reads the actual
`ffn_moe_topk-{layer}` tensors after execution. These are the selected expert
indices consumed by the expert FFN operations, produced by the model's router.
The callback does not calculate or replace routing decisions. It requests only
the top-k tensors and handles their GGML strides explicitly.

The prompt is evaluated as a batch with tracing disabled. Greedy sampling then
generates one token at a time; each generated token is supplied to one actual
`llama_decode` call, and all routed layers' expert IDs are recorded. The harness evaluates
exactly the requested number of generated tokens, including an EOS token if one
is generated. It does not stop early at EOS. The saved text contains these
generated tokens, excluding the prompt. Prompts are plain text, without an
instruction model chat template.

Each JSONL routing event contains:

```json
{"event":"routing","phase":"decode","token_index":0,"position":41,"token_id":203,"layer":0,"tensor_name":"ffn_moe_topk-0","expert_ids":[13,17,28,31,30,9,25,0]}
```

`token_index` is zero based among generated tokens, and `position` is the actual
context position, equal to prompt token count plus `token_index`. `expert_ids`
retains the router's top-k order. It is not sorted by numeric expert ID.

`scripts/inspect_gguf.py` uses the pinned source's GGUFReader to extract actual
GGUF tensor dimensions, quantization types, absolute data offsets and byte sizes.
For these two models' expert tensors, the expert dimension is dimension 2;
`data_offset + expert_id * bytes_per_expert` identifies an expert's contiguous
weight slice. `shape_ggml` follows GGML dimension order, with dimension 0
contiguous. Metadata includes a full-file SHA256.

## Models and storage

| Model | Layers | Hidden / expert intermediate | Total / selected experts | GGUF bytes |
| --- | ---: | ---: | ---: | ---: |
| OLMoE-1B-7B-0924 | 16 | 2048 / 1024 | 64 / 8 | 4,213,511,776 |
| Granite-3.1-1B-A400M-Instruct | 24 | 1024 / 512 | 32 / 8 | 821,847,360 |
| DeepSeek-V2-Lite-Chat | 27 (26 routed) | 2048 / 1408 | 64 / 6, plus 2 shared | 16,702,520,416 |

The OLMoE GGUF is the official AllenAI Q4_K_M release at Hugging Face revision
`70df85ed7132bf21b5acbcb33817f58e7d3cb949`, SHA256
`f3ae25f6c7e83efb0e3632ef068b321f4d4a644bcfb6c0e01dd8c00e94e53f14`.
The Granite GGUF is bartowski's Q4_K_M conversion at revision
`940d2e1f9f65330615c7c8e980e6c5ac73d3360c`, SHA256
`3a2ec1c2a78cb29d901e29bbf5162dcd03381e13803d2cbdcff838d4d08142eb`.
Download manifests and the vendors' original model configurations are stored
beside the GGUF files.

The OLMoE and Granite Q4_K_M files contain mixed tensor types. For example, layer 0's OLMoE
gate/up tensors are Q4_K with 1,179,648 bytes per expert per projection, and its
down tensor is Q6_K with 1,720,320 bytes per expert. Granite has 294,912-byte
Q4_K gate/up slices and a 430,080-byte Q6_K down slice in layer 0. Other layers'
actual types are in the metadata; do not assume a uniform 4-bit byte count.

## Reproduction

Run from the experiment directory:

```bash
python3 scripts/download_moe_models.py olmoe granite
bash scripts/build_llama_trace.sh
python3 scripts/inspect_gguf.py \
  /home/nfs_client/ericyu/expert_flash_experiment/models/olmoe-1b-7b-0924-q4_k_m.gguf \
  /home/nfs_client/ericyu/expert_flash_experiment/models/olmoe-metadata.json
python3 scripts/run_llama_traces.py \
  --model /home/nfs_client/ericyu/expert_flash_experiment/models/olmoe-1b-7b-0924-q4_k_m.gguf \
  --metadata /home/nfs_client/ericyu/expert_flash_experiment/models/olmoe-metadata.json \
  --output-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/olmoe-q4 \
  --steps 32 --threads 8
```

Use the Granite model and metadata paths with `runs/granite-q4` for its run.
The build script obtains the exact upstream source if it is missing and keeps
compiler temporary files under NFS. CMake, Ninja, a C++17 compiler, curl,
Python 3 and NumPy are required. Neither root nor a GPU is required.

The runner executes three distinct prompts (storage explanation, programming,
and mathematics). For each it runs a traced generation and an untraced greedy
baseline, requires byte-for-byte equal generated text, and validates one routing
record for every routed layer of every decode token, unique in-range expert IDs, and
the active-expert count. Run commands, prompt text, generated text, stderr,
timings and validation results are saved with the traces.

## Interpretation limits

These are real llama.cpp CPU routing traces from quantized models, not random
or uniform synthetic selections. Each model contributes three 32-token runs;
this is a small workload sample. The CPU timings include attention, FFN and
callback overhead, and memory mapped weights may be served by the host page
cache. They are not NVMeVirt or flash-PE performance measurements.

An AiF-style INT8 placement calculation must explicitly convert each logical
weight matrix to one byte per parameter, using its actual dimensions. The
quantized GGUF supplies routing and model geometry; it does not make its Q4/Q6/Q8_0
bytes an AiF INT8 weight image. Furthermore, routing may differ if inference
uses another quantization format.

Sources: [official OLMoE model](https://huggingface.co/allenai/OLMoE-1B-7B-0924),
[official OLMoE GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0924-GGUF),
[IBM Granite configuration](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct/blob/0da7a48b0276d500ce5922fd2b33944091fc6c09/config.json),
[llama.cpp routing implementation](https://github.com/ggml-org/llama.cpp/blob/927dada6c9143ba4c940b72004d2698fa5e4e930/src/llama-graph.cpp).

## Existing DeepSeek source model

The additional DeepSeek-V2-Lite-Chat experiment uses the existing safetensors in
`/home/nfs_client/ymtuan2/models/DeepSeek-V2-Lite-Chat`, with no changes to those
files. The source directory has all four indexed weight shards (31,412,968,448
parameter bytes). Its Hugging Face download metadata records revision
`85864749cd611b4353ce1decdb286193298f64c7`. A manifest in the experiment's NFS
`models/deepseek-source-manifest.json` records source paths, sizes, tensor-header
hashes, config hash, and the expected shard hashes from that download metadata;
those expected shard hashes are not presented as independently rehashed files.

The pinned converter can export Q8_0 directly, saving a roughly 31-GB F16
intermediate and another quantization pass. Reproduce with:

```bash
bash scripts/convert_existing_moe.sh deepseek
python3 scripts/run_llama_traces.py \
  --model /home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-v2-lite-chat-q8_0.gguf \
  --metadata /home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-metadata.json \
  --output-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/deepseek-q8 \
  --steps 32 --threads 8
python3 scripts/validate_hf_tokens.py \
  --source-model /home/nfs_client/ymtuan2/models/DeepSeek-V2-Lite-Chat \
  --runs /home/nfs_client/ericyu/expert_flash_experiment/runs/deepseek-q8 \
  --deepseek-compat
```

DeepSeek has 27 transformer layers: layer 0 has a dense FFN, and layers 1–26
each select six of 64 routed experts. Hidden size is 2048 and routed-expert
intermediate size is 1408. Each MoE layer also has two always-active shared
experts, represented by combined shared tensors with intermediate size 2816.
The trace validator derives the expected routed layer IDs from the actual
expert tensors rather than assuming every transformer layer has a router.
Metadata labels routed, shared, and dense FFN tensors separately. Shared/dense
weights must be explicitly included when reporting a full DeepSeek FFN service
time; they are absent from its selected-expert ID records.

The completed DeepSeek run contributes 2,496 actual routing records:
three prompts × 32 decode tokens × 26 routed layers. All three untraced greedy
baselines produce byte-identical generated text. The independent Hugging Face
comparison also matches all prompt token IDs for both variants (37, 23, and 34
tokens for explanation, programming, and mathematics). Evidence is under
`runs/deepseek-q8`, including `validation.json` and
`hf-tokenization-validation.json`.

The full converted GGUF SHA256 is
`09d9abe9170d14386b0a54fe3969035f6a770956cfefa7d2fbaa12e0900a2ba4`.
Actual routed projection slices are each 3,063,808 bytes in Q8_0, compared with
2,883,584 logical INT8 weight bytes; Q8_0's block scales account for the
difference. Full-hash metadata is `models/deepseek-metadata.json`. Geometry was
extracted into `models/deepseek-geometry.json` while hashing ran in parallel;
the final metadata's tensor geometry and model metadata were verified identical
to that earlier sidecar.

The installed Transformers 5.8 generic tokenizer backend omits this older
LlamaTokenizerFast config's `add_bos_token=True` behavior. The standalone
`convert_deepseek_compat.py` adapter restores the source config's declared BOS
policy in the in-memory tokenizer object. It changes neither source tokenizer
files nor the pinned converter's recognized hashes. After restoration, the
converter's full pre-tokenizer check matches its existing `deepseek-llm` hash
`049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754`.
The saved compatibility validation records this check. The tracing binary also
records complete prompt token IDs; `validate_hf_tokens.py` independently compares
them with the local Hugging Face tokenizer for every prompt and both variants.
