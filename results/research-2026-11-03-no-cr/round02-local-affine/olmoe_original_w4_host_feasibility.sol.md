# Original OLM Q4_K_M: necessary all-host baseline

2026-11-03; same Round2, source/metadata and **already closed resource outputs only**. No candidate/source changes, model/test/benchmark execution, quality computation, book/FIT/target/TEST bodies or new method clock. This is a feasibility ledger and a future measurement requirement, not proof of capacity or speed.

**Conclusion:** original OLM source-GGUF CPU inference remains a mandatory plausible 8GB competitor. The 4.213GB file alone is insufficient proof; the existing default loader repacks about3.152GB into additional anonymous CPU storage. Conversely the old quality-harness peak cannot prove that a source-only, memory-budgeted original implementation fails.

This competitor preserves the original mixed-GGUF function; it is not the routed-canonical W8A8 numerical reference. Its quality must remain labeled accordingly in a matched workload comparison rather than being called a bit-identical implementation of B64/C64.

## Pinned representation and deterministic bytes

`models/olmoe-metadata.json` and the download manifest under `/home/nfs_client/ericyu/expert_flash_experiment/` bind `olmoe-1b-7b-0924-q4_k_m.gguf`, SHA256 `f3ae25f6c7e83efb0e3632ef068b321f4d4a644bcfb6c0e01dd8c00e94e53f14`, **4,213,511,776 bytes**. This is mixed Q4_K_M, not uniform W4: metadata totals195 tensors, including3,209,969,664B Q4_K,992,839,680B Q6_K and8,921,088B F32. Those types already contain their quantization scales; do not add another standalone scale ledger or mandatory canonical-W8/F32 whole-weight copy.

Official config and GGUF metadata agree:16 layers, H2048,16 attention heads and16 KV heads,128 dimensions/head,64 routed experts/top8, no shared expert, vocabulary50304. For the existing single-sequence context1024/batch=ubatch128 configuration, pinned `llama_context_default_params` (`src/llama-context.cpp:3198`) uses F16 K and V. `llama-kv-cache.cpp:186–211` allocates both for each layer. Thus KV data is exactly `16*1024*2048*2(K,V)*2 = 134,217,728B` (128MiB), plus bookkeeping/alignment. This does not shrink to the populated prefix or batch1; context capacity is allocated. The closed OLM log confirms1024 cells/16 layers/F16 K64MiB+V64MiB.

| Conditional ledger, bytes | Total | Remaining under8,000,000,000B | Remaining under8GiB=8,589,934,592B |
| --- | ---: | ---: | ---: |
| Entire GGUF + F16 KV +128 full-vocabulary F32 logits | 4,373,485,152 | 3,626,514,848 | 4,216,449,440 |
| Above, **if** the complete source file remains resident alongside the logged3006MiB repack copy | 7,525,504,608 | 474,495,392 | 1,064,429,984 |

128 logits occupy25,755,648B (24.5625MiB); one B1 output is201,216B. A context that previously requested128 outputs retains its larger output buffer (`output_reserve`, context.cpp:1888–1959), so a B1 decode after128-token prefill cannot automatically claim the B1-only output allocation. Closed OLM logs show a26.5630MiB graph compute buffer at these settings, **in addition** to the table. CPU backend `ggml-cpu.cpp:173–184` can also retain a separate `cplan.work_size` allocation. Tensor/graph metadata, vocabulary/parser structures, thread stacks, allocator slack, sampler state, IO/loading peak and OS allowance remain unpaid. These table entries are neither final peak bounds nor additions to be summed with measured RSS.

## Why mmap and repack need measurement

Pinned loader/model sources (`llama-model.cpp:817–869,1445–1475,2093–2097`) default to mmap **and** extra CPU buffer types. Ordinary CPU tensors may point directly into the mapping; CPU_REPACK instead allocates a CPU buffer (`ggml-cpu/repack.cpp:4751`) and repacks from source data. The closed Stage3 OLM log reports **CPU_Mapped4016.62MiB + CPU_REPACK3006.00MiB**. The mapped figure is the min-to-max tensor address span (`llama-model-loader.cpp:1359–1372`), potentially including gaps whose weights were repacked. It is not a count of resident pages or an independent complete tensor copy.

File mapping virtual size is not RSS. A resident mmap file page and its page-cache backing are the same physical page: do not count both again. Anonymous repacked data and resident source pages are distinct and count separately. The loader's final cleanup unmaps unused ends (`1660–1670`); it does not establish that every interior repacked-source page is unmapped/reclaimed. Dropping such reclaimable pages under pressure may reduce steady residency, while initial repacking can still cause a larger loading peak. Disabling `use_extra_bufts` is an exposed possible memory-oriented source baseline, not a proven fast setting; disabling mmap alone does not remove file-cache or copy costs. Kernel/repack selection and floating reduction order require their own numerical qualification, not an assertion of old source-logit bit identity.

## What has actually been observed

There is an **OLM-specific** closed resource observation: `stage3-control-cost/dev-quality/olmoe/process/exit.json` reports aggregate sampled peakRSS **8,676,352,000B**, interval0.2s, explicitly a lower bound on the true peak. Its resource samples include the harness's maximum observed VmHWM8,223,472KiB and additional Python/strict-activation processes. The earlier `stage2-pg19-quality/QUALITY.resource_summary.json` also records an OLM harness. These are source-GGUF + replacement/IPC/control/reference-logit workloads; not standalone original-W4 deployment measurements. Aggregate process RSS may double-count shared physical pages; sampled sums and sums of per-process HWMs are different quantities. The observed exceedance of both budgets does **not** establish original source-only infeasibility, and subtracting worker or control standalone peaks would not reconstruct its peak.

I found no independently isolated **source-only OLM W4** whole-process loading/prefill/B1 peak in the inspected closed resource records. It remains unmeasured here.

## Minimum future measurement

After separate release, run one pinned source-only CPU path with n_gpu_layers0, context1024, batch=ubatch128, seq1 and8 physical cores; no replacement callbacks, Python numeric worker, canonical weight exports or stored full-history reference logits. Reuse an already authorized fixed TRAIN token sequence for128-token prefill followed by bounded B1 decode; no new quality search. Record exact build/kernel flags and model hash, loading and steady phases, backend buffers, kernel scratch, RSS/HWM plus mmap/file/anonymous PSS/residency and inclusive process-tree/cgroup accounting. Retained page cache is counted once, including reclaimability/ownership limitations; a short route trace does not prove every expert page is hot. Distinguish load peak from a budgeted steady execution, rather than omitting whichever exceeds the target.

Start with the current optimized extra-buffer setting. If its source-only inclusive budget fails, a separately pinned generic/no-extra-buffer setting is a legitimate bounded memory-oriented original baseline, with actual latency and numerical behavior measured independently; do not disable repacking merely to weaken the strongest feasible host implementation. Both8e9 and8GiB ledgers must close including chosen OS/runtime reserve. Current EPYC measurements are not i9 target measurements, and full DRAM fit neither means all weights fit LLC nor guarantees a speed/cache advantage. This note authorizes none of these runs and leaves the ongoing FIT untouched.
