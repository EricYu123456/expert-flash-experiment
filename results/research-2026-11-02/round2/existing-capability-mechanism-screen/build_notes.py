#!/usr/bin/env python3
"""Seal a bounded primary-evidence screen; no timing/model execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ROUND=HERE.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return str(Path(p).relative_to(ROOT))
def put(name,obj): (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

existing={
 'aif':ROOT/'paper/aif.txt',
 'cambricon':ROUND/'qwen38-placement/joint-prior/cambricon.html',
 'kvnand':ROUND/'chip-count-regime-prior/primary/KVNAND.v1.html',
 'nvllm':ROUND/'nvllm-conduit-audit/nvllm.html',
 'nasic_pdf':ROUND/'backend-capabilities/nasic-primary/2605.23294v1.pdf',
 'prior_backend_audit':ROUND/'backend-capabilities/2026_backend_audit.json',
 'prior_backend_matrix':ROUND/'backend-capabilities/capability_matrix.json',
 'prior_numeric_smoke':ROUND/'backend-capabilities/repository_smoke.json',
}
sources=[
 {'id':'AiF','url':'https://doi.org/10.1145/3695053.3731073','version':'ISCA 2025',
  'read_depth':'Existing local paper text, targeted reread lines around720–745 and section4.4; prior Fig14 audit retained.',
  'evidence':'be-enc uses fastest LSB pages. Idle latches after single sensing enable its input arrangement; CSB/MSB retain conventional data. Ordinary data-page read latency does not establish GEMV capability on those pages.',
  'page_class_gemv_status':'No native two-class GEMV contract established. Do not assign28/46us as two compute tiers.',
  'maturity':'Paper architecture and emulator, not commercial compute NAND evidence.'},
 {'id':'Cambricon-LLM','url':'https://arxiv.org/html/2409.15654v1','version':'MICRO 2024; arXiv v1 2024-09-24',
  'read_depth':'IV-A/B/C and V reread; VI ECC and VII-A/C configuration checked in complete legal preprint.',
  'evidence':'Weights flow array→data register→cache register→shared Compute Core; input has dedicated buffer. One plane occupies the die core; another plane may serve sliced ordinary reads to NPU. TableII has16KB pages and one30us tR.',
  'page_class_gemv_status':'Unlike AiF, no borrowed-latch LSB requirement is stated. No explicit LSB/CSB/MSB or mixed-mode command/timing/ECC contract was found; register-level compatibility is an inference, not a demonstrated tier.',
  'artifact':'Original SSDsim/NPU extensions not obtained; previously smoke-tested dengls24/IFC-cambricon-llm is explicitly third-party, not original numeric implementation.',
  'maturity':'Simulated architecture and synthesized compute logic; manufactured device not claimed.'},
 {'id':'SLIM','url':'https://arxiv.org/html/2507.09201v1','journal_url':'https://doi.org/10.1145/3750727',
  'version':'arXiv v1 2025-07-12; accepted2025-07-16/onlineAM2025-07-24; TECS25(3) article46 final2026-05-13',
  'read_depth':'Sections5.4,6.1–6.3,7.1,7.4,7.6; original PDF Table2 and extracted lines890–917/1029–1058 independently checked. Official indexed journal text cross-checked.',
  'evidence':'Section5.4 describes batched GEMM, input/output SRAM, output-stationary read-once weights, and reduce commands. Table2 die PE:16MAC,64KB SRAM,1GHz. SLC4KB/3us and TLC16KB/40us are two distinct SSD configurations. Section7.6 multi-plane support is prospective.',
  'page_class_gemv_status':'Same PE design evaluated with SLC or TLC devices, but not an established mixed-mode fast/slow pool within one fixed device.',
  'artifact':'Author wh-xu/sparse-ffn-llm Torch software previously inspected; no public SLIM FTL/PE simulator or RTL verified.',
  'maturity':'Modified MQSim/TransPIM plus synthesized HDL; not deployed compute NAND.'},
 {'id':'KVNAND','url':'https://arxiv.org/html/2512.03608v1','version':'v1 2025-12-03',
  'read_depth':'Prior IV-A–D/V methodology and TableI, now targeted reread SLC cost/reliability paragraphs.',
  'evidence':'TableI uses SLC4KB pages,4us tR. Cost analysis explicitly converts TLC density to SLC-mode NAND. Discrete weight/KV die grouping and compact shared-die mapping are already explored.',
  'page_class_gemv_status':'No two-class compute pool in the published configuration.',
  'artifact':'No original public executable integrated implementation verified.',
  'maturity':'Early-design system-level model; paper explicitly distinguishes this from RTL simulation.'},
 {'id':'NVLLM','url':'https://arxiv.org/html/2604.25699v2','version':'DAC 2026; author-prepared v2 2026-08-04',
  'read_depth':'Previously3.1–3.5/Algorithms1–2/4.1–4.3/Fig5; now reread3.1 raw-read buffering and4.1 timing.',
  'evidence':'Section3.1 explicitly covers MLC/TLC multiple read-voltage cycles using PBs, cluster FIFOs and segment buffers. Same raw-read dot engine supports BF16/INT8. Evaluation exposes one5.12us plane read time.',
  'page_class_gemv_status':'Positive architectural MLC/TLC support; not LSB-only. No per-class visible latency/address dataset or safe-tR table. Buffering is designed to hide variation, so physical fast-page benefit remains unquantified.',
  'artifact':'Prior bounded author/official repository search found no integrated source; third-party collectors are not author artifacts.',
  'maturity':'Synthesized/simulated architecture, not manufactured NAND.'},
 {'id':'NASiC','url':'https://arxiv.org/html/2605.23294v1','version':'v1 2026-05-22',
  'read_depth':'Already present in2026_backend_audit.json, previously sections3–4 and PDF Fig4–6. Targeted reread3.1–3.3/4.1 now; not a newly discovered unreviewed source.',
  'evidence':'CAM-selected interleaved experts, shared query broadcast, block-wise SSL activation, multilevel SL inputs and multivoltage multibit CIM are explicit. These are analog current/ADC mechanisms, not ordinary digital TLC page GEMV.',
  'page_class_gemv_status':'Multibit CIM does not prove interchangeable LSB/CSB/MSB digital compute pages. Expert-private Down inputs across interleaved experts remain unverified.',
  'artifact':'No author code verified in prior bounded search.',
  'maturity':'SPICE/Verilog-A/software study; no fabricated NASiC claimed.'},
 {'id':'Lincoln','url':'https://doi.org/10.1109/HPCA61900.2025.00128',
  'author_publications':'https://people.iiis.tsinghua.edu.cn/~gaomy/publications.html',
  'author_bib':'https://people.iiis.tsinghua.edu.cn/~gaomy/files/my_publications/publications.bib',
  'version':'HPCA 2025',
  'read_depth':'Author publication HTML and Lincoln bib entry only in this followup. Official IEEE endpoint returns robot-check text; legal guessed author PDF path404. No fulltext conclusion.',
  'evidence':'Author publication entry links IEEE10946816; no local PDF/artifact link in that entry.',
  'page_class_gemv_status':'Unknown. Do not infer a mixed SLC/TLC contract from KVNAND citations.',
  'access_attempts':[{'url':'https://ieeexplore.ieee.org/document/10946816','result':'robot-check page'}, {'url':'https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/lincoln.hpca25.pdf','result':'HTTP404'}]},
]

math=[]
for b in [2,4,8]:
 u=64*(1-(1-6/64)**b)
 math.append({'E':64,'K':6,'B':b,'expected_unique_experts':u,'separate_expert_evaluations':b*6,'ideal_weight_read_saving_fraction':1-u/(b*6)})
notes={
 'checked_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Bounded primary evidence plus arithmetic; no additional inference, arrivals, native timing, model modification or new hardware. Existing sealed studies untouched.',
 'main_conclusion':'No newly verified >5% same-capacity placement gain. Two distinct published-capability screens are possible; their gains must survive matched native controls. Mixed fast/slow GEMV page placement lacks a calibrated execution contract.',
 'page_classes':{
  'AiF':'No-go for substituting standard page classes into unchanged borrowed-latch GEMV.',
  'Cambricon':'Dedicated buffers remove the obvious AiF latch argument, but page-class support/timing is unverified rather than impossible.',
  'SLIM':'Positive SLC-or-TLC device portability; no verified co-resident heterogeneous pool. Table2 also says16channels while methodology prose says8: do not silently select a count.',
  'KVNAND':'Published target is SLC-mode.',
  'NVLLM':'Positive MLC/TLC multi-Vread support; finite buffers can mask latency. No calibrated per-class visible service model.',
  'Lincoln':'Fulltext not obtained; unknown.',
  'NASiC':'Direct MoE/interleave prior already in base; analog multi-state is not digital page-class capability.',
  'required_before_experiment':['A legal address→class map on one fixed configuration.','Same GEMV/input/ECC/numerical contract on both classes.','Measured or author-modeled class service after FIFO/ECC, including retention/temperature when claiming shortened tR.','Fixed capacity and same fast-first/frequency plus phase-critical controls.'],
 },
 'screens':[
  {'name':'SLIM native batched weight fetch, bounded SRAM',
   'native_evidence':'SLIM5.4 explicitly allows read-once GEMM across B inputs and reduce; no inference of AiF PB replay is used.',
   'what_is_newly_eligible':'Bounded multiple-request physical mapping under an already described batch engine, not a new batching operator.',
   'arithmetic_inference':math,
   'peak_only_check':{'formula':'PageBytes*B/(16 MAC/cycle*1GHz); excludes ECC, SRAM traffic, nonlinear ops and outputs.', 'B4_SLC4KiB_MAC_ns':1024,'SLC_tR_ns':3000,'B4_TLC16KiB_MAC_ns':4096,'TLC_tR_ns':40000},
   'strong_controls':['Identical batch/cohort and expert-centric read-once scheduler for equal-allwidth, simple-balanced and phase-coupled placements.','Fixed64KiB PE total SRAM, with input/output/scratch partition stated; not64KiB free input.','Dense unchanged model; no adaptive thresholding or invented activation quantizer.','Same GEMM/reduce/output costs and full expert-input dependencies for every placement.'],
   'cheap_falsifier':'Actual TRAIN only, same decode-step B4/B8 cohorts: count union-selected experts and per-die necessary weight pages; compare strongest same-batching placement floor. If ideal placement-only gain is below5%, stop before a serving simulator.',
   'blocking_unknowns':['Native command fields and batch limits are not published.','Output format/scales and standalone projection versus fused-FFN interface are not verified.','Dynamic full-vector A8 scaling must not be replaced by the paper fused-neuron flow without proving compatibility.'],
   'status':'Worth necessary-work bound only; no placement gain established; delegated evidence to boundary by root request.'},
  {'name':'Cambricon two-plane read/compute role placement with fixed NPU',
   'native_evidence':'IV-B/C permit one shared compute core and another plane serving sliced ordinary reads to NPU.',
   'hypothesis':'Routed row ownership may affect whether a coactive set simultaneously fills the existing core and sliced-read path; this resource asymmetry is absent from homogeneous AiF wave accounting.',
   'strong_controls':['Original sectionV optimal workload proportion and2D atomic tiling.','Same sliced-read scheduler, route choices, NPU capacity and fixed one-copy weights.','Best plane coloring/input-affinity baseline; do not compare compute-only weak baseline.'],
   'cheap_falsifier':'Before implementation, solve a zero-overhead perfect read/compute split relaxation for selected pages with one core per die and finite channel service. If it is at most5% below the best static split, stop. A surviving gap still requires legal row/page lowering and the actual buffer/numeric contract.',
   'caveat':'Original paper already optimizes hybrid distribution; no missing mechanism or >5% headroom is proven. Third-party simulator is not primary hardware validation.',
   'status':'Lower priority conditional screen, not a recommendation to build a new backend or claim novel hybrid mapping.'},
 ],
 'rejected_shortcuts':['Use28/46us as AiF fast/slow GEMV tiers.','Treat64KiB SLIM PE SRAM as64KiB dedicated input.','Count total batching/read-reuse gain as physical-placement gain.','Import SLIM multi-plane-read future-work as existing capability.','Treat NASiC CAM/ADC or new independent contexts as an AiF software configuration.','Use cell-error variation as visible latency without a safe tR/ready/ECC contract.'],
 'local_source_hashes':{k:{'path':rel(p),'sha256':sha(p)} for k,p in existing.items()},
 'prior_closures':{
  'same_allocated_witness':'g4-joint-prefix-replica-screen/same-allocated-witness/independent_review_receipt.json',
  'topology':'chip-count-regime-prior/completion.json',
  'note':'Both independent same-allocated reviews passed; no evidence in this screen changes that finite L0 conclusion.'},
}
put('sources.json',sources)
put('notes.json',notes)
(HERE/'report.md').write_text('''本輪沒有確認新的 >5% 同容量配置收益。\n\nSLIM §5.4 明示 batched GEMM 的 read-once 權重能力，可先做 TRAIN B4/B8 必要工作量上界篩選；Table2 的64KiB是 PE SRAM總量，不能全部當input。所有布局須用相同batching/reduce強控制，總batch收益不等於placement收益。\n\nNVLLM §3.1明示MLC/TLC多讀取電壓，因此不是LSB-only；但沒有頁類可見延遲/地址資料，FIFO還可能隱藏差異。Cambricon有獨立input buffer與共享die core，卻未給混合頁類契約。SLIM比較兩種SSD、KVNAND採SLC，均不能當同一器件兩級GEMV。Lincoln全文仍未取得。\n\nNASiC已在舊priorbase，本次重讀確認CAM+expert interleave直接相關；其analog/ADC能力不移植到普通NAND GEMV。另只保留Cambricon既有read/compute雙路角色的低優先理想bound，未證有新配置headroom。\n''')
files=[p for p in HERE.rglob('*') if p.is_file() and p.name!='completion.json']
put('completion.json',{'status':'SEALED_BOUNDED_PRIMARY_SCREEN_NO_NEW_GAIN_CLAIM','completed_utc':datetime.now(timezone.utc).isoformat(),'producer_source_sha256':sha(__file__),'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(files)}})
print(json.dumps({'notes_sha256':sha(HERE/'notes.json'),'sources_sha256':sha(HERE/'sources.json'),'completion_sha256':sha(HERE/'completion.json')},indent=2))
