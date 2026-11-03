#!/usr/bin/env python3
"""Bounded primary-prior audit and symbolic two-view inventory; no timings."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = HERE.parents[4]
P, B, C, R, E = 16384, 65536, 1536, 1024, 64


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def ceil(n, d):
    return (n + d - 1) // d


def construction(omitted_high_rows):
    b = [83] * 6 + [82] * 3 + [40] * 7
    omitted = next(i for i, n in enumerate(b[:9]) if n == omitted_high_rows)
    sources = [i for i in range(9) if i != omitted]
    a = [128 if i in sources else 0 for i in range(16)]
    core, cursor = [], 0
    for n in b:
        core.append(list(range(cursor, cursor + n)))
        cursor += n
    assert cursor == R
    packet = [r for i in range(16) if i not in sources for r in core[i]]
    image, cursor = [x.copy() for x in core], 0
    for i in sources:
        n = 128 - b[i]
        image[i] += packet[cursor:cursor + n]
        cursor += n
    assert cursor == len(packet)
    assert all(len(image[i]) == max(a[i], b[i]) for i in range(16))
    for counts in (a, b):
        selected = [row for i, n in enumerate(counts) for row in image[i][:n]]
        assert Counter(selected) == Counter(range(R))
    delta = sum(max(x, y) for x, y in zip(a, b)) - R
    assert delta == sum(abs(x - y) for x, y in zip(a, b)) // 2 == len(packet)
    def inventory(rows):
        pages = [ceil(n * C, P) for n in rows]
        waves = [ceil(n, 4) for n in pages]
        return dict(rows=rows, commands=sum(n > 0 for n in rows),
                    pages=pages, waves=waves, total_pages=sum(pages),
                    total_waves=sum(waves), useful_rows=sum(rows),
                    all_page_complete_outputs=sum(n * P // C for n in pages),
                    full_input_reload_bytes=sum(n > 0 for n in rows) * C)
    ia, ib, iu = inventory(a), inventory(b), inventory(list(map(len, image)))
    assert (ia['total_waves'], ib['total_waves'], iu['total_waves']) == (24, 25, 33)
    return dict(omitted_high_rows=omitted_high_rows, source_chips=sources,
                A=ia, B=ib, resident_union=iu, duplicate_rows=delta,
                duplicate_coefficient_bytes_per_expert=delta * C,
                duplicate_coefficient_bytes_64_experts=delta * C * E,
                duplicate_coefficient_MiB_64_experts=delta * C * E / 2**20,
                resident_wave_bytes_64_experts=iu['total_waves'] * B * E,
                additional_resident_vs_B=dict(waves_64_experts=8 * E, MiB=8 * E * B / 2**20),
                additional_resident_vs_A=dict(waves_64_experts=9 * E, MiB=9 * E * B / 2**20),
                scope_of_64_expert_totals='One MoE layer, not a40-layer model.',
                illustrative_40_layer_totals=dict(duplicate_coefficient_GiB=delta*C*E*40/2**30,
                    extra_reserved_vs_B_GiB=8*E*B*40/2**30,
                    extra_reserved_vs_A_GiB=9*E*B*40/2**30,
                    scope='Simple repeated inventory only; no all-layer global allocated-block packing proof.'),
                views_cover_each_logical_row_once=True,
                source_short_prefix_complete_outputs=131072 // C,
                source_short_prefix_partial_row_bytes=131072 % C,
                source_short_prefix_drop_complete_outputs={str(n):131072 // C-n for n in (82,83)},
                one_block_max_waves=3,
                limitations='Symbolic coefficient row identity and single-expert block inventory only; no numeric chip execution, all-layer physical packing, timing or learned layout claim.')


def main():
    now = datetime.now(timezone.utc).isoformat()
    write('geometry_inventory.json', {'created_utc': now, 'coefficient_bytes_per_row': C,
        'cases': [construction(82), construction(83)],
        'conditional_two_view_proof': 'With flexible derived row ownership, each chip stores its common shorter prefix and any longer-view appendix. A coherent whole-expert view covers all rows once. Resident rows sum max(Ac,Bc), matching the elementary per-chip count bound. Prescribed conflicting row sets need not admit the construction.',
        'three_view_triangle': {'R':2, 'views':[[1,1,0],[1,0,1],[0,1,1]],
            'all_assignments_checked':8,
            'satisfying_two_row_ID_assignments':sum(all(len({ids[i] for i,n in enumerate(v) if n})==2 for v in [[1,1,0],[1,0,1],[0,1,1]]) for ids in itertools.product(range(2),repeat=3)),
            'conclusion':'Infeasible under strict same-base/exact-useful-prefix counts at any deeper appendix capacity: all three chip pairs must have unequal first-row IDs, impossible with two IDs. Additional deeper slots are not consumed by these r=1 views.',
            'outside_model':'Changing per-view starts or reading extra rows and selecting valid logical IDs changes visibility. Paid NAND page fringe could provide such freedom, but no feasibility or extra-wave claim for that relaxed model was tested here.'}})

    sources = [
      dict(id='Tutel-paper',title='Tutel: Adaptive Mixture-of-Experts at Scale',version='arXiv2206.03382v2, 2023-06-05; MLSys2023',url='https://arxiv.org/html/2206.03382v2',
           actual_read='Re-read §3.1.1–3.1.2 and Figs6–8/Table4 discussion, §3.3 overview; not a new full evaluation reading.',
           evidence='Single distributed weight layout and adaptive:r permit changing parallelism without layout reformatting. DP/MP group choices use different within-group all-gathers, repeated inputs and final sums.',
           boundary='Direct prior for runtime parallel-degree selection on shared parameters. Does not show the inspected NAND same-base resident short/long row-prefix mechanism; runtime all-gather is substantive and cannot be assumed free in NAND.'),
      dict(id='Tutel-code',title='Official Tutel adaptive expert implementation',version='cbebd74b4e5b10b6ed25203f60dc441c5dc58b9a',url='https://github.com/microsoft/tutel/tree/cbebd74b4e5b10b6ed25203f60dc441c5dc58b9a',
           actual_read='README adaptive descriptions; impls/moe_layer.py initialization/adaptive:r and forward lines125–143,328–357; experts/ffn.py constructor and forward lines83–119. No GPU run.',
           evidence='forward accepts adaptive_r. ffn.py invokes zero_gather for both projections when DP/group gathering is required. valid_rs and hidden-size divisibility constrain the current implementation.',
           boundary='Executable official GPU implementation exists. Snapshot is current main, not claimed identical to 2023 artifact. No inspected physical NAND prefix-image allocator.'),
      dict(id='AnchorTP',title='AnchorTP: Resilient LLM Inference with State-Preserving Elastic Tensor Parallelism',version='arXiv2511.11617v1, 2025-11-05',url='https://arxiv.org/html/2511.11617v1',
           actual_read='Re-read §III-A–C/Algorithm1/Fig3, implementation and discussion excerpts; not full evaluation or a verified author-code execution.',
           evidence='Unequal floor/ceil shards over arbitrary GPU counts. Daemon preserves contiguous parameters; interval intersections generate Transfer tasks and missing ranges Reload tasks.',
           boundary='Direct prior for retaining common rows during TP reshaping, but it performs migration/reload in a freeze window. Does not establish statically resident zero-migration prefix alternatives selected per routed query.'),
      dict(id='ElasticMoE',title='ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models',version='arXiv2510.02613v1, 2025-10-02; venue not verified',url='https://arxiv.org/html/2510.02613v1',
           actual_read='Read §4.1,4.4–4.6, §5 lifecycle overview, AppendixC limitation and D.1/D.4–D.5/E details; not all evaluation or code.',
           evidence='TP explicitly stays fixed while DP/EP change. Shared HBM IPC references and virtual physical-page mappings reduce duplicate allocation; P2P moves expert weights during scaling.',
           boundary='A primary system implementation of shared-memory configurations, but not evidence of different expert TP degrees or static NAND prefix images.'),
      dict(id='ElasticMoE-RFC',title='RFC: Efficient Runtime-Reconfiguration of Parallelization Configuration of MoE Models',version='vllm-ascend issue6372 opened2026-01-29; accessed2026-11-02',url='https://github.com/vllm-project/vllm-ascend/issues/6372',
           actual_read='Entire issue body, status and milestone/development metadata; official API response saved.',
           evidence='Proposes concurrent different-TP instances sharing weights via HMM. TP dense/MoE support listed as M4/M5 milestones. Issue is open, with no linked branches/PRs in visible metadata.',
           boundary='Specific author/developer proposal, not verified merged implementation or a demonstrated same-base prefix solver. Broader shared weights across parallel configurations cannot be claimed first.'),
      dict(id='overlap-DRP',title='Ein Replikationsschema für multiple Fragmentierungen mit überlappenden Fragmenten',version='GvDB2016, CEUR1594 pp33–38',url='https://ceur-ws.org/Vol-1594/paper7.pdf',
           actual_read='Fresh read §2.1 correctness, §3 introduction/§3.1 Definition2/examples/formulation and §3.2–3.3 overview in downloaded official proceedings; not an experimental reproduction.',
           evidence='Different horizontal fragmentations share tuple intersections; tuple-level redundancy can be less than independently replicating complete fragments. Models capacity/conflicts and optional fragmentation constraints.',
           boundary='Direct prior for multiple overlapping physical fragmentations and duplicate minimization. Does not enforce one dense contiguous prefix per device or NAND setup/page work; reconstruction can combine fragment intersections. PDF footer2015 conflicts with official2016 proceedings date, retained as previously noted.'),
      dict(id='replicated-declustering',title='Query-Log Aware Replicated Declustering',version='IEEE TPDS24(5),2013; online2012; DOI10.1109/TPDS.2012.113',url='https://repository.bilkent.edu.tr/bitstreams/3e48b3b0-a708-4f57-9eb5-321cbd19928c/download',
           actual_read='Reused earlier primary reading documented in route-boundary-view-audit/sources.json: §2 and3.1–3.2. Fresh web request timed out; no new fulltext reading claimed.',
           evidence='Private/shared item decomposition, selective replicas under storage budgets, and query-dependent replica scheduling are established.',
           boundary='Strong native-adapted packet placement/selector control; independent item choices must be coupled into coherent A/B prefixes and charged native commands/inputs, not assumed feasible for free.'),
      dict(id='MCBC',title='Lower Bounds for Total Storage of Multiset Combinatorial Batch Codes Using Linear Programming',version='IEEE TIT67(1),2021; accepted/published2020',url='https://ymchee66.github.io/home/PDF/lbstoragejournal.pdf',
           actual_read='Read abstract/introduction and §II-A Definitions1–2 and example3 (pp255–257); no later LP constructions proof read.',
           evidence='Replication-based set systems minimize total storage while providing restricted per-server parallel retrieval of requested symbols.',
           boundary='Direct combinatorial storage/retrieval precedent. Its item-wise retrieval budget is not one-prefix, complete-expert GU/Down causal execution or integer page/setup service. Do not transfer its bounds without compound-query/prefix constraints.'),
      dict(id='HDA-reused',title='HDA-MoE: Hybrid Parallelism and Dynamic, Adaptive Scheduling for Mixture-of-Experts with 3D Near-Memory Processing',version='arXiv2609.08682v1; official code56ea3725b2c621d4aea248f16345c34f5e06b8ed',url='https://arxiv.org/html/2609.08682v1',code='https://github.com/PKU-SEC-Lab/HDA-MoE-TCAD26/tree/56ea3725b2c621d4aea248f16345c34f5e06b8ed',
           actual_read='Prior primary reading/code audit reused from coupled-closest-prior/sources.json, not a fresh fulltext/code reread.',
           evidence='Per-expert fractional P_ic allocations, capacity/load optimization and coactivation-related communication objective; inspected code has no GU/Down phase axis or row-address replica-view registry.',
           boundary='Direct heterogeneous-sharding/placement prior. Inspected formulation does not establish resident multiple-geometry prefix replicas; this is narrower than asserting HDA cannot express any such extended optimization.'),
      dict(id='AiF',title='AiF: Accelerating On-Device LLM Inference Using In-Flash Processing',version='ISCA2025, DOI10.1145/3695053.3731073',url='https://doi.org/10.1145/3695053.3731073',
           actual_read='Re-read local paper/aif.txt §4.4/Fig14 and §5.2 with prior numerical/streaming audit limits retained.',
           evidence='Dense command input dimension/block/offset/page count, sequential matrix stream and output FIFO; equal-submatrix stock aif_post.',
           boundary='Shared-prefix aliases/rowID scatter require controller/compiler metadata. Output scale/type, useful-row semantics, partial-row reset and finite FIFO/flow control remain unverified; two-view algebra does not prove the native ABI.'),
    ]
    write('sources.json', {'created_utc':now,'primary_only':True,'sources':sources})
    write('notes.json', {
      'created_utc':now,
      'scope':'Bounded closest-prior audit and symbolic row inventory only; no new inference/timings/model download, no modifications to old sealed artifacts, no subagents.',
      'conclusion':'Adaptive parallel degrees, shared parameter layouts, interval reuse, selective replication and overlapping alternative fragmentations all have direct prior. This audit did not find an inspected primary implementation of the exact static NAND common-core-plus-appendix two-prefix mechanism. That bounded observation is not a first/novelty claim. A plausible system question remains whether a small resident replica budget makes established adaptive parallelism worthwhile under actual ordinary-read integer work/depth, command/input and capacity costs.',
      'what_changes_from_old_single_edge':'A/B now change active-chip span8/16, GU work24/25waves, commands8/16 and job depth3/2waves. Previous single-edge views only redistributed fixed work/span. Down and whole-GU/full-original-vector quantization barriers remain unchanged.',
      'attribution':[
        'Two complete A/B copies execute exactly the same native jobs as shared prefixes if their address/block geometry is matched. Prefix sharing itself saves resident capacity; runtime latency comes from selecting an established alternative parallel profile, not faster shared coefficients.',
        'The relevant contribution, if measured, is the feasible capacity/latency envelope and native construction/control, not first elastic TP, first partial replication or a new half-L1 theorem.',
        'No broad system-value rejection follows merely from generic hypergraph/set-system expressibility. Require an actual matched implementation or a measured counterexample.'
      ],
      'strong_same_native_controls':[
        'A-only, B-only and best per-expert static A/B under the same row geometry/source masks/Down/shared controls and native scheduler. Preserve incumbent and equal TRAIN search budgets.',
        'Same resident image with route-blind hash/alternating views, current-queue/load greedy selection and TRAIN-fit packet/replica scheduling inspired by replicated declustering. Give all selectors the same legal prefixes and full native work/inputs/outputs.',
        'For topK6 enumerate64 whole-expert A/B choices using identical native cost; this is the small two-view family oracle, not optimal all-placements scheduling. Compare practical selector time and heldout only after useful TRAIN headroom.',
        'Full separate A/B backups: identical service geometry should match selected-view latency exactly, while using more storage. Do not credit the shared-prefix image with a runtime speedup over geometrically matched complete copies.',
        'Same globally allocated capacity budget applied to selective complete GU backups/hot experts or alternative row-packet replicas; choose which experts and layouts using the same TRAIN budget. Capacity reference must be stated: +512waves vs B or +576 vs A for64experts, before all-layer block packing.',
        'Prefix-aware prior control must reuse the same common-core/appendix coalescing if legal. A baseline that retrieves common/tail separately must pay extra commands/reloads, but that weaker baseline alone cannot establish a new selector contribution.',
        'Retain one-copy strongest mixed/static/free geometry and fair shared expert frontier. A dual-view gain over only g16/equal or only one fixed endpoint is insufficient.'
      ],
      'native_contract':[
        'Every consumed row uses original full C input and its unchanged coefficients/scales. Host logical rowID scatter restores all gate/up rows before activation; Down still uses a full expert input quantized once. No column partials or multiple contexts are introduced by the algebra.',
        'A positive command is one contiguous same-base prefix in a maximal image of at most3waves; single-block address existence is easy for one expert, but global physical packing/capacity must be independently proven.',
        'Short B source prefix reads8pages=131072bytes, emitting85complete row functions plus512bytes of a partial row: discard3 outputs for82usefulrows or2 for83. These fringe coefficients belong to the A appendix, not guaranteed zero padding. Charge all page outputs and command ownership through drain.',
        'Correct useful-prefix row order/separability, row-scale association and command accumulator reset remain required unknown native ABI conditions; paper FIFO alone does not validate them. Do not infer new hardware is needed merely because stock aif_post lacks aliases.'
      ],
      'low_cost_falsifiers':[
        '64-choice native TRAIN oracle under matched Down/shared and realistic input1/4 plus command250ns/5us leaves negligible whole-FFN headroom: stop without new heldout inference.',
        'Best static A/B assignment or same-capacity selective complete-backup/packet baseline matches dynamic image: attribute benefit to existing replication/geometry policy, not a new system mechanism.',
        'Short/long/short native numeric/reset test is inconsistent, or actual output contract depends on inactive appended rows: contract fails; no speculative decoder/reset extension.',
        'Global allocated blocks, metadata/scatter and selector overhead erase gain: reject practical envelope despite valid coefficient count proof.'
      ],
      'multi_view_boundary':'See geometry_inventory.json triangle. Two-view half-L1 construction does not imply arbitrary3+view same-base prefix images. Paid page-fringe overread/host selection is a separate untested relaxation.',
      'unresolved_fulltext_gaps':'DIAMoND ISCA2026 and Cambricon-FlexLLM TACO2026 full texts remain unavailable in this environment; see sibling fulltext-gap-followup. No exclusion of exact mechanism based on their abstracts.',
      'reused_primary_audit_input_hashes':{str(p):sha(p) for p in [BASE/'coupled-closest-prior/sources.json',BASE/'route-boundary-view-audit/sources.json',BASE/'fulltext-gap-followup/notes.json',REPO/'paper/aif.txt']}
    })
    files=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='completion.json':
            files.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)))
    write('completion.json',{'created_utc':now,'status':'bounded_audit_complete_no_timing_claim','files':files})
    print(json.dumps({'notes_sha256':sha(HERE/'notes.json'),'geometry_sha256':sha(HERE/'geometry_inventory.json'),'source_count':len(sources)},indent=2))


if __name__=='__main__':
    main()
