#!/usr/bin/env python3
"""Seal this bounded, primary-source-only fulltext access followup.

No model inference, controller experiment, or technical claim from an unread
paper is performed. Existing research artifacts are not modified.
"""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / 'primary'
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def source(sid, url, depth, result, files=(), kind='primary'):
    return dict(id=sid, url=url, accessed_date_utc='2026-11-02',
                source_kind=kind, actually_read=depth, result=result,
                local_files=list(files))


sources = [
    source('diamond-ieee', 'https://ieeexplore.ieee.org/abstract/document/11617910',
           'Official abstract and publication metadata through web search; direct publisher author-page response is HTTP 418, not full text.',
           'Confirms in-NAND/near-DRAM heterogeneous architecture, mask-based mapping and dynamic online expert selection. No algorithm/ISA details read.',
           ['primary/diamond_ieee_authors.html']),
    source('diamond-crossref', 'https://api.crossref.org/works/10.1109/ISCA66397.2026.00078',
           'Entire publisher-deposited Crossref JSON record; no abstract in this record.',
           'ISCA 2026, pp. 985–1000; issued June 2026, record created 2026-08-04. Only fulltext-style link is IEEE staging marked similarity-checking; not pursued as public access.',
           ['primary/diamond.crossref.json']),
    source('diamond-isca-program', 'https://iscaconf.org/isca2026/program/',
           'Official program, paper entry and authors/affiliations; surrounding links inspected.',
           'Confirms correct title and authors. No paper PDF/artifact link at the entry.',
           ['primary/isca_program.html']),
    source('flexllm-crossref', 'https://api.crossref.org/works/10.1145/3844618',
           'Entire publisher-deposited Crossref JSON, including complete abstract and license/date/link fields.',
           'Abstract explicitly names co-activation neuron-inspired weight reordering and sparsity-aware dynamic partitioning. Published-online 2026-10-13, record created 2026-08-29, print date registered 2026-12-31; CC-BY 4.0 start 2026-10-13. No full paper obtained.',
           ['primary/flexllm.crossref.json']),
    source('flexllm-standard-pdf', 'https://dl.acm.org/doi/pdf/10.1145/3844618',
           'One ordinary public PDF request in this followup, following the registered OA license; response is HTML access challenge.',
           'HTTP 403, 5501 bytes, not a PDF. No cookies, altered credentials, staging endpoint or challenge bypass attempted. No further URL variant loop.',
           ['primary/flexllm_standard_oa_pdf.response', 'primary/last.acquisition.json']),
    source('ma-author-home', 'https://matianyun.github.io/',
           'Full author homepage and complete /publications/ listing, including FlexLLM bibliography expansion.',
           'FlexLLM entry provides DOI and Bib only, no author PDF or code.',
           ['primary/ma_home.html', 'primary/ma_publications.response']),
    source('ma-author-bib', 'https://raw.githubusercontent.com/matianyun/matianyun.github.io/8b43cfe305f9cebf11000a9762e55e9c93e19542/_bibliography/papers.bib',
           'Full bibliography file, target entry inspected.',
           'Target has title/authors/venue/year/DOI, no PDF/arXiv/artifact URL.',
           ['primary/ma_papers_bib.response']),
    source('ma-author-repository', 'https://github.com/matianyun/matianyun.github.io',
           'Latest commit metadata and recursive tree at 8b43cfe305f9cebf11000a9762e55e9c93e19542; publication/PDF paths inspected.',
           'Commit dated 2026-09-07. Only assets/pdf entry is Tianyun_Ma_CV.pdf; no FlexLLM PDF in this tree. This is a bounded repository observation, not proof no private artifact exists.',
           ['primary/ma_commit.json', 'primary/ma_repository_tree.json']),
    source('flexllm-author-kudos', 'https://www.growkudos.com/publications/10.1145%25252F3844618/reader',
           'Author-attributed plain-language summary and Tianyun Ma perspective through search; not the paper.',
           'Groups data used together and adapts work; insufficient to determine any requested detailed mechanism. Not used as a substitute fulltext.', []),
    source('meng-author-page', 'https://mengli.me/',
           'Full homepage publication list and /publication/ archive searched for exact target name/title.',
           'DIAMoND not located. Other NAND papers are not substitutes for DIAMoND.',
           ['primary/meng_home.html', 'primary/meng_publication_archive.response']),
    source('zhong-author-page', 'https://shuzhangzhong.github.io/publication/',
           'Publications page read; primary homepage downloaded.',
           'No DIAMoND entry or author PDF found in this bounded page inspection.',
           ['primary/zhong_home.html']),
    source('sun-author-page', 'https://gsunpku.github.io/',
           'Full short author homepage; previous CECA homepage also checked.',
           'No DIAMoND author PDF found. Unrelated guangyusun.com identity not used.',
           ['primary/sun_home.txt', 'primary/sun_old.html']),
    source('zhao-author-page', 'https://yongwei.site/zh/about/',
           'Author profile and visible publication links read.',
           'No FlexLLM fulltext found.', ['primary/zhao_home.html']),
    source('arxiv-title-search', 'https://export.arxiv.org/api/query?search_query=ti:%22Cambricon-FlexLLM%22&start=0&max_results=20',
           'Entire official API Atom result; separate exact DIAMoND title-phrase query also inspected.',
           'Both target-specific queries returned totalResults=0. Does not exclude a differently titled/unindexed preprint.',
           ['primary/arxiv_flexllm_full_title.response', 'primary/arxiv_diamond_full_title.response', 'primary/final.acquisition.json']),
    source('github-repository-search', 'https://api.github.com/search/repositories?q=Cambricon-FlexLLM',
           'Official public GitHub repository search responses for Cambricon-FlexLLM and DIAMoND NAND.',
           'FlexLLM: zero repositories. DIAMoND NAND: two unrelated repositories. Not a GitHub code search or proof no differently named artifact exists. Connector search initially failed as unsupported; direct API results, not that error, support this observation.',
           ['primary/github_repo_search_flexllm.response', 'primary/github_repo_search_diamond.response']),
    source('pku-institutional-search', 'https://ir.pku.edu.cn/simple-search?query=DIAMoND',
           'One ordinary institutional repository search request.',
           'Connection timeout; no catalog or fulltext read. This is unavailable evidence, not a negative search result.',
           ['primary/additional.acquisition.json']),
    source('ict-institutional-search', 'https://ir.ict.ac.cn/simple-search?query=Cambricon-FlexLLM',
           'One ordinary institutional repository search request.',
           'DNS resolution failure; no catalog or fulltext read. This is unavailable evidence, not a negative search result.',
           ['primary/additional.acquisition.json']),
    source('old-cambricon-arxiv', 'https://arxiv.org/abs/2409.15654',
           'Official metadata and submission history read.',
           'Cambricon-LLM MICRO 2024, submitted 2024-09-24; only v1 listed. Not the TACO 2026 FlexLLM fulltext and not used to infer its added mechanisms.',
           ['primary/cambricon_arxiv.html']),
    source('openalex-access-discovery', 'https://api.openalex.org/works/https://doi.org/10.1145/3844618',
           'Both target DOI records and all listed locations inspected, solely to discover public fulltext URLs.',
           'FlexLLM marked diamond OA and DIAMoND closed; neither record supplied an accessible repository PDF. Metadata classification is not fulltext access or technical evidence.',
           ['primary/flexllm.openalex.json', 'primary/diamond.openalex.json'], kind='discovery-only, non-primary index'),
]
write('sources.json', {'generated_utc': NOW, 'sources': sources})

fields = ['whole_row_or_column_partition_and_integer_wave_geometry',
          'gate_up_and_down_ownership_or_correlated_row_counts',
          'same_input_dense_fusion_and_singleton_overlapping_read_views',
          'FIFO_prefix_outputs_and_independent_logical_expert_completion',
          'one_copy_or_replication_and_capacity_accounting',
          'runtime_mapping_and_route_fixed_scheduler_contract',
          'ordinary_read_vs_CR_read_and_plane_context_requirements',
          'numeric_output_scale_accumulator_and_partial_merge_ABI']
notes = {
    'generated_utc': NOW,
    'scope': 'Bounded 20-minute legal public fulltext/artifact followup; actual read depth only. No model inference, author contact, subagent, device action, or sealed-source modification.',
    'conclusion': 'Neither requested fulltext was obtained. No public author artifact was located within this search. Both remain unresolved closest-prior gaps; inaccessible details cannot be treated as absent.',
    'papers': [
        {'title': 'DIAMoND: Dynamic Inference for Adaptive Edge MOE with Heterogeneous In-NAND and Near-DRAM Compute Architecture',
         'doi': '10.1109/ISCA66397.2026.00078', 'venue': 'ISCA 2026',
         'fulltext_obtained': False, 'official_artifact_located': False,
         'actual_depth': 'Official abstract, program and publisher-deposited bibliographic metadata; author pages and access discovery only. No body sections, figures, tables or algorithm read.',
         'confirmed_primary_evidence': ['Heterogeneous in-NAND and near-DRAM architecture via 2.5D packaging.', 'Mask-based mapping addresses variable matrix sizes versus a fixed NAND array.', 'Dynamic online expert selection is proposed.'],
         'mechanism_matrix': {x: 'unverified: fulltext not read' for x in fields},
         'avoid_inference': ['Mask-based mapping does not establish our whole-row/no-new-decoder contract.', 'Dynamic expert selection does not establish fixed-routing runtime placement.', 'No manufactured-silicon status or timing portability inferred from abstract.']},
        {'title': 'Cambricon-FlexLLM: A Flexible Chiplet-Based Hybrid Architecture for On-Device 70B LLM Inference',
         'doi': '10.1145/3844618', 'venue': 'ACM TACO 2026',
         'fulltext_obtained': False, 'official_artifact_located': False,
         'actual_depth': 'Complete publisher-deposited abstract/metadata/license, author publication entry and bibliography, author repository tree, author-written plain-language summary. No body sections, figures, tables or algorithm read.',
         'confirmed_primary_evidence': ['Hybrid NPU/dedicated NAND design and hardware tiling reduce data movement.', 'Abstract explicitly proposes co-activation neuron-inspired weight reordering and sparsity-aware dynamic partitioning.', 'Author publication entry/bibliography provides DOI only; no public PDF/code there.'],
         'date_distinction': {'crossref_record_created': '2026-08-29', 'crossref_published_online': '2026-10-13', 'crossref_print_registration_future_date': '2026-12-31', 'CC_BY_4_0_license_start': '2026-10-13'},
         'mechanism_matrix': {x: 'unverified: fulltext not read' for x in fields},
         'avoid_inference': ['Cannot replace this paper with 2024 Cambricon-LLM or unrelated FlexLLM papers.', 'CC-BY metadata does not mean this environment successfully downloaded the paper.', 'Abstract reordering/partitioning is direct prior against broad first-coactivation-layout claims, but does not prove or disprove an exact match to ordinary whole-row overlapping views.']}
    ],
    'novelty_implication': {'supported': 'Retain both as unresolved direct threats. Restrict any candidate claims to documented native contracts, exact controls and measured evidence, without asserting these papers lack the same mechanism.',
                           'not_supported': 'No fulltext-based exclusion of row geometry, GU/Down coupling, same-x fusion, prefix output, copies or runtime mapping; no performance conclusion from secondary summaries.'},
    'stopping_rule': 'No repeated blocked publisher URL variants or staged/TDM access attempts. Exact-title arXiv and public GitHub searches, author publication pages/repository and institutional endpoints exhausted within the bound; stop with explicit missing-fulltext status.',
    'search_limitations': ['Web/arXiv/repository searches are not exhaustive; alternate titles and unindexed author manuscripts remain possible.', 'Institutional repository failures are access failures, not evidence the paper is absent.', 'No author contacted or private/subscription credentials used.'],
}
write('notes.json', notes)

attempts = []
for f in sorted(PRIMARY.glob('*acquisition.json')):
    attempts.extend([dict(log_file=str(f.relative_to(ROOT)), **x) for x in json.loads(f.read_text())])
write('access_attempts.json', {'generated_utc': NOW, 'recorded_attempts': attempts,
                             'additional_readings': 'See sources.json for web-tool readings and author pages with saved raw snapshots; not every web-tool access yielded a local response file.'})
manifest = []
for f in sorted(ROOT.rglob('*')):
    if f.is_file() and f.name != 'completion.json':
        manifest.append({'path': str(f.relative_to(ROOT)), 'bytes': f.stat().st_size,
                         'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
write('completion.json', {'completed_utc': NOW, 'status': 'bounded_search_complete_no_fulltexts_obtained',
                          'files': manifest})
print(json.dumps({'notes_sha256': hashlib.sha256((ROOT/'notes.json').read_bytes()).hexdigest(),
                  'sources_sha256': hashlib.sha256((ROOT/'sources.json').read_bytes()).hexdigest(),
                  'files': len(manifest)}, indent=2))
