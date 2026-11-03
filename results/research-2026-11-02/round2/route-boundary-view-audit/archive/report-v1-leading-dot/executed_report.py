#!/usr/bin/env python3
"""Evidence notes for the completed symbolic prefix construction audit."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
D=BASE/'route-boundary-view-audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(D/n).write_text(json.dumps(x,indent=2)+'\n')
r=json.loads((D/'construction_and_ledger.json').read_text())
assert r['status'].startswith('PASS')
sources=[
 {'id':'aif','title':'AiF: Accelerating On-Device LLM Inference Using In-Flash Processing','year':2025,'venue':'ISCA2025','url':'https://doi.org/10.1145/3695053.3731073','local_primary':'paper/aif.txt','local_sha256':sha(ROOT/'paper/aif.txt'),
  'read_depth':'Re-read section4.4/Fig14 and section5.2,local text lines704-889;source already examined in previous passes.',
  'evidence':'Native command takes input dimension,block ID,offset and page count. Sequential matrix data and input enter INT8 multipliers/adder tree. Ordered FIFO results are polled using SREG. Published aif_post uses equal submatrices;timing emulator returns dummy outputs.',
  'inference':'One contiguous same-C prefix is address-expressible using described chip primitives. Sharing two row maps and host scatter needs controller/software metadata. No disclosed valid-row/reset/output-scale ABI establishes actual numerical prefix execution.'},
 {'id':'overlap_drp_2016','title':'Ein Replikationsschema für multiple Fragmentierungen mit überlappenden Fragmenten','year':2016,'venue':'GvDB2016,CEUR-WS1594,pp33-38','url':'https://ceur-ws.org/Vol-1594/paper7.pdf','doi':None,
  'read_depth':'Read abstract/introduction and section3.1 overlap-DRP formulation/example,section3.2 conflict constraints,section3.3 optional conflicts/common subfragments(pp33-37 excerpts).Not a full evaluation read.Official CEUR proceedings confirms2016;PDF footer appears to have a2015 date typo.',
  'evidence':'Supports several horizontal fragmentations of one table. Uses fragment intersections and tuple-level replica constraints to reduce required servers;models common subfragments and optional conflicts.',
  'inference':'Direct precedent for overlapping alternative fragmentations and shared intersections;does not establish a NAND same-base prefix/page-rounded-GEMV construction or its numeric ABI. Neither shared fragments nor minimizing duplicate data is a new concept.'},
 {'id':'overlap_drp_2017','title':'A Replication Scheme for Multiple Fragmentations with Overlapping Fragments','year':2017,'online_year':2016,'venue':'The Computer Journal60(3):308-328','url':'https://doi.org/10.1093/comjnl/bxw041','publisher_url':'https://academic.oup.com/comjnl/article/60/3/308/2608047',
  'read_depth':'Publisher landing page redirected to inaccessibleCDN;publisherPDF not obtained. Search-indexed author-text excerpt and bibliographic metadata only;NOT counted as full-text method evidence. Related2016 CEUR author paper read directly above.',
  'evidence':'Indexed abstract describes multiple fragmentations with overlaps,replication/recovery and soft locality constraints. Online2016 versus volume2017 distinction retained.',
  'inference':'Strong related prior requiring explicit discussion;do not claim a precise algorithmic equivalence from unavailable full text. The suggested name overlapping range replication2016 was not independently identified as an exact paper title.'},
 {'id':'query_log_2013','title':'Query-Log Aware Replicated Declustering','year':2013,'online_year':2012,'venue':'IEEE TPDS24(5):987-995','doi':'https://doi.org/10.1109/TPDS.2012.113','url':'https://repository.bilkent.edu.tr/bitstreams/3e48b3b0-a708-4f57-9eb5-321cbd19928c/download',
  'read_depth':'Re-read abstract,section2 definitions/capacity and replication constraints,section3.1 two-way private/shared-item representation and replica scheduling,section3.2 multiway move/replication gain discussion. Publication year independently checked.',
  'evidence':'Selects items to replicate and destinations using query logs under capacity/replication budgets. Splits two-way storage into private A,private B and replicated AB,with query-dependent selection. Its retrieval model assumes equal unit item service times.',
  'inference':'A direct strong control is query-log-aware partial-row-packet replication/selection using the SAME prefix-legal physical layout and full native cost. Original unit-item formula alone omits page rounding,one-command prefix coalescing and GU/down readiness;those omissions do not make partial replication new.'},
 {'id':'replica_retrieval_2013','title':'Generalized Optimal Response Time Retrieval of Replicated Data from Storage Arrays','year':2013,'venue':'ACM TOS9(2),article5','doi':'https://doi.org/10.1145/2491472.2491474','url':'https://faculty.cse.louisville.edu/nihat/papers/TOS2013.pdf',
  'read_depth':'Read abstract and introduction/response-time formulation from primary authorPDF;previous pass examined LP/max-flow overview.Not a new full36page reading.',
  'evidence':'Replica selection accounts for heterogeneous disks,initial load and distributed network delay;LP and maxflow yield optimal retrieval under the paper model.',
  'inference':'Dynamic route-conditioned replica choice and load-aware selection are prior. A row-item flow relaxation can be a strong optimistic lower bound,butindependent row choices must not be presented as a feasible single-prefix view without packet coupling constraints.'},
 {'id':'replica_retrieval_2016','title':'Multithreaded Maximum Flow Based Optimal Replica Selection Algorithm for Heterogeneous Storage Architectures','year':2016,'online_year':2015,'venue':'IEEE TC65(5):1543-1557','doi':'https://doi.org/10.1109/TC.2015.2451620','url':'https://faculty.cse.louisville.edu/nihat/papers/TC2016.pdf',
  'read_depth':'Read abstract,introduction/model overview and section6 final partial-replication discussion(pp1543-1544,1556);not full algorithm/evaluation reading.',
  'evidence':'Explicitly states the retrieval algorithms also handle partial replication and nonuniform copies per bucket;single-copy buckets can be removed from flow graphs.',
  'inference':'A strong selector baseline should keep mandatory common rows fixed and optimize only replicated packets,while charging prefix constraints/native jobs. It does not itself show shared dense-GEMV prefix execution.'},
 {'id':'pdla_2020','title':'Optimizing Parallel I/O Accesses through Pattern-Directed and Layout-Aware Replication','year':2020,'online_year':2019,'venue':'IEEE TC69(2):212-225','doi':'https://doi.org/10.1109/TC.2019.2946135','url':'https://cs.iit.edu/~scs/assets/files/Opti_parallel_io.pdf','author_page':'https://grc.iit.edu/publications/he-2020-optimizing-parallel-5527/',
  'read_depth':'Read abstract,motivation and section3.1 overview/access-pattern definition;not full replication algorithm/evaluation.',
  'evidence':'Creates reorganized replicas according to access patterns and optimized layouts;runtime forwards I/O to selected replicas. Implemented within MPI-I/O/OrangeFS.',
  'inference':'Multiple access-pattern-specific physical replicas and runtime selection are established systems techniques. Our prefix constraint and row-function contract are a narrower implementation question,not first multi-layout replication.'},
 {'id':'tvm_layout','title':'Tensor Layout,ApacheTVM TIRx official documentation','year':2026,'version':'Live official documentation accessed2026-11-02;not a version-pinned research artifact','url':'https://tvm.apache.org/docs/tirx/layout.html',
  'read_depth':'Read layout overview/logical-to-physical mapping/design rationale.',
  'evidence':'Logical tensor indices map to sets of named physical coordinates,explicitly including shard,replica and offset axes.',
  'inference':'A compiler can already represent replication,offsets and multiple physical owners. The documentation is not an implementation of this NAND prefix allocation or proof of native legality.'}
]
write('sources.json',sources)

# Supplement the pre-frozen construction run with actual logical-range checks.
frozen=json.loads((D/'protocol.frozen.json').read_text());verified=0
for name,h in frozen['inputs_sha256'].items():
 p=Path(name);assert sha(p)==h
 if p.suffix!='json':continue
 doc=json.loads(p.read_text())
 for e in range(doc['shape'][2]):
  js=sorted([x for x in doc['logical'] if x['expert']==e and x['phase']==0],key=lambda x:x['chip']);cursor=0
  for x in js:assert x['row_start']==cursor;cursor+=x['rows'];verified+=1

table=[]
for key,ex in r['examples'].items():
 if key not in ('qwen38_101_75_singleedge','qwen38_101_76_singleedge_samechannel_reordered','granite4tiny.s0.t8','granite4tiny.s3.t11'):continue
 changed=[c for c in range(16) if ex['A']['rows'][c]!=ex['B']['rows'][c]]
 low=min(changed,key=lambda c:ex['A']['rows'][c]);short=next(v for v in ex['chips'][low]['views'] if v['name']=='A')
 table.append(dict(witness=key,delta_rows=ex['delta_rows'],extra_coefficient_bytes=ex['extra_coefficient_bytes'],
    extra_programmed_page_bytes=ex['extra_programmed_page_bytes'],extra_reserved_wave_bytes=ex['extra_reserved_wave_bytes'],
    native_commands_each_view=16,pages_each_view=ex['A']['total_pages'],waves_each_view=ex['A']['total_waves'],
    short_rows=short['valid_rows'],short_all_page_outputs=short['all_page_complete_outputs'],
    short_dropped_rows=short['drop_complete_output_rows'],short_partial_row_bytes=short['trailing_partial_row_bytes']))
notes={
 'status':'PASS conditional complete-row construction;native ABI unverified;no performance conclusion',
 'scope':'Main variant exchanges a SINGLE high/low pair s,t=s xor8,leaving14chips fixed. Whole-vector xor8 is only a separately labeled multi-edge control. Frozen v4 geometry is not silently substituted for v5/v6W8winners.',
 'proof':{
  'assumptions':'A,B are nonnegative integer rowcounts on the sameMchips,sumA=sumB=R;all rows have the SAME full input dimensionC and coefficient byte width;B logical ownership may be derived;each positive maximal prefix fits one block. Row outputs are separable,host knows valid output count and logical row identity.',
  'lower_bound':'Chipc needs at leastmax(Ac,Bc) resident logical rows to support both view counts. Therefore total physical rows>=summax(Ac,Bc)=R+0.5*sumabs(Ac-Bc). This is a basic count/transport bound,not a new theorem claim.',
  'construction':'Keep the original A physical row order. On every chip share the firstmin(Ac,Bc) rows. Collect each A-surplus contiguous tail packet and append its rows to B-deficit chips. A reads its originalAc-row prefix;B reads itsBc-row prefix. The common rows are private and every surplus logical row has exactly one source copy and one destination copy. Each chosen view covers each ofRlogical rows exactly once.',
  'single_edge':'For sourceAc=h,targetAt=l,h>l,Bexchanges these counts. Copy the SINGLE source tail ofh-lrows to the target after its original lrows. Only the target needs a new suffix;both start addresses are shared by A/B. A is unchanged;Bhost scatter restores the source tail after computing it at the target.',
  'limitations':'For prescribed A/B logical row sets,the count lower bound need not be tight. Actual union storage=sumc|SAc union SBc|. Equal-length but different row sets cannot both be the same-base prefix. Independently choosing long/short on each chip can duplicate or omit packets;choose one coherent view perexpert,or enforce packet-flow consistency explicitly.',
  'zeros':'If a view has zero rows on a chip,it issues no job there;zero-row GEMV is not assumed legal. All main real witnesses have16positive jobs.',
  'offline_append':'Append describes OFFLINE programming order in a newly packed resident image. No in-place NAND append,online migration,arbitrarypartial-pageprogram or freecopy command is assumed.'},
 'checks':{'exhaustive_A_B_vector_pairs':r['exhaustive_equal_sum_vector_pairs'],'actual_single_edge_geometry_certificates':r['actual_single_edge_certificates'],
           'actual_frozen_logical_row_starts_checked':verified,'actual_layers':[0,1,2],'actual_models':['qwen38','granite4tiny'],
           'routes_read':0,'heldout_read':False,'native_hardware_execution':False,'global_image_packed':False,
           'construction_ledger_sha256':sha(D/'construction_and_ledger.json')},
 'main_ledger_witnesses':table,
 'hardware_and_ABI':[
  'Each resident maximal prefix starts at plane0 of a wave-aligned offset;both views use that base and their own ordinary ceil(rows*C/pageB) pagecount. All shown maxima are2/4waves,wellwithin192wordlines. One standalone per-chip block address witness is included;global all-layer packing/capacity is NOT proven.',
  'Q38 101versus75:source26rows=66560B copied,but target marginal storage is4pages=65536B because its prior12-page prefix had4608B slack. 101versus76:25rows=64000B,also4extra pages. G4 83versus40:43rows=66048B;82versus40:42rows=64512B;both4extra pages. Keep useful-copy,programmed-page,reserved-wave and actual global allocated-block ledgers separate.',
  'The short prefix really reads into the appended packet. Q38 75rows,C2560,12pages emits76complete rows then2048bytes of a partial row:drop1complete append output. Q38 76rows uses12pages,emits76then2048partialbytes:drop0complete. G4 40rows,C1536,4pages emits42complete rows then1024partialbytes:drop2append outputs. Charge all requested-page outputs and drain/drop them before releasing ownership.',
  'Long prefix also has page fringe:Q38 101rows emits102complete outputs;G4 83/82rows emits85. Pad the maximal resident image safely through its final requested page. A partial row at a short boundary may contain real copied coefficients rather thanzeros;correct next-command row/accumulator reset cannot be inferred from the page-count interface.',
  'No partial sums,column split,multiple inputs or extra compute context is required by the algebra. Both views present the identical original full input to every full row. Host scatter restores the original gate/up logical row positions BEFORE activation and before the unchanged full Downinput quantization. Preserve per-row scale/bias metadata by logical rowID;shared host scale vectors need no new coefficient-page bytes.',
  'AiFchip command fields appear sufficient to express the two physical prefix ranges. Stock aif_post equal-submatrix allocation does not expose shared physical-prefix aliases,derived rowID scatter or viewregistry. Those need controller/compiler software. This is not proof the publishedhostAPI already executes them,nor evidence new chip hardware is necessary.',
  'Opaque row-function portability still requires rowseparability,input quantization once per original full vector,known output ordering/count,safe command reset and consistent per-row scale application. Global output quantization/statistics depending on all rows would violate the assumed contract. AiF outputdtype/scale/reset and numericalvalid-row behavior remain unverified.'
 ],
 'cost_upper_bound_and_invariants':[
  'Single swap preserves perexpert total useful rows,requested pages,ordinary read waves,jobcount,inputreload bytes and all-page outputcount in each view. Since s xor8 has the SAME channel(c%8),even per-channel input/output total work is invariant. Only one GU wave of occupancy and its small output timing are shifted between the two chip owners;Downmatrix ownership is unchanged.',
  'Therefore this mechanism cannot reduce total NAND work;its opportunity is reducing chip critical paths/readiness conflicts with later Downjobs. A pooled-work bound does not prove a speedup,and a zero simulated gain falsifies this geometry/cost case even if the copy proof is correct.',
  'For the optimal makespan of the SAME fixed job/dependency model,singleedge Q38 101vs75 changes one jobread by28000ns and allpageoutputcount by26scalars. Total decreased operation durations are28000+52channel+13PCIe=28065ns per switched expert;101vs76 has the SAME26-output fringe difference. Thus any optimal-schedule gain is bounded by the sum of such decreases over selected experts,by fixing a feasible resource/job order and applying longest-path monotonicity. This is a conditional processing-time perturbation bound,NOT a bound on differences between two independently chosen suboptimal greedy schedules.',
  'A one-layer view-selection screen should start with shared fixed;try both endpoint choices,allprefixgeometrymatched,copies1/4,andcommand250/5000. Include selectorCPUcost and pagefringe/copycapacity before any whole-model claim. TRAIN-onlyoraclebenefit is not heldout generalization.'
 ],
 'strong_controls':[
  {'name':'Same resident image/static view','control':'Aonly,Bonly,andperexpert fixed A/B chosen using TRAIN;identical replica bytes,addressimage,scatterhandling and allpagefringe. This separates runtime viewchoice from a better static physical mapping.'},
  {'name':'Routeblind view selection','control':'Sameimage with deterministic alternating/hash bits or static hotness balancing,plus trivial current-chip-load selector. Keep jobsarray/nodepriority order identical and charge selector time separately.'},
  {'name':'Query-log-aware replicated declustering','control':'Treat the h-l-row tail as replicated packet with mandatory private cores. Fit replica destination/selection using query-log objective,then evaluate its SAME legal prefix views in nativecost engine;equalTRAINbudget/capacity. Unit-row maxflow is an optimistic relaxation unless its packet choices form legal prefixes.'},
  {'name':'Exact small view oracle','control':'For topK6,exhaust64binary view combinations;topK10,1024combinations. Evaluate identical nativejobfamily/cost/scheduler. This bounds the chosen two-view family and tests whether faster heuristic control leaves any real headroom;it does not prove allplacements optimal.'},
  {'name':'Same-capacity geometry/copy budget','control':'Compare against previous strongest geometry/free staticmask controls with the SAME extra programmed/reserved/globalblockcapacity. Allow that budget to selectivehotexpert or alternativepacketreplicas. Do not compare against full2xexpertcopies or claim the newcopy fraction itself is novel.'},
  {'name':'Fragment retrieval with honest command cost','control':'If a prior-style packetselector assembles common+tail separately,charge extra densecommands/fullinputreload/outputs. Also include a prefix-aware prior control using the newsharedimage,so any gain cannot be attributed solely to withholding a legal coalescing implementation from baseline.'}
 ],
 'novelty_assessment':'No direct primary source inspected proves the identical ordinary-NAND shared-base two-prefix GEMV scheme with page-fringe drain and unchanged full-row contract. This bounded absence is not a priorityclaim. Selective replica placement,alternative horizontal fragmentations,shared intersections,query-dependent replica choice and multi-layout storage are direct priors. A possible contribution must demonstrate a useful capacity/command/latency envelope plus verified physical/numeric/controller contracts against the geometrymatched controls above;current evidence only establishes a conditional construction,not a paper breakthrough.',
 'low_cost_falsifiers':[
  'TRAINsmall-vieworacle gives negligible benefit once commoncontrols/Downreadiness/copies4/command5us are charged:archive thiscandidate without heldoutfitting.',
  'One-copyequal-capacitystronggeometryorprior-styleprefixselectionmatches theviewmethod:attributegain tostandardpartialreplication/coalescing anddocumentascontrol.',
  'Native two-command short/long/short test fails outputcount/order/reset or per-row output depends on appendedinactive rows:the opaque-row portabilitycontract fails;do notrepair by assuming new hardware.',
  'Global single-block packing cannot fit the max-prefix image at claimedcapacity,or hostscatter/selectorcost absorbs gain:rejectthecapacity/performanceclaim.'
 ],
 'source_scope':'Primary-only method conclusions use the accessible official/author papers and officialdocs in sources.json. The inaccessible2017journalfulltext and an unverified exact2016range-replication title are explicitly marked;no inference that they implement the same NAND construction.'}
write('notes.json',notes)
write('logical_range_check.json',{'status':'PASS','actual_frozen_jobs_row_start_matches_chiporder':verified,'source_sha256':sha(Path(__file__))})
write('completion.report.json',{'status':'complete','prior_construction_completion_sha256':sha(D/'completion.construction.json'),
 'postprocess_source_sha256':sha(Path(__file__)),'outputs_sha256':{str(D/p):sha(D/p) for p in ('notes.json','sources.json','logical_range_check.json','construction_and_ledger.json','protocol.frozen.json')}})
print(json.dumps({'notes_sha256':sha(D/'notes.json'),'sources_sha256':sha(D/'sources.json'),'witnesses':table},indent=2))
