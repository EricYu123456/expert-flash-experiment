#!/usr/bin/env python3
"""Bounded primary-evidence audit and tiny exhaustive GU view arithmetic."""
import hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/research-2026-11-02/round2/ordinary-gu-tile-sharing/closest-prior'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,z):assert not p.exists();p.write_text(json.dumps(z,indent=2)+'\n')
def main():
    # No route file is read. Four contiguous176-row experts,one copy,one block.
    rows=176;C=2048;B=65536;rpw=B//C
    views=[(e*rows//rpw,((e+1)*rows+rpw-1)//rpw) for e in range(4)]
    assert views==[(0,6),(5,11),(11,17),(16,22)]
    cases=[]
    for bits in itertools.product([0,1],repeat=4):
        selected=[e for e,b in enumerate(bits) if b];u=set()
        for e in selected:u.update(range(*views[e]))
        pairs=sum(bits[e]*bits[e+1] for e in [0,2]);assert len(u)==6*len(selected)-pairs
        loaded=set(r for w in u for r in range(w*rpw,(w+1)*rpw))
        needed=set(r for e in selected for r in range(e*rows,(e+1)*rows));assert needed<=loaded
        assert len(loaded-needed)==len(u)*rpw-len(selected)*rows
        cases.append(dict(selected=selected,union_waves=len(u),saved_waves_vs_separate_views=pairs,
          emitted_complete_rows=len(loaded),useful_rows=len(needed),inactive_rows_drained_and_dropped=len(loaded-needed)))
    def owner(w,n,copies=1):return 250+1024*copies+28000*w+2*n
    geometry=dict(C_bytes=C,wave_bytes=B,rows_per_wave=rpw,uniform_per_chip_rows=rows,
      singleton_views_wave_intervals=views,coefficient_copy_count=1,enumerated_cases=cases,
      singleton_owner_ns_input1=owner(6,192),coactive_pair_owner_ns_input1=owner(11,352),
      pair_owner_saved_vs_two_full_wave_views_ns=2*owner(6,192)-owner(11,352),
      pair_owner_saved_input4_ns=2*owner(6,192,4)-owner(11,352,4),
      old22page_singleton_owner_ns=owner(6,176),fullwave_view_singleton_extra_owner_ns=owner(6,192)-owner(6,176),
      important='Uniform176/chip already has88 useful waves/expert,but96individually rounded waves. Balanced192x8+160x8 achieves88 without fusion. Sharing is not an8wave improvement over this strong baseline.',
      NAND_work_identity='For a single chip containing pairs of176-row experts at wave-aligned pair starts,unionwaves=6*K-number_of_coactivated_pairs. The boundary between consecutive pairs saves no NANDwave,although merging can still save command/input costs.',
      matching_implication='Ignoring block boundaries/channel queues/causal latency and minimizing expected single-chip NANDwork,maximum-weight perfectmatching with edgeweight train coactivation count is exact for the fixed176-row geometry. Greedy Hamiltonian path is insufficient because only alternating boundaries share a physicalwave.',
      block_limit='Each352-row pair occupies11waves and can remain in one192WLblock;17pairs occupy187waves. Pair-aligned packing may waste5waves/block;globalcapacity must be compared to5/6wave balanced jobs,not to a padded96wave allocator.',
      cost_scope='Scalar owner work only;command250ns,ordinaryread28us,2GB/sONFI,4Boutputs. PCIe,controller queues,fullGU barrier,down readiness and hostscatter are outside this arithmetic. No hardware latency calibration.')
    write(D/'geometry_checks.json',geometry)
    sources=[
      dict(id='bigmomo',title='BigMoMo: Efficient Inference of Large-Scale MoE with Speculative Decoding on Mobile Devices',version='arXiv2609.14643v1,13Sep2026;preprint',url='https://arxiv.org/html/2609.14643v1',read_depth='Read3.2,4.1,4.3complete,4.4and5.1implementation;did not audit full evaluation orfigurepixels.',
        evidence='4.3 uses recent co-loading hyperedges,selects a contiguous anchorregion,models one anchor read plus scattered outside-expertreads,and swaps experts by cumulative profiled readtime benefit. Swap preserves total footprint;writes wait for bothweights inDRAM andyield tocriticalreads.',
        relation='Direct prior for single-copy query/co-loading-aware contiguous flashregions andruntime request consolidation.',
        unestablished='No rawchip/WL/planehalf-wave geometry,overlapping singleton denseGEMVviews,or in-flash computation in inspected sections. UFSoffload→DRAM→HexagonNPU;physicalNANDmapping is not established.',
        code='Paper describes approximately5000lines on llama.cpp/Hexagon;no authorBigMoMoartifact link verified in bounded paper/search. No code execution.'),
      dict(id='pimcomp',title='PIMCOMP: An End-to-End DNN Compiler for Processing-In-Memory Accelerators',version='arXiv2411.09159v1;TCAD2025,doi10.1109/TCAD.2024.3496847',url='https://arxiv.org/html/2411.09159v1',read_depth='RereadIII-A/B,V-Acomplete,VI-A/Bmethods;read entire official backend/InstructionOptimization.cpp at commit0cfbfa55cca22fcfebdb8dcfc75420dcbbbbdf77. No build/run.',
        evidence='V-A groups crossbars that share identical input under oneMVM pseudo-instruction with parallel outputs. Official OptimizeInstruction invokes MergePreparation/MergeLoadOperation:adjacentLD/LMV/LLDI are merged;smallinputloads are grouped into64-element loads pluslocalmoves.',
        relation='Direct same-input grouping andinstruction/data-load coalescing prior;generic fusion/inputreuse cannot be claimednew.',
        unestablished='The audited optimization file contains no MVMfusion transform. AGs are crossbar programming groups;no ordinaryNANDwave,overlappingflashview orsingle-copy half-wave tailsharing is shown.',
        code='https://github.com/sunxt99/PIMCOMP-NN/blob/0cfbfa55cca22fcfebdb8dcfc75420dcbbbbdf77/backend/InstructionOptimization.cpp'),
      dict(id='tilelens',title='TileLens: Efficiently Using Large-Granularity Memory Systems with Transparent Two-Dimensional Memory Layout',version='arXiv2607.04031v1,4Jul2026;preprint/manuscriptvenueplaceholder',url='https://arxiv.org/html/2607.04031v1',read_depth='Reread2.3–2.6,3.1–3.3complete,4.2/4.3;previouslocalaudit of5retained. No sourcebuild.',
        evidence='Astatic memorytile packs a 2Drectangle into one4KBregion;compute dimensions are multiples ofmemorytile dimensions. Alignment/padding andlogical-to-physical address translation are explicit. Discusses sharedrequests/straggler coupling betweenCTAs.',
        relation='Direct granularity-aware matrixlayout/addresscompiler andreadamplification prior.',
        unestablished='GPU/HBF matmul memoryfetches,not ordinaryNANDdenseGEMV. No cross-expert176row-tail pairing/selected-run fusedreadview shown. Evaluated system includes TMA/L2/MSHR/prefetch additions;cannot import those as existingAiF.',
        code='No authorrunnableartifact verified in paper/boundedsearch;no same-name repository assumed related.'),
      dict(id='recflash',title='RecFlash: Fast Recommendation System on In-Storage Computing with Frequency-Based Data Mapping',version='arXiv2604.25338v1,28Apr2026;extension ofAPCCAS2025',url='https://arxiv.org/html/2604.25338v1',read_depth='ReadII-B,III-A,III-B,III-C1/2;no new implementation/evaluation audit.',
        evidence='III-A/Fig4 remaps multiple requested embeddings into thesamephysicalpage,performs one arrayread andmultipledataouts;III-C addsfrequency remapping,planedistribution andpagecache.',
        relation='Closest inspected rawNAND prior for placement that amortizes onepage sensing operation across multiple selected logicalitems.',
        unestablished='Embedding SLS/indexeddataout,not full-inputdenseGEMV across complete outputrows or variable singleton/coactive overlappingreadviews. Controllercache is separateandmustnotbe assumedhere.',
        code='No authorrunnableartifact verified in inspectedtext.'),
      dict(id='llmflash',title='LLM in a flash: Efficient Large Language Model Inference with Limited Memory',version='arXiv2312.11514v2,4Jan2024;ACL2024',url='https://arxiv.org/html/2312.11514v2',read_depth='Read2.2and3.2complete,3.3pointer/rowmanagement;not newfull-evaluationread.',
        evidence='2.2 explicitly allows largerreads followed bydiscard.3.2 bundles corresponding up/down neuronvectors;closestfriend coactivationbundling was testedbutcaused repeatedloading ofhotneurons andreportednegative.',
        relation='Direct largercontiguousread/overfetch-discard andcoactivationbundling prior,includinganegativeresult.',
        unestablished='Flash→DRAM→CPU/GPUoffload;neither in-flashGEMV nor exactonecopy overlappingreadviews/hardwarewavegeometry is demonstrated.',
        code='No newartifact claim in this audit.'),
      dict(id='aif',title='AiF: Accelerating On-Device LLM Inference Using In-Flash Processing',version='ISCA2025,doi10.1145/3695053.3731073',url='https://doi.org/10.1145/3695053.3731073',read_depth='Localpaper/aif.txt4.4/Fig14surroundingtext,5.2,footnote9reread;notnewRTL/functionaldriverproof.',
        evidence='Nativecommand specifiesinputdimension,blockID,blockoffsetandmatrixpagecount;sequentialpages feedINT8PEs andoutputFIFO;cross-blockGEMV is splitinto commands.',
        relation='Direct existingdensefullinput command capability for acontiguousmulti-expertrowstream,conditionaloncomplete-row/reset/outputsemantics.',
        unestablished='Paperdoesnotexplicitlyimplementselectedexpertcoalescingviews ordisclosevalidrow/fringemasking/scale-outputABI. Callingtheprimitive oncompleterows issoftwarecomposition,not manufactureddeviceverification.',
        code='Localtimingbackendreturnsdummyvectors;notfunctionalPEevidence.')]
    write(D/'sources.json',dict(primary_only=True,sources=sources))
    controls=[
      dict(id='balanced88_same_native',required='Balanced88wave complete-row GU(8chips192rows+8chips160rows),sameDown/sharedgeometry,identicalfullnativeDAGandcosts,inputfanout,fullGUbarrierandcapacitylimits. Give it the same densepacking/selected-run commandcoalescing/outputdiscard capability.',falsifier='Ifthiscontrolmatchesorbeatsuniform176sharing,thereisnogainattributabletohalf-waveoverlap.'),
      dict(id='maxweight_matching_parity',required='Uniform176geometry with train coactivation maximumweight perfectmatching on64routedexperts,notonlygreedypath. Placeeachmatchedpair atanaligned11wavepacket;alsoallowpairorderingforcommandmerge. Test routeblindrandommatching andidentitymatching underidenticalcoalescer.',falsifier='IfMWM plusordinaryselected-runcoalescingmatchesneworder,thereisnonewplacementalgorithmcontribution.'),
      dict(id='bigmomo_native_adaptation',required='Fixedcopy/fixedroutingofflineanchor-region greedy swaps evaluatedonTRAINco-load sets,loweredthrough exactlythesame legal ordinaryGEMVviews andnativecostengine. NoUFS timings,pruning,speculationorbackgroundrelocation imported.',falsifier='Equal-budget anchorregion/adjacencyswap control matching thecandidate limitsnewness tobackendlowering/certification.'),
      dict(id='sharing_vs_command_reuse',required='Separate no-coalescing singletonviews,selected-adjacent coalescing onwave-aligned balancedrows,andhalf-wave overlappingviews;chargeallrequestedpages,auxiliaryoutputs/ECC/inputreload/hostscatterandfullGU->Downbarrier. Staticwholebundle is anadditionalweakcontrol,nottherequiredstrongone.',falsifier='Ifgainisjustcommand/inputamortizationonalreadyalignedrows,do notattributeit tophysicalwave sharing.'),
      dict(id='optimistic_bound',required='Per-chip unionwave orperfectmatching-work oracle may rejectearly,butisaschedulerrelaxation. Finaltestusesoneactualsharedcontroller/channels/chipownerandrow-specificGUreadiness;physicalpackalllayersbeforecapacityclaim.',falsifier='Ifoptimisticunionworkcannotbeatbalanced88,orfullDAG/capacityerasesgain,stopwithoutheldoutexpansion.')]
    write(D/'strong_controls.json',dict(controls=controls))
    notes=dict(status='bounded_audit_complete',conclusion='No inspectedprimarysourceexplicitlyshows theexact176rowhalf-wave,single-copy,overlapping-singleton/coactive ordinarydenseGEMV mechanism. This is not proof of firstness. Co-loading,overfetch/discard,sameinputgrouping andpagepacking arealreadydirectprior;anysystemclaimneedsstrongmatchednativecontrols.',
      hardware_boundary=['C2048divides64KiB:wordlineboundaryalwaysstartsacompleterow. Readviewsusefirstplane0andfull4plane waves;no arbitraryplaneoffset,newsparsemaskorinputcontext is neededalgebraically.','Eachsingletonreadmustcomputeanddrain16auxiliaryrows;both-activefusionreads11waves/emits352rows. HostscatterretainsselectedrowIDs/scales,discardstheauxiliaryrows,andrestoresfullGUbeforeactivation.','This usesonlyGUcommonfullinput. DownprivateinputsandthefullGU/fullvectorA8barrierremain;no crossnonlinear fusion.','Nativeaccumulatorreset/opaquecomplete-rowfunction,outputFIFO/backpressureandnumericoutputABIremainunverifiedinAiF. LackofhostAPI supportalone doesnotprove newhardwareisrequired.','Pairpacketsoverlappingreadviewsreferbacktothesamephysicalcoefficients;do notcountreadviewoverlapasweightreplication. Conversely,per-modelblockcapacitymustbeprovenafterglobalpacking.'],
      scope=dict(new_description_source='Parenttask:DeepC2048uniform176/chip;currentoldscriptisQwenonly.',old_script=str(ROOT/'scripts/research_20261103_gu_coalescing.py'),old_script_sha256=sha(ROOT/'scripts/research_20261103_gu_coalescing.py'),old_script_scope='QwenC2560scalarworkoracle,notDeepfullnativeFFN;readonly.',no_training_archive_read=True,no_heldout_or_inference=True,no_native_experiment_or_newhardware=True),
      cp_branch=dict(status='stopped_no_more_expansion',notes=str(ROOT/'results/research-2026-11-02/round2/compact-coupled-solver/notes.json'),notes_sha256=sha(ROOT/'results/research-2026-11-02/round2/compact-coupled-solver/notes.json'),coverage='G4L0input1actualW8TRAIN512only;188distinctmasks,96768nativecostrecords,567fulltraces. Bestmean0.279302%,P99TRAIN4.00291%;notglobalopt/notheldout.'),
      next_decision='Onlyconsiderexpandednativeexperiment iftheuniform176selected-viewmechanismbeatsbalanced88withmatching/parityandcoalescingcontrolsonthesamefixedcopycapacity. NoalgorithmfirstnessclaimforMWM/DP/hypergraphorverticalconcat.')
    write(D/'notes.json',notes)
    # Preserve directly inspected sources and scope of all written evidence.
    files=[p for p in D.rglob('*') if p.is_file()];write(D/'completion.json',dict(status='complete',audit_source_sha256=sha(__file__),files_sha256={str(p):sha(p) for p in sorted(files)}))
    print(json.dumps(dict(status='complete',notes_sha256=sha(D/'notes.json'),sources_sha256=sha(D/'sources.json'),geometry_cases=len(cases)),indent=2))
if __name__=='__main__':main()
