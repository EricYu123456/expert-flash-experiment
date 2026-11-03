#!/usr/bin/env python3
"""Bounded primary/code comparison and elementary category certificates."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/research-2026-11-02/round2/coupled-closest-prior'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):(D/name).write_text(json.dumps(x,indent=2)+'\n')
def ceil(a,b):return (int(a)+b-1)//b

cert=[]
for model,hg,hd,lowgu,highgu,lowdown,highdown in [
 ('granite4tiny',9,12,1,2,0,1),('deepseek',8,13,5,6,2,3)]:
    intersection=max(0,hg+hd-16)
    gu=np.zeros(16,dtype=int);down=np.zeros(16,dtype=int)
    gu[:hg]=1;down[:intersection]=1;down[hg:]=1
    assert gu.sum()==hg and down.sum()==hd and not np.any((gu==0)&(down==0))
    gw=np.where(gu,highgu,lowgu);dw=np.where(down,highdown,lowdown)
    cats=[{'gu_wave':int(g),'down_wave':int(d),'chips':int(np.count_nonzero((gw==g)&(dw==d)))} for g,d in sorted(set(zip(gw,dw)))]
    cert.append(dict(model=model,gu_high_chips=hg,down_high_chips=hd,minimum_high_intersection=intersection,
        categories=cats,joined_wave_histogram={str(int(v)):int(n) for v,n in zip(*np.unique(gw+dw,return_counts=True))},
        total_joined_waves=int((gw+dw).sum()),claim='Elementary intersection/count certificate,not a new theorem or proof of optimal routed makespan.'))
# A concrete counterexample to universal optimality of per-expert complement.
# This is abstract wave arithmetic,not measured/real shared geometry.
gu=np.zeros(16,dtype=int);down=np.zeros(16,dtype=int)
gu[:9]=1;down[3:15]=1;shared=np.zeros(16,dtype=int);shared[15]=2
joined=1+gu+down
assert gu.sum()==9 and down.sum()==12 and joined[15]==1
assert (joined+shared).max()==3
counterexample=dict(scope='Abstract single-query wave arithmetic only;not the actual shared expert distribution or native timing.',
    GU_high_mask=gu.tolist(),Down_high_mask=down.tolist(),fixed_shared_wave_offsets=shared.tolist(),
    independent_joined_waves=joined.tolist(),independent_max_with_shared=3,
    counterbalanced_minimum_max_with_shared=4,
    proof='Three-category complement forbids both-low,so joined>=2 on everychip. Fixed shared offset2 on chip15 forces max>=4. The provided legal independent masks put both-low there and all other totals<=3. Thus complement is a useful restriction/regularizer,not universally optimal with fixed offsets.')
write('category_certificates.json',dict(status='PASS elementary arithmetic',profiles=cert,shared_counterexample=counterexample,
    independent_superset='Every jointly permuted counterbalanced row assignment is also a legal independent GU/Down row assignment with the same per-phase rowhistograms. Therefore an unrestricted joint-objective optimum cannot be worse. Any restricted-search empirical advantage needs move-neighborhood/initialization/budget attribution.',
    exact_cost_warning='Same joined wave count does not imply same native ownedcost:input/command presence differs for G4 Downzero;Deep GUhigh/low andDownhigh/low have unequal all-page output increments. Use actual cost tables,not wave labels alone.'))

acq=json.loads((D/'source_acquisition.json').read_text())
assert acq['github']['official']['commit']==acq['github']['legacy']['commit']=='56ea3725b2c621d4aea248f16345c34f5e06b8ed'
for f,v in acq['github']['official']['files'].items():assert v['sha256']==acq['github']['legacy']['files'][f]['sha256']
sources=[
 {'id':'HDA_paper','title':'HDA-MoE: Hybrid Parallelism and Dynamic,Adaptive Scheduling for Mixture-of-Experts with3DNear-MemoryProcessing',
  'version':'arXiv2609.08682v1,2026-09-08;TCADonline2026','url':'https://arxiv.org/html/2609.08682v1','doi':'https://doi.org/10.1109/TCAD.2026.3733620',
  'read_scope':'Read full sectionsII,IV andV-A/V-B,includingEq1-12 andAlgorithms1-2;readV-C/V-D overview andVI-C/VI-Fcapacity/dynamic/gating ablations. Did not fully read detailedgating derivation or everyaccuracy/table experiment.',
  'evidence':'One expert fractionP_ic;expected fractional compute load;coactivation group-node communication indicator;capacity constraints;offlineNodeBalance thenphysicalLinkBalance. Shared experts described as deterministic;shared costs grouped withattention in breakdown.',
  'interpretation':'Direct prior for hybridexpert splitting,capacity/topology/load joint optimization. No separate GU/Down rowcount variables,ordinaryNANDwave rounding or the three joinedwave categories in the inspected formulation. One aggregatePcan encode totalexpertfraction but does not generate these two stage-specific row/addressmaps.'},
 {'id':'HDA_official_code','title':'HDA-MoE-TCAD26 official placement artifact','url':'https://github.com/PKU-SEC-Lab/HDA-MoE-TCAD26/tree/56ea3725b2c621d4aea248f16345c34f5e06b8ed',
  'legacy_url':'https://github.com/angerybob/HDA-MoE/tree/56ea3725b2c621d4aea248f16345c34f5e06b8ed','commit':'56ea3725b2c621d4aea248f16345c34f5e06b8ed',
  'read_scope':'Read README;node_allocation.py constructor/coactivation,compute_time/comm_time,communication event-model outline,ilp_solver_gurobi lines538-627,BO mapping857-938,dynamicpriority940-976;simulator.py configuration and placement/evaluation110-290;fullmoe_placement_utils.py. Downloadedbutdidnotfullyreadtopology.py/baseline.py/iter.py. No solver or fullsimulation run.',
  'evidence':'P shape[layer,expert,node],Zbinary;objective t_comp+2*t_comm. Compute usesmarginalfrequency;group frequencies entercommunication. BO/Hungarian permutes logicalnodes. No phaseaxis inP,no weightrowaddresses. Original solver requiresgurobipy/license andskopt.',
  'repository_relation':'Officialandlegacy tips have identicalcommit and all7downloadedfiles match SHA256. No version difference was found in this boundedsnapshot.',
  'caveat':'Papermarks3h*IS compute,codeuses2h*IS initscomputeproxy. Do not reuse eitherFLOPconstant orDRAMnetwork timings as NANDcalibration. This audit doesnotdiagnose paperperformance fromthat normalization difference.'},
 {'id':'Stratum','title':'Stratum:System-HardwareCo-Design withTieredMonolithic3D-StackableDRAM forEfficientMoEServing','version':'arXiv2510.05245v1,2025-10-06;MICRO2025',
  'url':'https://arxiv.org/html/2510.05245v1','doi':'https://doi.org/10.1145/3725843.3756043',
  'read_scope':'Read sections3.1-3.2,full4.1 ExpertProcessing/Fig8-9,4.3,5.2/Algorithm1 and6.1modelingoverview. Otherattention/prediction/evaluationdetailsnotfullyreread.',
  'evidence':'PU-levelGU output-neuron partitions align withDowninput partitions toavoid intermediatecommunication;Downpartialoutputsreduce-scatter. AllPUsprocessoneexpertatime;SFUsperformactivation/weightedaggregation. Placementrankswholeexperts intoDRAMtiers,includinghotsharedexperts.',
  'interpretation':'Explicitdirectprior forcross-projection ownership andcausaloperator mapping. Differentmechanism fromtwo independentlycomplete output-row projections with intentionallycomplementary wavehighs. ProposedMono3DDRAM/NMPmodel,not ordinaryNAND hardware evidence. No authorartifact wasverifiedinthis boundedpass.'},
 {'id':'AiF_native_scope','title':'AiF:AcceleratingOn-DeviceLLMInferenceUsingIn-FlashProcessing','year':2025,'venue':'ISCA2025',
  'url':'https://doi.org/10.1145/3695053.3731073','local_path':str(ROOT/'paper/aif.txt'),'sha256':sha(ROOT/'paper/aif.txt'),
  'read_scope':'Section4.4/Fig14 and5.2/nativevirtualdevice discussion readinprevious immediatelyprecedingaudit;not afreshfullpaperread.',
  'evidence':'DenseGEMV inputdimension/block/offset/pagecount,continuousrowstream,INT8multipliers/addertree,orderedFIFO;equalaif_postplacement. Emulator suppliesdummyoutputs.',
  'interpretation':'Publishedbasis for ordinarycomplete-row model;not observedthreecategoryjointMoEmapping. Outputdtype/scales/reset,FIFO/backpressure,inputfanout andPElatency remainunverifiednativecontracts.'}
]
write('sources.json',sources)

matrix=[
 {'system':'Currentcounterbalancedcandidate','complete_rows':'BothGUandDown outputrows;no columnpartials','joint_phase_ownership':'Explicit separate rowcounts,pairedmoves,threewavecategories','objective':'TRAIN E_query[max_chip summedGU+Downownedwork+fixedshared];nativeGFcost selectsinitial/final','granularity':'16KiBpages,4planes,192WLsingleblock lowerer','shared':'Samepool andsameorientationwithin shared_name/seedstratum','numeric_address_evidence':'Softwarelowerer/nativecost/virtualaddresses;thisauditnotanewnumericreplay;AiFABIstillunverified','read_depth':'Full rootdesignscript plus36existing sharedstrata checks'},
 {'system':'HDApaperandpinnedcode','complete_rows':'Continuousaggregateexpertfraction;no emittedperprojectionrows/addresscode inspected','joint_phase_ownership':'Aggregatewholeexpertcost/P;canencodeaggregatedfractionorbeextended,butnocurrentseparateGU/Downrowvariables','objective':'max_node expectedfractionalcompute pluscoactivationnodecommunicationproxy;topologyawarecommunicationreplay','granularity':'DRAMNMPclusterfractionsandNoCendpoints;noNANDceilwaves/oneblockrule','shared':'Deterministicsharedexpert concept;notoursame2projectionsharedrowpool','numeric_address_evidence':'Placement/timingandqualityroutingcode;noordinaryNANDnumeric/addresscertificate','read_depth':'PaperIV/V-Bandofficialplacementcode specifiedinsources'},
 {'system':'Stratum','complete_rows':'GUoutputpartition;DowninputpartitionandreductionatPUlevel;intraPUfurthermatrixsplit','joint_phase_ownership':'Explicitmatchedintermediate-neuronownershipavoidsGU-toDownmovement','objective':'Locality,TPutilization,communicationoverlap,hot/coldtierplacement','granularity':'DRAMbank/PU/PEandrowaddressinterval;noNAND4planeordinarywaves','shared':'Sharedexpertscountashotfortierplacement','numeric_address_evidence':'Hardwareproposal/circuitandprocessor/systemsmodeling;notNANDdeviceABI','read_depth':'Full4.1and5.2,microarchitecture/modelingoverview'}
]
write('capability_matrix.json',matrix)

checks=json.loads((D/'same_shared_existing_result_checks.json').read_text())
design_sha=sha(D/'design.snapshot.py')
notes={
 'status':'Bounded closest-prior/code audit complete;no claimofpriorityorbreakthrough',
 'usable_conclusion':'HDA-MoE covers fractionalexpertdeployment andload/communication/capacity optimization;Stratum explicitlyjointlymapsGUandDownintermediateownership. Neither inspectedmethod/code demonstrates the specificordinaryNANDcomplete-output-row,integer-wavecounterbalance. This is a meaningfulspecificdistinction,not evidence thatbasiccomplement/multi-stageplacement isnew.',
 'design_sha256':design_sha,
 'design_read':'Entire scripts/research_20261103_shared_counterbalanced.py frozenasdesign.snapshot.py,includingcanonical,coupled_random,projected,independent_initial,cyclic_pool,Fitandrun_case/artifact.',
 'mechanism':{
   'G4':'9GUhighand12Downhighon16chipsrequireatleast5bothhigh. Minimumintersection gives5(2,1),4(2,0),7(1,1):joined3waveson5chips,2on11;total37.',
   'Deep':'8GUhighand13Downhighrequireatleast5bothhigh. Minimumintersection gives5(6,3),3(6,2),8(5,3):joined9waveson5chips,8on11;total133.',
   'cost':'cost() ischipownedoccupancyproxy:ordinarywaves28us pluscommand/fullinputONFI/allpageoutputONFI. It isnot nativeDAGlatency;artifactreplays initialandfinal withwholeGUhostbarrierandfullDowninputs. Samewavecategorydoesnotalwaysmeansametime.',
   'what_is_standard':'Pigeonholeintersection andcomplementinghighloadsareelementary. Localannealing/coloringswapsarestandard. Theexperimentcouldestablish thatthisrestriction isuseful undernativeintegergranularity,not inventcomplementtheory.',
   'search_space':'counterbalanced isasubsetofindependentGU/Downpermutations. Counterbalanced local/globalmovesactonbothphases;currentindependentmovesactonephaseatatime. Henceafinitesearchwinmaybeneighborhood/initializationregularization,notanindependent-familyoptimumgap.'},
 'same_shared_checks':{
   'existing_result_files':12,'same_shared_strata':checks['shared_strata_checked'],'identical_seed0_initial_rows':checks['same_initial_seed0_row_arrays_checked'],
   'result':'Allfamilies withinshared_geometry/seed_index haveidenticalsharedrows. G4usesonesharedgeometry;Deepallows sourceandbalanced89. Finalwinnercandifferinsharedchoice;forcausalattributionalsoreportsameshared-stratumcomparisons.',
   'observed_same_stratum_gain_range_counterbalanced_vs_independent':[min(x['counterbalanced_gain_over_independent_same_shared_seed'] for x in checks['records']),max(x['counterbalanced_gain_over_independent_same_shared_seed'] for x in checks['records'])],
   'qualification':'Readandcheckedexistingsavedresults only;didnotindependentlyreplaythese newcosts. Noheldoutusedornewroutinginference.'},
 'precise_prior_distinctions':[
   'Expressibility:HDAoneP_iccanrepresentatotalexpertfractionandgeneraloptimizationcouldintroducecompoundstageitems. Thisdoesnotmeanitscurrentpaperorcodegeneratesthe samepairrowcounts. Conversely,themissingphaseaxisdoesnotproveitsarchitecturecannotexecutetheplacement.',
   'Explicitmechanism:HDAminimizesmaxexpectednodecompute(marginalf_i);coactivationf_gappearsinthecommunicationterm. Ourproxyisexpectedquery-wise maximumofstage-summedintegercosts. max(Eload)<=E(maxload);theobjectivesdifferatbatch/querygranularity,notbecauseHDAignoresallcoactivation.',
   'Explicitmechanism:StratumcoordinatesGUoutputandDowninputneurons forlocalityandreducesDownpartialoutputs. OursretainscompleteDownoutputrowsandhostfull-vectorbarrier;itscomplementlink isloadplacement,notretainingintermediateneuronsonPU.',
   'Nativeevidence:neitherHDA/StratumisordinaryNAND;HDAFLOPs/NoCandStratumSFU/reduce-scatter/tierrow-swapcapabilitiesmustnotbeaddedtoAiF. Allthree systemsexperimentclaimsneedtheir ownbackendcontract.',
   'No inspectedprimarysourceproves thesameonecopyordinaryNANDthreewavecategoryscheme. Thisboundedabsenceisnotaglobalsearchorfirstclaim. Genericrepresentabilityaloneisnotusedasarejection.'
 ],
 'strongest_executable_controls':[
   {'priority':1,'name':'Unrestrictedjoinedsearchwithpairedmovesandcounterbalancedincumbent',
    'recipe':'Keepbothphasehistograms,fullrowshapes,input/outputcontract,sharedrowgeometry/orientationandnativeenginefixed. Seedwiththebestcounterbalancedplan;allowbothsingle-phaseandjointtwo-phaseexpert/global swaps. Tracknativecostofincumbentsorretaininitial/finalnativecost. Mixtureofneighborhoodswithmatchedproposal/walltimebudgetmustincludethecandidateplan,soitcannotbeclaimedthatrestrictedfamilyisbetterthanunrestrictedoptimum.',
    'interpretation':'Ifitmatches,thecandidatehasprovidedausefulinitialization/restriction/compilepolicy,notanewfeasiblelayoutunavailabletofreeplacement. Ifitsbudgetconvergenceisbetter,reportthatwithmultipleseedsratherthanstrictdominance.'},
   {'priority':2,'name':'Samegeometryexpectedloadversusexpectedmaxandphasewiseversusjoined',
    'recipe':'Onthesamerowtemplatesandsharedoffsets,fittheHDA-stylemaxmeanownedloadproxy;separatelyfitphasewiseexpectedmax;comparewithjoinedexpectedmaxusingidenticalmoveclasses/initializers/budgets. Native-replayalloutputs. Do notuseunmodifiedDRAMFLOPcostorpoorad-hocfractionroundingasthestrongbaseline.',
    'interpretation':'Thisdirectlytestsclaimedadvantagefromroute-tail/stagecouplinginstead ofweakbaselinegeometry. Labelita geometrymatchedadaptationofpriorobjectives,notunchangedHDAreproduction.'},
   {'priority':3,'name':'Geometrymatchedintegerjointoptimizationreference',
    'recipe':'BinaryGUhighandDownhighvariableswithperexpertcounts(9,12)/(8,13),optionalcomplementconstraint,andqueryepigraphvariables t_q>=shared_c+sum_einq exactownedcost_e,c. Minimizeweighted sumt_qwithsamechipcapacitycaps. Exactceil/input/outputcosttablesarefixedbyrowtemplate. A boundedMILP/CPsolveorreportedgapcanbenchmarklocalheuristics;retainactualnativeDAGreplaybecauseepigraphobjectiveislowerboundproxy.',
    'interpretation':'Basicintegeroptimizationisstandard;itprovidesatrustworthysmallreferenceandseparatesconstraintstrengthfromsearchfailure. Notrequiredtolicense/runGurobiinthisliteratureaudit.'},
   {'priority':4,'name':'Sharedandcostmatchedablations',
    'recipe':'Reporteachsharedsource/balanced89andorientationstratumseparately,plusidenticalbest-of-poolaggregationforallfamilies. Retainstrongcyclicgeometryincludingtheactualfrozenpair,samecapacity/globalpacking,command/inputcopiesallpagefringe. Avoidcreditingasharedgeometrychangeoradditionalmultistartbudgettotheroutedcounterbalance.'},
   {'priority':5,'name':'Resource/DAGandnumericcontrols',
    'recipe':'SamefullGUhostbarrier/privateDowninputandpriorityforallexecutions;reportworkLBversusnativeGFgapandwavedrain/inputfanoutsensitivity. PreserveW8logicalrow/scales/addresscertificates. A separatedependency-aware/PIMCOMP-stylegenericmapperisusefulonlywhenloweredtothesamelegalnativejobsandservicecontract.'}
 ],
 'scope_boundaries':[
   'No largeexperiment,inference,solversearch,authorcontactorsubagents. Onlyprimaryretrieval,sourcereading,existing-resultconsistencychecksandasmallanalyticalcounterexample.',
   'Theknownroot606/606simplecyclic+balancedwavecoverage resultmeansDP/B&Bhasnoobservedextraqualitycontributioninthosecampaigns. ThisnewjointplacementtestmustnotreintroducefirstDP/exactgeometrynoveltyclaims.',
   'First3TRAINlayerscannotestablishfullmodeloutofsamplegain,samecapacityimprovementorhardwarecalibration. Thecentralremainingevidenceiswhetherjointstageplacementhasastablebenefitaftertheunrestrictedpaired-movecontrolandallnativecontracts/costs.'
 ],
 'files':{'primary_sources':'sources.json','pinned_acquisition':'source_acquisition.json','capabilities':'capability_matrix.json','category_proof_and_counterexample':'category_certificates.json','same_shared_result_checks':'same_shared_existing_result_checks.json','frozen_design':'design.snapshot.py'}}
write('notes.json',notes)
lines=[notes['usable_conclusion'],'',
 '最強可執行control：同rowhistogram、同sharedrows、同native成本，讓unrestricted GU/Down mapper以counterbalanced incumbent啟動，並同時允許單phase與雙phasepaired swaps。現在兩family的move neighborhood不同，有限搜尋差距不能當free-space最優差距。','',
 'G4三類為5×(2,1)、4×(2,0)、7×(1,1)；Deep為5×(6,3)、3×(6,2)、8×(5,3)。這是最小high-set交集，非新定理。已附固定shared-offset下complement非普遍最優的解析反例，明標不是實際硬體量測。','',
 '已核對36個相同shared strata、18組seed0相同initial rows；保存結果的counterbalanced相對independent改善範圍0.1123%–1.2207%，未在此重播新costs。','',
 'HDA新舊repo皆commit56ea3725b2c621d4aea248f16345c34f5e06b8ed，7個抽取檔案逐hash相同。[HDA原文](https://arxiv.org/html/2609.08682v1)與[Stratum原文](https://arxiv.org/html/2510.05245v1)的精確已讀範圍及code位置見sources.json。沒有把DRAM能力當AiF能力。']
(D/'notes.md').write_text('\n'.join(lines)+'\n')
write('completion.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),report_source_sha256=sha(Path(__file__)),
    files_sha256={str(p):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='completion.json'}))
print(json.dumps(dict(status='complete',notes_sha256=sha(D/'notes.json'),sources_sha256=sha(D/'sources.json'),category_sha256=sha(D/'category_certificates.json')),indent=2))

if __name__=='__main__':pass
