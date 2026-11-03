#!/usr/bin/env python3
"""Primary topology/prior evidence ledger; no execution of downloaded code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    now=datetime.now(timezone.utc).isoformat()
    write('web_primary_reading.json',dict(accessed_date_utc='2026-11-02',
      reason='Some publisher/vendor bytes could not be downloaded by requests. The official pages below were actually read via the web tool; this file hashes reading observations,not inaccessible original source bytes.',
      readings=[
        dict(url='https://ir.siliconmotion.com/news-releases/news-release-details/silicon-motion-launches-pcie-40-nvme-14-controller-solutions',depth='Official full press release,2020-10-20; product table lines126–130 and description120–124.',observation='SM2264 lists8NAND channels and8CE/channel;8channels up to1600MT/s. Release described sampling then. This establishes advertised controller connectivity,not AiF capability or allways populated hardware.'),
        dict(url='https://www.siliconmotion.com/download/3bn/a/SM2264_PB_EN.pdf',depth='Official product brief specifications obtained through web-search indexed text;full PDF download failed/timeout.',observation='2022brief specifies8NANDchannels and8CH/64CE. Do not call this a locally downloaded datasheet.'),
        dict(url='https://onfi.org/files/ONFI_6_0_Final.pdf',depth='Official indexed definitions on p3,§1.3.1.15,18,20. Direct open rejected;fullspec not read.',observation='HostTarget groups targets sharing hostCE when pinreduction is used;NANDTarget groupsLUNs sharingCE insidepackage;LUN is independently command/status addressable. These definitions say nothing about additionalPE/inputcontext independence.'),
        dict(url='https://www.arasan.com/products/nand-flash/onfi2-3/',depth='Official full short product page,features32–48.',observation='ControllerIP listsread/cacheinterleaving,multiLUN/dieoperations,LUNreset,CEpinreduction andvolumeaddressing. Evidence of storagecontrol mechanisms,not an AiFprocessor or measured hardware.'),
        dict(url='https://arxiv.org/abs/2603.28768',depth='Official currentmetadata/submissionhistory.',observation='v1dated2026-01-12,v2dated2026-05-29;MLSys2026. v2title differs fromv1. Dates are recorded as displayed,not inferred from numeric arXivID.'),
        dict(url='https://arxiv.org/abs/2512.03608',depth='Official currentmetadata/submissionhistory.',observation='Onlyv1,2025-12-03,listed at inspection.')]))

    sources=[
      dict(id='AiF',url='https://doi.org/10.1145/3695053.3731073',year=2025,venue='ISCA2025',
           actually_read='Local paper/aif.txt §4.4,§6.1/Table2,and§4.4Table1 area/power reread;not a new full paper reading.',
           evidence='Evaluated8channels×2chips/channel,4planes/chip,16KiBpage,2.0GB/s perchannel. Architecture adds per-chipECCLite/PEs/inputPB use;Table1 reports perchiplogic.',
           inference='8/16/32/64independent AiFcompute owners on8fixedchannels can be labeled1/2/4/8ways in a scale model,provided eachowner retains its own full AiFchip resources. CommodityNAND topology doesnot establish fabricatedAiFchips.'),
      dict(id='SMI',url='https://www.siliconmotion.com/download/3bn/a/SM2264_PB_EN.pdf',alternate_url='https://ir.siliconmotion.com/news-releases/news-release-details/silicon-motion-launches-pcie-40-nvme-14-controller-solutions',year='2022brief;2020release',
           actually_read='Officialrelease fullpage and productbrief indexedspecification. Seeweb_primary_reading.json;requestsDNSfailure explicitly retained.',
           evidence='8NANDchannels and8CE/channel,64CEtotal advertised.',
           inference='An8channelcontroller with4or8addressable targets/channel is a supported existing storagetopology. CEcount is not packagecount,diecount,or computecontextcount.'),
      dict(id='Cosmos',url='https://github.com/Cosmos-OpenSSD/Cosmos-plus-OpenSSD/tree/78601486bb5581e40628ec7e841dea8e97eff034',commit='78601486bb5581e40628ec7e841dea8e97eff034',
           actually_read='ftl_config.h macros,driver read-trigger/read-transfer/status functions,FMCTop.v parameter andreadybusy ports;recursive tree includesPrebuild/8Ch8Way-3.0.0. No firmware/FPGA build or device execution.',
           evidence='NSC_MAX_WAYS8,USER_WAYS8,up to8mappedchannels;driver selectsway percommand and separates asynchronousarray-readtrigger from transfer. RTLNumberOfWays8 andperwayReadyBusy. FirmwareUSER_DIES=channels×ways butalsoLUNS_PER_DIE2.',
           inference='Official publiccontroller artifact supports8-way topology/interleaving. Its DIE/way/LUN naming is not a physicaldie identity proof:do not multiplyUSER_DIES byLUNS_PER_DIE to claim free independentcomputeunits.'),
      dict(id='ONFI',url='https://onfi.org/files/ONFI_6_0_Final.pdf',
           actually_read='Onlyofficialindexeddefinitionsexcerpts,p3§1.3.1.15/18/20;fullspec accesses rejected.',
           evidence='Target/HostTarget/LUN hierarchy distinct,includingCEpinreduction.',
           inference='A packagedchip maycontainmultipleCEtargets;atargetmaycontainmultipleLUNs;physicaldie-to-LUN mapping is part-specific. Independentstandardcommand/status doesnot proveindependentIFPinput,PE,accumulator or FIFO.'),
      dict(id='CRAFT-paper',url='https://arxiv.org/html/2603.28768v2',version='v2,2026-05-29;MLSys2026',title='CRAFT: Fine-Grained Cost-Aware Expert Replication For Efficient Mixture-of-Experts Serving',
           actually_read='v1§4/AppendixAfirstread,thenlatestv2§4.1–4.3,§5.1/5.4/5.5excerpts,AppendixAalgorithmsandEartifact. Notallplots/evaluationread.',
           evidence='Replayperlayerreplicationbenefit;MCKP/DPallocatesglobalreplicabudget;capacity-awareinterleavingbalancesreplicasacrossdevices;workloadshiftperiodicrebalancing.',
           inference='Directprioragainstfirstcost-awarebudgetedreplication/perlayerglobalcapacityallocation/scalingclaims. AppropriateadaptedcontrolfitscopybudgetsacrosslayerswithcorrectNANDcosts;notwhole-rowwave/addressorprefixABIprior.'),
      dict(id='CRAFT-code',url='https://github.com/Accelsnow/CRAFT_core/tree/968ecd9882d020e7c36471ff51368a9862aab2c0',commit='968ecd9882d020e7c36471ff51368a9862aab2c0',archive='https://doi.org/10.5281/zenodo.19022696',
           actually_read='README/environment;craft_core.py functionscreate_balanced_tensor_interleaved,run_bal_trials,compute_replicate_benefits,round_final_alloc,craft_expert_replica_allocation; eplb_craft.py packing/replication interfaces. No execution or trace download.',
           evidence='CPU-onlycorepublicartifact;providedtraces areexpertload tensors. CurrentmainallocationCLI usesreplicamenu[0,4,8,16,32],benefitthresholdsandrounding;noexactbudgetedDPentryfoundinthiscorefunction. Replaydividesloadbycopycountusingfloor//.',
           inference='Canreuseplacement/profilecontrolideas,butcannotcallthisCLIanexactMCKPimplementation. Aone-tokenload1withcopies>1wouldfloor0:anativeadaptermustconservediscreteworkandchooselegalreplicas,notreusefractional/floorbalancednessaslatency. Thisdoesnotre-evaluatepaperGPUbatchresults.'),
      dict(id='BALANCE',url='https://www.aspdac.com/aspdac2026/archive/pdf/4B-2.pdf',doi='10.1109/ASP-DAC66049.2026.11420589',year=2026,venue='ASP-DAC2026',
           actually_read='Official19-slidepresentationtext,especiallypp6–12methodsandp16ECCthroughput. Published7-pagepaperfulltextnotobtained;diagramsnotfullyreverseengineered. Slidefooter2026-01-15isdeckdate,notassertedconferencedate.',
           evidence='Bits/layersensitivityranksSEUs;greedyECCassignment;slide12distributesSEUsacrossdies/planesandmapsimportantSEUstoreliablelowerpages. Slide16 6.4Gb/sisECCthroughput,notvalidatedGEMVMAClaneperformance.',
           inference='Importance-awareflashpageplacementplusreliabilityprotectionisexplicitprior. Proposinggenericcriticalweights-on-reliablepagescannotbeclaimednew. Inspectedslidesdonotestablishthecomplete-rowintegergeometrymechanism;unknownpublishedbodydetailsarenotexcluded.'),
      dict(id='KVNAND',url='https://arxiv.org/html/2512.03608v1',version='v1,2025-12-03;venue not verified',
           actually_read='§III-D,full§IV-A–D,§V-A/TableIandV-Bscalability/DSE;notallenergy/reliabilityexperiments. Noverifiedauthorartifactlinkfoundininspectedpaper.',
           evidence='Compute-enabledIFCdies,explicitweight/KVdiegroupDSEandcompactco-location,headgrouppipelining,pagelevelhead/layerKVmapping,access-awareblockallocationandrefresh. DSEcompares8/16dieconfigurations;32planes/4KiBpage/SLC/4usread/16FMACperplaneareitsproposedmodel.',
           inference='Diegroupallocation,headparallelism,page-localityandaccess-awarewearplacementareprior. Itisaproposed/synthesizedlogic+systemsimulationarchitecture,notcommercialordinaryNANDnorAiFtiming. Do nottransferitsplaneinput/computeindependenceor4usreadtoourmodel;itsnewlogic/buffersareexplicitresources.')]
    write('sources.json',dict(created_utc=now,sources=sources))
    write('notes.json',dict(created_utc=now,status='Primaryevidence supports storage topology sensitivity;compute scaling remains explicitly modeled',
      main_conclusion='8channels×4or8CE/ways hasprimarycontrollerandOpenSSDevidence. Testing8/16/32/64AiF-likecomputeowners on8channels is a defensibletopology sensitivity,notproofthatoff-the-shelfNANDalreadyexecutesAiF. EveryadditionalindependentownerrequiresitsownspecifiedPE/input/accumulator/FIFOresourcebundle.',
      terminology=dict(channel='Sharedcontroller↔NANDdata/commandbus;additionalwaysdonotmultiplyitsfixedI/Obandwidth.',way='Controller-specificselectableendpointindex;mapexplicitlytoCEtarget/LUNandcomputeowner.',target='OneormoreLUNsbehindNANDCEwithinpackage;CEpinreductionmaymultiplexseveraltargetsbehindahostCE.',LUN='Independentlystandard-command/statusaddressableunit;notautomaticallyindependentIFPcontext.',die='Physicalsiliconunit;checkdeviceorganizationratherthanassumingonepackage=onedie=oneCE=oneLUN.',plane='SubdivisioninsideLUN/die;ordinarymultiplanereadrestrictionsandsharedperipheryremainarchitecture-specific.'),
      regime_matrix=[dict(compute_owners=n,channels=8,owners_per_channel=n//8,channel_assignment='owner%8',way_assignment='owner//8',status='PublishedAiFevaluation' if n==16 else 'Scale-modelsensitivityusingrepeatedsameper-ownerarchitecture',compute_resource_cost='Ntimespublishedper-ownerPE/ECC/inputstorage;notfreefromCEcount') for n in (8,16,32,64)],
      keep_fixed_in_each_regime=[
        'SameNandtopologyforallplacement/controlfamilies;regeneratelegalrowprofiles/masksforN,notweakg16versusotherwidthlabels.',
        '8sharedchannelswithsameperchannelbandwidth,PCIelimit,controllerdispatchassumption,inputfanout1/4andordinaryreadlatency. Extraownerscanoverlapinternalworkbutstillserializebusinput/output.',
        'Onefulloriginalinputcontextpermodeledowner;do notpromoteplanes/LUNsbehindsharedPEtoindependentcontextswithoutarchitectureevidence.',
        'CapacityperownerheldfixedinprimaryscalesweepmeanslargerNalsogiveslargertotalcapacity/logic. ReportthisandholdphysicalbudgetequalwithineachN. Anisocapacityvariantwouldbeaseparateexperiment.',
        'ReportNANDwork/depth/commands,input/outputtrafficandperchannelloadbounds. Controllercommand250ns/globalserverisnotcalibratedbycontrollerCEspecification;retaincostsensitivityandavoiduniversalhardwarewinclaims.',
        'NoPElanelearningfromECCGb/s. Perdiecomputeprovisioningmustbeassumedexplicitlyandcharged;commoditycontrollerconnectivitydoesnotvalidateoutputscale/resetABI.'
      ],
      novelty_boundaries=dict(CRAFT='NofirstperlayercopybudgetDP/capacity-awareplacement/scalingclaim. Nativeprefix/waveaddressproblemisnarrower;useadaptedglobalbudgetcontrolswithdiscreteworkconservation.',BALANCE='Nofirstimportance/reliabilitypageplacementclaim. ECC/bitencodingchangesareoutsidetheuserexistinghardware-onlyscope.',KVNAND='NofirstdiegroupDSE/page-locality/headpipeline/wear-awareallocationclaim. Differentproposedbackendcanbeaseparatecapabilitymodel,butcannotborrowitsadditionalcompute/buffersasAiFfacts.'),
      source_access_limitations='SMIprimarywebpagewasreadbutrequestsDNSfailed;briefwasindexedreadonly. ONFIfullPDFwasblockedandonlyofficialindexeddefinitionsexcerptsused. BALANCEofficialslidesread,publishedbodyunread. No data/device operations,hardwarechanges,codeexecution,modelinference or new datasets.',
      source_hashes={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'primary').iterdir()) if p.is_file()},
      AiF_local_source_sha256=sha(ROOT/'paper/aif.txt')))
    write('completion.json',dict(created_utc=now,files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(dict(notes_sha256=sha(HERE/'notes.json'),sources_sha256=sha(HERE/'sources.json')),indent=2))


if __name__=='__main__':main()
