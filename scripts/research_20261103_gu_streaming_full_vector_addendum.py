#!/usr/bin/env python3
"""Independent full-vector controller gather contract;no new timing runs."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/ordinary-gu-tile-sharing'
OLD=BASE/'streaming-capability-audit';D=BASE/'streaming-full-vector-addendum'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,z):assert not p.exists();p.write_text(json.dumps(z,indent=2)+'\n')
def main():
    old={str(p):sha(p) for p in OLD.rglob('*') if p.is_file()};assert not (D/'addendum.json').exists()
    routed_rows=2816;shared_rows=5632;K=6;bytes_per_output=4
    assert K*routed_rows+shared_rows==22528
    total=(K*routed_rows+shared_rows)*bytes_per_output;assert total==90112==88*1024
    capacity=dict(model='DeepSeekLite,one layer,one selectedtop6+shared GU batch',routed_experts=K,
      routed_GU_rows_per_expert=routed_rows,shared_GU_rows=shared_rows,bytes_per_output=bytes_per_output,
      routed_vector_bytes=routed_rows*4,routed_vector_KiB=11,shared_vector_bytes=shared_rows*4,shared_vector_KiB=22,
      full_batch_valid_output_rows=22528,full_batch_valid_output_payload_bytes=total,full_batch_valid_output_payload_KiB=88,
      meaning='Sufficient payloadcapacity toholdallvalidGUoutputsfromthisonebatchbeforeanyhostDMA,under4-byteelementassumption. Full-vectorDMAs canfreeeachcompletedexpertbuffer. NotapublishedAiFbuffersizeorameasuredliveoccupancy.',
      exclusions=['Ingress/FIFO/DMA burststaging,alignment,metadata/rowmaps/completionstate,inputsandothercontrollerusers areadditional.','Auxiliaryfringerowsmustbedrainedanddiscarded;88KiBdoesnotassertthemaximumtotalFIFOtrafficorstaging.','Multipleconcurrenttokens/layers orsimultaneousDownoutputs requireadditionalreservedspace;notboundedbythissinglebatchledger.','AiFnativeoutputdtype remainsunpublished;4B/F32isthestudycontract,notverifiednativewireformat.'])
    write(D/'capacity_ledger.json',capacity)
    sources=[
      dict(id='aif',url='https://doi.org/10.1145/3695053.3731073',version='ISCA2025',sections='4.1p533;4.4pp536–537/Fig14;5.1p537/Fig15;5.2p538;6.1/Table2p538',
        read_depth='Rereadlocalpaper/aif.txtoverview,fullFIFOparagraph,headparallelism,hostAPIandTable2;searchalltextforcontrollerDRAM/buffer. DidnotreadnewRTL.',
        evidence='4.1controllergatherssubvectorsandreturnsvector;4.4sequentialFIFO/SREGpollingduringcomputation;5.1head-wisehostoverlap;5.2outputaddressperlogicalGEMV;Table2giveschannels/chips/planes/pages/readlatency/bandwidth.',
        absent='No controllerRAMcapacity/freeallocationpublishedininspectedpaper. FIFOasgenerateddoesnotdefineper-waveoutputreadytimesorpausing. Head-wiseoverlapdoesnotidentifyhostrequestgranularity.'),
      dict(id='nvme',url='https://nvmexpress.org/wp-content/uploads/NVM-Express-Base-Specification-Revision-2.1-2024.08.05-Ratified.pdf',version='Base2.1,5Aug2024',
        sections='1.5command-completiondefinition;3.1.3completionordering;3.4.1commandordering;3.4.2FUSEdistinction;3.4.4completion;4.2.1/Figs96–98CQE',
        read_depth='RereadtheseexactsectionsinpreviouslydownloadedofficialPDFtext;notfullspec.',
        evidence='EachcompletedlogicalcommandidentifiedbySQID+CID;completionsmayreorder. RetainingoneCIDperexpertandpostingonefinalCQEafteritsfulloutputtransferusesordinarycompletionsemantics.',
        boundary='NVMedoesnotrequireinternalnativeGEMVcoalescing,anddoesnotcertifyAiFprefixECCorcontrollerbuffers.'),
      dict(id='localabi',url=None,sections='kernel/aif.h:11–57;aif.c:48–198;scripts/aif_replay.c:62–70',
        read_depth='Rereadmetadata-onlypayload/process/completionandsequentialioctlloop;sourcehashesrecorded.',
        evidence='No realoutputvectoraddress/data/gatherbufferorin-flightlogicalexperts. Oneaggregatedtimingtargetperrequest. No localfunctionalimplementationdemonstratesthiscontract.',
        boundary='Missinglocalemulationisnotproofnewnativehardwareisneeded.'),
      dict(id='cosmos',url='https://github.com/Cosmos-OpenSSD/Cosmos-plus-OpenSSD/tree/78601486bb5581e40628ec7e841dea8e97eff034/source/software/GreedyFTL-3.0.0',
        commit='78601486bb5581e40628ec7e841dea8e97eff034',sections='memory_map.h;data_buffer.h;ftl_config.h;nvme/host_lld.c',
        read_depth='Officialrepositorypinnedcommit;readfourfiles,includingDMA/CQEfunctions. Nohardware/sourcebuildrun.',
        evidence='DRAMstaticaddressmapthrough0x3fffffff;data-bufferentries16*USER_DIESwith16384bytes/slice(16MiBwhen8channels*8ways). Existinghost_lldprovidesdevice-addressDMA,donechecks,andset_nvme_cpl(sqId,cid,...). DirectDMAhas<=4KiB/16B-alignmentconstraints.',
        boundary='Concreteexistingcontrollercapacity/DMA/CQEexampleonly. NotAiFcontroller,notfreeavailablememoryproof,notanAiF-compatibleinflashprototype. OnefullvectorlogicaltransfermayusenormalmultipleDMA/PRPdescriptors.')]
    write(D/'sources.json',dict(primary_only=True,sources=sources))
    retained=[
      'OutputFIFOhasdeterministiccomplete-roworder/width/reset/scaleassociation,andcontrollerknowsphysicalrow->logicalexpert/offsetmapping.',
      'AllcoalescedlogicalGUrequestsuseidenticalfullinputbytes/dimension;oneinputcontextremainsresidentforentirefusednativecommand.',
      'Foranexpert,allitsgate+uprowsacrossallchipsmustbereceivedandplacedexactlyonceinitsgatherbuffer;auxiliaryrowsaredropped,notmistakenforvalidoutputs.',
      'Theexpertprefixhasfinalsuccess/ECCvaliditybeforeitssuccessfulCQE. Ifonlywholefusedcommand-endstatusisreliable,waitforit;bufferingalonecannotremove thisblock.',
      'ExistingcontrollerRAMhasareservedavailablebudget>=88KiBvalidpayloadforthisonebatch,plusunquantifiedrequiredingress/metadata/inputresources;internalbufferwrite/readtrafficmustnotbeassumedfree.',
      'FinitechipFIFOserviceislosslessunderchannelandcontrollercontention. Noautomaticpause,backpressure,unboundedFIFOorPEprogressportisassumed.',
      'Logicalrequests/CIDsremainindependent;onefullvectorlogicalDMAthenonefinalCQE,afterdataisvisible. Buffersandinputlifetime/abort/timeoutstateareretainedsafely.',
      'Thefusednativejobstillownseachchip/inputPBuntilitslastoutputdrain/end. EarlylogicalexpertcompletiondoesnotreleasethechiporallownewDowninputthere.',
      'Hostperexpertactivation/fullDownA8vectorbarrierremains. Currentbarrieralreadyisperexpert,notallselectedexperts.'
    ]
    add=dict(status='complete_bounded_readonly_addendum',conclusion='Yes:controllergatherfollowedbyonefull-vectorlogicalDMAandoneoriginalCIDfinalCQEremovesthehostpartial-vectorstreaming/progressrequirement. ItdoesnotrequiremultiplecompletionsofonesingleCID. Thisisamoreconservativeconditionalcontroller-softwarecontract,notverifiedAiFfirmware.',
      removed_requirements=['Hostneednotobserve/usepartialGUvectors.','NoDMAofprefixdatawhilethatsameexpertvectorisstillbeinggathered.','NoextraCQE/progressnotificationforoneCID.','NoDMAaddressswitchbetweenexpertsduringanindividualfull-vectorlogicaltransfer.'],
      DMA_precision='Full-vectorDMAmeansonelogicaltransferafterfullvectoravailability,notonephysicallyatomicPCIetransactionoroneunboundedDMAinstruction. OrdinaryPRP/SGL/descriptorfragmentationremainsnormalandmustfinishbeforeCQE.',
      retained_conditions=retained,capacity=capacity,
      resource_path='ChipFIFO->existingcontrollergatherbuffer->onefullvectorhostDMA->oneCQE->hostscatter/scaleifneeded->fullactivation/A8->Downqueue. NativeGUchipownerremainsactiveindependentlyuntilitsfusedviewends.',
      AiF_capacity_finding='No numericalcontrollerDRAM/gatherallocationinAiF4.1/4.4/5.2/Table2. Do not claim88KiBexistsorisfree. OfficialCosmos+showsordinarycontrollerscanhaveadequateconfiguredDRAMandDMA/CQEprimitives,butdoesnotverifyAiF.',
      paper_precision='AiFexplicitlypollsFIFOasoutputsaregenerated;itdoesnotexplicitlypromiseonewaveoutputbatch,readytime,nativepauseorFIFOcapacity. Asoftwarewave-sizeddrainpolicyneedsactualFIFOavailability,notaninventedper-wavecompletionevent.',
      experimental_scope='No newtimings/native replays/inference/deviceoperations. Atmostafutureexplicitconditionaloptimisticsensitivity,withallfamiliesgivenidenticalcompletionpolicyandcost/capacityassumptions.',
      old_seal_unchanged=all(sha(p)==h for p,h in old.items()),old_files_sha256=old)
    assert add['old_seal_unchanged']
    write(D/'addendum.json',add)
    local=[ROOT/'paper/aif.txt',ROOT/'kernel/aif.h',ROOT/'kernel/aif.c',ROOT/'scripts/aif_replay.c']
    write(D/'completion.json',dict(status='complete',audit_source_sha256=sha(__file__),local_sources_sha256={str(p):sha(p) for p in local},
      old_streaming_directory_unchanged=True,outputs_sha256={str(p):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(dict(status='complete',addendum_sha256=sha(D/'addendum.json'),capacity_bytes=total,old_seal_unchanged=True),indent=2))
if __name__=='__main__':main()
