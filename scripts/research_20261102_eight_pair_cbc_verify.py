#!/usr/bin/env python3
"""Independent universal Hall, address, row and native-contract certificates."""
from collections import Counter,defaultdict
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr'
BUILDER=ROOT/'scripts/research_20261102_eight_pair_cbc.py'
REF=ROOT/'scripts/research_20261102_cr_ipr_pipeline_audit.py'
OUT=BASE/'eight-pair-cbc-degree4'
PAGE=16384


def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def independent_matching(supports,route):
    # Different finite-state reference: reachable sets of occupied groups.
    states={0:[]}
    for e in route:
        nxt={}
        for occupied,assignment in states.items():
            for g in supports[e]:
                if not occupied&(1<<g):nxt.setdefault(occupied|(1<<g),assignment+[g])
        states=nxt
    return next(iter(states.values())) if states else None


def selected_audit(result,layout):
    need(result['feasible'] and result['capacity']==1,'dispatchfeasibility')
    ids=result['route'];groups=result['assignment'];halves=result['shared_half_assignment']
    need(set(groups)==set(ids) and len(set(groups.values()))==6,'sixdistinctexpertgroups')
    need(set(groups.values()).isdisjoint(halves.values()) and len(set(halves.values()))==2,'sharedhalfunusedgroups')
    need(set(groups.values())|set(halves.values())==set(range(8)),'eightgroupcoverage')
    jobs=result['selected_jobs'];need(len(jobs)==16 and {p['chip'] for p in jobs}==set(range(16)),'onenativejobperphysicalchip')
    row_intervals=defaultdict(list)
    for p in jobs:
        e=p['expert'];g=p['resident_copy'];local=p['chip']%2;half=p['shared_half']
        need(p['copy']==0 and p['chip']//2==g and p['pages']==24 and p['reserved_WL']==6,'executionviewbinding')
        if e<64:
            need(e in ids and groups[e]==g and g in layout['supports'][e] and half is None,'private expert/groupbinding')
            owner=f'e{e}';rows=768;C=512;start=local*768
        else:
            need(half in [0,1] and halves[half]==g,'sharedhalf/groupbinding')
            owner=f'S{half}';rows=384;C=1024;start=half*768+local*384
        entry=next(j for j in layout['groups'][g]['jobs'] if j['owner']==owner)
        need(p['resident_job_id']==f'{owner}.g{g}.c{p["chip"]}' and p['block']==entry['block'] and p['wl_start']==entry['WL_start'],'selectedphysicalresidentbinding')
        need(p['rows']==rows and p['columns']==C and p['row_start']==start,'fullinput/row/scalepositionbinding')
        row_intervals[e].append((start,start+rows))
    for e in ids+[64]:
        cursor=0
        for start,end in sorted(row_intervals[e]):need(start==cursor,'selectedrowgap/overlap');cursor=end
        need(cursor==1536,'full1536outputrowsexpert')
    need(sum(p['rows'] for p in jobs)==10752,'unchangedfullDownrowcount')


def main():
    began=time.monotonic();builder=load('sealed_joint_cbc_builder',BUILDER)
    frozen=read(OUT/'protocol.frozen.json')
    for p,h in frozen['inputs_sha256'].items():need(sha(p)==h,'sealedbuilderinputchanged')
    paths=[Path(__file__).resolve(),BUILDER,REF,OUT/'protocol.frozen.json',OUT/'layout.frozen.json',OUT/'resident_jobs.frozen.json']
    hashes={str(p):sha(p) for p in paths}
    write(OUT/'verification.protocol.frozen.json',dict(status='frozen_before_independent_verification',inputs_sha256=hashes,
        seed=2026110284,random_route_tests=1000,scope='All256server-subset Hall certificate, actualphysicaladdresses/fullrowextents and deterministicruntime matching tests. NoTRAIN/heldoutfiles or fullFFN execution.',
        chronology=frozen['chronology']))
    layout=read(OUT/'layout.frozen.json');supports=layout['supports']
    removed={frozenset(s) for s in [(0,1,2,3),(4,5,6,7),(0,2,4,6),(1,3,5,7),(0,1,4,5),(2,3,6,7)]}
    independent=sorted(tuple(g for g in range(8) if mask&(1<<g)) for mask in range(256) if mask.bit_count()==4 and frozenset(g for g in range(8) if mask&(1<<g)) not in removed)
    need(supports==[list(x) for x in independent] and len(set(map(tuple,supports)))==64,'exactbalanceddegree4incidence')
    need(Counter(g for s in supports for g in s)==Counter({g:32 for g in range(8)}),'pergroup32expertcopies')
    masks=[sum(1<<g for g in s) for s in supports];hall=[]
    for server_set in range(256):
        contained=sum(not(mask&~server_set) for mask in masks)
        # A valid request contains exactly6expert jobs andtwofullyflexible
        # sharedhalves. Proper server subsets contain neither sharedhalf.
        demand=min(6,contained)+(2 if server_set==255 else 0)
        need(demand<=server_set.bit_count(),'universalcapacitatedHallcounterexample')
        hall.append(dict(server_mask=server_set,server_count=server_set.bit_count(),resident_experts_contained=contained,
                         max_valid_request_jobs_contained=demand))
    write(OUT/'universal_Hall_certificate.json',dict(status='PASS',server_subsets=hall,unordered_top6_covered=math.comb(64,6),
        proof='Everyvalidrequest is6distinctexperts plusS0/S1. For anyserver subsetS, at mostmin(6,containedexperts) requestexperts have allneighbors inS; S0/S1 onlycontribute ifS=all8. All256subsetdemands<=|S| certifyHall for everyvalidrequest. Degree4 uniqueness implies inside4set<=1 andinside5set<=C(5,4)=5. StandardCBC fact, not a newtheorem.'))
    own=set();ledger=[];job_records=[];source_rows=Counter();native=[]
    def add(chip,block,plane,wl,kind,e,half,g,lp,source,C):
        key=(chip,block,plane,wl);need(key not in own and 0<=chip<16 and 0<=block<2 and 0<=plane<4 and 0<=wl<192,'physicalcollision/capacity')
        own.add(key);ledger.append([chip,block,plane,wl,kind,e,half,g,lp,source,C])
    for chip in range(16):
        for e in range(64):
            for lp,plane in enumerate([0,1,2,0,1,3]):
                wl=2*e+lp//3 if plane<2 else e
                add(chip,0,plane,wl,'GU',e,None,None,lp,chip*64*1536+lp*PAGE,1536)
        for lp in range(12):add(chip,0,[3,2][lp%2],64+lp//2,'GU',64,None,None,lp,chip*128*1536+lp*PAGE,1536)
    need(len(own)==6336,'actualonecopyGU99MiB')
    ref=load('reused_independent_native_reference',REF)
    cfg=dict(cache_copy_ns=500,recycle_ns=6040,cold_discharge_ns=0,cold_ns=28000,cr_ns=9700,
             array_mode='cr',overwrite_at='evaluation_start',copy_overlaps_recycle=True,issue_policy='strict_no_interval',
             availability_phase_ns=0,unselected_policy='reject',weight_slots=2,copy_scope='per_plane',source_policy='fixed',ecc_bytes_per_second=6400000000)
    for g in range(8):
        need(len(layout['groups'][g]['jobs'])==34,'34residentlogicaljobsperpair')
        for rank,entry in enumerate(layout['groups'][g]['jobs']):
            expected_block=0 if rank<10 else 1;expected_wl=128+6*rank if rank<10 else 6*(rank-10)
            need(entry['block']==expected_block and entry['WL_start']==expected_wl,'closedformphysicalbands')
            for local in range(2):
                bundle=builder.buildjob(entry['owner'],g,local,layout);meta=bundle['metadata'];e=meta['expert'];half=meta['shared_half'];C=meta['columns'];chip=2*g+local
                start=local*768 if half is None else half*768+local*384;rows=768 if half is None else 384
                need(meta['physical_chip']==chip and meta['source_row_start']==start and meta['source_row_end']==start+rows,'canonicalrows')
                need(meta['input_bytes']==C==(512 if half is None else 1024),'completeinputcolumnsdimension')
                need(meta['WL_count']==6 and meta['pages']==24 and rows*C==24*PAGE,'wholepagewhole-rowjob')
                for wi,wave in enumerate(bundle['waves']):
                    need(wave['source_order']==[0,1,2,3] and len(wave['pages'])==4 and len({p['wl'] for p in wave['pages']})==1,'nativefixedsourcesameWL')
                    for pi,p in enumerate(wave['pages']):
                        lp=4*wi+pi;source=start*C+lp*PAGE
                        need(p['plane']==pi and p['block']==expected_block and p['wl']==expected_wl+wi and p['page_index']==lp,'actualnativeaddress')
                        need(p['physical_chip']==chip and p['source_byte_offset']==source and p['source_row_bytes']==C,'actualcanonicalsourcebytes')
                        need(source%C==0 and source+PAGE<=1536*C,'sourcefullrowpagebounds')
                        add(chip,p['block'],p['plane'],p['wl'],'Down',e,half,g,lp,source,C)
                for row in range(start,start+rows):source_rows[e,row]+=1
                view=builder.execution_view(bundle);job_records.append(view)
                rr=ref.finite_reference(bundle['waves'],cfg)
                need(rr['timing_feasible'] and rr['makespan']==89940,'existingtwo-slotnativeprimitive')
                reads=[[] for _ in range(4)];caches=[[] for _ in range(4)];ecc={p:(a,b) for p,a,b in rr['ecc']}
                copies={p:(a,b) for p,a,b in rr['copies']}
                for wi,wave in enumerate(bundle['waves']):
                    for p in wave['pages']:
                        a,b=copies[p['page_id']];end=ecc[p['page_id']][1];reads[p['plane']].append((rr['array'][wi][3],b));caches[p['plane']].append((a,end))
                peaks=[ref.peak(reads[p]+caches[p]) for p in range(4)]
                need(max(peaks)<=2 and all(ref.no_overlap(x) for x in caches),'twophysicalweightslots')
                native.append(dict(resident_job_id=view['resident_job_id'],duration_ns=89940,peak_slots_per_plane=peaks))
    need(job_records==read(OUT/'resident_jobs.frozen.json'),'executionviewfreezeidentity')
    for e in range(65):
        for row in range(1536):need(source_rows[e,row]==(4 if e<64 else 8),'fullcanonicalrowreplicamultiplicity')
    need(len(job_records)==544 and len(own)==19392 and len(own)*PAGE==303*(1<<20),'303MiBfullpayload')
    perchip=[[sum(c==chip and p==plane for c,b,p,w in own) for plane in range(4)] for chip in range(16)]
    need(perchip==[[332,332,274,274]]*16,'actualperplaneGUplusDowncounts')
    write(OUT/'physical_address_ledger.json',dict(status='PASS',columns=['chip','block','plane','wl','kind','expert','shared_half','copy_group','page_index','canonical_source_byte','row_bytes'],pages=ledger,
        actual_payload_bytes=len(own)*PAGE,actual_payload_MiB=303,GU_MiB=99,Down_MiB=204,reservation_plus_Down_MiB=332,
        actual_nonempty_chipblocks=len({(c,b) for c,b,p,w in own}),common_budget_chipblocks=32,
        source_identity='Canonicalmatrixbyte ranges androw/scalepositions only; no weightvalues programmed in this proof.'))
    synthetic=[list(range(6)),list(range(58,64)),list(range(40,46)),[0,63,1,62,2,61]]
    # Tight5-group Hall boundaries: fivecontainedsupport types plus any6thE.
    for servers in itertools.combinations(range(8),5):
        inside=[e for e,s in enumerate(supports) if set(s)<=set(servers)]
        if len(inside)==5:
            synthetic += [inside+[e] for e in range(64) if e not in inside]
    rng=np.random.Generator(np.random.PCG64(2026110284));synthetic += [rng.choice(64,6,replace=False).tolist() for _ in range(1000)]
    tested=[]
    for route in synthetic:
        need(independent_matching(supports,route) is not None,'independentfinite-statematching')
        result=builder.assign_route(route,layout);selected_audit(result,layout)
        reverse=builder.assign_route(list(reversed(route)),layout)
        need(result==reverse,'sortedIDruntimepolicy')
        if len(tested)<12:tested.append(result)
    controls=[]
    bad=copy.deepcopy(tested[0]);bad['selected_jobs'][-1]['columns']=512
    bad2=copy.deepcopy(tested[0]);bad2['selected_jobs'][-1]['row_start']=0
    bad3=copy.deepcopy(tested[0]);bad3['selected_jobs'][0]['block']=1-bad3['selected_jobs'][0]['block']
    for label,result in [('shortsharedinput',bad),('sharedrowoverlap',bad2),('wrongnativeblock',bad3)]:
        try:selected_audit(result,layout)
        except RuntimeError as e:controls.append(dict(name=label,detected=True,diagnostic=str(e)))
        else:raise RuntimeError('undetectedmutation'+label)
    for label,call in [('GU186cannotfit',lambda:builder.build_layout(gu_reserved_wl=186)),
                       ('duplicatedexpertID',lambda:builder.assign_route([0,0,1,2,3,4],layout))]:
        try:call()
        except ValueError as e:controls.append(dict(name=label,detected=True,diagnostic=str(e)))
        else:raise RuntimeError('undetectedinvalidcase'+label)
    # Larger degree5 design is a mathematical/storage comparison only.
    degree5=[tuple(s) for s in itertools.combinations(range(8),5)]+[tuple(sorted((i+j)%8 for j in range(5))) for i in range(8)]
    need(len(degree5)==64 and max(Counter(degree5).values())==2,'degree5multiplicity')
    need(Counter(g for s in degree5 for g in s)==Counter({g:40 for g in range(8)}),'degree5column40')
    need(hashes=={p:sha(p) for p in hashes},'sealedbuilderchanged')
    report=dict(status='PASS',builder_sha256=sha(BUILDER),verification_source_sha256=sha(__file__),inputs_sha256=hashes,
        universal_Hall_server_subsets=256,all_top6_covered=math.comb(64,6),expert_supports_unique=True,expert_degree=4,experts_per_group=32,
        runtime_synthetic_routes_tested=len(synthetic),runtime_reference='Independentreachable-group-maskDP plus sealedaugmentingdispatcher; reversedrouteorder identical.',
        resident_logical_jobs_per_pair=34,resident_chip_jobs=544,certified_native_job_references=len(native),native_duration_ns=89940,
        actual_full_payload_MiB=303,extra_vs_g2_MiB=13.5,actual_payload_bytes=len(own)*PAGE,
        extra_vs_g2_breakdown_MiB=dict(routed=3,shared=10.5),per_chip_plane_pages=perchip,
        input_columns=dict(routed=512,each_shared_half=1024),shared_global_rows=dict(S0=[0,768],S1=[768,1536]),
        I32_abs_dot_bounds=dict(routed=512*128*128,shared=1024*128*128),
        degree5_comparison=dict(expert_degree=5,copies=320,experts_per_group=40,jobs_per_pair=42,Down_WL_per_chip=252,total_MiB=351,saved_MiB_by_degree4=48,
            proof='Atmosttwoexperts shareany5support. Subsets<=5have5neighbors; sixexperts cannot allshare one5set. No degree5builder or actualreplay performed.'),
        negative_controls=controls,sample_dispatches=tested[:4],elapsed_seconds=time.monotonic()-began,
        chronology=frozen['chronology'],actual_commands=[['python',str(BUILDER)],['python',str(Path(__file__).resolve())]],
        limitations=['KnownCBC/replicateddeclustering/matching ideas; no newtheorem,solver,hardware or globalstorageoptimality claim.',
            'Onlymathematics, byte-address/rowextent certificates and isolatedconditionalnative recurrence are tested; no fullFFN, taskquality or actualflashmeasurement.',
            'Canonicalsource offsets identify exactfullrows and unchangedoriginalscale positions, but no newphysicalweightprogramming/numericalcheckpointtest is performed here.',
            'Allsharedhalfjobs retain completeC1024 input after thefullsharedGU/A8 barrier; half meansoutputrows, notinputdimension. Nohostpartialsum, inputfragment or additionaldotcontext.',
            'Conditionalnativekernel includes cold28us firstavailability,9.7usCR,500nscopy,twoPBslots,implicit6.4GB/sPE/ECCsustain; firstphase/discharge split remainsunknown.',
            'Defaultlayout reservesGU<=128WL; replicatedGU186WL cannotfit with34sixWLDownjobs underthislocal2blockallocator.',
            'Designfollowsobservedprospectivefailures. Anyreplay ofpreviouslyseenprospectivedata must belabeleddiagnosticreuse, notnewheldoutvalidation.'])
    write(OUT/'verification_report.json',report)
    write(OUT/'native_reference_summary.json',native)
    (OUT/'README.md').write_text(
        f"PASS: balancedunique degree4CBC supports all{math.comb(64,6):,}top6 requests plusS0/S1 by an independent256server-subset Hallcertificate. {len(synthetic):,} syntheticruntime routes also agree with a separate reachable-group-maskDP; thedispatcher places6experts in distinctpairs andsharedhalves in the2unusedpairs. This is standardCBC, not a newtheorem.\n\n"
        'Canonicalsupports are all70four-of-eightsets minus0123/4567,0246/1357,0145/2367. Eachpair holds32experts plusS0/S1=34jobs. Tenjobs occupyblock0WL128..187;24occupyblock1WL0..143. Everyjob is24pages/sixsameWLfixed0123waves withinoneblock. The544residentchipjobs and19392actualGU+Downpages are collisionfree. Total303MiB/layer=GU99+routed192+shared12; this is13.5MiB above thepreviousg2layout. Actualplane counts332/332/274/274 fit384; all32chipblocks are used.\n\n'
        'Eachselectedroutedexpert emits1536rows across two768rowbands with fullC512input. S0/S1 emit rows0..767/768..1535 across two384rowbands each, bothusing fullC1024input. The16selectedchipjobs cover eachphysicalchip exactlyonce andall10752Downrows, withoriginalrow/scalepositions. All544isolated nativejobs fit theexistingconditionaltwo-slotrecurrence at89940ns; this does not include fullFFNcommand/input/output contention.\n\n'
        'API in scripts/research_20261102_eight_pair_cbc.py: build_layout(); build_private_job(expert,group,local_chip,layout); build_shared_job(half,group,local_chip,layout); assign_route(route,layout). Dispatchreturns sixexpert assignment, twohalf assignment and16selected_jobs in existingexpert/copy0executionviewformat. Sharedjobs all useexpert64; resident_job_id identifiesS0/S1 androw_start is0/384/768/1152.\n\n'
        + '\n'.join(report['limitations'])+'\n')
    print(json.dumps({k:report[k] for k in ['status','builder_sha256','universal_Hall_server_subsets','all_top6_covered','runtime_synthetic_routes_tested','resident_chip_jobs','actual_full_payload_MiB','extra_vs_g2_MiB','elapsed_seconds']},indent=2))


if __name__=='__main__':main()
