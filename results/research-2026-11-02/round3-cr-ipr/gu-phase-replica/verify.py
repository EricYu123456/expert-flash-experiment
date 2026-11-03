#!/usr/bin/env python3
"""Count/ledger and sealed-PB bridges for two equal-byte GU replica schemes."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib
import importlib.util
import json
import math
import random
import sys
import phase_builder as phase
import group_builder as group
from all_dummy_reference import simulate_fullplane

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def freeze():
 assert not (HERE/'verification.protocol.frozen.json').exists()
 paths=[Path(__file__),HERE/'phase_builder.py',HERE/'group_builder.py',HERE/'all_dummy_reference.py',HERE/'plan.frozen.json',HERE/'protocol.frozen.json',HERE/'group.protocol.frozen.json',HERE/'resident_ledger.json',HERE/'group.resident_ledger.json',PIPE]
 dump(HERE/'verification.protocol.frozen.json',dict(status='frozen_before_aggregate_verification',utc=datetime.now(timezone.utc).isoformat(),inputs_sha256={str(p):sha(p) for p in paths},
   availability='Exhaustall16availabilitycountclasses, weightedexactlyC(64,6); additionally32fixedseedrandomroutes. NoTRAINorheldout.',
   PBbridge='Phaseandbothg8groups: everyroute/plain100/plain500/split500 compareindependentfullplanereference tosealedfinitePB perpageECCtimes andlatchviolations. Savecanonicaltraces.',
   all_dummy='Oncanonicalrouteg8group0/1, test13possibleall-dummyboundaries withonecomplete realunselectedsameWLreadwave. PrimarysameWLcontrol fixesafter10usefulwavesbeforeoutcomes;compare155680ns. Noall-dummywaveissilentlypassedtooldsimulator.',
   joint_capacity='FullGU744pagestemplate/chip. TripleroutedDown: chips0..11 block1,64experts×12pages(each3wholeWL); sharedDownchips12..15block0WL186..191,C1024,24pages. Expandphysicaladdressesandrequirezerocollision/budget2blocks.'))

def sealed():
 sp=importlib.util.spec_from_file_location('verify_phase_sealed',PIPE);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m

def compare(a,b):
 keys=('page_id','wave','plane','bytes','start_ns','end_ns')
 assert [tuple(p[k] for k in keys) for p in a['page_completions']]==[tuple(p[k] for k in keys) for p in b['page_completions']]
 assert a['feasible']==b['feasible'] and a['makespan_ns']==b['makespan_ns']
 keys=('wave','plane','release_ns','overwrite_ns','excess_ns')
 assert sorted(tuple(p[k] for k in keys) for p in a['violations'])==sorted(tuple(p[k] for k in keys) for p in b['violations'])

def availability():
 records=[];routes=[]
 for a in range(4):
  for b in range(4):
   d=6-a-b;route=list(range(a))+list(range(3,3+b))+list(range(6,6+d))
   chosen=phase.assign_phases(route)
   assert len(chosen['phase0_experts'])==len(chosen['phase2_experts'])==3
   count=math.comb(3,a)*math.comb(3,b)*math.comb(58,d)
   records.append(dict(phase0only=a,phase2only=b,dual=d,routes=count,status='PASS'));routes.append(route)
 assert sum(r['routes'] for r in records)==math.comb(64,6)==74974368
 rng=random.Random(20261102);routes += [sorted(rng.sample(range(64),6)) for _ in range(32)]
 return records,routes

def ledger_check():
 L=phase.build_layout();templates=[L]+[group.build_layout(g) for g in [0,1]]
 for t in templates:
  pages=t['resident_pages'];assert len(pages)==744
  assert Counter(p['plane'] for p in pages)==Counter({0:186,1:186,2:186,3:186})
  assert len({(p['plane'],p['block'],p['wl']) for p in pages})==744
  for plane in range(4):assert sorted(p['wl'] for p in pages if p['plane']==plane)==list(range(186))
 for e,copies in L['copies'].items():
  expected=1 if e<6 else 2;assert len(copies)==expected
  for pages in copies.values():
   assert [(p['source_page_id'],p['source_byte_offset']) for p in pages]==[(f'e{e}.p{j}',j*16384) for j in range(6)]
 chips=[];count=0
 for chip in range(16):
  # BothGUtemplates have identical186WLoccupancy, thoughlogicalsource shards differ.
  used={(p,b,w) for p in range(4) for b in [0] for w in range(186)};jobs=[]
  if chip<12:
   for e in range(64):
    band=dict(expert=e,block=1,WL_start=3*e,WL_count=3,pages=12,columns=512,rows=384,source_row_start=(chip%4)*384,copy_group=chip//4)
    jobs.append(band)
    for p in range(4):
     for w in range(3*e,3*e+3):assert (p,1,w) not in used;used.add((p,1,w))
  else:
   jobs=[dict(expert=64,block=0,WL_start=186,WL_count=6,pages=24,columns=1024,rows=384,source_row_start=(chip-12)*384)]
   for p in range(4):
    for w in range(186,192):assert (p,0,w) not in used;used.add((p,0,w))
  assert all(0<=b<2 and 0<=w<192 for p,b,w in used)
  count+=len(used);chips.append(dict(chip=chip,GU_pages=744,Down_jobs=jobs,total_pages=len(used),nonempty_blocks=len({b for p,b,w in used})))
 assert count==21216 and count*16384/2**20==331.5
 out=dict(status='PASS',chips=chips,total_pages=count,total_MiB=331.5,GU_MiB=186.0,extra_GU_MiB=87.0,
   triple_private_Down_MiB=144.0,shared_Down_MiB=1.5,capacity_budget_MiB=384.0,nonempty_blocks=28,
   assumption='GU rowbyte1536 andsourceW8 unchanged; replicated fullGU tensors differonlyphysicalplacement. G8templatesexpertshard128rowschip%8;phaseexpertshard64rowschip. Shared128rowschip inbothschemes. AllDownwhole384rows.' )
 dump(HERE/'joint_capacity_ledger.json',out)
 return {k:v for k,v in out.items() if k!='chips'}

def run():
 proto=read(HERE/'verification.protocol.frozen.json')
 for p,h in proto['inputs_sha256'].items():assert sha(Path(p))==h,p
 p=sealed();classes,routes=availability();capacity=ledger_check();records=[];bridges=0
 for route in routes:
  for kind,g in [('phase',None),('group',0),('group',1)]:
   for mode,c in [('plain',100),('plain',500),('split',500)]:
    build=(phase.build_fixedphase_plain if mode=='plain' else phase.build_fixedphase_split) if kind=='phase' else (group.build_fixedgroup_plain if mode=='plain' else group.build_fixedgroup_split)
    out=build(route) if g is None else build(route,g)
    a=p.simulate(out['waves'],cache_copy_ns=c,source_policy='fixed',unselected_policy='drop_before_ecc_conditional')
    b=simulate_fullplane(out['waves'],c);compare(a,b);bridges+=1
    assert a['feasible']==(mode=='split' or c==100)
    if a['feasible']:assert a['makespan_ns']==(155140 if mode=='split' else 150980)
    records.append(dict(route=route,kind=kind,group=g,mode=mode,copy_ns=c,feasible=a['feasible'],makespan_ns=a['makespan_ns'],source0123=True))
 boundary=[]
 for g in [0,1]:
  base=group.build_fixedgroup_whole_dummy([0,1,2,3,4,5],g)
  dummy=base['waves'][10];plain=group.build_fixedgroup_plain([0,1,2,3,4,5],g)['waves']
  for pos in range(13):
   waves=plain[:pos]+[dummy]+plain[pos:];r=simulate_fullplane(waves,500)
   assert all(len({(x['block'],x['wl']) for x in w['read_pages']})==1 for w in waves)
   boundary.append(dict(group=g,dummy_after_useful_waves=pos,feasible=r['feasible'],makespan_ns=r['makespan_ns'],violations=r['violations']))
   if pos==10:
    assert r['feasible'] and r['makespan_ns']==155680
    dump(HERE/f'group{g}.trace.whole_dummy.copy500.json',dict(builder=base,trace=r,source='all_dummy_reference.py',mode='sameWL skipWLCR only; noIPR'))
 dump(HERE/'availability_proof.json',dict(status='PASS',classes=classes,total_routes=74974368,replication_minimality='Underexactlytwofullphase/groupcopiesandcapacity3perphase/group,atmost3singlecopyexpertseachside;otherwiseatop6containing4exclusiveexpertsfails. Thusatleast58dualexpertsnecessaryandconstructionattainsit. NotaglobalbyteLBoutsidecontract.'))
 dump(HERE/'sealed_reference_bridge.json',dict(status='PASS',cases=bridges,records=records))
 dump(HERE/'whole_dummy_boundaries.json',boundary)
 report=dict(status='PASS',verification_protocol_sha256=sha(HERE/'verification.protocol.frozen.json'),availability_classes=16,all_top6_routes=74974368,sealed_reference_bridges=bridges,capacity=capacity,
   kernel_results=dict(phase_fixed0123_split500_ns=155140,group_fixed0123_IPRsplit500_ns=155140,group_fixed0123_sameWLwholeDummy500_ns=155680,phase_or_group_plain100_noDrop_ns=150980,plain500='strictfinitePBfails'),
   interpretation='Two-servercapacity3 CBC/grouploadbalancingprior; standardg8wholeGUreplicasareequal87MiBstrongcontrolandmatchphaseIPRsplitkerneltiming. Phase-specificnoveltynotestablished. SameWLg8whole-dummy needsnoIPR,onlyskipWLCRandconditionalpreECCdrop.',
   limits=['NoTRAIN,heldout,fullFFNreplayormodelvaluechange.','NativepreECCdropunverified; fixedPBsource0123usedthroughout.','IPRsplitmodeandtrueSAME-WLwhole-dummymodeareseparatecontrols.','All-dummyemptyusefulwavecannotbe passedtosealedv1;restrictedreferenceexplicitlyextendsthisinputdomainandbridges432nonemptycases.','NewPEselector,inputcontext,FIFO,idleholdorpartialdotreductionnotintroduced.','Firstavailabilityconservative28us,notmeasured;ECC6.4GB/sandCR9.7uskeptseparate.'])
 dump(HERE/'report.json',report)
 lines=['# Equal-byte GU replica controls','','The two-phase and standard g8 whole-GU constructions both add58completeGUcopies (87MiB/layer), use186WL/plane forGU, and admit every top6route. The ordinary g8 control prevents a phase-specific novelty claim.','','| Kernel | Source/capability | waves | copy ns | latency µs |','|---|---|---:|---:|---:|',
 '| phase replicas + split10/prefix1 | fixed0123, IPR, conditional pre-ECC drop | 13 | 500 | 155.140 |',
 '| g8 whole-GU replicas + split10/prefix1 | fixed0123, IPR, conditional pre-ECC drop | 13 | 500 | 155.140 |',
 '| g8 whole-GU replicas + whole dummy after10usefulwaves | fixed0123, everywave SAME-WL, conditional pre-ECC drop | 13 | 500 | 155.680 |',
 '| either construction, plain | fixed0123, no drop | 12 | 100 | 150.980 |', '',
 'Plain12waves atcopy500 fails the finite two-weight-latch deadline. No cold or idle rescue is inserted. The sealed-compatible split primary is checked directly by the unchanged pipeline; the whole-dummy same-WL control uses a small explicit full-plane recurrence, with every ECCcompletion and latch hazard bridged against the sealed simulator on432nonemptycases.', '',
 'There are16availabilitycountclasses representing all74,974,368top6sets. Eachside has3exclusiveexperts; dualexperts fillremainingcapacity3. Fourexclusiveexperts ononeside would give acounterexample, so58dualexperts is minimal within this two-full-copy/capacity3 contract, not a newgeneral CBC theorem.', '',
 'Capacity jointly includes triple routed Down and dedicated sharedg4 Down: GU186MiB + privateDown144MiB + sharedDown1.5MiB=331.5MiB/layer, below common384MiB. Privatechips use GUblock0andDownblock1; sharedchips fill GU0..185 andDown186..191 inblock0. EveryDownjob remainswhole384rows, onecommand/onecoldstart; 28blocks are nonempty.', '',
 'The source ledgers preserve source page/byte identity. Phase replicas use64GUrows/chip across16chips; groupreplicas use128rows/chip across8chips. Shared remains128rows/chip across16chips. Complete owners consume their full6/12pages without insertinganotherowner. No quantization, dotproduct, PEselector, extra inputcontext orFIFO is changed.', '',
 'Pre-ECC drop remains conditional. Skip-WL CR is the user premise; same-WL whole-dummy g8 does not need IPR, while split controls do. Firstpage28us is a conservative accounting convention, not measured availability. These are conditional kernel/compiler witnesses; fullFFN results and external independent audit live separately.']
 (HERE/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
