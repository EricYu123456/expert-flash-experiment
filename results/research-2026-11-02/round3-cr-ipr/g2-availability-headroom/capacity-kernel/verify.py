#!/usr/bin/env python3
"""Independent capacity, complete-row and finite-PB checks; no route training."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import random
import sys
import numpy as np
import kernel_builder as build

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def freeze():
 assert not (HERE/'verification.protocol.frozen.json').exists()
 paths=[Path(__file__),HERE/'kernel_builder.py',HERE/'plan.frozen.json',HERE/'protocol.frozen.json',HERE/'layout.frozen.json',PIPE]
 dump(HERE/'verification.protocol.frozen.json',dict(status='frozen_before_fullverification',inputs_sha256={str(p):sha(p) for p in paths},
   work='All252completecopybands×2chips +4sharedchips, sealedstrict500, actualphysicaladdresses andGUreservation noncollision. Firstandlastblockbands saved. NoTRAINorheldout.',
   corners='copy0/100/500/1940/1941/2000sixwavekernel; independentmaxplus recurrence;1940/1941deadline boundary. Explicit6expertmissinggroup counterexample and512seededcapacity2fixtures, notdistributionestimates.',
   numeric='FourdeterministicfullshapeW8_A8syntheticprivatefixtures plusonesharedfixture. Physicallypacksourcebytesintoactualplane/block/WLpages, fixed0123read,reassemblefullrows,dotint32andscatterexactlyagainsthost. No realmodelnumericclaim orquantizationchange.'))

def pipeline():
 spec=importlib.util.spec_from_file_location('g2_capacity_pipe',PIPE);p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p);return p

def six_wave_reference(copy):
 cache=[0]*4;read=[0]*4;ecc=0;done=[];violations=[]
 for wave in range(6):
  ready=28000+9700*wave;overwrite=0 if wave==0 else 28000+9700*(wave-1)+6040
  for plane in range(4):
   if read[plane]>overwrite:violations.append((wave,plane,read[plane]-overwrite))
  for plane in range(4):
   read[plane]=max(ready,cache[plane])+copy
   ecc=max(ecc,read[plane])+2560;cache[plane]=ecc;done.append(ecc)
 return dict(feasible=not violations,makespan_ns=None if violations else max(76500,ecc),ECC_end_ns=done,violations=violations)

def addresses_and_kernels(p,L):
 rows=[];records=[];kernels=0;source_rows=0
 for group in L['groups']:
  for local in [0,1]:
   chip=2*group['group']+local;used={(plane,0,wl) for plane in range(4) for wl in range(128)};pages=[]
   for job in group['jobs']:
    out=build.build_private_job(job['expert'],group['group'],local,L);waves=out['waves']
    r=p.simulate(waves,row_bytes=512,source_policy='fixed',cache_copy_ns=500)
    assert r['feasible'] and r['makespan_ns']==89940 and r['pages']==r['physical_read_pages']==24 and r['cold_waves']==1 and r['cr_waves']==5
    assert r['stream_validation']['owner_rows']=={f'e{job["expert"]}':768}
    assert len({x['block'] for w in waves for x in w['pages']})==1
    for w in waves:
     assert len({(x['block'],x['wl']) for x in w['pages']})==1
     for x in w['pages']:
      a=(x['plane'],x['block'],x['wl']);assert a not in used;used.add(a)
      assert x['source_byte_offset']==local*768*512+x['page_index']*16384
      pages.append([job['expert'],x['page_index'],*a,x['source_byte_offset']])
    kernels+=1;source_rows+=768
    if local==0 and job['rank'] in [0,9,10,41]:
     dump(HERE/f'trace.g{group["group"]}.rank{job["rank"]}.json',dict(builder=out,trace=r))
   assert len(pages)==1008 and len(used)==1520
   assert Counter(z[3] for z in pages)==Counter({0:240,1:768})
   rows.append(dict(chip=chip,rows=pages,reserved_GU_pages=512,total_reserved_plus_Down_pages=1520))
 for local in range(4):
  out=build.build_shared_job(local);r=p.simulate(out['waves'],row_bytes=1024,source_policy='fixed',cache_copy_ns=500)
  assert r['feasible'] and r['makespan_ns']==89940 and r['stream_validation']['owner_rows']=={'s':384}
  pages=[]
  for w in out['waves']:
   for x in w['pages']:
    assert x['block']==0 and 128<=x['wl']<134
    assert x['source_byte_offset']==local*384*1024+x['page_index']*16384
    pages.append([64,x['page_index'],x['plane'],x['block'],x['wl'],x['source_byte_offset']])
  rows.append(dict(chip=12+local,rows=pages,reserved_GU_pages=512,total_reserved_plus_Down_pages=536));kernels+=1;source_rows+=384
 dump(HERE/'physical_address_ledger.json',dict(status='PASS',page_columns=['expert','local_page_index','plane','block','WL','source_byte_offset'],chips=rows,
    reservation='GUblock0WL0..127eachplane; reservationisnotfictitiousdataorreadpages.',physical_Down_pages=12096+96,
    private_Down_MiB=189.0,shared_Down_MiB=1.5,conservative_GU_reserved_MiB=128.0,total_reserved_plus_Down_MiB=318.5,
    actual_total_if_onecopy_GU99MiB=289.5,capacity_budget_MiB=384.0,nonempty_blocks=28))
 return dict(kernels=kernels,source_output_rows_checked=source_rows,copy_instances=252,privatejobs_eachsingleblock=True)

def packed_dot_fixture(L):
 rng=np.random.default_rng(20261102);records=[]
 for e in [0,40,60,63]:
  W=rng.integers(-128,128,size=(1536,512),dtype=np.int16).astype(np.int8);A=rng.integers(-128,128,size=512,dtype=np.int16).astype(np.int8)
  expected=W.astype(np.int32)@A.astype(np.int32);source=W.tobytes(order='C')
  for g in L['supports'][e]:
   actual=np.empty(1536,dtype=np.int32);covered=np.zeros(1536,dtype=np.int8)
   for local in [0,1]:
    out=build.build_private_job(e,g,local,L);memory={}
    for w in out['waves']:
     for p in w['pages']:
      address=(p['plane'],p['block'],p['wl']);offset=p['source_byte_offset'];memory[address]=source[offset:offset+16384]
    stream=b''.join(memory[p['plane'],p['block'],p['wl']] for w in out['waves'] for p in sorted(w['pages'],key=lambda x:x['plane']))
    values=np.frombuffer(stream,dtype=np.int8).reshape(768,512).astype(np.int32)@A.astype(np.int32)
    begin=local*768;actual[begin:begin+768]=values;covered[begin:begin+768]+=1
   assert np.all(covered==1) and np.array_equal(actual,expected)
   records.append(dict(expert=e,copy_group=g,rows=1536,exact=True,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
 W=rng.integers(-128,128,size=(1536,1024),dtype=np.int16).astype(np.int8);A=rng.integers(-128,128,size=1024,dtype=np.int16).astype(np.int8)
 expected=W.astype(np.int32)@A.astype(np.int32);source=W.tobytes();actual=np.empty(1536,dtype=np.int32)
 for local in range(4):
  out=build.build_shared_job(local);memory={}
  for w in out['waves']:
   for p in w['pages']:
    offset=p['source_byte_offset'];memory[p['plane'],p['block'],p['wl']]=source[offset:offset+16384]
  stream=b''.join(memory[p['plane'],p['block'],p['wl']] for w in out['waves'] for p in sorted(w['pages'],key=lambda x:x['plane']))
  actual[local*384:(local+1)*384]=np.frombuffer(stream,dtype=np.int8).reshape(384,1024).astype(np.int32)@A.astype(np.int32)
 assert np.array_equal(actual,expected);records.append(dict(expert=64,rows=1536,exact=True,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
 dump(HERE/'synthetic_W8_A8_packed_rows.json',dict(status='PASS',scope='Deterministicfullshapesyntheticfixtures;notrealmodelvalidation.',records=records,exact_rows=sum(r['rows'] for r in records)))
 return sum(r['rows'] for r in records)

def run():
 proto=read(HERE/'verification.protocol.frozen.json')
 for path,h in proto['inputs_sha256'].items():assert sha(Path(path))==h,path
 p=pipeline();L=build.build_layout();assert L==read(HERE/'layout.frozen.json')
 audit=addresses_and_kernels(p,L);exactrows=packed_dot_fixture(L);corners=[]
 for copy in [0,100,500,1940,1941,2000]:
  out=build.build_private_job(0,0,0,L);r=p.simulate(out['waves'],row_bytes=512,source_policy='fixed',cache_copy_ns=copy);ref=six_wave_reference(copy)
  assert r['feasible']==ref['feasible'] and r['makespan_ns']==ref['makespan_ns']
  assert [x['end_ns'] for x in r['page_completions']]==ref['ECC_end_ns']
  assert [(v['wave'],v['plane'],v['excess_ns']) for v in r['violations']]==ref['violations']
  corners.append(dict(copy_ns=copy,feasible=r['feasible'],makespan_ns=r['makespan_ns'],independent_reference='EXACT',violations=r['violations']))
 missing=[e for e,s in enumerate(L['supports']) if 0 not in s];assert len(missing)==22
 counter=missing[:6];fail=build.assign_route(counter,L['supports'],1);good=build.assign_route(counter,L['supports'],2)
 assert not fail['feasible'] and good['feasible']
 rng=random.Random(20261102);matching=[]
 for route in [counter,list(range(6)),[0,40,60,61,62,63]]+[sorted(rng.sample(range(64),6)) for _ in range(512)]:
  b=build.assign_route(route,L['supports'],2);assert b['feasible']
  matching.append(dict(route=route,capacity2=b,capacity1=build.assign_route(route,L['supports'],1)))
 dump(HERE/'matching_cornercases.json',dict(status='PASS',scope='Deterministicfixturesonly,notTRAINoradistributionestimate.',records=matching))
 try:build.build_layout(gu_reserved_wl=186);raise AssertionError('Expectedlocalcapacityfailure')
 except ValueError as e:replica_failure=str(e)
 report=dict(status='PASS',verification_protocol_sha256=sha(HERE/'verification.protocol.frozen.json'),kernel_audit=audit,synthetic_exact_rows=exactrows,
    private_g2_kernel_ns=89940,shared_g4_kernel_ns=89940,private_owner_occupation_input1_ns=90510,shared_owner_occupation_input1_ns=90734,
    resource_accounting='Kernel89940=28000+500+24*2560ns; arrayends76500ns. ECCwork61440ns. Startup+drainarecausal;notmax(array,ECC). Owneroccupationaddscommand250+inputC/2+lastpage4Browoutput/2.',
    copy_corners=corners,capacity=dict(group_resident_copies=[42]*6,total_copy_instances=252,Down_WL_per_private_chip=252,GU_reserved_WL=128,
      privatechip_block0_GU='0..127',privatechip_block0_Down='10whole6WLjobs128..187',privatechip_block1_Down='32whole6WLjobs0..191',privatechip_unused_WL='block0 188..191',
      private_Down_MiB=189.0,shared_Down_MiB=1.5,actual_total_with_onecopy_GU99MiB=289.5,conservative_total_with_GUreservation128MiB=318.5,budget_MiB=384.0,nonempty_blocks=28,
      phase_or_group_GU186_rejected=replica_failure,local_overflow_WL_with_GU186=54),
    matching=dict(counterexample_route=counter,counterexample_union_groups=sorted({g for e in counter for g in L['supports'][e]}),capacity1_failure=fail,capacity2_witness=good,
      universal_capacity2_proof='Anyselectedsubsethasatmost6experts. Anyoneexpertalreadyhasatleast3neighbors,so2*|N(S)|>=6>=|S|;capacity2Hallholds. Atmosttwoseparatejobs/groupareexecutedserially,notconcurrentlywithmultipleprivateinputs.',
      universal_capacity1_impossible='Eachgroupmisses22experts;choose6missingonegroup,then6expertsseeatmost5groups. Moregenerallyuniversalcapacity1wouldrequireeachgroupstore>=59experts,atleast354copyinstances;252isnotenough.',
      fixture_count=len(matching),fixture_scope='NotTRAINandnotanestimateofrealmatchingfrequency.'),
    g4_triples_kernel_comparison=dict(two_g4_jobs_ns=118440,one_g2_job_ns=89940,reduction_ns=28500,reduction_fraction=1-89940/118440,
      scope='OnlybareprivateDownchipcompute: twoseparate12pageg4privatejobsagainstsingle24pageg2jobwhenperfectmatchingexists. FullFFNspeeduprequiresrealrouteavailabilityandcausalscheduling.'),
    interpretation='Standardreplication/availability/sharddegreebaseline;notnewCBC,solverorhardware. NoTRAIN,heldoutorfullFFN. OnlyGU<=128WLjointplacementisverified.')
 dump(HERE/'report.json',report)
 lines=['# g2 Down availability: capacity and kernel check','','All252completeDowncopybands andbothchipshards were expanded and checked:508private/sharedkernels pass sealedfixed0123/strict500 timing. Eachprivatejob has768completeC512rows; eachsharedjob has384completeC1024rows. No job crossesblocks.','','| Quantity | Verified result |','|---|---:|','| Private g2 / shared g4 kernel | 89.940µs |','| Private owner occupation, input1 | 90.510µs |','| Shared owner occupation, input1 | 90.734µs |','| Copy instances / resident experts pergroup | 252 / 42 |','| PrivateDownWL/chip + GUreservation | 252 +128 =380 |','| Actualweights with99MiBonecopyGU | 289.5MiB/layer |','| ConservativeGUreservation +Down | 318.5MiB/layer |','| Commoncapacitybudget | 384MiB/layer |','| SyntheticpackedW8_A8exactrows | '+str(exactrows)+' |','',
 'Eachprivatechip places10whole6WLjobs inblock0WL128..187 and32 inblock1WL0..191. Thefourremainingblock0WL areunused. SharedDown usesblock0WL128..133 onchips12..15. Everyjob hasonecommand andonecoldstart. GUreservationsarenotfictitiousdataorreadpages.', '',
 f'Capacity1 is not universal. The explicit route{counter} avoids group0 and sees onlyfivegroups. Everygroupmisses22experts,so suchacounterexampleexists foreachgroup. Universalcapacity1wouldrequireeachgroupmissatmost5experts, at least354totalcopies, beyond252.', '',
 'Capacity2 is universal byHall: anynonempty subsetofatmostsixselectedexperts hasatleastthreeneighborgroups,so twiceitsneighborcount isatleastthesubsetsize. This permitsatmosttwoseparateprivatejobs pergroup, executedserially. It doesnotgrant extra PE inputcontexts.', '',
 'The six-wavecopy boundary is1940ns pass /1941ns fail underthedeclaredwholepagecache-onlymodel. The89940nskernel accountsfor28000nsstartup,500nscopy,and61440nsECC; arrayworkendsat76500ns. It is not inferredbymax(array,ECC).', '',
 'The phase/groupGUreplica186WL maps do not fit this privateg2layout:186+252=438>384locally. Aggregateweightbytes do not remove thatperchipoverflow. ThiswitnessusesGU<=128WL andmustbecomparedwiththecorrespondingfeasibleGUmenus.', '',
 'A perfectsixgroupmatching wouldreduceprivatechipkernelwork fromtwog4jobs118.44µs tooneg2job89.94µs, saving28.5µs. This is kernelheadroom only; actualTRAINmatchingandfullFFNreplayareseparateroot/numericwork. The construction is astandardreplication/shardingbaseline, not newCBC.']
 (HERE/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:report[k] for k in ['status','kernel_audit','synthetic_exact_rows','private_g2_kernel_ns','capacity','matching']}),flush=True)

if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
