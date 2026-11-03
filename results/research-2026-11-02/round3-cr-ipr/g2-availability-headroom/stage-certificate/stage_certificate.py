#!/usr/bin/env python3
"""Conditional whole-row Down service-stage arithmetic certificate."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import hashlib
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
CAP=HERE.parent/'capacity-kernel'
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
B=Fraction(32,5) # usefulECCbytes/ns
TOTAL=6*1536*512+1536*1024

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def exact(x):
 f=Fraction(x)
 return int(f) if f.denominator==1 else dict(numerator=f.numerator,denominator=f.denominator)

def freeze():
 assert not (HERE/'protocol.frozen.json').exists()
 paths=[Path(__file__),HERE/'plan.frozen.json',CAP/'kernel_builder.py',CAP/'protocol.frozen.json',CAP/'layout.frozen.json',PIPE]
 dump(HERE/'protocol.frozen.json',dict(status='frozen_before_checks',inputs_sha256={str(p):sha(p) for p in paths},
    objective='Coldcommon-release pureDown array/cache/ECC completiononly; noinputbus/command/output/MAC/GUcost. Allnativejobscold,oneGEMV/chip. AggregateproofonlyneedseachchipfirstECC>=s.',
    sensitivities=dict(cold_ns=28000,copy_ns=[0,100,500],abstract_startup_units=[0,1,2,7]),
    frozen_attainment_route=[0,4,8,12,16,20],numeric='ExactFractionarithmetic; integersealedtimestamps. Syntheticabstractoraclesonly,notTRAINorheldout.'))

def arithmetic(copy):
 s=28000+copy;threshold=393216
 service=Fraction(TOTAL,B);T=Fraction(s)+Fraction(service,16)
 maxr=threshold//512-1;maxs=threshold//1024-1
 nr=6*((1536+maxr-1)//maxr);ns=(1536+maxs-1)//maxs;n=nr+ns
 total_work=n*s+service;capacity=16*T
 assert TOTAL==6*2**20 and service==983040 and T==s+61440
 assert (maxr,maxs,nr,ns,n)==(767,383,18,5,23)
 assert total_work-capacity==7*s>0
 return dict(copy_ns=copy,startup_ns=s,total_useful_bytes=TOTAL,total_ECC_service_ns=exact(service),
    chip_count=16,bytes_per_chip_at_equality=TOTAL//16,ECC_tail_per_chip_ns=61440,lower_bound_ns=exact(T),
    aggregate_strict_proof='IfT<Tstar,16B(T-s)<6MiBsoallrequiredcoefficientbytescannotbeECC-served.',
    original_job_count=dict(under_strict_Tstar_routed_max_rows=maxr,shared_max_rows=maxs,min_routed_jobs=nr,min_shared_jobs=ns,min_total_jobs=n,
        serialized_job_work_lb_ns=exact(total_work),capacity_at_Tstar_ns=exact(capacity),strict_margin_ns=exact(total_work-capacity),
        extra_assumption='Entirejobstartup/ECCintervalsonasinglechiparedisjoint. Aggregateproofdoesnotneedthisassumption.'))

def compositions(total,n):
 if n==1:yield (total,);return
 for first in range(total+1):
  for rest in compositions(total-first,n-1):yield (first,)+rest

def exhaustive_small(H,startups):
 C=(1,1,2);chips=4;comps=list(compositions(H,chips));best={s:None for s in startups};count=0
 for allocation in product(comps,repeat=3):
  count+=1
  w=[sum(C[e]*allocation[e][c] for e in range(3)) for c in range(chips)]
  n=[sum(allocation[e][c]>0 for e in range(3)) for c in range(chips)]
  for s in startups:
   cost=max(w[c]+s*n[c] for c in range(chips))
   if best[s] is None or cost<best[s]['optimum']:
    best[s]=dict(optimum=cost,allocation=[list(a) for a in allocation],chip_work=[w[c]+s*n[c] for c in range(chips)],jobs=sum(n))
 records=[]
 for s in startups:
  lb=Fraction(s)+Fraction(H*sum(C),chips);r=best[s]
  assert r['optimum']>=lb
  if H==4:assert r['optimum']==s+4==lb
  records.append(dict(rows_each_matrix=H,columns=list(C),chips=chips,startup_units=s,total_allocations=count,
      aggregate_lower_bound=exact(lb),bound_attained=(r['optimum']==lb),**r))
 return records

def attainment(p,b,copy):
 route=[0,4,8,12,16,20];L=b.build_layout();matching=b.assign_route(route,L['supports'],1)
 assert matching['feasible'] and matching['group_load']==[1]*6
 jobs=[]
 for e,g in sorted(matching['assignment'].items()):
  for local in [0,1]:
   out=b.build_private_job(e,g,local,L);r=p.simulate(out['waves'],row_bytes=512,source_policy='fixed',cache_copy_ns=copy)
   jobs.append(dict(expert=e,chip=2*g+local,rows=768,columns=512,builder=out,trace=r))
 for local in range(4):
  out=b.build_shared_job(local);r=p.simulate(out['waves'],row_bytes=1024,source_policy='fixed',cache_copy_ns=copy)
  jobs.append(dict(expert=64,chip=12+local,rows=384,columns=1024,builder=out,trace=r))
 T=89440+copy
 assert len(jobs)==16 and len({j['chip'] for j in jobs})==16
 assert sum(j['rows']*j['columns'] for j in jobs)==TOTAL
 assert all(j['trace']['feasible'] and j['trace']['makespan_ns']==T for j in jobs)
 assert all(j['trace']['pages']==j['trace']['physical_read_pages']==24 for j in jobs)
 for e in route+[64]:assert sum(j['rows'] for j in jobs if j['expert']==e)==1536
 dump(HERE/f'attainment.copy{copy}.json',dict(status='PASS',route=route,matching=matching,parallel_start_ns=0,makespan_ns=T,jobs=jobs))
 return dict(copy_ns=copy,parallel_jobs=16,job_useful_bytes=393216,physical_pages_per_job=24,
    cold_per_job=1,CR_per_job=5,makespan_ns=T,source='fixed0123',drop=False,IPR=False,
    guard='Thisroutehasresidentperfectmatching;notallroutesdo.Allinputsreadyandcomputeonlyexcludesbus/commands/output.')

def run():
 proto=read(HERE/'protocol.frozen.json')
 for path,h in proto['inputs_sha256'].items():assert sha(Path(path))==h,path
 p=module('stage_certificate_sealed',PIPE);b=module('stage_certificate_capacity',CAP/'kernel_builder.py')
 values=[arithmetic(c) for c in [0,100,500]];witnesses=[attainment(p,b,c) for c in [0,100,500]]
 small=exhaustive_small(4,[0,1,2,7])+exhaustive_small(3,[0,1,2,7])
 dump(HERE/'small_whole_row_oracles.json',dict(status='PASS',model='Abstractcomplete-row additivejobcost=startup+rows*columns;B1unit/time;allinputsready.Onejob/owner/chipbymergingdominance.Arbitraryunequalrowsandmultipleowners/chipenumerated.',records=small))
 report=dict(status='PASS',protocol_sha256=sha(HERE/'protocol.frozen.json'),source_sha256=sha(__file__),
    objective='Coldcommon-release pureDown array/cache/ECC completion,whole-rowABI. NotfullFFNornativehardwareoptimality.',
    result_copy500_ns=89940,primary_proof='6MiBrequired;eachchipfirstusefulECC>=28.5us,then<=6.4GB/s. 6MiB<=16*6.4GB/s*(T-28.5us),soT>=89.94us.',
    arithmetic=values,finite_PB_attainment=witnesses,small_oracle_cases=len(small),
    assumptions=['NoDownweightsalreadyECC-readyorprestageservice;everychipfirstusefulECCstartsnoearlierthancold+copy.','6MiBdensecoefficientbytesmustbeprocessed;replicationonlychangesresidentaddresses.','EachchipusefulECCrate<=6.4GB/s,16homogeneouschips.','Purecompute excludescommand/input/outputbuses,GUcalendars,powerconstraintsanduncalibratedMACcost.','Attainmentrequiresresidentg2matchingandg4shared;completeprivateinputs/rows,oneGEMV/chip,noextraPEselector.','Theoriginal23jobproofadditionallyrequiresdisjointwhole-jobintervals;theaggregatecertificateavoidsperjobstartupsummation.'],
    limits=['AfullFFNcanreleaseexpertsatdifferenttimesandoverlapGU/Down;thisstageoptimumdoesnotprovewholeFFNoptimalityorallowaddingaGUbound.','WarmPBpages,prefetchbeforestage0,earlierfirstavailabilityorchangedthroughputinvalidatethecorrespondingpremise.','Measuredhardwarefirstready28us/copy500nsnotestablished;thisisthesamesealedconditionalserviceenvelope.','Whole-rowattainmentdoesnotgrantanewcolumnsplit/reductionABI.','Knownstartup/workthroughputcertificate,notnewgeneraltheory.'])
 dump(HERE/'report.json',report)
 lines=['# Conditional whole-row Down stage certificate','',
   'Under the declared cold common-release array/cache/ECC model, the pure Down stage optimum is **89.94µs** at copy500ns. This is a service-envelope certificate, not fullFFN or measured-hardware optimality.','',
   'Six routed1536×512 W8 matrices plus one shared1536×1024 matrix require6,291,456bytes =6MiB. Each chip first supplies useful ECC data no earlier than28.5µs. By timeT,16chips can supply at most16×6.4GB/s×(T−28.5µs) useful bytes. ThereforeT≥28.5µs+61.44µs=89.94µs. This aggregate proof does not require summing startup over multiple jobs.','',
   'The original job-count proof also checks out. ForT<89.94µs, a job has fewer than393,216 useful bytes. Complete rows imply at most767routed rows or383shared rows/job, hence at least18+5=23jobs. If whole-job intervals on a chip are disjoint, total work is at least23×28.5+983.04=1638.54µs, exceeding16×89.94=1439.04µs. The margin is199.5µs. The aggregate proof avoids this extra serialization premise.','',
   '| Copy ns | First ECC startup µs | Lower bound µs | Sealed16-job witness µs |','|---:|---:|---:|---:|']
 for a,w in zip(values,witnesses):lines.append(f'| {a["copy_ns"]} | {a["startup_ns"]/1000:.3f} | {a["lower_bound_ns"]/1000:.3f} | {w["makespan_ns"]/1000:.3f} |')
 lines += ['',
   'Attainment uses12routedjobs on six2-chipgroups plus4sharedjobs on fourchips. Everyjob reads393,216bytes in24pages/sixfullsame-WLwaves, with onecoldstart and fiveCRwaves. Fixed0123 and finite2weightlatches pass at copy0/100/500; no drop, IPR, extra inputcontext, FIFO orhold is used. Every expert receives all1536 complete output rows. A resident perfectmatching is required; it is not guaranteed for every route.','',
   'Two reduced exhaustive whole-row allocation oracles enumerate arbitrary unequal row counts and multiple owners/chip for3matrices on4chips, at startup0/1/2/7units. The balanced H4 geometry attains its aggregate bound; the H3 geometry demonstrates that row granularity and additional jobs can make the bound unattainable. These are abstract arithmetic sanity checks, not native timing simulations.','',
   'The stage starts with all private inputs available and no prior Down service. Command/input/output transfer, GU calendars, extra MAC latency and other shared resources are excluded. If a fullFFN prefetches Down weights or begins some experts before the last input is ready, this common-release stage certificate cannot be applied to its remaining Down interval or added to a GU lower bound. First-ready28us/copy500ns remain conditional model parameters.']
 (HERE/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS',copy_cases=values,finite_PB_attainment=witnesses,small_oracles=small)),flush=True)

if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
