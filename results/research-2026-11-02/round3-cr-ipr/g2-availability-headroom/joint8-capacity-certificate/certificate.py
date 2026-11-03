#!/usr/bin/env python3
"""Exact arithmetic + bounded solver cross-check for joint8 full-copy CBC."""
from pathlib import Path
from fractions import Fraction
from itertools import combinations
from math import comb,ceil
import hashlib
import json
import sys
import time
import numpy as np
import scipy
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_array

HERE=Path(__file__).resolve().parent
SUPPORTS=list(range(1,256))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def exact(x):
 f=Fraction(x);return int(f) if f.denominator==1 else dict(numerator=f.numerator,denominator=f.denominator)
def freeze():
 assert not (HERE/'protocol.frozen.json').exists()
 paths=[Path(__file__),HERE/'plan.frozen.json']
 dump(HERE/'protocol.frozen.json',dict(status='frozen_before_solver',inputs_sha256={str(p):sha(p) for p in paths},scipy_version=scipy.__version__,time_limit_seconds_per_call=2.0,
   model='Eightgroups,64variableexpert supports,twofixedsharedsupports. All24union8symmetryrepresentativesLP;3directMILPs. No realroutesoroutcomesread.',
   proof='Exactrationalcountinequalitiesprimary;solverstatuscrosscheckonly. Known34witnesscheckedbyall255Hallconditions.'))

def arithmetic():
 small=[]
 coeff={1:(Fraction(11,2),Fraction(1,2)),2:(Fraction(9,2),Fraction(1,2)),3:(Fraction(4),Fraction(1)),4:(Fraction(3),Fraction(1)),5:(Fraction(2),Fraction(1))}
 for a in range(1,6):
  r=8-a;h=6-a;w=[comb(r-d,h-d) if d<=h else 0 for d in range(r+1)]
  alpha,beta=coeff[a]
  assert all(Fraction(d)>=alpha-beta*w[d] for d in range(r+1))
  totalHall=5*comb(r,h);lb=64*alpha-beta*totalHall;capacity=32*r
  assert lb>capacity
  small.append(dict(shared_support_size=a,outside_groups=r,T6_supersets=comb(r,h),outside_degree_weights=w,
    per_expert_linear_inequality=dict(alpha=exact(alpha),beta=exact(beta),statement='outside_degree>=alpha-beta*Hallweight'),
    Hall_weight_upper=totalHall,outside_expert_copy_LB=exact(lb),integer_copy_LB=ceil(lb),outside_copy_capacity=capacity,contradiction=True))
 degree=[]
 for d in range(1,9):
  w=comb(8-d,5-d) if d<=5 else 0
  assert 6*d+w>=28
  degree.append(dict(degree=d,T5_containment_count=w,six_degree_plus_count=6*d+w))
 assert Fraction(28*64-56*5,6)==252
 return dict(status='PASS',union_cover_proof='IfgoutsideAunionB,T=allgroupsminusgcontainsthetwosharedsupports.HallforcesnE(T)<=5,sogstores>=64-5=59E,contradictingcap33.',
   small_shared_support_exclusion=small,global_degree_inequality=degree,
   total_expert_copy_LB=252,cap33_total_copies=264,shared_copy_LB_after_exclusions=12,
   equality_forces='a=b6,Ecopies252,all56T5 have nE(T)=5.',last_contradiction=dict(sum_T5_inside_shared6=30,maximum_from_Hall_shared6=25,
    explanation='Shared6supportAcontains6five-subsets,eachwith5E. HallnE(A)<=5;anynonemptyexpertsupportinsideAiscontainedinatmost5ofthosefive-subsets. Thus30<=5*5=25 isimpossible.'),
   conclusion='No supportcounts,evenfractionalLPcounts,canmeetcap33. Wholecopyminmax>=34.')

def representative(a,b):
 assert 1<=a<=b<=8 and a+b>=8
 overlap=a+b-8;A=(1<<a)-1;B=((255^A)|((1<<overlap)-1))
 assert A.bit_count()==a and B.bit_count()==b and (A|B)==255
 return A,B

def model(A,B,cap=33):
 rr=[];cc=[];data=[];upper=[];labels=[]
 def row(columns,rhs,label):
  index=len(upper);upper.append(rhs);labels.append(label)
  for col in columns:rr.append(index);cc.append(col);data.append(1.)
 row(range(255),64,'total_E')
 for g in range(8):
  shared=int(bool(A&(1<<g)))+int(bool(B&(1<<g)))
  row([i for i,S in enumerate(SUPPORTS) if S&(1<<g)],cap-shared,f'group{g}')
 for T in SUPPORTS:
  shared=int((A&T)==A)+int((B&T)==B);rhs=T.bit_count()-shared
  if rhs<6:row([i for i,S in enumerate(SUPPORTS) if (S&T)==S],rhs,f'Hall_T{T}')
 M=coo_array((np.array(data), (np.array(rr,dtype=np.int32),np.array(cc,dtype=np.int32))),shape=(len(upper),255)).tocsc()
 lb=np.full(len(upper),-np.inf);lb[0]=64
 return LinearConstraint(M,lb,np.array(upper,dtype=float)),labels

def solve(a,b,integer):
 A,B=representative(a,b);constraints,labels=model(A,B)
 start=time.monotonic()
 out=milp(np.zeros(255),integrality=np.ones(255) if integer else np.zeros(255),bounds=Bounds(np.zeros(255),np.full(255,64.)),
          constraints=constraints,options=dict(time_limit=2.0,presolve=True,mip_rel_gap=0.0))
 rec=dict(shared_sizes=[a,b],overlap=a+b-8,shared_masks=[A,B],integer=integer,status=int(out.status),message=str(out.message),
          wall_seconds=time.monotonic()-start,variables=255,constraints=len(labels),time_limit_seconds=2.0)
 if out.status==0:raise AssertionError(('Unexpectedfeasiblecasecontradictsarithmeticcertificate',rec,out.x.tolist()))
 return rec

def known34():
 masks=[sum(1<<g for g in C) for C in combinations(range(8),4)]
 pairs=[]
 for S in masks:
  if S<(255^S):pairs.append([S,255^S])
 deleted={S for pair in pairs[:3] for S in pair};experts=[S for S in masks if S not in deleted]
 assert len(experts)==64
 counts=[sum(bool(S&(1<<g)) for S in experts) for g in range(8)]
 assert counts==[32]*8
 checks=[]
 for T in SUPPORTS:
  n=sum((S&T)==S for S in experts);s=2 if T==255 else 0
  assert min(n,6)+s<=T.bit_count()
  checks.append(dict(T=T,size=T.bit_count(),expert_containment=n,shared_containment=s,Hall_lhs=min(n,6)+s))
 dump(HERE/'known34.witness.json',dict(status='PASS',deleted_complement_pairs=pairs[:3],expert_support_masks=experts,shared_masks=[255,255],
      expert_residents_per_group=counts,total_residents_per_group=[34]*8,all_Hall_checks=checks,
      Down_WL_per_chip=204,Down_MiB=204.0,with_onecopy_GU99_MiB=303.0,with_GUreplica186_MiB=390.0,
      payload_optimality_claimed=False))
 return dict(status='PASS',minmax_resident_jobs=34,total_resident_job_copies=272,expert_copy_instances=256,shared_copy_instances=16,
             Down_WL_per_chip=204,GU186_plus_Down_WL=390,per_chip_WL_budget=384,overflow_WL=6)

def run():
 protocol=json.loads((HERE/'protocol.frozen.json').read_text())
 for p,h in protocol['inputs_sha256'].items():assert sha(p)==h,p
 proof=arithmetic();dump(HERE/'arithmetic_certificate.json',proof)
 results=[]
 for a in range(1,9):
  for b in range(a,9):
   if a+b<8:continue
   r=solve(a,b,False);results.append(r);print(json.dumps(r),flush=True)
 for a,b in [(4,4),(6,6),(8,8)]:
  r=solve(a,b,True);results.append(r);print(json.dumps(r),flush=True)
 unknown=[r for r in results if r['status']!=2]
 witness=known34()
 report=dict(status='PASS',protocol_sha256=sha(HERE/'protocol.frozen.json'),source_sha256=sha(__file__),
   exact_arithmetic_certificate='cap33 impossible;casepartitioncoversallnonemptysharedsupports up togrouppermutation/sharedswap.',
   LP_representatives=24,direct_integer_MILP_representatives=3,solver_infeasible=sum(r['status']==2 for r in results),solver_unknown=unknown,
   solver_results=results,known34=witness,conditional_optimum_minmax_resident_jobs=34,
   interpretation='KnownHall/CBC capacityaccountingcertificateunderfull24pagecopycontract,notsolverortheoremnovelty.',
   limits=['NoTRAIN,prospectiveinputs/outcomes,heldoutorfullFFNread.','Sharedobjectsarefixedoutputhalves andeveryresidentcopyis24pages/6WLperchip.','Doesnotruleoutotherrowmaps,partialcopies,projectionmappingorABIs.','Known34witnesspayload303MiBwithGU99isnotclaimedstorage-optimal.','GU186+Down33wouldfit384WLbut33isimpossible;known34needs390WL.'])
 dump(HERE/'report.json',report)
 lines=['# Joint eight-pair CBC: cap33 is impossible','',
   '**Within the fixed24-page full-copy contract, the minimum possible maximum residentjobs/group is34.** A33-job cap is impossible by exact Hall counting; a known34-job construction passes all255Hall constraints. This is a standard CBC capacity certificate, not a newtheorem or fullFFN result.','',
   'LetA,B be the nonempty supports of the two fixed sharedhalf objects. For everygroup subsetT, universal top6routed plusbothshared matching requires min(nE(T),6)+s(T)≤|T|.','',
   'First,A∪B must cover all8groups. Ifg is outside their union, T=allgroups exceptg contains both shared supports, hence nE(T)≤5. Groupg would have to store at least59routedexperts, contradicting33.','',
   'If a sharedsupport has sizea≤5, let r=8−a andh=6−a. Sum Hall over all size6supersets of that support. An expert havingd supportgroups outside it is counted C(r−d,h−d)times. Exact scalar inequalities give the following lower bounds on routedcopies outside the sharedsupport; eachoutsidegroup already stores the other shared object, leaving at most32expertcopies.','',
   '| Shared support size | Outsidegroups | Required expertcopies, integer LB | Capacity |','|---:|---:|---:|---:|']
 for r in proof['small_shared_support_exclusion']:lines.append(f'| {r["shared_support_size"]} | {r["outside_groups"]} | {r["integer_copy_LB"]} | {r["outside_copy_capacity"]} |')
 lines += ['',
   'Thus both sharedsupports have size≥6. Summing the56size5Hall inequalities gives ΣE C(8−dE,5−dE)≤280. For every nonempty support,6d+C(8−d,5−d)≥28, so the64experts require at least252copies. A33-job cap allows264copies total; sharedcopies require at least12. Equality is forced: both supports have size6, expertcopies=252, and every size5subset contains exactly5expert supports.','',
   'Take one shared6support. Its6five-subsets therefore have30expert-containment incidences. Hall on the6support allows at most5expert supports inside it, and every nonempty expert support is contained in at most5of itsfive-subsets. There can be at most25incidences, contradicting30. This proves33impossible even for fractional support-count LPvariables.','',
   f'Solver cross-check:24shared-size symmetry representatives use LP relaxations;3representatives also use directintegerMILP. Time limit is2seconds/call. Infeasible statuses: {report["solver_infeasible"]}/27; unknown statuses: {len(unknown)}. Exact arithmetic, rather than solver status, is the certificate.','',
   'The34-job witness retains64distinct four-subsets after deleting3complementpairs from all70four-subsets; eachgroup stores32experts andbothsharedobjects. All255Hall inequalities pass. Down uses204WL/chip. With GU186WL, total390WL exceeds the384WL two-block budget by6WL; the hoped-for33-job/198WL composition is unavailable under this contract. The known witness has303MiB/layer with onecopyGU99, and390MiB with GU186. Its total payload is not claimed globally minimal.','',
   'No TRAIN, prospective data/outcomes, heldout orfullFFN was accessed. This does not reject partialcopy/variable-row layouts or other ABIs beyond the fixed full24pagecopy model.']
 (HERE/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS',minmax=34,LP_MILP_infeasible=report['solver_infeasible'],unknown=len(unknown),known34=witness)),flush=True)

if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
