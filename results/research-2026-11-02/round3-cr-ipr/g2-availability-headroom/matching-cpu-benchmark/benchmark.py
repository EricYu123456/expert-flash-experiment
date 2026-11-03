#!/usr/bin/env python3
"""Fresh synthetic matching CPU benchmark; no TRAIN or prospective I/O."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROUND=HERE.parents[1];CAP=HERE.parent/'capacity-kernel'
SEED=202611020805;N=4096;SAMPLES=31;CALLS=16384;RUNS=3
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
def reference_module():
 p=CAP/'kernel_builder.py';s=importlib.util.spec_from_file_location('cpu_matching_reference',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def check_semantics(path):
 ref=reference_module();records=0;fail=0
 with Path(path).open() as f:
  for row in csv.DictReader(f,delimiter='\t'):
   route=[int(row[f'e{i}']) for i in range(6)];mode=int(row['mode'])
   a=ref.assign_route(route,capacity=1);b=ref.assign_route(route,capacity=2)
   expected=a if mode==0 or (mode==2 and a['feasible']) else b
   attempts=2 if mode==2 and not a['feasible'] else 1
   assert int(row['feasible'])==int(expected['feasible']) and int(row['attempts'])==attempts
   assignment=expected.get('assignment',expected.get('partial_assignment',{}))
   assert [int(row[f'g{i}']) for i in range(6)]==[assignment.get(e,-1) for e in route]
   records+=1;fail+=int(mode==0 and not a['feasible'])
 return dict(status='PASS',records=records,cap1_failures_including_deliberate_control=fail,semantic_sha256=sha(path))

def prepare():
 assert not (HERE/'protocol.frozen.json').exists()
 # These files contain only prior synthetic fixtures, never TRAIN or prospective data.
 excludes=[CAP/'matching_cornercases.json',ROUND/'gu-phase-replica/sealed_reference_bridge.json',HERE.parent/'stage-certificate/protocol.frozen.json']
 seen=set()
 for path in excludes:
  data=read(path)
  for row in data.get('records',[]):
   if 'route' in row:seen.add(tuple(sorted(row['route'])))
  if 'frozen_attainment_route' in data:seen.add(tuple(data['frozen_attainment_route']))
 old_count=len(seen);hall=(40,41,42,43,44,45);seen.add(hall)
 rng=random.Random(SEED);routes=[]
 while len(routes)<N:
  route=tuple(sorted(rng.sample(range(64),6)))
  if route not in seen:routes.append(route);seen.add(route)
 (HERE/'synthetic_routes.txt').write_text(''.join(' '.join(map(str,r))+'\n' for r in routes))
 dump(HERE/'synthetic_routes.json',dict(status='frozen_fresh_synthetic_inputs',seed=SEED,routes=routes,
   old_synthetic_fixture_routes_excluded=old_count,exclusion_inputs_sha256={str(p):sha(p) for p in excludes},
   data_scope='SyntheticexpertIDswithcanonicalavailabilityonly.NoTRAIN,prospectiveprompts/routes,outcomesorweightsread.',deliberate_Hall_failure=list(hall)))
 compiler=shutil.which('g++');assert compiler
 flags=['-O3','-std=c++20','-DNDEBUG','-Wall','-Wextra','-Wpedantic']
 cmd=[compiler,*flags,str(HERE/'matching_bench.cpp'),'-o',str(HERE/'matching_bench')]
 built=subprocess.run(cmd,capture_output=True,text=True);(HERE/'compiler.stdout.txt').write_text(built.stdout);(HERE/'compiler.stderr.txt').write_text(built.stderr)
 assert built.returncode==0,built.stderr
 subprocess.run([str(HERE/'matching_bench'),str(HERE/'synthetic_routes.txt'),str(HERE/'semantic_check'),'0','1','0'],check=True)
 semantics=check_semantics(HERE/'semantic_check.semantic.tsv');dump(HERE/'semantic_check.json',semantics)
 symbols=subprocess.check_output(['nm','-C',str(HERE/'matching_bench')],text=True)
 disassembly=subprocess.check_output(['objdump','-d','-C',str(HERE/'matching_bench')],text=True)
 names=['matching<1>','matching<2>','fallback(','control(']
 assembly={name:dict(symbol_present=name in symbols,call_sites=sum('call' in line and name in line for line in disassembly.splitlines())) for name in names}
 assert all(r['symbol_present'] and r['call_sites']>0 for r in assembly.values())
 dump(HERE/'anti_optimization_check.json',dict(status='PASS',assembly=assembly,
   method='No-inlineAPIfunctions,percallcompilerfences,all8-byteoutputsreadintoFNVchecksumaftertimedblockandobservedatomic,untimedassemblycallsandsymbolschecked.',
   hash_outside_timing=True,percall_output_stores_and_cursor_included=True))
 allowed=sorted(os.sched_getaffinity(0));cpu=allowed[-1]
 cpu_model='unknown'
 for line in Path('/proc/cpuinfo').read_text().splitlines():
  if line.startswith('model name'):cpu_model=line.split(':',1)[1].strip();break
 paths=[Path(__file__),HERE/'matching_bench.cpp',HERE/'matching_bench',HERE/'synthetic_routes.txt',HERE/'synthetic_routes.json',HERE/'semantic_check.json',HERE/'anti_optimization_check.json',CAP/'kernel_builder.py']
 protocol=dict(status='frozen_before_timing',utc=datetime.now(timezone.utc).isoformat(),inputs_sha256={str(p):sha(p) for p in paths},
   compile_command=cmd,compiler_version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],platform=platform.platform(),cpu_model=cpu_model,
   allowed_CPU_ids=allowed,benchmark_pinned_CPU=cpu,frequency_or_priority_changed=False,
   samples_per_mode_profile_per_process=SAMPLES,calls_per_sample=CALLS,process_repetitions=RUNS,warmup_batches=4,
   profile_order='Rotate4profiles/4modesbyprocessindex;checksumsorder-independent.',
   profiles=['mixed_fresh_synthetic','cap1_success','cap1_failure','deliberate_Hall_failure'],modes=['cap1','cap2','cap1_then_cap2','harness_control'],
   timing='std::chrono::steady_clock aroundcallbatch;matchinggiven6residentbitmasks. Includesdirectnoinlinecall,cursor/outputstore/compilerfence;excludesroutepreparation,hash,classification,I/O. Rawtimesreported;harnesscontrolnot subtracted.',
   interpretation='Distributionofbatch-average ns/call,notindividualcallP99. HostCPUonly,notfirmware/controller/nativeflashmeasurement.NoTRAIN/prospectiveoutcomeread.')
 dump(HERE/'protocol.frozen.json',protocol)
 print(json.dumps(dict(status='PREPARED',profile_sizes=read(HERE/'semantic_check.receipt.json')['profile_sizes'],CPU=cpu,protocol_sha256=sha(HERE/'protocol.frozen.json'))),flush=True)

def quantile(a,q):
 a=sorted(a);p=(len(a)-1)*q;i=int(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
def stats(a):return dict(n=len(a),minimum=min(a),median=statistics.median(a),p95=quantile(a,.95),p99=quantile(a,.99),maximum=max(a))

def run():
 p=read(HERE/'protocol.frozen.json')
 for path,h in p['inputs_sha256'].items():assert sha(path)==h,path
 all_rows=[];receipts=[];wall=[]
 def pin():os.sched_setaffinity(0,{p['benchmark_pinned_CPU']})
 for run in range(RUNS):
  prefix=HERE/f'run{run}';start=time.monotonic()
  subprocess.run([str(HERE/'matching_bench'),str(HERE/'synthetic_routes.txt'),str(prefix),str(SAMPLES),str(CALLS),str(run)],check=True,preexec_fn=pin)
  wall.append(time.monotonic()-start);receipts.append(read(str(prefix)+'.receipt.json'))
  assert sha(str(prefix)+'.semantic.tsv')==sha(HERE/'semantic_check.semantic.tsv')
  with Path(str(prefix)+'.samples.tsv').open() as f:
   rows=list(csv.DictReader(f,delimiter='\t'))
   for row in rows:row['run']=run;all_rows.append(row)
  print(json.dumps(dict(run=run,status='PASS',wall_seconds=wall[-1],checksum=receipts[-1]['checksum'])),flush=True)
 assert len({r['checksum'] for r in receipts})==1 and len({r['sink'] for r in receipts})==1
 groups=defaultdict(list);checks=defaultdict(set)
 for row in all_rows:
  key=(row['profile'],row['mode']);groups[key].append(int(row['elapsed_ns'])/int(row['calls']))
  checks[(row['profile'],row['mode'],row['sample'])].add(row['checksum'])
 assert all(len(v)==1 for v in checks.values())
 results=[dict(profile=k[0],mode=k[1],batch_mean_ns_per_call=stats(v),calls_per_sample=CALLS,processes=RUNS) for k,v in sorted(groups.items())]
 report=dict(status='PASS',protocol_sha256=sha(HERE/'protocol.frozen.json'),profiles=receipts[0]['profile_sizes'],fresh_synthetic_routes=N,
    semantic_reference=read(HERE/'semantic_check.json'),checksum=receipts[0]['checksum'],sink=receipts[0]['sink'],checksums_identical_across_processes=True,
    per_profile_mode_sample_checksums_identical=True,total_timed_calls=sum(int(r['calls']) for r in all_rows),process_wall_seconds=wall,results=results,
    environment={k:p[k] for k in ['compiler_version','cpu_model','platform','benchmark_pinned_CPU','frequency_or_priority_changed']},
    scope='HostCPUfunctionmicrobenchmarkusingcanonicalsyntheticavailability.NoTRAIN/prospectiveinputoroutcomeread.No firmware/controller timing claim.',
    limitations=['Percentilesaredistributionsofrepeatedbatchmeans,notindividualfunctionlatencytails.','Timesincludeharnesscursor,callboundary,compilerfence,outputstores;hashisoutsidetiming.Controlfloorreportedseparatelyandnotsubtracted.','SyntheticcanonicalgraphstatisticsarenotrealroutingorTRAINmatchingfrequencyestimates.','Frequencyandpriorityunchanged;OS/loadnoiseandcompiler/CPUchoiceslimittransferability.'])
 dump(HERE/'report.json',report)
 lines=['# Deterministic6×6 matching: host CPU microbenchmark','',
   f'Fresh synthetic top6sets: {N}. Eachsidecapacity1 or2 uses the same standard deterministic augmenting-path order as the frozen Python helper. All{report["semantic_reference"]["records"]:,} semantic outputs match, including exact assignments/failed partial results. The separate deliberate Hall failure route is40,41,42,43,44,45.', '',
   f'CPU: {p["cpu_model"]}; pinned logical CPU{p["benchmark_pinned_CPU"]}. Compiler: {p["compiler_version"]}, -O3 C++20. Threeprocessrepetitions,31samples/profile/mode/process,16,384calls/sample. No CPUfrequency orpriority changes.', '',
   '| Synthetic profile | Mode | median ns/call | P95 ns/call | P99 ns/call |','|---|---|---:|---:|---:|']
 for r in results:
  st=r['batch_mean_ns_per_call'];lines.append(f'| {r["profile"]} | {r["mode"]} | {st["median"]:.2f} | {st["p95"]:.2f} | {st["p99"]:.2f} |')
 lines += ['',
   'These percentiles describe repeated batch-average ns/call, not individual-call P99. Timed intervals include the direct no-inline call, route cursor, compiler fences and8-byte output stores; inputmask preparation, classification, file I/O and checksum hashing are outside timing. The harness_control floor is reported separately and is not subtracted.', '',
   f'All{report["total_timed_calls"]:,} timedcalloutputs feed observable checksums; globalchecksum{report["checksum"]} and eachprofile/mode/samplechecksum are identical acrossallthreeprocesses. Disassembly confirms matching/fallback/control calls remain; no-inline functions and compiler fences prevent loop elimination/hoisting.', '',
   'Cap2 is ordinary matching to12capacityslots acrosssixgroups, not extra PE inputcontexts. cap1_then_cap2 first attempts exactcapacity1 and falls back fromscratch onlywhen needed. Profileclassifications are arithmetic on syntheticcanonical supports, not estimates of TRAINorprospective route behavior.', '',
   'No TRAIN, prospective prompts/routes/outcomes, weights or frozen main drivers are read or changed. This is a host CPU software microbenchmark, not firmware, controller, flash or end-to-end scheduling measurement.']
 (HERE/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS',checksum=report['checksum'],results=results)),flush=True)

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
