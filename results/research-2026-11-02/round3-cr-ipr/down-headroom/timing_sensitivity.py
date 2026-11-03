#!/usr/bin/env python3
"""Fixed-map Down work sensitivity only; never an end-to-end replay."""
from pathlib import Path
import json
import hashlib
import numpy as np
import oracle

HERE=Path(__file__).resolve().parent
RUN=HERE/'final-run'
DEST=HERE/'timing-sensitivity'
MODES={
 'nominal_CR9700_copy500':dict(cache_copy_ns=500),
 'ordinary_28us_every_wave_copy500':dict(cache_copy_ns=500,array_mode='ordinary'),
 'conditional_CR10240_copy500_recycle6040':dict(cache_copy_ns=500,cr_ns=10240),
 'nominal_CR9700_copy0':dict(cache_copy_ns=0),
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):return dict(mean_ns=float(a.mean()),p99_ns=float(np.percentile(a,99)),max_ns=int(a.max()))
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def run():
 DEST.mkdir(exist_ok=True);assert not (DEST/'report.json').exists()
 paths=[Path(__file__),RUN/'selection.frozen.json',RUN/'menus.frozen.npz',oracle.PIPE,oracle.TRAIN]
 save(DEST/'plan.frozen.json',dict(status='frozen_fixed_maps_before_sensitivity',inputs_sha256={str(p):sha(p) for p in paths},modes=MODES,
   scope='Only private/shared Down mandatory chip-owner work. Exact frozen fullFFN row maps and TRAIN512 routes; no new fit/selection, no GU or global resource replay.',
   prices='unchanged sealed finite PB pipeline makespan +250ns command +columns*inputcopies/2 +lastpage output bytes/2. Add owner prices on each chip; take max. Difference of relaxations is not a native FFN speedup.',
   cadence='10.24us conditional source-parameter sensitivity, not measured legal timer support; recycle remains6040ns. Ordinary array mode pays cold28us everywave.'))
 pipe=oracle.load_pipeline();sel=json.loads((RUN/'selection.frozen.json').read_text())
 with np.load(oracle.TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3]
 with np.load(RUN/'menus.frozen.npz') as z:maps={k:z[k].copy() for k in z.files}
 records=[];proof=[];arrays={}
 for mode,cfg in MODES.items():
  prices={}
  for columns in [512,1024]:
   for count in [6,12,24]:
    waves=[]
    for begin in range(0,count,4):
     pages=[dict(page_id=f'x.p{j}',owner='x',page_index=j,plane=j%4,block=0,wl=j//4,bytes=16384) for j in range(begin,min(begin+4,count))]
     waves.append(dict(pages=pages,source_order=[p['plane'] for p in pages]))
    out=pipe.simulate(waves,source_policy='fixed',row_bytes=columns,**cfg)
    assert out['feasible'],(mode,columns,count)
    prices[columns,count]=out['makespan_ns']
    proof.append(dict(mode=mode,columns=columns,pages=count,pipeline_makespan_ns=out['makespan_ns'],physical_pages=out['physical_read_pages'],feasible=out['feasible']))
  for GU in ['candidate','RR','g8_fixed']:
   incumbent='original_g2_shared16' if GU=='g8_fixed' else 'original_g4_shared16'
   for copies in [1,4]:
    costs={}
    for label,menu in [('incumbent',incumbent),('partition',sel['winners'][GU])]:
     allW=[]
     for layer in range(3):
      M=maps[menu][layer];P=M*np.array([512]*64+[1024])[:,None]//16384
      table=np.zeros_like(P,dtype=np.int64)
      for e,c in enumerate([512]*64+[1024]):
       for chip in np.flatnonzero(P[e]):
        p=int(P[e,chip]);table[e,chip]=250+c*copies//2+prices[c,p]+(16384//c*4)//2
      A=oracle.incidence(routes[layer]);allW.append(A@table)
     W=np.array(allW);costs[label]=W.max(axis=2)
     arrays[f'{mode}.{GU}.input{copies}.{label}.critical']=costs[label]
     records.append(dict(mode=mode,GU=GU,input_copies=copies,map=label,menu=menu,critical_work=stats(costs[label]),total_work=stats(W.sum(axis=2))))
    records.append(dict(mode=mode,GU=GU,input_copies=copies,comparison='partition_vs_incumbent_work_relaxation',mean_critical_reduction=float(1-costs['partition'].mean()/costs['incumbent'].mean()),p99_critical_reduction=float(1-np.percentile(costs['partition'],99)/np.percentile(costs['incumbent'],99))))
 np.savez_compressed(DEST/'arrays.npz',**arrays)
 save(DEST/'report.json',dict(status='PASS',inputs_sha256={str(p):sha(p) for p in paths},proof=proof,records=records,scope='Mandatory Down chip-owner work relaxation. No end-to-end speedup inference from relative lower bounds; no new capabilities granted.'))
 lines=['# Fixed-map Down work timing sensitivity','','This is a chip-owner work relaxation, not fullFFN replay. All maps remain frozen and no timing case is used to reselect them.','','| Timing model | GU | input copies | critical-work reduction | P99-work reduction |','|---|---|---:|---:|---:|']
 for r in records:
  if 'comparison' in r:lines.append(f'| {r["mode"]} | {r["GU"]} | {r["input_copies"]} | {100*r["mean_critical_reduction"]:.3f}% | {100*r["p99_critical_reduction"]:.3f}% |')
 lines += ['', 'The ordinary model pays 28us for every wave. CR10.24us is conditional cadence sensitivity with recycle6040ns, not evidence that hardware permits changing the phase. Copy0 removes modeled copy cost. The comparisons show whether the same static resource partition reduces mandatory Down work under each model; they do not establish corresponding native fullFFN speedups.']
 (DEST/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS',comparisons=[r for r in records if 'comparison' in r])))
if __name__=='__main__':run()
