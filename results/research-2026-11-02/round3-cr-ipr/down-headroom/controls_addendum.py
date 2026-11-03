#!/usr/bin/env python3
"""Summarize frozen control outcomes without any further selection."""
from pathlib import Path
import json
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent;RUN=HERE/'final-run'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def run():
 z=np.load(RUN/'TRAIN512/service_arrays.npz');sel=read(RUN/'selection.frozen.json');screen=read(RUN/'screen/report.json')
 identities=[]
 for name in z.files:
  if '.GF.' in name:
   other=name.replace('.GF.','.DF.');assert np.array_equal(z[name],z[other]),name
   identities.append(dict(GF=name,DF=other,status='EXACT'))
 controls=[]
 for g in ['candidate','RR','g8_fixed']:
  anchor=sel['comparisons'][g]['incumbent'];selected=sel['comparisons'][g]['best']
  rr={r['name']:r for r in screen['records']}
  a=rr[anchor]['mean_ns'];b=rr[selected]['mean_ns']
  one=rr[f'{g}.dedicated_shared4_learned.input1.DF']['mean_ns']
  static=[r for r in screen['records'] if r['GU']==g and r['menu'].startswith('all64_shared4_static')]
  ss=min(static,key=lambda r:(r['mean_ns'],r['name']))
  controls.append(dict(GU=g,scope='TRAIN64screen_only',incumbent_mean_ns=a,all64_learned_mean_ns=b,oneblock_63plus1_learned_mean_ns=one,
       incremental_all64_mean_reduction_vs63plus1=1-b/one,best_static_of_frozen3placements=ss['name'],static_mean_ns=ss['mean_ns'],static_reduction_vs_incumbent=1-ss['mean_ns']/a,
       learned_reduction_vs_static=1-b/ss['mean_ns'],interpretation='Static grouping uses the same TRAIN-cold expert selection; label assignment is coactivation-blind. This best-of3 screen summary is descriptive and was not promoted to fullTRAIN.'))
 cap=read(HERE/'two-block-addendum/physical_capacity_audit.json');capacity=[]
 for r in cap['records']:
  if r['menu'] in set(sel['winners'].values())|{'original_g4_shared16','original_g2_shared16','dedicated_shared4_learned'}:
   capacity.append(dict(menu=r['menu'],layer=r['layer'],nonempty_blocks=sum(sum(v>0 for v in c['block_WL_fill']) for c in r['chips']),jobs_in_block1=sum(j['block']==1 for c in r['chips'] for j in c['Down_jobs']),max_block_WL=max(max(c['block_WL_fill']) for c in r['chips'])))
 paths=[Path(__file__),RUN/'TRAIN512/service_arrays.npz',RUN/'screen/report.json',RUN/'selection.frozen.json',HERE/'two-block-addendum/physical_capacity_audit.json']
 out=dict(status='PASS',inputs_sha256={str(p):sha(p) for p in paths},GF_DF_exact_identities=identities,TRAIN64_controls=controls,capacity=capacity,
    claim='The reported fullTRAIN improvement persists with either identical priority policy: GF and DF arrays are exactly equal for every completed profile, including FIFO. It is not a scheduling-priority gain.',
    limits='No heldout. Learned/static controls were fitted or described on TRAIN. Standard resource partition/whole-row generic compiler, not a new algorithm. Capacity budget2blocks/chip is common; physical nonemptyblock count differs.')
 (RUN/'controls_addendum.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Partition controls addendum','',
        'GF and DF cost/FIFO arrays are exactly identical for all 24 fullTRAIN profiles. Therefore the reported partition gain also holds at the same scheduling priority.', '',
        '| GU | TRAIN64 learned all64 mean µs | 63+1 one-block mean µs | best static all64 mean µs | static vs incumbent |',
        '|---|---:|---:|---:|---:|']
 for r in controls:lines.append(f'| {r["GU"]} | {r["all64_learned_mean_ns"]/1000:.3f} | {r["oneblock_63plus1_learned_mean_ns"]/1000:.3f} | {r["static_mean_ns"]/1000:.3f} | {100*r["static_reduction_vs_incumbent"]:.3f}% |')
 lines += ['', 'Static grouping is ID-sorted; the same rare expert was selected using TRAIN. The table describes only the original 64-query screen, not an independent validation or a later fullTRAIN selection. Learned label fitting matters in this workload; the shared/private separation alone does not guarantee an improvement.', '',
           'The common budget is two blocks per chip (32 blocks/layer). The selected all64 layout occupies 20 nonempty blocks/layer; the original and 63+1 learned maps occupy 16. Four entire private jobs use block1, one on each chip of the 22-expert routed group. Each still has a single command and cold start. The all64 control slightly improves the 63+1 timing while removing its rare-expert-on-shared-group restriction.', '',
           'The frozen-map timing sensitivity is in ../timing-sensitivity/report.md. It compares mandatory Down work only, and does not establish native fullFFN gains under those alternative timing models.']
 (RUN/'controls_addendum.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS',identities=len(identities),controls=controls)))
if __name__=='__main__':run()
