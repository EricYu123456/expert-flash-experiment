#!/usr/bin/env python3
"""Paired summary across immutable primary, g8 and shaping campaigns."""
from pathlib import Path
import hashlib,json
import numpy as np

OUT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(folder):
    with np.load(folder/'service_arrays.npz') as z:arrays={k:z[k].copy() for k in z.files}
    return json.loads((folder/'report.json').read_text()),arrays
def metric(a,b):
    return dict(mean_gain_pct=100*(1-float(a.mean()/b.mean())),p99_gain_pct=100*(1-float(np.percentile(a,99)/np.percentile(b,99))),
                candidate_mean_ns=float(a.mean()),baseline_mean_ns=float(b.mean()),
                candidate_p99_ns=float(np.percentile(a,99)),baseline_p99_ns=float(np.percentile(b,99)),
                slower_fraction=float(np.mean(a>b)),slower_queries=int(np.count_nonzero(a>b)),
                equal_queries=int(np.count_nonzero(a==b)))
def compare(name,a,af,other,b,bf):
    ok=(a>=0)&(b>=0);sums=ok.all(axis=0)
    result=dict(candidate=name,baseline=other,requested_event_queries=int(a.size),
                candidate_infeasible_queries=int(np.count_nonzero(a<0)),baseline_infeasible_queries=int(np.count_nonzero(b<0)),
                jointly_feasible_event_queries=int(ok.sum()),validity_notice='Subset-only statistics are not all-query performance when either operand is infeasible.',
                jointly_valid_FIFO_event_queries={str(k):int(np.count_nonzero(ok&(af<=k)&(bf<=k))) for k in [64,128,256,2048]})
    if ok.any():result['event_metrics']=metric(a[ok],b[ok])
    result['jointly_feasible_three_layer_inputs']=int(sums.sum())
    if sums.any():result['three_layer_sum_metrics']=metric(a[:,sums].sum(axis=0),b[:,sums].sum(axis=0))
    return result


primary,pa=load(OUT/'TRAIN512');g8,ga=load(OUT/'g8-addendum/TRAIN512');shaped,sa=load(OUT/'shaping-addendum/TRAIN512')
assert primary['queries_per_layer']==512 and g8['limit']==shaped['limit']==512
comparisons=[]
for profile in ['strict100','cold_reset500']:
    for down in ['g16_equal','g12_cyclic']:
        for copies in [1,4]:
            for priority in ['GF','DF']:
                prefix=f'{profile}.{down}.input{copies}.{priority}.a0.'
                candidate=prefix+'candidate'
                for kind in ['identity','learned']:
                    baseline=prefix+'g8_'+kind
                    comparisons.append(compare(candidate,pa[candidate+'.cost'],pa[candidate+'.fifo'],baseline,ga[baseline+'.cost'],ga[baseline+'.fifo']))
for r in shaped['records']:
    if r['variant']!='candidate_IPR_uniform_split10':continue
    candidate=r['name'];prefix=candidate.rsplit('.',1)[0]+'.'
    baselines=[(prefix+kind,sa) for kind in ['matching_layout_IPR_uniform_split10','matching_layout_IPR_fill_only','RR_IPR_uniform_split10']]
    oldprefix=prefix.replace('strict500.','cold_reset500.')
    baselines += [(oldprefix+kind,pa) for kind in ['candidate','sameWL_matching_keep_hot']]
    for baseline,arrays in baselines:
        comparisons.append(compare(candidate,sa[candidate+'.cost'],sa[candidate+'.fifo'],baseline,arrays[baseline+'.cost'],arrays[baseline+'.fifo']))
inputs=[OUT/'TRAIN512/report.json',OUT/'TRAIN512/service_arrays.npz',OUT/'g8-addendum/TRAIN512/report.json',OUT/'g8-addendum/TRAIN512/service_arrays.npz',OUT/'shaping-addendum/TRAIN512/report.json',OUT/'shaping-addendum/TRAIN512/service_arrays.npz']
result=dict(status='complete_paired_TRAIN_summary',analysis_source_sha256=sha(__file__),input_sha256={str(p):sha(p) for p in inputs},
            scope='1536layer-events from the same512aligned TRAIN inputs. Three-layer sums use512inputs, not1536independentprompts; only32TRAIN prompts. No heldout.',
            comparisons=comparisons,generic_compiler='Exact candidate alias:0%gain in everyprofile; no newoptimizer attribution.',
            limits=['Strict100 andcopy500 are unmeasured conditionalcopyparameters, selected after pipeline sensitivity.','Every128B peak page-output burst violates64BFIFO; no backpressure rescue.','All virtual full-model controls have equal2blocks/chip reservation; oldwhole-rowpadding is explicit.','Fixed9.7us producer sensitivity is not the only possiblehardwaremodel: a controllably extended10us phase may outperformshaping; timer/analog feasibility remainsunverified.','Readshaping is sameuniform structuralrule for allkernels; forcedone-split is not a globallyoptimal baselinepolicy. RRfill-only/coldreset may be a strongerfollowup control.','Frozenmain Recipes key excludesblock, so it is not a genericcrossblockcache. Everyactualprimaryrecipe staysblock0; static Down relocation is a whole freshcold job and does not reusecrossblocktiming. Newshapingcache includescompleteaddresses/readpages/configuration.'])
(OUT/'paired_all_controls.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Paired conditional FFN controls','','TRAIN first3layers ×512inputs;32prompts. Genericcompiler exactlytiescandidate. Statistics with fewer than1536jointlyvalid events are subset-only, with infeasible queries retained.','','| Candidate | Baseline | Joint events | Event mean gain | Three-layer P99 gain | Slower events |','|---|---|---:|---:|---:|---:|']
for c in comparisons:
    m=c.get('event_metrics');s=c.get('three_layer_sum_metrics')
    if not m:continue
    p99=f"{s['p99_gain_pct']:.4f}%" if s else 'unavailable'
    lines.append(f"| {c['candidate']} | {c['baseline']} | {c['jointly_feasible_event_queries']}/1536 | {m['mean_gain_pct']:.4f}% | {p99} | {m['slower_queries']} |")
lines+=['','Everytrace fails64B under the declared page-completion burst model. Common-GU chain controls peak at128B; some input4/DF separate-GU andlegacywhole-row cases require256B. Validity must follow each recorded FIFOcount. Realistic timing/source/FIFO support remains conditional. Native measured latency, generalization and publication readiness are not established.','','Readshaping is bounded by the fixed9.7us cadence assumption; conditional extendedphase/cadence controls can be faster. Uniformone-split kernels are controls, not globalnativeoptima.']
(OUT/'paired_all_controls.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='complete',comparisons=len(comparisons))))
