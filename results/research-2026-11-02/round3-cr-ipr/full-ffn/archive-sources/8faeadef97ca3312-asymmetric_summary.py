#!/usr/bin/env python3
"""Paired effects and provenance for immutable asymmetric-Down outputs."""
from pathlib import Path
import json,hashlib
import numpy as np

OUT=Path(__file__).resolve().parent;DEST=OUT/'asymmetric-down'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def compare(name,a,other,b):
    assert a.shape==b.shape==(3,512) and np.all(a>=0) and np.all(b>=0)
    aa=a.sum(axis=0);bb=b.sum(axis=0)
    return dict(candidate=name,baseline=other,paired_layer_events=1536,aligned_three_layer_inputs=512,
                mean_gain_pct=100*(1-float(a.mean()/b.mean())),event_p99_gain_pct=100*(1-float(np.percentile(a,99)/np.percentile(b,99))),
                three_layer_sum_p99_gain_pct=100*(1-float(np.percentile(aa,99)/np.percentile(bb,99))),
                slower_layer_events=int(np.count_nonzero(a>b)),equal_layer_events=int(np.count_nonzero(a==b)),
                slower_aligned_three_layer_inputs=int(np.count_nonzero(aa>bb)),
                candidate_mean_ns=float(a.mean()),baseline_mean_ns=float(b.mean()))

report=json.loads((DEST/'TRAIN512/report.json').read_text());selection=json.loads((DEST/'selection.frozen.json').read_text())
ar=load(DEST/'TRAIN512/service_arrays.npz');shaped=load(OUT/'shaping-addendum/TRAIN512/service_arrays.npz');prior=load(OUT/'policy-addendum/TRAIN512/service_arrays.npz')
comparisons=[]
for copies in [1,4]:
    for pr in ['GF','DF']:
        ca=f'candidate.{selection["winners"]["candidate"]}.input{copies}.{pr}'
        rr=f'RR.{selection["winners"]["RR"]}.input{copies}.{pr}'
        g8=f'g8_fixed.{selection["winners"]["g8_fixed"]}.input{copies}.{pr}'
        old=f'strict500.g16_equal.input{copies}.{pr}.a0.candidate_IPR_uniform_split10'
        for other,arrays in [(rr,ar),(g8,ar),(old,shaped)]:comparisons.append(compare(ca,ar[ca+'.cost'],other,arrays[other+'.cost']))
old_rr='first_feasible500.g16_equal.input1.GF.a0.RR';rr=f'RR.{selection["winners"]["RR"]}.input1.GF'
comparisons.append(compare(rr,ar[rr+'.cost'],old_rr,prior[old_rr+'.cost']))
old_report=json.loads((OUT/'shaping-addendum/TRAIN512/report.json').read_text())
new=next(r for r in report['records'] if r['GU']=='candidate' and r['input_copies']==1 and r['priority']=='GF')
old=next(r for r in old_report['records'] if r['variant']=='candidate_IPR_uniform_split10' and r['Down']=='g16_equal' and r['copies']==1 and r['priority']=='GF')
attribution={}
for key in ['command','input_ONFI_bytes','output_ONFI_bytes','output_PCIe_bytes','pipeline_pages','pipeline_physical_read_pages','pipeline_ecc_bytes','pipeline_cold_waves','pipeline_cr_waves','array_ECC_busy_ns','command_busy_ns','input_ONFI_busy_ns']:
    attribution[key]=dict(wide_g16_Down=old['counts_and_work_per_valid_query'].get(key),compact_g4_Down=new['counts_and_work_per_valid_query'].get(key))
result=dict(status='complete_TRAIN_paired_summary',analysis_source_sha256=sha(__file__),
            input_sha256={str(p):sha(p) for p in [DEST/'protocol.frozen.json',DEST/'selection.frozen.json',DEST/'screen/report.json',DEST/'TRAIN512/report.json',DEST/'TRAIN512/service_arrays.npz',OUT/'shaping-addendum/TRAIN512/report.json',OUT/'policy-addendum/TRAIN512/report.json']},
            comparisons=comparisons,resource_work_attribution_candidate_same_GU=attribution,
            inference='Compact privateDown improves both candidateGU andRR byabout20.7%; remainingcandidateGUlayoutincrementoverequallyreoptimizedRR isabout0.4%. Attribute themajorresult toprivateoperandsharding/coldstarts,notsharedlayoutnovelty.',
            selection_disclosure='Same13Downmenusforall3GUfamilies onfrozenTRAIN64indices/3layers; g4oldTRAINlabels selectedforcandidateandRR,g2oldlabels forg8fixed. All512remainpreviouslyobservedTRAIN, nottest.',
            limits=['No measuredhardware/nativeABI/inferencespeed/heldoutclaim.','FinitePB/copy500/source/drop/readcadence areconditional; genericcompilercandidatealias remains0%gain.','G1privateDownstrict12fullwaves infeasible andretained; no rescue.','Fixedg8source0123 needsnonewPBpermutationbutGUshaping/drop/controllercontinuationremainconditional.','Countsandresourcework explainmechanism butdo notaddlinearlytoquerylatency.','Alloptimizedtracespeak128B outputFIFO underpageburstgranularity;64Binvalid.','Selectedwinneroneblockaddresscertificate supplementscommon2blockscreenbudget; notacapacityconfound.'])
(DEST/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Wide common-input GU and compact private-input Down','','TRAINonly:32prompts ×16decodeinputs ×3MoElayers. Same13Downmenus were screenedfor everyGUfamily on64fixedindices/layer, thenwinnersfrozenbeforefull512TRAINreplay. Thisis aconditionalservice-modelresult, nothardwaremeasurement.','','| GU / selectedDown | copies | priority | meanus | P99us | validqueries | peakFIFO |','|---|---:|---|---:|---:|---:|---:|']
for r in report['records']:
    s=r['cost_ns'];lines.append(f"| {r['GU']} / {r['Down_menu']} | {r['input_copies']} | {r['priority']} | {s['mean']/1000:.6f} | {s['p99']/1000:.6f} | {r['completed_queries']}/1536 | {r['FIFO_peak_bytes']['maximum']:.0f}B |")
lines+=['','Candidate+g4 meanimproves20.7205% overthesameGUwithg16Down atinput1/GF. RR+g4 similarlyimproves20.6545% overitsg16Downcontrol. CandidateoverequallyreoptimizedRR isonly0.4016%input1 and0.3942%input4;378events areslower and533equal. Genericcompilerexactlytiescandidate.','','TheprivateDownC512vectorstayscompleteon everyjob. g2/g4 full0123waves arelegalunderstrict500nsPB model;g1twelve-waveburstisinfeasible. Coldstarts, commandcount andfullinputcopiesdecline; usefulcoefficients/readbytes andoutputsareunchanged. Resourceworkcounters areattributionevidence, notanadditivelatencydecomposition.','','Alloptimizedtracesrequire128B underdeclaredpagecompletionburstgranularity;64B isinvalid. Everyselectedfullmodelconfigurationhas onecopy andfitsone192WLblock/chip in theadditionaladdresscertificate. Thecomparisons retained thesame2blockcapacitybudgetduringselection.','','Themajorresultis operand-awareasymmetricsharding underCR, notanewsharedlayoutalgorithm. OldTRAINlearnedgroups areessentialstrongcontrols andcarrynoalgorithmnoveltyclaim. Noheldout,measurednativeperformanceorpublicationreadinessclaim.']
(DEST/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='complete',comparisons=len(comparisons),attribution=attribution),indent=2))
