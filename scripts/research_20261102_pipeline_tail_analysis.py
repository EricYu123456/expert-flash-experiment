#!/usr/bin/env python3
"""First-three-layer tail attribution; never extrapolates P99 to full model."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_granite4_pipeline import SOURCE,EXPECTED,profiles
from research_20261102_pipeline_joint_replay import rows_from_labels
from run_corpus_traces import ROOT,sha256

BASE=ROOT/'results/research-2026-11-02/round2'
DEST=BASE/'pipeline-tail-analysis'


def summary(x):
    return dict(n=len(x),mean_ns=float(x.mean()),p95_ns=float(np.percentile(x,95)),p99_ns=float(np.percentile(x,99)),
        minimum_ns=int(x.min()),maximum_ns=int(x.max()),variance_ns2=float(x.var()))


def main():
    DEST.mkdir(parents=True,exist_ok=True);assert sha256(SOURCE)==EXPECTED;d=json.loads(SOURCE.read_text())
    route=Path(d['training_routes']);assert sha256(route)==d['training_routes_sha256']
    with np.load(route) as f:routes=f['routes'][:32,:64].copy();layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
    assert len(ids)==32 and all(s.startswith('train_') for s in ids)
    jpath=BASE/'pipeline-joint/joint_work_screen.json';j=json.loads(jpath.read_text())
    sources={'source':(BASE/'pipeline-joint-replay/zero/schedule_costs.npz','source'),
        'matched_phase':(BASE/'pipeline-joint-replay/zero/schedule_costs.npz','matched_phase_best'),
        'joint':(BASE/'pipeline-joint-replay/zero/schedule_costs.npz','joint_best'),
        'wave':(BASE/'pipeline-gu-screen/full/schedule_costs.npz','gu_wave_rotations'),
        'opposite':(BASE/'pipeline-joint-replay/opposite-full/schedule_costs.npz','g8_opposite')}
    latency={};provenance={}
    for name,(path,prefix) in sources.items():
        with np.load(path) as f:
            np.testing.assert_array_equal(f['indices'],np.arange(2048));latency[name]=f[prefix+'__ready__gate_first'].copy()
        assert latency[name].shape==(3,2048);provenance[name]=dict(path=str(path),sha256=sha256(path))
    home_m=[];bounds={name:[] for name in latency};per_layer=[]
    for li,layer in enumerate(layers[:3]):
        ps=[p for p in d['phases'] if p['layer']==layer];selected=routes[:,:,layers.index(layer)].reshape(-1,d['active'])
        gu=np.asarray(ps[0]['rows']);down=np.asarray(ps[1]['rows']);home=(gu[:,0]>0).astype(int)
        n=home[selected].sum(axis=1);m=np.maximum(n,6-n);home_m.append(m)
        wave=np.asarray(next(c for c in ps[0]['candidates'] if c['label']=='wave_rotations')['rows'])
        qq=next(q for q in j['results'] if q['layer']==layer)
        opposite=np.zeros_like(down)
        for e,h in enumerate(home):opposite[e,8:] = 192 if h else 0;opposite[e,:8] = 192 if not h else 0
        shapes={'source':(gu,down),'matched_phase':(gu,rows_from_labels(qq['matched_phase_best']['labels'],1536)),
            'joint':(gu,rows_from_labels(qq['joint_best']['labels'],1536)),'wave':(wave,down),'opposite':(gu,opposite)}
        record=dict(layer=layer,home_max_occupancy_counts={str(k):int((m==k).sum()) for k in range(3,7)},families={})
        for name,(g,dd) in shapes.items():
            gw=profiles(g,1536)['waves'][selected].sum(axis=1);dw=profiles(dd,512)['waves'][selected].sum(axis=1)
            bound=(gw+dw).max(axis=1)*28000;bounds[name].append(bound)
            if name=='opposite':np.testing.assert_array_equal(bound,(12+m)*28000)
            x=latency[name][li];coef=np.polyfit(m,x,1)
            record['families'][name]=dict(latency=summary(x),joined_nand_bound=summary(bound),
                native_schedule_minus_nand_bound=summary(x-bound),
                latency_slope_ns_per_extra_home_expert=float(coef[0]),
                pearson_home_m_vs_latency=float(np.corrcoef(m,x)[0,1]),
                routing_bins={str(k):summary(x[m==k]) for k in range(3,7) if np.any(m==k)})
        per_layer.append(record)
    home_m=np.asarray(home_m);N=2048;ntop=int(np.ceil(.01*N));aggregate={};tailsets={}
    for name,x in latency.items():
        total=x.sum(axis=0);top=np.argsort(total,kind='stable')[-ntop:];tailsets[name]=set(top.tolist())
        cov=np.cov(x,bias=True);varsum=float(total.var());assert np.isclose(varsum,cov.sum())
        per_top=[set(np.argsort(row,kind='stable')[-ntop:].tolist()) for row in x]
        aggregate[name]=dict(paired_three_layer_sum=summary(total),
            sum_of_individual_layer_p99_ns=float(sum(np.percentile(row,99) for row in x)),
            covariance_ns2=cov.tolist(),sum_of_layer_variances_ns2=float(np.trace(cov)),
            cross_layer_covariance_contribution_ns2=float(cov.sum()-np.trace(cov)),
            variance_identity='Var(sum layers)=sum Var(layers)+2sum pairwise Cov; numerically verified.',
            top1_exact_rank_count=ntop,top1_route_indices=top.tolist(),
            top1_distinct_prompts=len(set((top//64).tolist())),
            top1_mean_original_home_m_by_layer=home_m[:,top].mean(axis=1).tolist(),
            top1_extreme_home_m_ge5_occurrences=int((home_m[:,top]>=5).sum()),
            per_layer_top1_intersection_counts={f'{a}-{b}':len(per_top[a]&per_top[b]) for a,b in [(0,1),(0,2),(1,2)]})
    source_top=tailsets['source'];paired_tail={}
    for name,x in latency.items():
        if name=='source':continue
        delta=x.sum(axis=0)-latency['source'].sum(axis=0)
        paired_tail[name]=dict(on_source_sum_top1=summary(delta[sorted(source_top)]),
            overlap_with_source_sum_top1=len(tailsets[name]&source_top),
            on_each_source_layer_top1={str(li):summary((x[li]-latency['source'][li])[np.argsort(latency['source'][li],kind='stable')[-ntop:]]) for li in range(3)})
    npz=DEST/'prefix3_vectors.npz';np.savez_compressed(npz,home_m=home_m,**{name:arr for name,arr in latency.items()},
        **{'bound_'+name:np.asarray(arr) for name,arr in bounds.items()})
    result=dict(status='descriptive train-only first3layer tail attribution; no full-model extrapolation',created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_plan_sha256=EXPECTED,root_joint_sha256=sha256(jpath),training_routes_sha256=sha256(route),prompt_ids=ids,
        source_arrays=provenance,per_layer=per_layer,paired_three_layer_aggregation=aggregate,paired_source_tail=paired_tail,
        mechanism=dict(opposite='With K6 and original GUhome occupancy m=max(nA,6-nA), joined NAND work is exactly(12+m)waves. GU3wave jobs and down2wave jobs use opposite chip halves. Extra total work reduces sensitivity to routing imbalance; it does not shorten an expert own chain.',
            g12_best_possible_given_gu_home='Even query-adaptive omitted colors give joined minima m3:14,m4:15,m5:18,m6:21waves. Opposite gives15,16,17,18 respectively. Thus for m>=5 its load bound beats any g12omitted-color plan under these fixed g8GU homes, while balanced queries can lose.',
            opposite_additional_precedence_bound='The last GU expert on the busier half cannot finish before3m waves and still needs its2wave down job. Thus the resourceDAG lower bound is max(12+m,3m+2), giving15,16,17,20waves for m3..6. The m6 load-only18wave bound is unattainable; do not confuse it with feasible tail latency.',
            wave='GU max native-job depth decreases3to2waves and expert uses all16chips, trading GUtotal24to25waves. Own-expert GU/down chain decreases4to3waves; this mechanism differs from opposite stage complementarity.',
            caution='A joined-work lower bound does not establish an achievable schedule. The event results supply separate feasibility evidence; no new hardware or optimal scheduler is claimed.'),
        recommendation=dict(full40train='Yes: evaluate fixed source/wave/opposite families across40layers before dismissing tail or claiming a robust mapping. Do not fit a new tail mapping first.',
            efficient_gate='First compute all32x64x40 exact work/depth and routing-occupancy diagnostics; then a predeclared prompt-stratified schedule screen across40layers. Only if it persists run all2048routes for stable quantiles.',
            necessary_evidence=['Per-layer P99 must improve broadly, not only the three-layer sum through covariance.',
                'Report full40paired routed-FFN latency sum, mean/P95/P99 and each-layer tails; percentile sums are not a full-model P99.',
                'Use prompt-block uncertainty and unseen routing distributions only after all family/profile choices freeze.',
                'Preserve same scheduler, buffer/context and actual increased wave/page/command costs. No host I/O/full-model inference claim.',
                'Compare mean-tail Pareto frontier: current joint improves mean while opposite/wave improve tail. A mean winner is not a universal winner.'],
            scope='No full40schedule or new fit is run here; current evidence covers three layers,32training prompts only.'),
        limitations=['P99 uses2048correlated decode positions from32training prompts; descriptive tail, not an independent test.',
            'Exact-rank top1uses21observations and deterministic tie order; quantile thresholds can contain more tied events.',
            'The three-layer routing sequence and covariance do not establish full40-layer behavior.',
            'All latencies remain conditional native-job service simulations with activation/combine0 and uncalibrated PE throughput.'],
        source_sha256=sha256(Path(__file__)),vectors_sha256=sha256(npz))
    path=DEST/'prefix3.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(aggregate=aggregate,paired_source_tail=paired_tail),indent=2))


if __name__=='__main__':main()
