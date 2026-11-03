#!/usr/bin/env python3
"""Train-only best-mean g16 horizontal-pair NAND oracle, no host computation.

Integer-weight maximum perfect matching exactly maximizes co-selected pairs.
All inactive bundle weights are read. Every candidate non-NAND cost is then
set to zero as an intentionally favorable impossibility screen. No new operator,
output contract, alpha quantization, host assembly or correctness is asserted.
"""
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from research_20261102_projection_study import evaluate
from run_corpus_traces import ROOT, sha256

DEST=ROOT/'results/research-2026-11-02/round2/horizontal-bundle-oracle'
STRONG=ROOT/'results/research-2026-11-02/round2/unbalanced-group'


def stats(v):
    return dict(mean_ns=float(v.mean()),p95_ns=float(np.percentile(v,95)),
        p99_ns=float(np.percentile(v,99)),min_ns=int(v.min()),max_ns=int(v.max()))


def independent_milp(edge_weight,expected,perfect=True):
    E=len(edge_weight);edges=list(itertools.combinations(range(E),2));N=len(edges)
    row=np.asarray([e for pair in edges for e in pair]);col=np.repeat(np.arange(N),2)
    A=csc_matrix((np.ones(2*N),(row,col)),shape=(E,N))
    w=np.asarray([edge_weight[a,b] for a,b in edges],dtype=float)
    result=milp(-w,integrality=np.ones(N),bounds=Bounds(0.,1.),
        constraints=LinearConstraint(A,np.ones(E) if perfect else np.zeros(E),np.ones(E)),options=dict(time_limit=30.,mip_rel_gap=0.))
    assert result.success,result.message
    assert abs(-result.fun-expected)<1e-7
    assert abs(-result.mip_dual_bound-expected)<1e-7
    return dict(status=int(result.status),message=result.message,score=int(round(-result.fun)),
        dual_bound=float(-result.mip_dual_bound),mip_gap=float(result.mip_gap),nodes=int(result.mip_node_count))


def main():
    DEST.mkdir(parents=True,exist_ok=True);models=[]
    extra_path=ROOT/'results/research-2026-11-02/round2/adversarial-design/mixed-columns/screen.json'
    extra=json.loads(extra_path.read_text())
    for model in ['granite','qwen3']:
        source=STRONG/(model+'.screen.strong.frozen.json');d=json.loads(source.read_text())
        source_sha=sha256(source);route=Path(d['training_routes']);assert sha256(route)==d['training_routes_sha256']
        with np.load(route) as f:
            routes=f['routes'][:32,:64].copy();layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
        assert len(ids)==32 and all(s.startswith('train_') for s in ids)
        E=d['experts'];K=d['active'];phases=[];baseline_phases=[]
        candidates=[('screen_strong',source,d)]
        full=STRONG/(model+'.full.strong.frozen.json')
        if full.exists():candidates.append(('full_strong',full,json.loads(full.read_text())))
        for p in d['phases']:
            selected=routes[:,:,layers.index(p['layer'])].reshape(-1,K)
            options=[]
            for label,path,doc in candidates:
                q=next(q for q in doc['phases'] if q['layer']==p['layer'] and q['name']==p['name'])
                rows=np.asarray(q['chosen_rows']);cost=evaluate(selected,rows,p['columns'],p['common'])
                options.append((float(cost['model_ns'].mean()),label,path,rows,cost))
            if model=='qwen3' and p['name']=='down':
                em=next(m for m in extra['models'] if m['model']==model)
                ep=next(q for q in em['phases'] if q['layer']==p['layer'])
                g=np.asarray(ep['winners']['whole_rows_03']['g']);rows=np.zeros((E,16),dtype=np.int64)
                for e in range(E):rows[e,:8] = p['total_rows']//8 if g[e]==3 else 0;rows[e,8:] = p['total_rows']//8 if g[e]==0 else 0
                cost=evaluate(selected,rows,p['columns'],p['common'])
                options.append((float(cost['model_ns'].mean()),'shared_binary_refit_03',extra_path,rows,cost))
            _,label,path,rows,base=min(options,key=lambda q:q[0])
            baseline_phases.append(dict(layer=p['layer'],name=p['name'],mean_ns=float(base['model_ns'].mean())))
            if p['name']!='down':continue
            assert np.all(np.count_nonzero(rows,axis=1)==8)
            R,C=p['total_rows'],p['columns'];assert R%16==0 and E%2==0 and K%2==0
            shard_bytes=R//16*C;pair_pages=(2*shard_bytes+16383)//16384;pair_waves=(pair_pages+3)//4
            coact=np.zeros((E,E),dtype=np.int64)
            for s in selected:
                for a,b in itertools.combinations(sorted(s.tolist()),2):coact[a,b]+=1;coact[b,a]+=1
            graph=nx.Graph();graph.add_nodes_from(range(E))
            graph.add_weighted_edges_from((a,b,int(coact[a,b])) for a,b in itertools.combinations(range(E),2))
            pairs=sorted(tuple(sorted(e)) for e in nx.max_weight_matching(graph,maxcardinality=True,weight='weight'))
            assert len(pairs)==E//2 and sorted(e for pair in pairs for e in pair)==list(range(E))
            score=sum(int(coact[a,b]) for a,b in pairs);milp_result=independent_milp(coact,score)
            pair_id=np.empty(E,dtype=np.int64)
            for i,(a,b) in enumerate(pairs):pair_id[a]=pair_id[b]=i
            triggered=np.asarray([len(set(pair_id[s])) for s in selected],dtype=np.int64)
            coselected=K-triggered;assert int(coselected.sum())==score
            # Literal whole-supermatrix read, including inactive expert weights.
            pages=triggered*pair_pages;waves=triggered*pair_waves
            flash=np.maximum(waves*28000,pages*2560)
            assert np.all(flash==triggered*pair_waves*28000)
            baseline=stats(base['model_ns']);baseline_flash=stats(base['flash_ns']);candidate=stats(flash)
            single_pages=(shard_bytes+16383)//16384;single_waves=(single_pages+3)//4
            frequency=np.bincount(selected.ravel(),minlength=E)
            benefit=pair_waves*coact-(pair_waves-single_waves)*(frequency[:,None]+frequency[None,:])
            optional_graph=nx.Graph();optional_graph.add_nodes_from(range(E))
            optional_graph.add_weighted_edges_from((a,b,int(benefit[a,b])) for a,b in itertools.combinations(range(E),2))
            optional_pairs=sorted(tuple(sorted(e)) for e in nx.max_weight_matching(optional_graph,maxcardinality=False,weight='weight'))
            optional_score=sum(int(benefit[a,b]) for a,b in optional_pairs)
            optional_milp=independent_milp(benefit,optional_score,False)
            optional_id=np.full(E,-1,dtype=np.int64)
            for i,(a,b) in enumerate(optional_pairs):optional_id[a]=optional_id[b]=i
            optional_wave=[];optional_page=[]
            for s in selected:
                active_pair={int(optional_id[e]) for e in s if optional_id[e]>=0}
                singleton=sum(optional_id[e]<0 for e in s)
                optional_wave.append(pair_waves*len(active_pair)+single_waves*singleton)
                optional_page.append(pair_pages*len(active_pair)+single_pages*singleton)
            optional_wave=np.asarray(optional_wave);optional_page=np.asarray(optional_page)
            assert int((K*single_waves-optional_wave).sum())==optional_score
            optional_flash=np.maximum(optional_wave*28000,optional_page*2560)
            assert np.all(optional_flash==optional_wave*28000)
            optional_stats=stats(optional_flash)
            oracle_ns=(K//2)*pair_waves*28000
            phases.append(dict(layer=p['layer'],name='down',rows=R,columns=C,experts=E,active=K,
                baseline_label=label,baseline_source=str(path),baseline_source_sha256=sha256(path),
                baseline_group_width=8,baseline_phase=baseline,baseline_nand=baseline_flash,
                pair_layout=dict(chips=16,experts_per_horizontal_bundle=2,per_expert_chip_bytes=shard_bytes,
                    horizontal_matrix_rows=R//16,horizontal_matrix_columns=2*C,pages_per_bundle_per_chip=pair_pages,
                    waves_per_bundle_per_chip=pair_waves,pairs=pairs,same_total_useful_bytes=True,same_total_pages=(E//2)*pair_pages==E*((shard_bytes+16383)//16384)),
                maximum_weight_perfect_matching=dict(integer_coactivation_score=score,mean_coselected_pairs=score/len(selected),
                    independent_scipy_highs_milp=milp_result),
                favorable_candidate_nand_only=candidate,
                optional_pair_stronger_control=dict(pairs=optional_pairs,singleton_experts=np.flatnonzero(optional_id<0).tolist(),
                    edge_benefit_formula='pair_waves*coact-(pair_waves-single_waves)*(freq_a+freq_b)',
                    integer_saved_wave_score=optional_score,independent_scipy_highs_milp=optional_milp,
                    favorable_candidate_nand_only=optional_stats,
                    mean_headroom_with_all_candidate_non_nand_costs_zero_ns=baseline['mean_ns']-optional_stats['mean_ns']),
                mean_headroom_with_all_candidate_non_nand_costs_zero_ns=baseline['mean_ns']-candidate['mean_ns'],
                mean_nand_saving_vs_baseline_ns=baseline_flash['mean_ns']-candidate['mean_ns'],
                unattainable_query_dependent_pair_oracle_ns=oracle_ns,
                optimality_scope='Exact training-mean NAND minimum over static perfect pairs in this g16 B2 horizontal family. Even allowing different pairing on each chip cannot improve this mean optimum: E[max_c work_c]>=max_c E[work_c], and the same optimal matching on every chip attains equality. This is not a p99-optimality claim or a bound for arbitrary bundle sizes/shard geometries.',
                limitations=['All inactive bundle experts are truly read/computed; only NAND time charged here.',
                    'Every candidate command, input/output, alpha weighting/quantization, zero-input assembly, partial summation and PE compute cost is optimistically zero.',
                    'No existing-output correctness, A8 quality or native ABI capability established by this oracle.']))
            print(model,p['layer'],'mean-pairs',score/len(selected),'baseline NAND/full',baseline_flash['mean_ns'],baseline['mean_ns'],
                'best pair NAND',candidate['mean_ns'],'zero-overhead headroom',baseline['mean_ns']-candidate['mean_ns'],
                'optional-pair NAND',optional_stats['mean_ns'],'optional headroom',baseline['mean_ns']-optional_stats['mean_ns'],flush=True)
        baseline_total=sum(p['mean_ns'] for p in baseline_phases)
        best_candidate_total=baseline_total+sum(p['favorable_candidate_nand_only']['mean_ns']-p['baseline_phase']['mean_ns'] for p in phases)
        optional_total=baseline_total+sum(p['optional_pair_stronger_control']['favorable_candidate_nand_only']['mean_ns']-p['baseline_phase']['mean_ns'] for p in phases)
        assert sha256(source)==source_sha
        models.append(dict(model=model,training_routes=str(route),training_routes_sha256=sha256(route),
            source_plan=str(source),source_plan_sha256=source_sha,prompt_ids=ids,steps=64,
            phases=phases,baseline_all_examined_phases=baseline_phases,
            baseline_routed_ffn_screen_ns=baseline_total,
            candidate_routed_ffn_with_all_down_non_nand_free_ns=best_candidate_total,
            favorable_whole_ffn_mean_gain_pct=100*(1-best_candidate_total/baseline_total),
            optional_pair_candidate_routed_ffn_with_all_down_non_nand_free_ns=optional_total,
            optional_pair_favorable_whole_ffn_mean_gain_pct=100*(1-optional_total/baseline_total),
            optional_pair_decision='No-go even allowing optional pairs/singletons with all candidate non-NAND costs free.' if optional_total>=baseline_total else 'Optional pairs retain headroom; additional existing-command cost gate needed.',
            decision='No-go for this mean-latency family: even exact optimal pairs with all candidate non-NAND costs removed lose.' if best_candidate_total>=baseline_total else 'Headroom remains; native cost/quality gate needed.'))
    src=Path(__file__);nxsrc=inspect.getsource(nx.algorithms.matching.max_weight_matching)
    out=dict(status='train-only exact g16 horizontal-pair NAND oracle; no heldout/no implementation/new-method claim',
        created_utc=datetime.now(timezone.utc).isoformat(),models=models,
        formulas=dict(coselected_pair_count='M',triggered_pair_count='K-M',nand_ns='28000*ceil(2*expert_chip_bytes/65536)*(K-M)',
            training_mean='Perfect matching maximizes sum over pair coactivation counts, hence minimizes exact training-mean NAND waves.',
            free_candidate_bound='Replace complete baseline down phase by candidate NAND alone; all candidate non-NAND costs are zero. A negative gain therefore rules out a mean win for this pair family under the service model.'),
        solver=dict(networkx_version=nx.__version__,networkx_source_sha256=hashlib.sha256(nxsrc.encode()).hexdigest(),scipy_version=scipy.__version__,
            primary_documentation='https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html',
            verification='Integer blossom matching cross-checked by a binary edge MILP with degree-one constraints; equal primal and dual optimum.'),
        source_sha256=sha256(src),
        data_scope='Only original training cache[:32,:64], first2down. Whole-FFN denominator covers only examined layers, not physical full-model inference. Baseline strengthened by previously frozen source candidates; no main plans changed.')
    target=DEST/'oracle.json';target.write_text(json.dumps(out,indent=2)+'\n')
    completion=dict(status='complete',artifacts={src.name:sha256(src),'oracle.json':sha256(target)},
        decisions={m['model']:dict(decision=m['decision'],favorable_whole_ffn_mean_gain_pct=m['favorable_whole_ffn_mean_gain_pct'],
            optional_pair_decision=m['optional_pair_decision'],optional_pair_favorable_whole_ffn_mean_gain_pct=m['optional_pair_favorable_whole_ffn_mean_gain_pct']) for m in models})
    (DEST/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion,indent=2))


if __name__=='__main__':main()
