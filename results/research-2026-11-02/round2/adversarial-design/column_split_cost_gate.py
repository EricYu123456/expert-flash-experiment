#!/usr/bin/env python3
"""Independent train-only algebraic gate for the proposed2:1 native rectangles.

Reuses frozen binary g8 colors; no new fitting or heldout reads. Native partial
output contract and actual host-reduction time remain unverified.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
from research_20261102_projection_study import evaluate
from run_corpus_traces import sha256


def main():
    items=[]
    for model,phase_name,big,small in [('granite4tiny','gate_up',1024,512),('qwen3','down',512,256)]:
        path=ROOT/'results/research-2026-11-02/round2/unbalanced-group'/(model+'.screen.strong.frozen.json')
        doc=json.loads(path.read_text());route=Path(doc['training_routes'])
        assert sha256(route)==doc['training_routes_sha256']
        with np.load(route) as f:
            routes=f['routes'][:32,:64];layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
        assert all(i.startswith('train_') for i in ids)
        K=doc['active'];records=[]
        for p in doc['phases']:
            if p['name']!=phase_name:continue
            rows=np.asarray(p['chosen_rows']);R,C=p['total_rows'],p['columns'];assert C==big+small
            assert np.all(np.count_nonzero(rows,axis=1)==8)
            color=(rows[:,0]==0).astype(int);selected=routes[:,:,layers.index(p['layer'])].reshape(-1,K)
            n=color[selected].sum(axis=1);m=np.maximum(n,K-n)
            old=evaluate(selected,rows,C,p['common'])
            np.testing.assert_array_equal(old['flash_ns'],3*m*28000)
            new=dict(flash_ns=(K+m)*28000,control_ns=np.full(len(m),16*K*250,dtype=np.int64),
                host_ns=np.full(len(m),2000,dtype=np.int64),
                input_ns=np.full(len(m),max((C*(1 if p['common'] else K)+7)//8,(K*C+1)//2),dtype=np.int64),
                output_ns=np.full(len(m),K*R,dtype=np.int64))
            candidate=sum(new.values());gain=old['model_ns']-candidate
            delta_nonflash=float((new['control_ns']+new['input_ns']+new['output_ns']-old['control_ns']-old['input_ns']-old['output_ns']).mean())
            assert np.all((R//8)*big==2*65536) and np.all((R//8)*small==65536)
            records.append(dict(layer=p['layer'],K=K,R=R,C=C,mean_m=float(m.mean()),
                original_peak_waves='3*m',candidate_peak_waves='K+m',
                original_mean_ns=float(old['model_ns'].mean()),candidate_excluding_host_sum_ns=float(candidate.mean()),
                original_p99_ns=float(np.percentile(old['model_ns'],99)),candidate_excluding_host_sum_p99_ns=float(np.percentile(candidate,99)),
                nonflash_penalty_mean_ns=delta_nonflash,break_even_mean_host_sum_ns=float(gain.mean()),
                break_even_mean_m=K/2+delta_nonflash/56000,
                scalar_adds_per_query=K*R,
                minimum_three_array_pass_bytes_for_host_sum=12*K*R,
                profile=dict(row_blocks=8,rows_per_block=R//8,big_columns=big,small_columns=small,
                    big_waves=2,small_waves=1,native_commands_per_query=16*K),
                costs_before_host_sum={k:float(v.mean()) for k,v in new.items()}))
        items.append(dict(model=model,phase=phase_name,source_plan=str(path),source_plan_sha256=sha256(path),
            training_routes_sha256=sha256(route),records=records,
            aggregate_mean_host_sum_budget_ns=sum(p['break_even_mean_host_sum_ns'] for p in records)))
    output=dict(status='algebraic cost gate only; not a native-legality or novelty certificate',items=items,
        formulas=dict(flash_saving_ns='28000*(2*m-K)',cost_gate='mean host_sum_ns <56000*(E[m]-K/2)-E[extra command/input/output ns]',
            color_learning='Both old3*m and newK+m objectives are monotonic affine functions of m. Reusing the same learned binary colors is not a new orientation-learning algorithm.'),
        critical_controls=['Cambricon-LLM hardware-aware2D page tiling, mapped onto existing AiF commands without importing its extra hardware features.',
            'Uniform64KiB512-column atoms with generic static coactivation declustering and exact native-command/partial-output costs.',
            'The same atom baseline with legal rectangular fusion/repacking into128x1024 or256x512 matrices; this feasible family contains the proposed2:1layout, so no universal gap over its optimum can be claimed.',
            'Equal-column halves; all whole-row strong baselines; random/frequency/learned orientation under identical rows and storage.',
            'Measured host sum, per-command input reload, partial output dtype/dequantization and shape-dependent PE throughput.'],
        qualifications=['No new routing fit, test read, quantization, replica, or hardware implementation.',
            'The predicted margins can disappear after host reduction, native ABI, column-dependent arithmetic throughput or a stronger2D rectangle-coalescing baseline.',
            'Previous no-go results apply to whole-output-row layouts, not to all compositions of existing denseGEMV.'],
        source_sha256=sha256(Path(__file__)))
    out=Path(__file__).with_name('column_split_cost_gate.json');out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
