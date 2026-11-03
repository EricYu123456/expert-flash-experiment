#!/usr/bin/env python3
"""No-CR supplement: reevaluate frozen training-only supermatrix candidates.

No bundle memberships are learned from test routes. Existing training-only
memberships/chip groups are retained; g/B is reselected with no-CR training
latency. The old model is parameterized in this process only by setting its
successive-read latency equal to its ordinary-read latency, leaving the
max(NAND,ECC) overlap and all command/vector costs unchanged.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import aif_model as flash
import static_wave_bundle_study as bundle
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DATA/'runs/corpus-300-v1')
    parser.add_argument('--previous', type=Path, default=ROOT/'results/plane-wave-v1/static-bundles')
    parser.add_argument('--output', type=Path, default=ROOT/'results/plane-wave-v2/no-cr-bundles')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    original_cr = flash.CR_NS
    flash.CR_NS = flash.COLD_NS
    summaries, candidates_out, prompts_out, receipts = [], [], [], []
    try:
        for name in args.models:
            analysis = args.run_dir/name/'analysis'
            metadata_path = DATA/'models'/f'{name}-metadata.json'
            metadata = json.loads(metadata_path.read_text())
            dims = flash.tensor_dimensions(metadata)
            layers, experts, _ = geometry(metadata)
            previous_path = args.previous/f'{name}.frozen_plan.json'
            previous = json.loads(previous_path.read_text())
            old_path = analysis/'frozen_plan.json'
            old = json.loads(old_path.read_text())
            assert previous['source_frozen_plan_sha256'] == sha256(old_path)
            with np.load(analysis/'train_routes.npz', allow_pickle=False) as data:
                train, training_ids = data['routes'], data['prompt_ids'].tolist()
                assert data['layers'].tolist() == layers
            assert train.shape[:2] == (200,128)
            assert training_ids == previous['training_ids']
            unique = [np.unique(np.sort(train[:,:,li].reshape(-1,train.shape[-1]),axis=1),
                                axis=0,return_counts=True) for li in range(len(layers))]
            candidates = []
            for g in (1,2,4,8,16):
                mappings = old['mappings'][f'g{g}']
                for size in (1,2,4):
                    tag = f'g{g}_b{size}'
                    labels = previous['labels'][tag]
                    weighted = 0
                    for li,layer in enumerate(layers):
                        routes,weights = unique[li]
                        costs,_,_ = bundle.layer_costs(routes,mappings[str(layer)],g,dims[layer],labels[str(layer)])
                        weighted += int(costs['model_ns'] @ weights)
                    item = dict(model=name,tag=tag,g=g,bundle_size=size,
                                training_ms=weighted/(200*128)/1e6)
                    candidates.append(item)
            winner = min(candidates,key=lambda r:(r['training_ms'],r['bundle_size'],r['g']))
            forced = min((r for r in candidates if r['bundle_size']>1),key=lambda r:(r['training_ms'],r['bundle_size'],r['g']))
            frozen_path = args.output/f'{name}.frozen_plan.json'
            frozen = dict(model=name,use_cr=False,nand_ns_per_wave=flash.COLD_NS,
                          pipeline='max(NAND,ECC), unchanged',baseline='g16_width4_stacked_identity',
                          previous_bundle_training_plan_sha256=sha256(previous_path),
                          original_chip_plan_sha256=sha256(old_path),
                          train_cache_sha256=sha256(analysis/'train_routes.npz'),
                          training_ids=training_ids,candidates=candidates,winner=winner,forced_bundle_winner=forced,
                          membership_learning='Retain old training-only fixed memberships; only g/B reselected with noCR training objective.',
                          test_loaded_before_freeze=False)
            write_json(frozen_path,frozen)
            frozen_hash = sha256(frozen_path)
            with np.load(analysis/'test_routes.npz',allow_pickle=False) as data:
                test,test_ids = data['routes'],data['prompt_ids'].tolist()
                assert data['layers'].tolist() == layers
            assert test.shape[:2] == (100,128)
            assert set(training_ids).isdisjoint(test_ids)
            identity = {str(layer):list(range(experts)) for layer in layers}
            baseline = bundle.evaluate(test,layers,dims,identity,identity,16)
            specs = dict(baseline=(16,1,None), selected=(winner['g'],winner['bundle_size'],winner['tag']),
                         forced_bundle=(forced['g'],forced['bundle_size'],forced['tag']),
                         fixed_g16_b2=(16,2,'g16_b2'),fixed_g16_b4=(16,4,'g16_b4'))
            values = {'baseline':baseline}
            for label,(g,_,tag) in specs.items():
                if label != 'baseline':
                    values[label] = bundle.evaluate(test,layers,dims,old['mappings'][f'g{g}'],previous['labels'][tag],g)
            arrays = {'prompt_ids':np.asarray(test_ids)}
            for label,(costs,upper,metrics) in values.items():
                for key,value in costs.items():
                    arrays[f'{label}__{key}'] = value
                for window in (32,128):
                    described = bundle.describe(costs,upper,metrics,window,baseline[0],baseline[1])
                    described = {key:value for key,value in described.items() if not key.startswith('conservative')}
                    summaries.append(dict(model=name,policy=label,window=window,g=specs[label][0],
                                          bundle_size=specs[label][1],**described))
                    for i,prompt in enumerate(test_ids):
                        prompts_out.append(dict(model=name,policy=label,window=window,prompt_id=prompt,
                                                mean_ms=float(costs['model_ns'][i,:window].mean()/1e6)))
            np.savez_compressed(args.output/f'{name}.token_costs.npz',**arrays)
            assert sha256(frozen_path) == frozen_hash
            receipts.append(dict(model=name,previous_plan=str(previous_path),previous_plan_sha256=sha256(previous_path),
                                 train_cache_sha256=sha256(analysis/'train_routes.npz'),
                                 test_cache_sha256=sha256(analysis/'test_routes.npz'),
                                 frozen_sha256=frozen_hash,test_used_for_selection=False))
            candidates_out.extend(candidates)
            write_json(args.output/'summary.json',summaries)
            print(name,'noCR bundle winner',winner,'forced',forced,flush=True)
        with (args.output/'per_prompt.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(prompts_out[0]))
            writer.writeheader();writer.writerows(prompts_out)
        write_json(args.output/'training_candidates.json',candidates_out)
        write_json(args.output/'manifest.json',dict(date='2026-11-01',use_cr=False,
            no_cr_implementation='Process-local CR_NS=COLD_NS=28000; all reused code files remain unchanged.',
            software_sha256={name:sha256(ROOT/'scripts'/name) for name in
                             ('evaluate_no_cr_bundles.py','static_wave_bundle_study.py','aif_model.py','corpus_analysis.py')},
            sources=receipts,wall_seconds=time.time()-started,
            limitation='Fixed old training-only group/bundle maps; no new joint bundle membership search. Entire active bundle read/computed/returned.'))
    finally:
        flash.CR_NS = original_cr


if __name__ == '__main__':
    main()
