#!/usr/bin/env python3
"""Resume five-model, real 128-step routing collection with full greedy controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np

MODELS = ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral')
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment')
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def geometry(metadata):
    arch = metadata['metadata']['general.architecture']
    layers = sorted({t['layer'] for t in metadata['tensors'] if
                     'projection' in t and 'expert_count' in t})
    return layers, metadata['metadata'][arch+'.expert_count'], metadata['metadata'][arch+'.expert_used_count']


def validate_prompt(stem, row, metadata, steps=128, save_prefix=True):
    layers, experts, active = geometry(metadata)
    traced = json.loads(Path(str(stem)+'.traced.summary.json').read_text())
    baseline = json.loads(Path(str(stem)+'.baseline.summary.json').read_text())
    assert traced['trace_enabled'] and not baseline['trace_enabled']
    assert traced['decoded_tokens'] == baseline['decoded_tokens'] == steps
    assert traced['prompt_token_ids'] == baseline['prompt_token_ids']
    assert traced['generated_token_ids'] == baseline['generated_token_ids'], f'{stem}: greedy IDs changed'
    assert len(traced['generated_token_ids']) == steps
    text = Path(str(stem)+'.traced.output.txt').read_bytes()
    assert text == Path(str(stem)+'.baseline.output.txt').read_bytes(), f'{stem}: greedy bytes changed'
    assert Path(str(stem)+'.traced.prefix32.txt').read_bytes() == Path(str(stem)+'.baseline.prefix32.txt').read_bytes()
    routing = Path(str(stem)+'.routing.jsonl')
    routes = np.zeros((steps, len(layers), active), dtype=np.int16)
    seen = set(); layer_index = {layer:i for i,layer in enumerate(layers)}; prefix = []
    with routing.open() as f:
        for line in f:
            e = json.loads(line)
            t, layer = e['token_index'], e['layer']
            key = t, layer
            assert key not in seen and 0 <= t < steps and layer in layer_index
            assert e['event'] == 'routing' and e['phase'] == 'decode'
            assert e['position'] == traced['prompt_tokens'] + t
            assert e['token_id'] == traced['generated_token_ids'][t]
            selected = e['expert_ids']
            assert len(selected) == len(set(selected)) == active
            assert all(0 <= x < experts for x in selected)
            routes[t,layer_index[layer]] = selected
            seen.add(key)
            if t < 32:
                prefix.append(line)
    assert len(seen) == traced['routing_records'] == steps*len(layers)
    if save_prefix:
        prefix_dir = stem.parent/'prefix32'; prefix_dir.mkdir(exist_ok=True)
        (prefix_dir/(stem.name+'.routing.jsonl')).write_text(''.join(prefix))
    result = dict(id=row['id'], split=row['split'], task=row['task'], length_bucket=row['length_bucket'],
                  prompt_sha256=row['prompt_sha256'], prompt_tokens=traced['prompt_tokens'],
                  decode_tokens=steps, prefix_tokens=32, routing_records=len(seen),
                  routed_layers=layers, active_experts=active, experts=experts,
                  baseline_output_identical=True, baseline_token_ids_identical=True,
                  routing_sha256=sha256(routing), output_sha256=hashlib.sha256(text).hexdigest(),
                  sequence_batch=traced['sequence_batch'])
    return result, routes


def run_model(name, args, rows):
    metadata_path = args.data/'models'/(name+'-metadata.json')
    metadata = json.loads(metadata_path.read_text())
    model_path = Path(metadata['model_path'])
    destination = args.run_dir/name/'traces'; destination.mkdir(parents=True, exist_ok=True)
    validation_dir = args.run_dir/name/'batches'; validation_dir.mkdir(exist_ok=True)
    identity = dict(corpus_sha256=sha256(args.corpus), metadata_sha256=sha256(metadata_path),
                    binary_sha256=sha256(args.binary), steps=args.steps, batch_size=args.batch_size,
                    threads=args.threads)
    if args.reuse_prefill:
        identity['baseline_prefill_state_reused'] = True
    complete = []
    pending = []
    for start in range(0,len(rows),args.batch_size):
        group = rows[start:start+args.batch_size]
        checkpoint = validation_dir/f'{start:04d}.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['identity'] != identity or saved['ids'] != [r['id'] for r in group]:
                raise ValueError('Checkpoint provenance changed; use a new run directory')
            complete.extend(saved['validation'])
        else:
            pending.append((start,group))
    jobs = [r for _,group in pending for r in group]
    print(f'{name}: {len(complete)}/{len(rows)} validated; {len(jobs)} pending', flush=True)
    if jobs:
        write_json(args.run_dir/name/'provenance.json',dict(identity=identity,model=name,
                   model_path=str(model_path),model_sha256=metadata['sha256'],
                   source_commit='927dada6c9143ba4c940b72004d2698fa5e4e930',
                   routing='Actual llama.cpp ffn_moe_topk callback, independent sequence IDs, CPU, greedy, fixed EOS-inclusive length.',
                   checkpoint_policy='Resume whole original sequence batches to preserve numerical configuration.'))
        progress_lock=threading.Lock()
        def worker(rank,assigned):
            suffix='' if args.inference_workers==1 else f'.worker{rank}'
            tsv=args.run_dir/name/('pending'+suffix+'.tsv')
            tsv.write_text(''.join(str(args.corpus.parent/'prompts'/(r['id']+'.txt'))+'\t'+
                          str(destination/r['id'])+'\n' for _,group in assigned for r in group))
            command=[str(args.binary),str(model_path),str(tsv),str(args.steps),str(args.threads),
                     str(args.batch_size),'both-cache' if args.reuse_prefill else 'both']
            write_json(args.run_dir/name/('command'+suffix+'.json'),command)
            error_path=args.run_dir/name/('stderr'+suffix+'.log')
            with error_path.open('a') as err, (args.run_dir/name/('stdout'+suffix+'.log')).open('a') as log:
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
                try:
                    for line in process.stdout:
                        log.write(line);log.flush()
                        if line.startswith('BATCH_COMPLETE '):
                            offset,size=map(int,line.split()[1:])
                            start,group=assigned[offset//args.batch_size]
                            assert size==len(group)
                            validation=[validate_prompt(destination/r['id'],r,metadata,args.steps)[0] for r in group]
                            write_json(validation_dir/f'{start:04d}.json',dict(identity=identity,
                                       ids=[r['id'] for r in group],validation=validation))
                            with progress_lock:
                                complete.extend(validation)
                                write_json(args.run_dir/name/'progress.json',dict(model=name,completed=len(complete),
                                           total=len(rows),updated_unix=time.time()))
                                print(f'{name}: {len(complete)}/{len(rows)} complete, greedy outputs and all routes verified',flush=True)
                    if process.wait()!=0:raise RuntimeError(f'{name}: inference failed; see {error_path}')
                finally:
                    if process.poll() is None:process.terminate();process.wait()
        with ThreadPoolExecutor(max_workers=args.inference_workers) as pool:
            tasks=[pool.submit(worker,rank,pending[rank::args.inference_workers])
                   for rank in range(min(args.inference_workers,len(pending)))]
            for task in as_completed(tasks):task.result()
    assert len(complete) == len(rows)
    assert {r['id'] for r in complete} == {r['id'] for r in rows}
    write_json(args.run_dir/name/'validation.json',sorted(complete,key=lambda r:r['id']))
    print(f'{name}: all {len(rows)} prompts validated',flush=True)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--run-dir',type=Path,default=DATA/'runs/corpus-300-v1')
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--binary',type=Path,default=DATA/'build/llama-route-trace/llama-corpus-trace')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--inference-workers',type=int,default=1,
                        help='Concurrent processes per model, distributing whole original sequence batches.')
    parser.add_argument('--reuse-prefill',action='store_true',
                        help='Restore the identical prefill KV state into a separate untraced context; decode is rerun fully.')
    args = parser.parse_args()
    args.corpus = args.corpus or args.run_dir/'corpus/prompts.jsonl'
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    if args.steps != 128 or len(rows) != 300:
        parser.error('Production run requires exactly 300 prompts and 128 decode inputs')
    (args.run_dir/('-'.join(args.models)+'.collector.pid')).write_text(str(os.getpid())+'\n')
    artifacts=args.run_dir/'artifacts';artifacts.mkdir(exist_ok=True)
    pinned=artifacts/('llama-corpus-trace-'+sha256(args.binary))
    if not pinned.exists():shutil.copy2(args.binary,pinned)
    assert sha256(pinned)==sha256(args.binary)
    args.binary=pinned
    # Stable batch membership and prompt-level split are immutable run inputs.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_model,name,args,rows):name for name in args.models}
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                failures.append(f'{futures[future]}: {e}')
                print(failures[-1],flush=True)
        if failures:
            raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    main()
