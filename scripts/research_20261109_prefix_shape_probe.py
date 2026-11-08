#!/usr/bin/env python3
"""Read-only old/new native-harness and prefix batch-shape probe on TRAIN."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from research_20261109_shared_feature_targets import DATA, MODEL, OLD, ROOT, dump, sha


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def compare(a,b):
    n=min(len(a),len(b))
    assert all(a[i]['input_token_id']==b[i]['input_token_id'] and a[i]['target_token_id']==b[i]['target_token_id'] for i in range(n))
    delta=[b[i]['nll_F64']-a[i]['nll_F64'] for i in range(n)]
    return {'compared_scored_positions':n,'records_identical':a[:n]==b[:n],
            'NLL_deltas_second_minus_first':delta,'max_abs_NLL_delta':max(map(abs,delta)),
            'top1_equal_positions':sum(a[i]['top1_token_id']==b[i]['top1_token_id'] for i in range(n)),
            'BOS_record_identical':a[0]==b[0]}


def activation_comparison():
    result=[]
    for layer in (1,13,26):
        for field in ('x','shared_z','selected_experts'):
            values=[]
            for root in (OLD/'captures/train/shard0',DATA/'captures64/train/shard0'):
                m=json.loads((root/'capture_manifest.json').read_text())
                item=m['layers'][str(layer)]['arrays'][field]
                values.append(np.asarray(np.memmap(root/item['path'],dtype=item['dtype'],shape=tuple(item['shape']),mode='r')[:16]))
            a,b=values
            row={'layer':layer,'field':field}
            if a.dtype==np.float32:
                row['bit_equal_by_position']=np.all(a.view(np.uint32)==b.view(np.uint32),axis=1).tolist()
                row['max_abs_difference']=float(np.max(np.abs(a-b)))
                row['relative_squared_difference']=float(np.sum((a.astype(np.float64)-b)**2)/np.sum(a.astype(np.float64)**2))
            else:
                row['equal_by_position']=np.all(a==b,axis=1).tolist()
            result.append(row)
    return result


def main():
    folder=DATA/'prefix-shape-probe'
    folder.mkdir()
    jobs=folder/'train1.jobs.tsv'
    first=(DATA/'smoke/jobs2.tsv').read_text().splitlines()[0]
    jobs.write_text(first+'\n')
    prompt_id=first.split('\t')[1]
    binaries={'old':OLD/'build/shared-feature-quality','newproto':DATA/'build/shared-feature-quality-proto'}
    capture=OLD/'build/shared-feature-capture'
    hashes={str(p):sha(p) for p in (*binaries.values(),capture)}
    report={'status':'executing','prompt_id':prompt_id,'jobs_sha256':sha(jobs),'binary_sha256':hashes,
            'source_sha256':sha(Path(__file__)),'CPU_only':True,'CPU_affinity':'40-47','threads':8,'AR_steps':0,
            'cases':{},'comparisons':{},'existing_same_binary_capture_activation_comparison':activation_comparison()}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name,binary in binaries.items():
        for length in (16,64):
            label=f'{name}{length}'
            out=folder/label
            cmd=['taskset','-c','40-47',str(binary),str(MODEL),str(jobs),str(out),'%d'%length,'8','0','-','w8']
            start=time.monotonic()
            with (folder/f'{label}.stdout.log').open('w') as stdout,(folder/f'{label}.stderr.log').open('w') as stderr:
                subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,check=True)
            summary=json.loads((out/f'{prompt_id}.summary.json').read_text())
            report['cases'][label]={'argv':cmd,'wall_seconds':time.monotonic()-start,'summary':summary,
                                    'positions_sha256':sha(out/f'{prompt_id}.positions.jsonl')}
            print(json.dumps({'event':'quality_case_complete','case':label}),flush=True)
    # Source-GGUF observation uses no FFN override and no offroute fixed targets.
    for length in (16,64):
        label=f'sourcegguf{length}'
        out=folder/label
        cmd=['taskset','-c','40-47',str(capture),str(MODEL),str(jobs),str(out),str(length),'8','source-gguf','1:;13:;26:']
        start=time.monotonic()
        with (folder/f'{label}.stdout.log').open('w') as stdout,(folder/f'{label}.stderr.log').open('w') as stderr:
            subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,check=True)
        report['cases'][label]={'argv':cmd,'wall_seconds':time.monotonic()-start,
                               'summary':json.loads((out/'scores'/f'{prompt_id}.summary.json').read_text())}
        print(json.dumps({'event':'source_case_complete','case':label}),flush=True)
    def pos(label):
        root=folder/label
        path=root/'scores'/f'{prompt_id}.jsonl' if label.startswith('sourcegguf') else root/f'{prompt_id}.positions.jsonl'
        return records(path)
    for left,right in [('old16','newproto16'),('old64','newproto64'),('old16','old64'),('newproto16','newproto64'),('sourcegguf16','sourcegguf64')]:
        report['comparisons'][f'{left}_vs_{right}']=compare(pos(left),pos(right))
    report['status']='complete'
    for path,identity in hashes.items():assert sha(Path(path))==identity
    report['interpretation']='Within-length old/new checks isolate adapter drift. Cross-length checks compare identical causal token prefixes with different batch shapes; source-GGUF observations are separate from canonical W8 FFN reference. Existing captures localize any pre-MoE1 divergence but do not identify an attention kernel by themselves.'
    dump(folder/'report.json',report)
    dump(ROOT/'results/research-2026-11-09-shared-features/prefix_shape_probe.json',report)
    print(json.dumps({'status':'complete','comparisons':report['comparisons']}),flush=True)


if __name__=='__main__':main()
