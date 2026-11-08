#!/usr/bin/env python3
"""CPU worker scheduling benchmark, not compressed-model/AiF performance."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

from research_20261109_shared_feature_targets import DATA, MODEL, ROOT, dump, sha


def load(path):
    return json.loads(path.read_text())


def main():
    folder = DATA / 'benchmark/threads4_vs8'
    folder.mkdir()
    binary = DATA / 'build/shared-feature-quality-proto'
    jobs = DATA / 'smoke/jobs2.tsv'
    report = dict(status='complete', purpose='Choose CPU experimental worker/thread scheduling; no model compression or AiF speedup inference.',
                  binary_sha256=sha(binary), jobs_sha256=sha(jobs), input_tokens_max=64, AR_steps_max=16, cases={})
    for threads in (4, 8):
        out = folder / f'threads{threads}'
        out.mkdir()
        cpus = f'40-{39 + threads}'
        cmd = ['/usr/bin/time', '-v', '-o', str(out / 'time.txt'), 'taskset', '-c', cpus,
               str(binary), str(MODEL), str(jobs), str(out / 'data'), '64', str(threads), '16', '-', 'w8']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LC_ALL='C')
        dump(out / 'command.json', dict(argv=cmd, CUDA_VISIBLE_DEVICES='', CPU_affinity=cpus))
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            start = time.monotonic()
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
            warm_status = None
            while proc.poll() is None:
                if warm_status is None and 'QUALITY_COMPLETE 0 ' in (out / 'stdout.log').read_text():
                    try:
                        children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                        if children:
                            status = Path(f'/proc/{children[0]}/status').read_text()
                            warm_status = {k: int(v) for k, v in re.findall(r'^(VmRSS|VmHWM|RssAnon|RssFile|RssShmem):\s+(\d+)\s+kB', status, re.M)}
                    except FileNotFoundError:
                        pass
                time.sleep(.5)
            assert proc.returncode == 0
            wall = time.monotonic() - start
        text = (out / 'time.txt').read_text()
        peak = int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', text).group(1))
        summaries = [load(p) for p in sorted((out / 'data').glob('*.summary.json'))]
        assert len(summaries) == 2
        report['cases'][str(threads)] = dict(cpu_group=cpus, wall_seconds_including_model_load=wall,
                maxRSS_KiB=peak, warm_process_status_KiB=warm_status,
                prompts={v['prompt_id']: {k: v[k] for k in ('seconds', 'sum_nll_F64', 'mean_nll_F64', 'actual_generated_length', 'stopped_eog', 'generated_token_ids')} for v in summaries})
        print(json.dumps(dict(threads=threads, **report['cases'][str(threads)])), flush=True)
    comparisons = []
    for p in sorted((folder / 'threads4/data').glob('*.summary.json')):
        a, b = load(p), load(folder / 'threads8/data' / p.name)
        pa = p.with_name(a['prompt_id'] + '.positions.jsonl')
        pb = folder / 'threads8/data' / pa.name
        records_a = [json.loads(s) for s in pa.read_text().splitlines()]
        records_b = [json.loads(s) for s in pb.read_text().splitlines()]
        same = records_a == records_b and a['token_ids'] == b['token_ids'] and a['generated_token_ids'] == b['generated_token_ids']
        comparisons.append(dict(prompt_id=a['prompt_id'], NLL_top1_token_records_identical=records_a == records_b,
                                input_and_generated_tokens_identical=a['token_ids'] == b['token_ids'] and a['generated_token_ids'] == b['generated_token_ids'],
                                all_exactly_equal=same))
    report['comparisons'] = comparisons
    report['status'] = 'passed' if all(c['all_exactly_equal'] for c in comparisons) else 'outputs_differ'
    report['memory_interpretation'] = 'MaxRSS includes GGUF file mappings, including possible duplicate mappings inside one worker, and cannot be multiplied by worker count to estimate physical RAM. Warm RssAnon is a useful private-weight/workspace proxy; count source GGUF shared file residency once and add each arm private C/transform payload separately. Two prompts are a small scheduling pilot.'
    dump(folder / 'report.json', report)
    dump(ROOT / 'results/research-2026-11-09-shared-features/cpu_worker_benchmark.json', report)
    print(json.dumps(dict(status=report['status'], report=str(folder / 'report.json'))), flush=True)


if __name__ == '__main__':
    main()
