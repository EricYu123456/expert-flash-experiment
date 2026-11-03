#!/usr/bin/env python3
"""Independent full-wave burst bound for one explicitly declared PB contract.

No device or AiF RTL inference: array completion releases a whole page, read
latch may be reused after the next recycling phase, read-to-cache copy retains
its whole source, and ECC reads only the cache at the declared bandwidth.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import research_20261102_cr_ipr_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-2026-11-02/round3-cr-ipr/burst-bound'
COPIES = (0, 100, 320, 321, 500, 1000, 1940, 1941, 2000, 3020, 6040, 6041)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent(n, copy_ns):
    """Max-plus recurrence written without the tested simulator's helpers."""
    cache_available = [0] * 4
    consumer_available = 0
    hazards = []
    for epoch in range(n):
        ready = 28000 + epoch * 9700
        for bank in range(4):
            copied = max(ready, cache_available[bank]) + copy_ns
            consumer_available = max(consumer_available, copied) + 2560
            cache_available[bank] = consumer_available
            if epoch < n - 1 and copied > ready + 6040:
                hazards.append((epoch, bank, copied - ready - 6040))
    return dict(feasible=not hazards, hypothetical_makespan_ns=consumer_available,
                hazards=hazards)


def closed_form(n, copy_ns):
    if n == 1:
        return True
    if n == 2:
        return copy_ns <= 6040
    return copy_ns <= 6040 and 2 * copy_ns + (n - 2) * 540 <= 6040


def waves(n):
    return [dict(pages=[dict(owner='matrix', page_index=4*w+p,
                            page_id=f'matrix.p{4*w+p}', plane=p, block=0,
                            wl=w, bytes=16384) for p in range(4)],
                 source_order=[0, 1, 2, 3]) for w in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = OUT/'execution.frozen.json'
    if freeze.exists():
        raise RuntimeError('Use a new output directory instead of overwriting a sealed run.')
    inputs = {str(Path(__file__).resolve()): sha(__file__),
              str(Path(pipeline.__file__).resolve()): sha(pipeline.__file__),
              str(OUT.parent/'experiment_plan.frozen.json'): sha(OUT.parent/'experiment_plan.frozen.json')}
    freeze.write_text(json.dumps(dict(inputs_sha256=inputs, copies_ns=COPIES,
        lengths=list(range(1, 65)), cases=64*len(COPIES),
        contract='fixed 0123 drain, four full pages each wave, two whole-page slots per plane, cache-only ECC, concurrent plane copies allowed during recycle, eval-start overwrite, strict no idle',
        derivation='For N>=3, necessary 2*c+(N-2)*(4*s-r)<=u; fixed 0123 attains this condition here. s=2560ns,r=9700ns,u=6040ns. N=1 has no following overwrite; N=2 requires c<=u.',
        scope='Conditional microarchitectural bound, not a claim about undisclosed AiF latch timing or a new general queueing theorem.'), indent=2)+'\n')
    records = []
    for copy_ns in COPIES:
        for n in range(1, 65):
            cfg = pipeline.PipelineConfig(row_bytes=4096, cache_copy_ns=copy_ns,
                                          source_policy='fixed')
            actual = pipeline.simulate(waves(n), cfg)
            reference = independent(n, copy_ns)
            formula = closed_form(n, copy_ns)
            assert actual['timing_feasible'] == reference['feasible'] == formula, (n, copy_ns)
            assert actual['hypothetical_makespan_ns'] == reference['hypothetical_makespan_ns'], (n, copy_ns)
            assert actual['functional_ok']
            records.append(dict(waves=n, copy_ns=copy_ns, feasible=formula,
                                hypothetical_makespan_ns=reference['hypothetical_makespan_ns'],
                                hazard_count=len(reference['hazards'])))
    maximum = {str(c): max(r['waves'] for r in records if r['copy_ns']==c and r['feasible'])
               for c in COPIES}
    result = dict(status='PASS', cases=len(records), independent_recurrence=True,
                  closed_form_matched=True, maximum_strict_full_wave_burst=maximum,
                  records=records, inputs_sha256=inputs,
                  conclusion='The 320ns window for twelve full waves is a producer/consumer/latch limit under this contract, rather than a unique property of expert IDs or the G4 stencil.',
                  limits='Actual PB transfer timing, source selection, ECC/MAC coupling and cold precharge/discharge phases remain unverified; this does not establish measured hardware latency.')
    (OUT/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    lines=['# Conditional finite-PB full-wave burst bound', '',
           'Independent max-plus recurrence, closed form and event simulator agree in all 768 cases. This is a declared microarchitecture sensitivity, not a device measurement.', '',
           'For N>=3 full waves: `2*c + (N-2)*540 <= 6040 ns`; fixed 0,1,2,3 drain attains this condition in the tested contract.', '',
           '| Copy ns | Maximum strict full-wave burst |', '|---:|---:|']
    lines += [f'| {c} | {maximum[str(c)]} |' for c in COPIES]
    lines += ['', 'A twelve-wave burst admits c<=320ns; a six-wave burst admits c<=1940ns. Longer chains need a documented different buffering/streaming path or explicit cold restarts under this contract. Plane load balance alone does not prove continuous CR feasibility.', '',
              'Both read-latch deadline and consumer buffering assumptions are explicit in execution.frozen.json. No change to any hardware is asserted.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    assert all(sha(p)==h for p,h in inputs.items())
    print(json.dumps(dict(status='PASS', cases=len(records), max_burst=maximum,
                          source_sha256=sha(__file__))))


if __name__ == '__main__':
    main()
