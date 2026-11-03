#!/usr/bin/env python3
"""Read saved JSON only; independently check execution and physical selection."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify(t, d, resident):
    ns = t['nodes']
    resources, owners, ee = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(lambda: [0, 0]))
    assert t['cost_ns'] == max(n['end_ns'] for n in ns)
    for i, n in enumerate(ns):
        assert n['index'] == i and n['end_ns']-n['start_ns'] == n['duration']
        assert all(ns[q]['end_ns'] <= n['start_ns'] for q in n['deps'])
        if n['resource'] >= 0 and n['duration']:
            resources[n['resource']].append((n['start_ns'], n['end_ns']))
        if n['kind'] == 'output_ONFI':
            assert len(n['deps']) == 1
            p = ns[n['deps'][0]]
            assert p['kind'] == 'row_produced' and p['chip'] == n['chip'] and p['bytes'] == n['bytes']
            assert n['duration']*2 == n['bytes']
            ee[n['chip']][p['end_ns']][0] += n['bytes']
            ee[n['chip']][n['start_ns']][1] += 2
            ee[n['chip']][n['end_ns']][1] -= 2
    down = []
    for j, job in enumerate(t['jobs']):
        lo, hi = ns[job['command']]['start_ns'], ns[job['release']]['end_ns']
        owners[job['chip']].append((lo, hi))
        assert ns[job['compute']]['start_ns'] == ns[job['input']]['end_ns']
        assert all(lo <= n['start_ns'] <= n['end_ns'] <= hi for n in ns if n['job'] == j)
        assert job['pipeline']['finite_PB_feasible']
        if 'physical_down_job' in job:
            p = job['physical_down_job']; down.append(p)
            assert p == resident[p['resident_job_id']] and p['copy'] == 0
            assert job['chip'] == p['chip'] and job['columns'] == p['columns']
            assert job['pipeline']['pages'] == job['pipeline']['ecc_pages'] == 24
            assert sum(ns[q]['bytes'] for q in job['output_ONFI']) == p['rows']*4
    for spans in list(resources.values()) + list(owners.values()):
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    peak = 0
    for events in ee.values():
        previous = backlog = rate = 0
        for time, (arrivals, delta) in sorted(events.items()):
            backlog -= rate*(time-previous)
            assert backlog >= 0
            backlog += arrivals; peak = max(peak, backlog)
            rate += delta; previous = time
            assert rate >= 0
        assert backlog == rate == 0
    assert peak <= 128 and peak == t['audit']['peak_output_fifo_bytes']
    decision = d['decision']
    assert sorted(p['resident_job_id'] for p in down) == sorted(p['resident_job_id'] for p in decision['selected_jobs'])
    assert len(down) == 16 and sum(p['rows']*p['columns'] for p in down) == 6*1024*1024
    assert len(set(d['route'])) == 6
    for e in d['route'] + [64]:
        spans = sorted((p['row_start'], p['row_start']+p['rows']) for p in down if p['expert'] == e)
        assert spans[0][0] == 0 and spans[-1][1] == 1536
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    for p in down:
        if p['expert'] == 64:
            assert p['resident_copy'] == 6+p['shared_half']
        else:
            assert p['resident_copy'] < 6
    return dict(nodes=len(ns),jobs=len(t['jobs']),down_jobs=len(down),peak_fifo_bytes=peak,capacity=decision['capacity'])

def main():
    protocol = read(HERE/'protocol.frozen.json')
    for p, h in protocol['inputs_sha256'].items():
        assert sha(p) == h, p
    ledger = read(HERE/'resident_jobs.reference.frozen.json')
    resident = {p['resident_job_id']:p for p in ledger}
    assert len(ledger) == len(resident) == 544
    records = []
    for stage in ['screen', 'TRAIN512']:
        for item in read(HERE/stage/'trace_index.json'):
            p = Path(item['path']); d = p.with_name(p.stem+'.dispatch.json')
            checks = verify(read(p), read(d), resident)
            records.append(dict(path=str(p),trace_sha256=sha(p),dispatch_sha256=sha(d),**checks))
    result = dict(status='PASS',scope='Saved JSON semantic/calendar/FIFO/address/row check, independent of causal executor and its audit function; finite PB uses saved sealed recipe certificate, not a new latch resimulation.',
                  source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.frozen.json'),saved_traces=len(records),records=records)
    (HERE/'saved_trace_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',saved_traces=len(records),cap2_traces=sum(x['capacity']==2 for x in records))))

if __name__ == '__main__':
    main()
