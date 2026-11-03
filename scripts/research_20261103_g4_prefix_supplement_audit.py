#!/usr/bin/env python3
"""Pin parent's rational audit and independently verify CP32 actual row work.

Adds a final index; leaves all earlier sealed sources and reports unchanged.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round2/g4-elastic-prefix-screen/work-v3-independent-prefix'
DEST=OUT/'independent'
CP=OUT/'cp32-integer-audit'
EXACT=OUT/'integer-dual-audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def work(nr,nc):
    if not nr:return 0
    pages=(int(nr)*int(nc)+16383)//16384
    waves=(pages+3)//4
    emitted=pages*16384//int(nc)
    return 250+(int(nc)+1)//2+28000*waves+2*emitted


def main():
    assert not (DEST/'final-completion.json').exists()
    for proto in [CP/'protocol.frozen.json',EXACT/'protocol.frozen.json']:
        p=json.loads(proto.read_text())
        for x in p['inputs']:
            assert sha(x['path'])==x['sha256']
    for seal in [DEST/'completion.json',CP/'completion.json',EXACT/'completion.json']:
        for x in json.loads(seal.read_text())['files']:
            path=Path(x['path'])
            if not path.is_absolute():path=seal.parent/path
            assert sha(path)==x['sha256']
    with np.load(OUT/'contract_arrays.npz') as z:
        rows,routes=z['rows'],z['routes']
    with np.load(CP/'arrays.npz') as z:
        cp={k:z[k] for k in z.files}
    ledger=json.loads((OUT/'rowid_ledgers.json').read_text())[3]['experts']
    report=json.loads((CP/'report.json').read_text())
    assert np.array_equal(cp['indices'],np.linspace(0,511,32,dtype=int))
    for i,q in enumerate(cp['indices']):
        loads=np.zeros(16,np.int64)
        for ordinal,e in enumerate(np.r_[routes[q],64]):
            gu=rows[e,0].copy()
            if e<64:
                entry=ledger[e]
                u=cp['bits'][i,ordinal]
                for j,c in enumerate(entry['sources']):gu[c]=128 if u[j] else gu[c]
                for j,c in enumerate(entry['low_destinations']):gu[c]=0 if u[j] else 40
                c=entry['high_destination']
                gu[c]=rows[e,0,c] if not u[7] else entry['high_prefix_rows'] if not u.all() else 0
            for chip in range(16):
                loads[chip]+=work(gu[chip],1536)+work(rows[e,1,chip],1024 if e==64 else 512)
        assert loads.max()==cp['best_work'][i]==cp['best_bound'][i]
        assert report['cases'][i]['status']=='OPTIMAL'
    improvement=cp['greedy_work']-cp['best_work']
    exactreport=json.loads((EXACT/'report.json').read_text())
    summary=dict(status='PASS_final_bounded_no_go',source_sha256=sha(__file__),
                 initial_native_audit_completion_sha256=sha(DEST/'completion.json'),
                 exact_rational_completion_sha256=sha(EXACT/'completion.json'),
                 exact_rational_protocol_sha256=sha(EXACT/'protocol.frozen.json'),
                 rational_certificates=exactreport['certificates'],
                 integer_CP32_completion_sha256=sha(CP/'completion.json'),
                 CP32_independent_actualrow_cost_checks=32,CP32_optimal_count=32,
                 mean_CP32_work_improvement_ns=float(improvement.mean()),
                 max_CP32_work_improvement_ns=int(improvement.max()),CP32_cases_gt3us=int(np.sum(improvement>3000)),
                 erratum='Initial independent/report.md sentence that no small integer oracle was run is superseded by this separately authorized CP32 supplement. Initial native results/source/seal unchanged.',
                 limits='CP32 optimality is solver reported; this independent audit verifies actual row-job objective/assignment and zero solver gap, not a formal proof trace. No native replay of CP32 masks, new fitting, heldout or selector benchmark.')
    (DEST/'supplement.json').write_text(json.dumps(summary,indent=2)+'\n')
    text='''Final bounded result: the independent-prefix native sensitivity improves mean 1.725484% and P99 0.600511% on G4 L0 TRAIN512, with zero selector/scatter/software cost. It fails the 2% mean or 5% tail expansion gate. No production selector, large-copy controls, heldout or all-layer expansion followed.

The independent main audit passed all 65,536 expert views, 2,048 floating box-dual checks, 4,096 greedy endpoint costs, 3,072 consumed-row ownership maps, and nine full Python native schedules. Root additionally supplied exact rational dual certificates for all 2,048 LP cases (denominator 10^9, Python arbitrary precision). Their floor bounds differ from the original float floor by only -1/0 ns. These certificates support only a necessary continuous relaxation; they do not prove feasible speed.

A subsequently authorized CP-SAT check used 32 predetermined linspace TRAIN indices, one fixed layout, a 0.5 s / one-thread cap and retained greedy incumbents. All 32 were solver-OPTIMAL in total 0.54 s; actual row-job objective was independently recomputed for every solution. Mean improvement over greedy is only 784.0625 ns, maximum 2,326 ns, with zero cases above 3 us. Thus the sampled LP-to-greedy gap is mostly an integer-relaxation gap. This does not infer a whole512 exact result or native speed for those 32 alternatives.

Provenance correction: the initial report's sentence that no small integer oracle was run is superseded by this separately sealed CP32 supplement. The initial native source and results remain unchanged. Refer to supplement.json/final-completion.json for the final joined provenance.
'''
    (DEST/'supplement.md').write_text(text)
    paths=[Path(__file__),DEST/'supplement.json',DEST/'supplement.md',DEST/'completion.json',
           EXACT/'completion.json',CP/'completion.json']
    (DEST/'final-completion.json').write_text(json.dumps(dict(status='sealed_final_bounded_no_go',files=[
        dict(path=str(x),sha256=sha(x)) for x in paths]),indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],final_completion_sha256=sha(DEST/'final-completion.json')),indent=2))


if __name__=='__main__':main()
