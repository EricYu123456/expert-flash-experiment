#!/usr/bin/env python3
"""Constructive generic replicated-tile/coalescing control for fixed pair copies."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round2/deep-shared-boundary'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sdwork(rows):
    rows=np.asarray(rows,dtype=np.int64)
    return ((rows*2816+65535)//65536)*28000+np.where(rows>0,1658+rows*2,0)


def main():
    doc=json.loads((OUT/'full/oracle.json').read_text());checked=0;maximum_difference=0
    for record in doc['records']:
        with np.load(record['npz']) as f:
            occupied=f['routed_native_chip_ns']
            for r in (0,1,2,3,11):
                label='r'+str(r);u=occupied+sdwork(f[label+'_down_rows'])
                lo,hi=11-r,11+r
                minima=[]
                for p in range(8):
                    target=(22*28064+u[:,p+8]-u[:,p])/(2*28064)
                    # Endpoints handle the positive-shard fixed input/command fee.
                    candidates=np.stack([np.floor(target),np.ceil(target),
                        np.full(len(u),lo),np.full(len(u),hi)],axis=1).astype(np.int64)
                    candidates=np.clip(candidates,lo,hi)
                    values=np.maximum(u[:,p,None]+28064*candidates+1274*(candidates>0),
                        u[:,p+8,None]+28064*(22-candidates)+1274*(candidates<22))
                    minima.append(values.min(axis=1))
                    checked+=len(u)
                reconstructed=np.stack(minima,axis=1).max(axis=1)+2000
                original=f[label+'_native_chip_bound_ns']
                maximum_difference=max(maximum_difference,int(np.abs(reconstructed-original).max()))
                assert np.array_equal(reconstructed,original)
    result=dict(status='PASS constructive fixed-copy generic-control equality',source_sha256=sha(__file__),
        original_oracle_sha256=sha(OUT/'full/oracle.json'),checked_pair_queries=checked,
        maximum_native_bound_difference_ns=maximum_difference,
        model='Each pair has22 full32row×2048I8 tiles; first11-r onlyA, last11-r onlyB, middle2r on both owners.',
        proof=[
            'Any generic independent two-choice assignment of the2r replicated tiles gives A some a in[11-r,11+r] tiles and B22-a.',
            'A positive owner must pay at least one native command and one full input. Each tile pays28000ns read/compute plus64ns output-channel drain.',
            'For every feasible a, choose A the first a tiles and B the remaining suffix. Both are within fixed stored ranges; each positive owner uses exactly one contiguous native call.',
            'Therefore optimal coalescing attains the lower per-chip occupancy for that tile-count assignment, with no new contexts, sparse instruction, or physical concatenation.',
            'Minimizing max(uA+28064a+1274[a>0],uB+28064(22-a)+1274[a<22]) is exactly the fixed-pair generic replicated-tile retrieval control under these occupancy costs.',
            'The interior optimum is nearest to(22+(uB-uA)/28064)/2, clipped to[11-r,11+r], with endpoints checked for zero-shard command elimination.'
        ],
        implication='The fixed-copy split selector is not a new allocation algorithm. A cost-aware generic tile replica selector with optimal contiguous coalescing can exactly match it.',
        limits=['Equality covers the fixed paired owner/copy graph and native-chip occupancy objective; it is not a theorem that all placements or true DAG schedules are solved.',
            'uA/uB include future routed/shared-down occupancy, not measured initial availability. Full DAG replay is still needed.',
            'A systems contribution can remain in physical image lowering, owner/budget selection, runtime costs and validated dependence-aware planning; expressibility alone does not disprove it.'])
    p=OUT/'generic_control_equivalence.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(p),sha256=sha(p),checked=checked,max_difference_ns=maximum_difference)))


if __name__=='__main__':main()
