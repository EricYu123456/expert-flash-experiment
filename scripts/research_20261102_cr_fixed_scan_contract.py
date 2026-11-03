#!/usr/bin/env python3
"""Small exhaustive certificate for a narrowly stated fixed-scan obstruction.

The symbolic reduction is supplied explicitly, not replaced with a tiny-model
solver claim. This does not establish what the actual AiF selector supports.
"""
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round3-cr-ipr/fixed-scan-contract'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'certificate.json').exists()
    profiles=[]
    triples=[tuple((start+j)%4 for j in range(3)) for start in range(4)]
    for v in itertools.product(range(3),repeat=4):
        if sum(v)!=6:continue
        shared=tuple(12-6*n for n in v)
        assert sum(shared)==12 and min(shared)>=0
        support={p for p,n in enumerate(shared) if n}
        possible=[t for t in triples if set(t)<=support]
        assert 1<=len(support)<=2 and not possible
        profiles.append(dict(routed_pages=v,shared_pages=shared,
                             shared_support=sorted(support),
                             legal_three_shared_page_fixed_scan_starts=possible))
    assert len(profiles)==10 and len(triples)==4
    result=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statement='No one-copy G4 layout can guarantee the 12-wave/48-page lower bound for every top6 under the exact fixed cyclic drain and zero-overfetch row-stream contract below.',
        conditions=[
            'E=64 homogeneous routed experts, K=6, six coefficient pages/expert/chip and twelve always-read shared pages/chip.',
            'All logical coefficients stored once; no unwanted weight bytes, padding, coding or partial-column reduction read.',
            'One active row accumulator; changing owner only at a complete row. Each 1536-byte row uses its original column order.',
            'ECC/PE consumes a strict unmasked infinite cyclic plane sequence 0,1,2,3; it cannot skip a source, permute sources or reorder buffered pages.',
            'A 12-wave optimum consumes 48 useful pages with each plane supplying exactly 12.'
        ],
        reduction=[
            'Zero unwanted bytes for every top6 implies a physical page cannot mix two routed experts or a routed expert and shared weights: a query can include one owner while excluding the other.',
            'Compare two top6 sets differing in one expert. Exact saturation of every plane forces all routed experts to have the same integer page vector v.',
            'For every plane 6*v[p]+shared[p]=12, hence v[p]<=2 and sum(v)=6. The ten possibilities have type(2,2,1,1) or(2,2,2,0).',
            'Shared weights therefore occupy at most two planes.',
            'Because gcd(16384,1536)=512, an owner switch between exclusive full pages requires a multiple of three pages. Shared has twelve pages and must contain a run of at least three consecutive shared pages.',
            'Every three consecutive entries of the strict cyclic scan visit three distinct planes. Such a shared run is impossible when its support has at most two planes.'
        ],profiles=profiles,fixed_scan_triples=triples,exhaustive_profile_triple_checks=40,
        exclusions=[
            'Does not prohibit a 12-sense-wave construction when existing source skipping, a programmable selector, or buffered page reordering is available.',
            'Does not establish latency or source-selector properties of a real chip.',
            'Does not prove a larger universal minimum once overfetch, changing profile, or a different execution contract is allowed.'
        ])
    (OUT/'certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Fixed cyclic source contract obstruction','',result['statement'],'',
          'The certificate enumerates all ten forced routed profiles and all four cyclic three-page starts. All forty shared-run checks fail, matching the symbolic reduction in certificate.json.','',
          'This is a capability boundary for the proposed lowering, not a claim that AiF has a fixed selector and not a stand-alone novelty claim. The 12-wave stencil uses programmable source permutations and is outside this restrictive contract.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='PASS',profiles=len(profiles),checks=40)))


if __name__=='__main__':main()
