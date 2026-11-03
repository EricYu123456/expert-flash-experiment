#!/usr/bin/env python3
"""Small integer clique-demand addendum to the TRAIN closure screen."""
import argparse
import hashlib
import itertools
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/m32-pair-closure-placement-screen'
OUT=BASE/'clique-demand-addendum'
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),TRAIN,BASE/'protocol.frozen.json',BASE/'completion.json',BASE/'graph_arrays.npz',BASE/'maximal_cliques.json',BASE/'graph_summary.json']
    refs=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    shutil.copy2(__file__,OUT/Path(__file__).name)
    dump(OUT/'protocol.frozen.json',dict(status='analytic_addendum_after_scalar_capacity_screen',utc=datetime.now(timezone.utc).isoformat(),
      source_sha256=sha(__file__),references=refs,
      origin='The requested uniform capacity upper bound469>=320 was inconclusive. This stronger subset-demand observation was made after viewing TRAIN graph/solver output, before this certificate check; not claimed preregistered.',
      deterministic_witness='First saved maximal clique of size16, then first12 sorted experts. No dev/fresh graph or route is opened.',
      proof='For a cliqueC with12experts, all6subsets are protected. For cap1/2/3, sumC X mustbe<=cap: anycap+1 marked experts can extendto a6subset. Therefore32chip capacity55 cannot meet12*5=60.',
      checks='All66pairs have originalTRAINsupport with earliestquery witness, all924sixsubsets are graphcliques, integertruth table min(6,m)<=cap iffm<=cap for m0..12/cap1..3.',
      scope='Only coupled X5/O7 union-complete perexpert profile with fixed sharedlayout and pair-closure routefamily. Not independent phasegeometry, allrouter robustness or newgraph theorem. No fitting/native/inference/devcoverage.'))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__)),indent=2))


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
    for ref in p['references']:assert sha(ref['path'])==ref['sha256']
    cliques=json.loads((BASE/'maximal_cliques.json').read_text());large=next(c for c in cliques if len(c['experts'])==16);C=large['experts'][:12]
    with np.load(TRAIN) as z:
        assert str(z['split'])=='train';routes=z['routes'][:,:,0].reshape(512,6)
    present=np.zeros((512,64),bool);present[np.arange(512)[:,None],routes]=True
    pairs=[]
    for a,b in itertools.combinations(C,2):
        queries=np.flatnonzero(present[:,a]&present[:,b]);assert len(queries)>0
        pairs.append(dict(a=a,b=b,TRAIN_joint_count=len(queries),earliest_TRAIN_query=int(queries[0])))
    assert len(pairs)==66
    with np.load(BASE/'graph_arrays.npz') as z:
        adjacency=z['adjacency_masks'];shared=z['shared_waves']
        assert all(int(adjacency[a])&(1<<b) for a,b in itertools.combinations(C,2))
    six=0
    for route in itertools.combinations(C,6):
        assert all(int(adjacency[a])&(1<<b) for a,b in itertools.combinations(route,2));six+=1
    assert six==924
    truth=[]
    for cap in (1,2,3):
        for marked in range(13):
            worst=min(6,marked);assert (worst<=cap)==(marked<=cap)
            truth.append(dict(cap=cap,total_X_in_C=marked,maximum_X_in_six_subset=worst,allowed=marked<=cap))
    caps=4-shared;assert np.all((caps>=1)&(caps<=3))
    capacity=int(caps.sum());demand=5*len(C);assert capacity==55 and demand==60 and demand>capacity
    report=dict(status='CERTIFIED_PEAK10_IMPOSSIBLE_FOR_STATED_CLOSURE_AND_COUPLED_PROFILE',
       protocol_sha256=sha(OUT/'protocol.frozen.json'),clique=C,parent_maximal_clique=large,
       pair_support=pairs,protected_six_subsets_checked=six,integer_truth_table=truth,
       perchip_shared_waves=shared.tolist(),perchip_C_X_cap=caps.tolist(),total_C_X_capacity=capacity,
       perexpert_required_X=5,clique_required_X=demand,deficit=demand-capacity,
       integer_contradiction='Sum_c Sum_e_in_C X[e,c] = 12*5 = 60, but <=12*1+17*2+3*3=55.',
       requested_uniform_screen='U1=9,U2=17,U3=24 andweighted469>=320 remains true buttoo loose because singlechip maxima canfavor differentexpert subsets.',
       interpretation='A standard clique-demand/pigeonhole certificate. Closure conservatism has a real physical cost; no jointplacement fit isneeded to ruleout peak10 inthisfamily.',
       limits=p['scope'],dev_read=False,no_fit=True,no_native_or_inference=True)
    dump(OUT/'certificate.json',report)
    (OUT/'proof.md').write_text('For C='+str(C)+', all66pairs occur in original TRAIN; hence every one of its924six-subsets belongs to the frozen pair closure.\n\n'
      'Assume every protected route has NAND peak<=10. On chip c with shared waves S_c, every six-subset has at most4-S_c X experts. Since4-S_c is1,2or3, any larger number of marked C experts could be extended to a violating six-subset. Thus sum_{e in C}X[e,c]<=4-S_c.\n\n'
      'Summing all32chips gives at most12*1+17*2+3*3=55 X incidences. But12experts each require5X placements, so the same sum is60. Contradiction.\n\n'
      'This applies only to the stated pair closure, coupledX5/O7 profile and fixed shared layout; no dev data or universal-router robustness claim. The earlier uniform469>=320 necessary screen remains inconclusive on its own.\n')
    dump(OUT/'completion.json',dict(status='sealed_integer_clique_demand_certificate',utc=datetime.now(timezone.utc).isoformat(),
       files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(OUT.rglob('*')) if x.is_file()]))
    print(json.dumps(dict(status=report['status'],clique=C,capacity=capacity,demand=demand,
                         completion_sha256=sha(OUT/'completion.json')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));a=parser.parse_args();{'freeze':freeze,'run':run}[a.mode]()
