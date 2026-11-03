#!/usr/bin/env python3
"""Bounded 32 fixed-TRAIN-query integer-work oracle; not a native selector fit."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ortools
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2/g4-elastic-prefix-screen/work-v3-independent-prefix'
OUT = BASE/'cp32-integer-audit'
INDICES = np.linspace(0,511,32,dtype=int)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    paths = [Path(__file__),BASE/'protocol.frozen.json',BASE/'contract_arrays.npz',
             BASE/'greedy_arrays.npz',BASE/'native-v1/report.json',
             BASE/'integer-dual-audit/protocol.frozen.json',
             BASE/'integer-dual-audit/completion.json',BASE/'integer-dual-audit/certificates.npz']
    shutil.copy2(__file__,OUT/'source.py')
    dump(OUT/'protocol.frozen.json',dict(status='frozen_before_CP32',utc=datetime.now(timezone.utc).isoformat(),
         source_sha256=sha(__file__),inputs=[dict(path=str(x),sha256=sha(x)) for x in paths],
         solver=dict(python=sys.executable,ortools_version=ortools.__version__,max_time_s=0.5,workers=1,random_seed=20261112),
         fixed_layout=3,indices=INDICES.tolist(),index_rule='linspace(0,511,32,dtype=int), fixed without looking at improvements',
         model='48Booleanu+6Booleanz AND8+integerT,16actualnative chipownershipwork constraints; minimizeT',
         incumbent='Greedy complete assignment is a hint and hard upper bound; retain if solver UNKNOWN. No incumbent loss.',
         scope='Only32 TRAIN integerwork diagnostic. No native rerun/512fit/heldout/source relocation/selector benchmark. Report FEASIBLE separately fromOPTIMAL.'))
    print(sha(OUT/'protocol.frozen.json'))


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for x in p['inputs']:
        assert sha(x['path'])==x['sha256']
    assert ortools.__version__==p['solver']['ortools_version']
    assert not (OUT/'report.json').exists()
    a=np.load(BASE/'contract_arrays.npz');g=np.load(BASE/'greedy_arrays.npz')
    cert=np.load(BASE/'integer-dual-audit/certificates.npz')
    rows=[];outbits=[]
    t0=time.monotonic()
    for q in p['indices']:
        selected=a['routes'][q]
        D=a['D'][3,selected];Z=a['Z'][selected];base=a['base'][q]
        initial=g['feasible_bits'][3,q]
        initial_work=int(g['feasible_work'][3,q])
        lower=int(cert['work_lower_bound'][3,q])
        model=cp_model.CpModel()
        u=[[model.NewBoolVar(f'u{e}_{i}') for i in range(8)] for e in range(6)]
        z=[model.NewBoolVar(f'z{e}') for e in range(6)]
        T=model.NewIntVar(lower,initial_work,'T')
        for e in range(6):
            for i in range(8):
                model.Add(z[e]<=u[e][i])
                model.AddHint(u[e][i],int(initial[e,i]))
            model.Add(z[e]>=sum(u[e])-7)
            model.AddHint(z[e],int(initial[e].all()))
        model.AddHint(T,initial_work)
        for chip in range(16):
            model.Add(int(base[chip])+sum(int(D[e,i,chip])*u[e][i] for e in range(6) for i in range(8))+
                      sum(int(Z[e,chip])*z[e] for e in range(6))<=T)
        model.Minimize(T)
        solver=cp_model.CpSolver()
        solver.parameters.max_time_in_seconds=0.5
        solver.parameters.num_search_workers=1
        solver.parameters.random_seed=20261112
        status=solver.Solve(model)
        bits=initial.copy();best=initial_work
        found=status in (cp_model.OPTIMAL,cp_model.FEASIBLE)
        if found:
            candidate=np.asarray([[solver.Value(u[e][i]) for i in range(8)] for e in range(6)],np.int8)
            loads=base+np.einsum('ei,eic->c',candidate,D)+(candidate.all(1)[:,None]*Z).sum(0)
            assert loads.max()==solver.Value(T)<=initial_work
            best=int(loads.max());bits=candidate
        # Integer objective: floor solver bound conservatively, plus exact LPcertificate.
        bound=max(lower,int(np.floor(solver.BestObjectiveBound())))
        assert bound<=best
        if status==cp_model.OPTIMAL:
            assert best==bound
        outbits.append(bits)
        entry=dict(query=q,status=solver.StatusName(status),greedy_work_ns=initial_work,best_work_ns=best,
                   best_bound_ns=bound,exact_rational_LP_bound_ns=lower,improvement_ns=initial_work-best,
                   optimality_gap_ns=best-bound,wall_seconds=solver.WallTime(),
                   branches=int(solver.NumBranches()),conflicts=int(solver.NumConflicts()),
                   retained_greedy_incumbent=not found)
        rows.append(entry)
        print(json.dumps(entry),flush=True)
    saved=np.asarray(outbits,np.int8)
    np.savez_compressed(OUT/'arrays.npz',indices=np.asarray(p['indices'],np.int32),bits=saved,
                        greedy_work=np.asarray([r['greedy_work_ns'] for r in rows]),
                        best_work=np.asarray([r['best_work_ns'] for r in rows]),
                        best_bound=np.asarray([r['best_bound_ns'] for r in rows]))
    improvements=[r['improvement_ns'] for r in rows]
    report=dict(status='bounded_integerwork_audit_not_native',protocol_sha256=sha(OUT/'protocol.frozen.json'),
                source_sha256=sha(__file__),cases=rows,optimal_count=sum(r['status']=='OPTIMAL' for r in rows),
                feasible_count=sum(r['status']=='FEASIBLE' for r in rows),unknown_count=sum(r['status']=='UNKNOWN' for r in rows),
                mean_greedy_work_ns=float(np.mean([r['greedy_work_ns'] for r in rows])),
                mean_best_work_ns=float(np.mean([r['best_work_ns'] for r in rows])),
                mean_improvement_ns=float(np.mean(improvements)),max_improvement_ns=max(improvements),
                cases_improved_gt3us=sum(x>3000 for x in improvements),seconds=time.monotonic()-t0,
                limits='32fixedTRAINcases only, not full512expectation or native latency. FEASIBLE cases retain nonzero bounds/gaps. No newphysicalplacement or samebudget control was fit.')
    dump(OUT/'report.json',report)
    dump(OUT/'completion.json',dict(status='sealed_bounded_CP32',files=[dict(path=str(x),sha256=sha(x)) for x in [
        OUT/'protocol.frozen.json',OUT/'source.py',OUT/'report.json',OUT/'arrays.npz']]))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
