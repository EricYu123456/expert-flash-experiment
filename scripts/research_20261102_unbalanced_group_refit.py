#!/usr/bin/env python3
"""Add matched-proposal-count balanced swaps to the frozen relabel baseline.

Writes a new audit version; never changes the original frozen screen/main plan.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from research_20261102_unbalanced_group import DEST, inputs, inventory, mean, write
from run_corpus_traces import ROOT,sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True)
    ap.add_argument('--full',action='store_true');a=ap.parse_args()
    label='full' if a.full else 'screen'
    path=DEST/(a.model+'.'+label+'.frozen.json');source_sha=sha256(path)
    source=json.loads(path.read_text())
    source_path,doc,route_path,routes,ids,layers,all_phases=inputs(a.model)
    assert source['source_plan_sha256']==sha256(source_path)
    cpp=ROOT/'scripts/research_20261102_geometry_solver.cpp'
    target=DEST/('balanced-swap-'+sha256(cpp)+'.so')
    if not target.exists():subprocess.run(['g++','-std=c++17','-O3','-march=native','-shared','-fPIC',str(cpp),'-o',str(target)],check=True)
    lib=ctypes.CDLL(str(target));p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.optimize_geometry.argtypes=[p32,p32,p64]+[ctypes.c_int]*6+[ctypes.c_uint64,ctypes.c_double,p64]
    lib.optimize_geometry.restype=ctypes.c_double
    updated=[dict(p) for p in all_phases];out=[]
    for p in source['phases']:
        li=layers.index(p['layer']);pi=next(i for i,q in enumerate(all_phases) if q['layer']==p['layer'] and q['name']==p['name'])
        selected,weights=np.unique(np.sort(routes[:,:,li].reshape(-1,doc['active']),axis=1),axis=0,return_counts=True)
        selected=np.ascontiguousarray(selected,dtype=np.int32);weights=np.ascontiguousarray(weights,dtype=np.int64)
        controls=[]
        for c in p['candidates']:
            g=c['g'];seeds=[s for s in all_phases[pi]['seeds'] if np.count_nonzero(s['rows'][0])==g]
            seed=min(seeds,key=lambda s:mean(selected,weights,np.asarray(s['rows']),p))
            best=np.ascontiguousarray(seed['rows'],dtype=np.int32);best_mean=mean(selected,weights,best,p);search=[]
            for ti,s in enumerate(c['search']):
                fitted=np.ascontiguousarray(seed['rows'],dtype=np.int32);stats=np.zeros(2,dtype=np.int64)
                # Match evaluated single-move proposals, including coordinate descent.
                iterations=s['valid']
                cost=lib.optimize_geometry(fitted,selected,weights,doc['experts'],len(selected),doc['active'],p['columns'],int(p['common']),iterations,20261102+pi*1009+g*31+ti,s['temperature_ns'],stats)
                exact=mean(selected,weights,fitted,p)
                assert abs(cost-exact)<1e-6
                np.testing.assert_array_equal(np.sort(fitted,axis=0),np.sort(np.asarray(seed['rows']),axis=0))
                search.append(dict(temperature_ns=s['temperature_ns'],proposal_iterations=iterations,valid=int(stats[0]),accepted=int(stats[1]),mean_ns=exact))
                if exact<best_mean:best_mean=exact;best=fitted.copy()
            controls.append(dict(g=g,mean_ns=best_mean,rows=best.tolist(),search=search))
        balanced=min(controls,key=lambda c:c['mean_ns']);unbalanced=min(p['candidates'],key=lambda c:c['mean_ns'])
        use_unbalanced=unbalanced['mean_ns']<balanced['mean_ns']
        chosen=unbalanced if use_unbalanced else balanced
        updated[pi]['rows']=chosen['rows']
        phase=dict(layer=p['layer'],name=p['name'],total_rows=p['total_rows'],columns=p['columns'],common=p['common'],
            original_group_ns=p['group_ns'],original_main_ns=p['original_main_ns'],
            balanced_refit_ns=balanced['mean_ns'],balanced_refit_rows=balanced['rows'],balanced_refit_g=balanced['g'],
            single_relabel_ns=unbalanced['mean_ns'],single_relabel_rows=unbalanced['rows'],
            group_ns=balanced['mean_ns'],group_rows=balanced['rows'],
            chosen_ns=chosen['mean_ns'],chosen_rows=chosen['rows'],chosen_g=chosen['g'],
            chosen_kind='single_relabel' if use_unbalanced else 'balanced_refit',balanced_controls=controls,
            unbalanced_candidates=p['candidates'])
        out.append(phase)
        print(a.model,p['layer'],p['name'],'old',p['group_ns'],'balancedrefit',balanced['mean_ns'],'single',unbalanced['mean_ns'],'chosen',phase['chosen_kind'],flush=True)
    cap=inventory(updated,doc['experts']);assert cap['within_actual_lsb_cap']
    assert cap['total_physical_pages']==source['baseline_whole_model_inventory']['total_physical_pages']
    assert sha256(path)==source_sha
    result=dict(model=a.model,status='training-only stronger baseline audit; no main plan changes',phases=out,
        original_relabel_plan=str(path),original_relabel_plan_sha256=source_sha,
        source_plan=str(source_path),source_plan_sha256=sha256(source_path),
        training_routes=str(route_path),training_routes_sha256=sha256(route_path),
        train_ids=ids,steps=64,layers=source['layers'],experts=doc['experts'],active=doc['active'],
        selected_training_slice=source['selected_training_slice'],frozen_utc=datetime.now(timezone.utc).isoformat(),
        training_totals_ns={k:sum(p[k] for p in out) for k in ['original_group_ns','balanced_refit_ns','single_relabel_ns','chosen_ns','original_main_ns']},
        updated_whole_model_inventory=cap,
        whole_model_rows=[{k:p[k] for k in ['layer','name','total_rows','columns','common','rows']} for p in updated],
        comparison='Balanced swapping uses the same temperatures and a number of loop iterations equal to all evaluated single-relabel proposals in the corresponding anneal+descent run. Some swap iterations are invalid, so valid proposal counts are reported. Equal proposal count does not imply equal wall-clock work or global optimality.',
        limitations=source['limitations'],source_sha256={Path(__file__).name:sha256(Path(__file__)),cpp.name:sha256(cpp)})
    final=DEST/(a.model+'.'+label+'.matched.frozen.json');write(final,result)
    print(str(final),json.dumps(result['training_totals_ns']),flush=True)


if __name__=='__main__':main()
