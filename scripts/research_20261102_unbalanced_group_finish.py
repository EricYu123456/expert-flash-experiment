#!/usr/bin/env python3
"""Close strong balanced-refit controls under greedy single-expert moves."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from research_20261102_projection_study import equal_rows
from research_20261102_unbalanced_group import DEST,inputs,inventory,mean,write
from run_corpus_traces import sha256


def library():
    cpp=Path(__file__).with_name('research_20261102_unbalanced_group.cpp')
    target=DEST/('single-relabel-'+sha256(cpp)+'.so')
    if not target.exists():
        temporary=target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
        subprocess.run(['g++','-std=c++17','-O3','-march=native','-shared','-fPIC',str(cpp),'-o',str(temporary)],check=True)
        os.replace(temporary,target)
    lib=ctypes.CDLL(str(target))
    p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.optimize_unbalanced_group.argtypes=[p32,p32,p32,p64]+[ctypes.c_int]*8+[ctypes.c_uint64,ctypes.c_double,p64]
    lib.optimize_unbalanced_group.restype=ctypes.c_double
    return lib


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--full',action='store_true');a=ap.parse_args()
    label='full' if a.full else 'screen'
    path=DEST/(a.model+'.'+label+'.matched.frozen.json');before=sha256(path);doc=json.loads(path.read_text())
    original,meta,route_path,routes,ids,layers,all_phases=inputs(a.model)
    assert sha256(original)==doc['source_plan_sha256']
    lib=library();updated=[dict(p) for p in all_phases];out=[]
    for p in doc['phases']:
        li=layers.index(p['layer']);pi=next(i for i,q in enumerate(all_phases) if q['layer']==p['layer'] and q['name']==p['name'])
        selected,weights=np.unique(np.sort(routes[:,:,li].reshape(-1,meta['active']),axis=1),axis=0,return_counts=True)
        selected=np.ascontiguousarray(selected,dtype=np.int32);weights=np.ascontiguousarray(weights,dtype=np.int64)
        candidates=[]
        for c in p['balanced_controls']:
            g=c['g'];templates=np.ascontiguousarray(equal_rows(p['total_rows'],16//g,g),dtype=np.int32)
            lookup={tuple(row):i for i,row in enumerate(templates)}
            labels=np.ascontiguousarray([lookup[tuple(row)] for row in c['rows']],dtype=np.int32)
            stats=np.zeros(3,dtype=np.int64)
            cost=lib.optimize_unbalanced_group(labels,templates,selected,weights,meta['experts'],len(templates),len(selected),meta['active'],p['columns'],int(p['common']),0,20,20261102+pi*1009+g,0.,stats)
            rows=templates[labels];exact=mean(selected,weights,rows,p);assert abs(exact-cost)<1e-6
            proposed=[dict(q) for q in updated];proposed[pi]['rows']=rows.tolist();cap=inventory(proposed,meta['experts'])
            candidates.append(dict(g=g,mean_ns=exact,rows=rows.tolist(),group_counts=np.bincount(labels,minlength=16//g).tolist(),
                valid_proposals=int(stats[0]),greedy_moves=int(stats[2]),within_actual_cap=cap['within_actual_lsb_cap'],
                max_reserved_lsb_bytes_per_chip=cap['max_reserved_lsb_bytes_per_chip']))
        eligible=[c for c in candidates if c['within_actual_cap']]
        winner=min(eligible,key=lambda c:c['mean_ns'])
        if winner['mean_ns']<p['chosen_ns']:
            rows=winner['rows'];chosen_ns=winner['mean_ns'];g=winner['g'];kind='greedy_from_balanced_refit'
        else:rows=p['chosen_rows'];chosen_ns=p['chosen_ns'];g=p['chosen_g'];kind=p['chosen_kind']
        updated[pi]['rows']=rows
        phase=dict(p,chosen_ns=chosen_ns,chosen_rows=rows,chosen_g=g,chosen_kind=kind,greedy_from_balanced_controls=candidates)
        out.append(phase)
        print(a.model,p['layer'],p['name'],'balanced',p['balanced_refit_ns'],'strongrelaxed',chosen_ns,'g',g,'kind',kind,flush=True)
    cap=inventory(updated,meta['experts']);assert cap['within_actual_lsb_cap']
    assert cap['total_useful_bytes']==doc['updated_whole_model_inventory']['total_useful_bytes']
    assert cap['total_physical_pages']==doc['updated_whole_model_inventory']['total_physical_pages']
    assert sha256(path)==before
    final=dict(doc,phases=out,parent_matched_plan=str(path),parent_matched_plan_sha256=before,
        frozen_utc=datetime.now(timezone.utc).isoformat(),updated_whole_model_inventory=cap,
        training_totals_ns={k:sum(p[k] for p in out) for k in ['original_group_ns','balanced_refit_ns','single_relabel_ns','chosen_ns','original_main_ns']},
        whole_model_rows=[{k:p[k] for k in ['layer','name','total_rows','columns','common','rows']} for p in updated],
        finish_source_sha256=sha256(Path(__file__)),
        finish_rule='Every balanced-refit candidate is additionally closed under 20 greedy coordinate sweeps of single-expert relabels, then the best feasible old or new result is retained. No heldout routes opened.')
    target=DEST/(a.model+'.'+label+'.strong.frozen.json');write(target,final)
    print(str(target),json.dumps(final['training_totals_ns']),flush=True)


if __name__=='__main__':main()
