#!/usr/bin/env python3
"""Separate v2 strong controls; never modifies the frozen v1 campaign."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import research_20261103_shared_counterbalanced as v1
from research_20261103_topology_placement import sha
from research_20261103_topology_placement_replay import Replay,load_python

OUT=v1.OUT/'v2'


class PairedFit(v1.Fit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.move_counts={k:0 for k in ['single_global','paired_global','single_local','paired_local','single_label','paired_label']}

    def global_move(self,p,a,b,temp,greedy=False):
        paired=bool(self.rng.integers(2));self.move_counts['paired_global' if paired else 'single_global']+=1
        original=self.kind
        if paired:self.kind='coupled'
        try:return super().global_move(p,a,b,temp,greedy)
        finally:self.kind=original

    def local_move(self,p,e,a,b,temp,greedy=False):
        paired=bool(self.rng.integers(2));self.move_counts['paired_local' if paired else 'single_local']+=1
        original=self.kind
        if paired:self.kind='coupled'
        try:return super().local_move(p,e,a,b,temp,greedy)
        finally:self.kind=original

    def label_move(self,p,e,newlabel,temp,greedy=False):
        if not self.rng.integers(2):
            self.move_counts['single_label']+=1
            return super().label_move(p,e,newlabel,temp,greedy)
        self.move_counts['paired_label']+=1;ii=self.members[e]
        if not len(ii):return False
        q=1-p;other=int(self.rng.integers(len(self.variants[q])))
        phases=[p,q];labels=[newlabel,other];changes=[];newrows=[];newcost=[]
        for phase,label in zip(phases,labels):
            rr=self.variants[phase][label,self.perm[phase]];cc=v1.cost(rr,self.columns[phase],self.s)
            changes.append((phase,ii,np.broadcast_to(cc-self.cc[e,phase],(len(ii),16))))
            newrows.append(rr);newcost.append(cc)
        def mutation():
            for phase,label,rr,cc in zip(phases,labels,newrows,newcost):
                self.rows[e,phase]=rr;self.cc[e,phase]=cc;self.labels[phase][e]=label
        return self.apply(changes,mutation,temp,greedy)

    def run(self,*args,**kwargs):
        rows,statistics=super().run(*args,**kwargs)
        statistics['move_attempts']=self.move_counts.copy()
        statistics['unrestricted_single_and_paired_phase_moves']=True
        return rows,statistics


def orbit(base):
    return min(tuple(np.roll(base,k).tolist()) for k in range(16))


def menu_candidates(rec,p):
    result=[]
    for c in rec['fitting'][p]:
        base=np.asarray(c['base_rows']);ll=np.asarray(c['labels']);width=int(np.count_nonzero(base))
        shifts=ll*width if c['family']=='disjoint' else ll
        rows=np.asarray([np.roll(base,int(v)) for v in shifts])
        result.append(dict(rows=rows,label=c['label'],geometry=orbit(base),family=c['family']))
    assert any(c['family']=='whole_rows' for c in result)
    return result


def full_menu_pool(rec,routes,shared,columns,s,incumbent,limit=4):
    gu=menu_candidates(rec,0);down=menu_candidates(rec,1)
    gc=[v1.cost(c['rows'],columns[0],s)[routes].sum(1) for c in gu]
    dc=[v1.cost(c['rows'],columns[1],s)[routes].sum(1) for c in down]
    best={}
    for a,g in enumerate(gu):
        for b,d in enumerate(down):
            key=(g['geometry'],d['geometry'])
            for rotation in range(16):
                value=float((gc[a]+np.roll(dc[b],rotation,axis=1)+shared).max(1).mean())
                if key not in best or value<best[key]['initial_work_ns']:
                    best[key]=dict(gu=g,down=d,rotation=rotation,initial_work_ns=value)
    incumbent_key=(orbit(incumbent[0,0]),orbit(incumbent[0,1]))
    ranked=sorted(best.items(),key=lambda pair:(pair[1]['initial_work_ns'],pair[0]))
    selected=[c for key,c in ranked if key!=incumbent_key][:limit-1]
    pool=[dict(rows=incumbent.copy(),label='frozen_whole_rows_simple_incumbent',
               geometry=incumbent_key,initial_work_ns=float(v1.work(incumbent,routes,shared,columns,s).mean()))]
    for c in selected:
        rows=np.stack([c['gu']['rows'],np.roll(c['down']['rows'],c['rotation'],axis=1)],axis=1)
        pool.append(dict(rows=rows,label=c['gu']['label']+'+'+c['down']['label']+'@'+str(c['rotation']),
                         geometry=(c['gu']['geometry'],c['down']['geometry']),initial_work_ns=c['initial_work_ns']))
    assert len(pool)==limit
    return pool,dict(phase_candidates=[len(gu),len(down)],distinct_geometry_pairs=len(best),
                     contains_whole_rows_templates=True,rotation_trials=16,
                     selection='top3distinct cyclic-orbit geometrypairs plus actual frozen whole_rows/simple incumbent',
                     guarantee='finite pool/heuristic only, not fullmenu global permutation optimum')


def best_v1_context(case,context,families):
    result=[]
    for r in case['records']:
        if (r['shared_geometry'],r['seed_index'])!=context or r['family'] not in families:continue
        for stage in ['initial','final']:
            result.append((r[stage]['mean_ns'],r[stage],r['family'],stage))
    return min(result,key=lambda x:x[0])


def run_case(args):
    started=time.monotonic();protocol=json.loads((OUT/'protocol.frozen.json').read_text())
    assert sha(__file__)==protocol['source_sha256']
    assert sha(v1.__file__)==protocol['v1_source_sha256']
    assert sha(v1.OUT/'completion.v1.json')==protocol['v1_completion_sha256']
    for source in protocol['sources']:assert sha(source['path'])==source['sha256']
    sourcecase=v1.OUT/f'{args.model}.L{args.layer}.input{args.copies}/result.json'
    old=json.loads(sourcecase.read_text());s=old['service'];shape=old['shape'];columns=shape[:2]
    summary=json.loads(Path(old['source_summary']).read_text())
    rec=next(r for r in summary['records'] if r['model']==args.model and r['layer']==args.layer and r['command_ns']==250)
    with np.load(old['training']['path']) as z:routes=np.ascontiguousarray(z['routes'][:,:,args.layer].reshape(-1,6),np.int32)
    assert sha(old['training']['path'])==old['training']['sha256']
    unique,weights=np.unique(np.sort(routes,axis=1),axis=0,return_counts=True)
    art=next(a for a in rec['artifacts'] if a['family']=='whole_rows')
    with np.load(v1.ROOT/art['arrays']) as z:pp={k:z[k] for k in z.files}
    incumbent=v1.rows_from_plan(pp)[:-1]
    native=Replay();backend=load_python();dest=OUT/f'{args.model}.L{args.layer}.input{args.copies}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'result.json').exists()
    contexts={}
    for r in old['records']:
        if r['family']=='counterbalanced':contexts[(r['shared_geometry'],r['seed_index'])]=r
    records=[]
    for context,guide in contexts.items():
        shared_name,si=context;seed=v1.SEEDS[si];shared_rows=np.asarray(guide['shared_rows'])
        shared=sum(v1.cost(shared_rows[p],C,s) for p,C in enumerate([shape[0],shape[5]]))
        with np.load(guide['initial']['arrays']) as z:routeblind=z['rows'][:-1].copy()
        _,retained,retained_family,retained_stage=best_v1_context(old,context,['counterbalanced','independent_permutation'])
        with np.load(retained['arrays']) as z:guide_winner=z['rows'][:-1].copy()
        # Shared pool is literally inherited from v1; no family-specific masks.
        initial_independent=routeblind if si==0 else guide_winner
        pool,poolproof=full_menu_pool(rec,routes,shared,columns,s,incumbent)
        families=[('joint_simple_menu',[(c['label'],c['rows']) for c in pool]),
                  ('independent_paired',[('routeblind_coupled' if si==0 else 'retained_v1_incumbent',initial_independent)])]
        for family,starts in families:
            inherited=best_v1_context(old,context,['joint_cyclic'] if family=='joint_simple_menu' else ['counterbalanced','independent_permutation'])
            for pi,(label,initial) in enumerate(starts):
                kind='cyclic' if family=='joint_simple_menu' else 'independent'
                fit=PairedFit(initial,kind,unique,weights,shared,columns,s,seed)
                final,statistics=fit.run(20000//len(starts),4096//len(starts))
                stem=shared_name+'.seed'+str(si)+'.'+family+'.pool'+str(pi)
                ini,_=v1.artifact(native,backend,initial,shared_rows,shape,routes,s,dest/(stem+'.initial.npz'))
                fin,_=v1.artifact(native,backend,final,shared_rows,shape,routes,s,dest/(stem+'.final.npz'))
                options=[('initial',ini),('final',fin),('inherited',inherited[1])]
                selected=min(options,key=lambda x:x[1]['mean_ns'])[0]
                records.append(dict(family=family,kind=kind,shared_geometry=shared_name,shared_rows=shared_rows.tolist(),
                    shared_permutation=guide['shared_permutation'],seed=seed,seed_index=si,pool_index=pi,pool_label=label,
                    menu_pool=poolproof if family=='joint_simple_menu' else None,fit=statistics,initial=ini,final=fin,
                    inherited=inherited[1],inherited_family=inherited[2],inherited_stage=inherited[3],selected=selected,
                    inheritance_note='Retains prior fitted incumbent without charging its earlier search to v2; report inherited versus incremental gains separately.'))
            print(args.model,args.layer,args.copies,context,family,'complete',flush=True)
    winners={}
    for family in ['joint_simple_menu','independent_paired']:
        choices=[r for r in records if r['family']==family]
        winners[family]=min(choices,key=lambda r:r[r['selected']]['mean_ns'])
    result=dict(model=args.model,layer=args.layer,copies=args.copies,shape=shape,service=s,training=old['training'],
        v1_case_path=str(sourcecase),v1_case_sha256=sha(sourcecase),records=records,winners=winners,
        elapsed_seconds=time.monotonic()-started,script_sha256=sha(__file__),
        limits='Additional matched-move baseline strengthening on TRAIN only. Independent retains coupled incumbent and therefore cannot be said to lose to it on mean. Simplemenu includes balanced/depth profiles; finite fitted pool, no global optimum. No new heldout or v1 mutation.')
    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(case=str(dest),seconds=result['elapsed_seconds'],winners={k:v[v['selected']]['mean_ns'] for k,v in winners.items()})),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['granite4tiny','deepseek'],required=True)
    ap.add_argument('--layer',type=int,required=True);ap.add_argument('--copies',type=int,choices=[1,4],required=True)
    run_case(ap.parse_args())
