#!/usr/bin/env python3
"""Bounded full-native elastic whole-row geometry control, TRAIN only.

Standard finite-menu elastic/fractional sharding plus local search. No new
optimizer, numerical representation, native operator or hardware claim.
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

import research_20261103_native_joint_refine as fixed

ROOT=fixed.ROOT
BASE=fixed.BASE


def sha(path):return fixed.sha(path)


def profile_key(row):return tuple(sorted(map(int,row)))


def menus(record, service):
    """All saved base profiles, quotienting only arbitrary chip permutation."""
    result=[]
    for p in range(2):
        found={};columns=record['shape'][p]
        expected=2*record['shape'][1] if p==0 else record['shape'][0]
        for c in record['fitting'][p]:
            rr=np.asarray(c['base_rows'],np.int64)
            assert rr.shape==(16,) and np.all(rr>=0) and rr.sum()==expected
            pages=(rr*columns+service['page_bytes']-1)//service['page_bytes']
            waves=(pages+service['planes']-1)//service['planes']
            if np.max(waves)>service['block_wordlines']:continue
            key=profile_key(rr)
            if key not in found:found[key]=dict(rows=list(key),labels=[],families=[])
            found[key]['labels'].append(c['label']);found[key]['families'].append(c['family'])
        result.append([found[k] for k in sorted(found)])
        assert result[-1]
    return result


class MixedFit(fixed.ExactFit):
    def __init__(self,rows,shape,service,routes,seed,menu,checkpoint_every=128):
        super().__init__(rows,shape,service,routes,'independent',seed)
        self.menu=menu
        self.checkpoint_every=checkpoint_every
        self.checkpoints=[]
        self.validation_calls=0;self.validation_queries=0
        self.geometry_events=[];self.geometry_active_counts_seen=set()
        self.optimization_calls=0;self.optimization_queries=0
        for p in range(2):
            eligible={profile_key(c['rows']) for c in menu[p]}
            assert all(profile_key(rr) in eligible for rr in rows[:-1,p])

    def checkpoint(self,tag):
        fresh=self.lower()
        for k in ['offsets','jobs','input_columns','total_rows']:
            assert np.array_equal(fresh[k],self.plan[k]),(tag,k)
        values,bounds=self.native.run(fresh,self.unique,self.config)
        assert np.array_equal(values,self.values),tag
        assert np.array_equal(bounds,self.bounds),tag
        assert int(values@self.weights)==self.total,tag
        self.validation_calls+=1;self.validation_queries+=len(self.unique)
        self.checkpoints.append(dict(tag=tag,unique_queries=len(self.unique),mean_ns=self.total/len(self.routes)))

    def proposal(self,temperature):
        global_move=self.rng.random()<.04
        paired=self.rng.random()<.5
        phases=[0,1] if paired else [int(self.rng.integers(2))]
        a,b=map(int,self.rng.choice(16,2,replace=False))
        experts=list(range(self.shape[2])) if global_move else [int(self.rng.integers(self.shape[2]))]
        geometry=(not global_move) and self.rng.random()<.25
        changes=[]
        if global_move:
            tag='global_paired' if paired else 'global_single'
            for e in experts:
                for p in phases:
                    old=self.rows[e,p].copy();new=old.copy();new[[a,b]]=new[[b,a]]
                    changes.append((e,p,old,new))
        elif geometry:
            tag='geometry_paired' if paired else 'geometry_single'
            e=experts[0]
            for p in phases:
                old=self.rows[e,p].copy();current=profile_key(old)
                available=[c for c in self.menu[p] if profile_key(c['rows'])!=current]
                if not available:continue
                chosen=available[int(self.rng.integers(len(available)))]
                new=np.empty(16,np.int64)
                # Monotone alignment keeps existing chip-load orientation.
                new[np.argsort(old,kind='stable')]=chosen['rows']
                changes.append((e,p,old,new))
        else:
            tag='swap_paired' if paired else 'swap_single'
            for p in phases:
                old=self.rows[experts[0],p].copy();new=old.copy();new[[a,b]]=new[[b,a]]
                changes.append((experts[0],p,old,new))
        changes=[c for c in changes if not np.array_equal(c[2],c[3])]
        self.moves.setdefault(tag,dict(attempts=0,valid=0,accepted=0))
        self.moves[tag]['attempts']+=1
        if not changes:return
        ii=np.arange(len(self.unique)) if global_move else self.members[experts[0]]
        if not len(ii):return
        previous_plan=self.plan
        for e,p,old,new in changes:self.rows[e,p]=new
        # Geometry can change active-chip count: rebuild all ragged offsets.
        self.plan=self.lower()
        values,bounds=self.native.run(self.plan,self.unique[ii],self.config)
        self.optimization_calls+=1;self.optimization_queries+=len(ii)
        change=int((values-self.values[ii])@self.weights[ii])
        accept=change<=0 or (temperature>0 and self.rng.random()<math.exp(-change/(len(self.routes)*temperature)))
        self.valid+=1;self.moves[tag]['valid']+=1
        if accept:
            self.accepted+=1;self.moves[tag]['accepted']+=1
            self.values[ii],self.bounds[ii]=values,bounds;self.total+=change
            if global_move:
                for p in phases:self.permutations[p,[a,b]]=self.permutations[p,[b,a]]
            if self.total<self.best:
                self.best=self.total;self.best_rows=self.rows.copy();self.best_permutations=self.permutations.copy()
        else:
            for e,p,old,new in changes:self.rows[e,p]=old
            self.plan=previous_plan
        if geometry:
            counts=tuple((p,int(np.count_nonzero(old)),int(np.count_nonzero(new))) for e,p,old,new in changes)
            self.geometry_events.append(dict(expert=experts[0],phases=phases,accepted=bool(accept),mean_change_ns=change/len(self.routes),active_counts=counts))
            if counts not in self.geometry_active_counts_seen:
                self.geometry_active_counts_seen.add(counts)
                self.checkpoint('first_geometry_active_counts_'+str(counts))

    def forced_ragged_smoke(self):
        """Pure correctness probe: force narrow geometry, then restore exactly."""
        self.checkpoint('before_forced_ragged')
        original=self.rows.copy();oldplan=self.plan
        e=int(self.unique[0,0]);ii=self.members[e]
        for p in range(2):
            narrow=min(self.menu[p],key=lambda c:np.count_nonzero(c['rows']))
            self.rows[e,p]=narrow['rows']
        self.plan=self.lower()
        values,bounds=self.native.run(self.plan,self.unique[ii],self.config)
        full,fb=self.native.run(self.plan,self.unique,self.config)
        assert np.array_equal(values,full[ii]) and np.array_equal(bounds,fb[ii])
        untouched=np.ones(len(self.unique),bool);untouched[ii]=False
        assert np.array_equal(full[untouched],self.values[untouched])
        assert np.array_equal(fb[untouched],self.bounds[untouched])
        changed_job_count=len(self.plan['jobs'])!=len(oldplan['jobs'])
        assert changed_job_count
        self.rows=original;self.plan=oldplan
        self.checkpoint('restored_forced_ragged')
        return dict(expert=e,changed_job_count=True,affected_queries=len(ii),unaffected_queries=int(untouched.sum()))

    def run_mixed(self,iterations,temperature):
        started=time.monotonic();cpu_started=time.process_time()
        self.checkpoint('initial')
        for it in range(iterations):
            temp=temperature*(.0001**(it/max(1,iterations-1)))
            self.proposal(temp)
            if (it+1)%self.checkpoint_every==0:self.checkpoint('proposal_'+str(it+1))
        self.checkpoint('terminal')
        terminal=self.total/len(self.routes)
        self.rows=self.best_rows.copy();self.permutations=self.best_permutations.copy();self.plan=self.lower()
        self.values,self.bounds=self.native.run(self.plan,self.unique,self.config)
        self.total=int(self.values@self.weights)
        self.validation_calls+=1;self.validation_queries+=len(self.unique)
        assert self.total==self.best
        self.checkpoint('restored_best')
        values,bounds=self.native.run(self.plan,self.routes,self.config)
        self.validation_calls+=1;self.validation_queries+=len(self.routes)
        assert int(values.sum())==self.best and self.best<=self.initial_total
        assert np.array_equal(self.rows[-1],self.initial_rows[-1])
        for p in range(2):
            eligible={profile_key(c['rows']) for c in self.menu[p]}
            assert all(profile_key(rr) in eligible for rr in self.rows[:-1,p])
            assert np.array_equal(self.rows[:,p].sum(1),self.initial_rows[:,p].sum(1))
        stats=dict(objective='exact TRAIN mean of complete native DAG; ordered routing tuples',proposals=iterations,valid=self.valid,accepted=self.accepted,move_counts=self.moves,optimization_native_calls=self.optimization_calls,optimization_native_query_evaluations=self.optimization_queries,validation_native_calls=self.validation_calls,validation_native_query_evaluations=self.validation_queries,initial_temperature_ns=temperature,initial_mean_ns=self.initial_total/len(self.routes),terminal_mean_ns=terminal,final_mean_ns=self.best/len(self.routes),initial_retained=True,unique_ordered_routes=len(self.unique),original_routes=len(self.routes),seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu_started,checkpoints=self.checkpoints,geometry_events=self.geometry_events,shared_unchanged=True,profile_changes=sum(profile_key(self.rows[e,p])!=profile_key(self.initial_rows[e,p]) for e in range(self.shape[2]) for p in range(2)),profile_histograms=[dict(Counter(str(profile_key(rr)) for rr in self.rows[:-1,p])) for p in range(2)])
        return self.plan,values,bounds,stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    protocol=json.loads(args.protocol.read_text());assert sha(__file__)==protocol['source_sha256']
    assert sha(fixed.__file__)==protocol['fixed_source_sha256']
    for src in protocol['inputs']:assert sha(src['path'])==src['sha256']
    assert args.seed in protocol['seeds'] and str(args.case.resolve()) in protocol['cases']
    assert args.smoke==protocol['smoke']
    rec=json.loads(args.case.read_text());winner=rec['winners']['independent_paired'];art=winner[winner['selected']]
    assert sha(art['arrays'])==art['sha256']
    with np.load(art['arrays']) as z:rows=z['rows'];routes=z['routes'];initial_cost=z['cost']
    with np.load(rec['training']['path']) as z:
        assert z['split'].item()=='train' and z['completed'].all()
        assert np.array_equal(routes,z['routes'][:,:,rec['layer']].reshape(-1,rec['shape'][3]))
    v1=json.loads(Path(rec['v1_case_path']).read_text());summary=json.loads(Path(v1['source_summary']).read_text())
    source=next(r for r in summary['records'] if r['model']==rec['model'] and r['layer']==rec['layer'] and r['command_ns']==250)
    menu=menus(source,rec['service']);assert menu==protocol['menus'][rec['model']]
    fit=MixedFit(rows,rec['shape'],rec['service'],routes,args.seed,menu,protocol['checkpoint_every'])
    assert fit.initial_total==int(initial_cost.sum())
    forced=fit.forced_ragged_smoke() if args.smoke else None
    plan,values,bounds,stats=fit.run_mixed(protocol['iterations'],protocol['initial_temperature_ns'])
    args.output.mkdir(parents=True,exist_ok=False)
    physical=fit.backend.physical_addresses(plan,fit.service)
    addresses=np.asarray([[a[k] for k in ['chip','block','wordline','reserved_wordlines','pages','first_plane']] for a in physical.pop('addresses')],np.int32)
    logical=np.asarray([[a[k] for k in ['expert','phase','chip','row_start','rows','columns']] for a in plan['logical']],np.int32)
    C=np.tile(rec['shape'][:2],(rec['shape'][2]+1,1));C[-1,1]=rec['shape'][5]
    pages=(fit.rows*C[:,:,None]+16383)//16384;waves=(pages+3)//4
    ledger=dict(useful_bytes_by_chip=(fit.rows*C[:,:,None]).sum((0,1)).tolist(),programmed_page_bytes_by_chip=(pages.sum((0,1))*16384).tolist(),reserved_wave_bytes_by_chip=(waves.sum((0,1))*65536).tolist(),allocated_lsb_block_bytes_by_chip=[n*192*65536 for n in physical['chip_blocks']])
    profile_choices=np.asarray([[next(i for i,c in enumerate(menu[p]) if profile_key(c['rows'])==profile_key(fit.rows[e,p])) for p in range(2)] for e in range(rec['shape'][2])],np.int32)
    arrays=args.output/'arrays.npz'
    initial_profiles=np.asarray([[next(i for i,c in enumerate(menu[p]) if profile_key(c['rows'])==profile_key(rows[e,p])) for p in range(2)] for e in range(rec['shape'][2])],np.int32)
    np.savez_compressed(arrays,rows=fit.rows,initial_rows=rows,initial_row_multisets=np.sort(rows,axis=2),routes=routes,initial_cost=initial_cost,cost=values,lower_bound=bounds,offsets=plan['offsets'],jobs=plan['jobs'],input_columns=plan['input_columns'],total_rows=plan['total_rows'],addresses=addresses,logical=logical,profile_choices=profile_choices,initial_profile_choices=initial_profiles)
    checks=[]
    for q in [0,255,511]:
        direct=fit.backend.ReferenceGraph.build(plan,np.r_[routes[q],rec['shape'][2]],fit.service).run();assert direct['cost']==int(values[q])
        checks.append(dict(query=q,cost_ns=int(values[q]),nodes=len(direct['trace'])))
    result=dict(case=str(args.case),case_sha256=sha(args.case),protocol=str(args.protocol),protocol_sha256=sha(args.protocol),seed=args.seed,source_sha256=sha(__file__),model=rec['model'],layer=rec['layer'],copies=rec['copies'],shape=rec['shape'],service=rec['service'],training=rec['training'],baseline_incumbent=art,shared_rows=rows[-1].tolist(),menus=menu,fit=stats,physical=physical,per_chip_ledger=ledger,arrays=str(arrays),arrays_sha256=sha(arrays),mean_ns=float(values.mean()),p99_ns=float(np.percentile(values,99)),python_checks=checks,forced_ragged_smoke=forced,scope='TRAIN only, finite standard elastic whole-row geometry search; no hardware or algorithm novelty claim')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(model=rec['model'],seed=args.seed,mean_ns=result['mean_ns'],p99_ns=result['p99_ns'],profile_changes=stats['profile_changes'],seconds=stats['seconds'],cpu_seconds=stats['cpu_seconds'])),flush=True)


if __name__=='__main__':main()
