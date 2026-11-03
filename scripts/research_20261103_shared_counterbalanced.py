#!/usr/bin/env python3
"""W8 TRAIN: matched joint-cyclic / independent / counterbalanced controls.

Standard coloring and local-search controls, not a new optimizer. Shared
candidate pool and proposal budgets are identical across routed families.
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from research_20261103_topology_placement import sha
from research_20261103_topology_placement_replay import Replay,load_python

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'shared-counterbalanced'
SOURCE=BASE/'ordinary-read-compiler'
SEEDS=[20261103,20261104]


def cost(rows,C,s):
    r=np.asarray(rows,np.int64);pages=(r*C+16383)//16384;waves=(pages+3)//4
    return waves*28000+(r>0)*(250+C*s.get('input_channel_copies',1)//2)+2*(pages*16384//C)


def work(rows,routes,shared,columns,s):
    cc=np.stack([cost(rows[:,p],C,s) for p,C in enumerate(columns)],axis=1)
    return (cc[routes].sum((1,2))+shared).max(1)


def rows_from_plan(plan):
    rr=np.zeros((len(plan['offsets']),2,16),np.int64)
    for e in range(len(rr)):
        for p in range(2):
            lo,hi=plan['offsets'][e,p]
            for c,n,*_ in plan['jobs'][lo:hi]:rr[e,p,c]+=n
    return rr


def canonical(model,highgu,highdown):
    E=len(highgu);rr=np.zeros((E,2,16),np.int64)
    if model=='granite4tiny':
        rr[:,0]=40;rr[:,1]=np.where(highdown,128,0)
        for e in range(E):
            hh=np.flatnonzero(highgu[e]);assert len(hh)==9
            rr[e,0,hh]=82;rr[e,0,hh[:6]]=83
    else:
        rr[:,0]=np.where(highgu,192,160);rr[:,1]=np.where(highdown,137,89)
    return rr


def coupled_random(model,seed):
    rng=np.random.default_rng(seed);O=4 if model=='granite4tiny' else 3
    cats=np.zeros((64,16),np.int8)
    for e in range(64):
        order=rng.permutation(16);cats[e,order[:5]]=2;cats[e,order[5:5+O]]=1
    return canonical(model,cats!=0,cats!=1)


def projected(model,independent):
    gu=independent[:,0]>=(82 if model=='granite4tiny' else 192)
    down=independent[:,1]>=(128 if model=='granite4tiny' else 137)
    count=4 if model=='granite4tiny' else 3
    out=np.ones_like(down)
    for e in range(64):
        keep=list(np.flatnonzero(gu[e]&~down[e]))
        available=[int(c) for c in np.flatnonzero(gu[e]) if c not in keep]
        omit=(keep+available)[:count];out[e,omit]=False
    return canonical(model,gu,out)


def independent_initial(rec,model,routes,s):
    columns=[rec['shape'][0],rec['shape'][1]];high=[]
    targets=([1]*7+[2]*9,[0]*4+[1]*12) if model=='granite4tiny' else ([5]*8+[6]*8,[2]*3+[3]*13)
    for p in range(2):
        selected=[]
        for c in rec['fitting'][p]:
            base=np.asarray(c['base_rows']);ww=(base*columns[p]+65535)//65536
            if sorted(ww.tolist())!=list(targets[p]):continue
            ll=np.asarray(c['labels']);width=int(np.count_nonzero(base))
            shifts=ll*width if c['family']=='disjoint' else ll
            rr=np.asarray([np.roll(base,int(v)) for v in shifts])
            value=cost(rr,columns[p],s)[routes].sum(1).max(1).mean()
            selected.append((value,rr,c['label']))
        if not selected:
            raise AssertionError(('missing frozen phase geometry',model,p))
        _,rr,label=min(selected,key=lambda x:x[0])
        ww=(rr*columns[p]+65535)//65536
        high.append(ww==(2 if model=='granite4tiny' and p==0 else 1 if model=='granite4tiny' else 6 if p==0 else 3))
    return canonical(model,*high)


def source_cyclic_candidates(rec,p):
    result=[]
    for c in rec['fitting'][p]:
        if c['family'] not in ['cyclic','disjoint']:continue
        base=np.asarray(c['base_rows']);ll=np.asarray(c['labels']);width=int(np.count_nonzero(base))
        shifts=ll*width if c['family']=='disjoint' else ll
        rr=np.asarray([np.roll(base,int(v)) for v in shifts])
        result.append(dict(rows=rr,width=width,label=c['label']))
    return result


def cyclic_pool(rec,routes,shared,columns,s,incumbent,limit=4):
    gu=source_cyclic_candidates(rec,0);down=source_cyclic_candidates(rec,1)
    gc=[cost(c['rows'],columns[0],s)[routes].sum(1) for c in gu]
    dc=[cost(c['rows'],columns[1],s)[routes].sum(1) for c in down]
    best={}
    for a,g in enumerate(gu):
        for b,d in enumerate(down):
            for rotation in range(16):
                value=float((gc[a]+np.roll(dc[b],rotation,axis=1)+shared).max(1).mean())
                key=(g['width'],d['width'])
                if key not in best or value<best[key]['initial_work_ns']:
                    best[key]=dict(gu=g,down=d,rotation=rotation,initial_work_ns=value)
    ranked=sorted(best.values(),key=lambda c:(c['initial_work_ns'],c['gu']['width'],c['down']['width']))
    # Always retain the actual frozen strong cyclic pair, not only work-ranked
    # alternatives that may have worse whole-expert causal latency.
    oldwidth=tuple(int(np.count_nonzero(incumbent[0,p])) for p in range(2))
    selected=[c for c in ranked if (c['gu']['width'],c['down']['width'])!=oldwidth][:limit-1]
    out=[dict(rows=incumbent.copy(),label='frozen_cyclic',widths=oldwidth,
              initial_work_ns=float(work(incumbent,routes,shared,columns,s).mean()))]
    for c in selected:
        rr=np.stack([c['gu']['rows'],np.roll(c['down']['rows'],c['rotation'],axis=1)],axis=1)
        out.append(dict(rows=rr,label=c['gu']['label']+'+'+c['down']['label']+'@'+str(c['rotation']),
                        widths=(c['gu']['width'],c['down']['width']),initial_work_ns=c['initial_work_ns']))
    assert len(out)==limit
    return out


class Fit:
    def __init__(self,rows,kind,routes,weights,shared,columns,s,seed):
        self.rows=rows.copy();self.initial=rows.copy();self.kind=kind;self.routes=routes;self.weights=weights
        self.shared=shared;self.columns=columns;self.s=s;self.rng=np.random.default_rng(seed)
        self.members=[np.flatnonzero((routes==e).any(1)) for e in range(64)]
        self.order=sorted(range(64),key=lambda e:(-int(weights[self.members[e]].sum()),e))
        self.perm=np.tile(np.arange(16),(2,1));self.variants=[];self.labels=[]
        if kind=='cyclic':
            for p in range(2):
                vv=np.unique(np.stack([np.roll(rows[0,p],k) for k in range(16)]),axis=0)
                labels=[]
                for rr in rows[:,p]:
                    match=np.flatnonzero(np.all(vv==rr,axis=1));assert len(match)==1;labels.append(match[0])
                self.variants.append(vv);self.labels.append(np.asarray(labels,np.int32))
        self.rebuild();self.initial_value=self.total/int(weights.sum());self.initial_phase=self.phase_bounds()
        self.best=self.total;self.snapshot=self.capture();self.valid=self.accepted=self.greedy_valid=0

    def rebuild(self):
        self.cc=np.stack([cost(self.rows[:,p],C,self.s) for p,C in enumerate(self.columns)],axis=1)
        self.lp=self.cc[self.routes].sum(1).transpose(1,0,2)
        self.loads=self.lp.sum(0)+self.shared;self.values=self.loads.max(1)
        self.total=int(self.values@self.weights)

    def phase_bounds(self):return [float(self.lp[p].max(1)@self.weights/self.weights.sum()) for p in range(2)]
    def capture(self):return (self.rows.copy(),self.perm.copy(),[x.copy() for x in self.labels])
    def remember(self):
        if self.total<self.best:self.best=self.total;self.snapshot=self.capture()

    def apply(self,changes,mutation,temp,greedy=False):
        # changes are(phase,queryindices,delta peraffectedquery/chip).
        if not changes:return False
        ii=changes[0][1];dd=np.zeros((len(ii),16),np.int64);proposal_phase=[]
        for p,jj,delta in changes:
            assert np.array_equal(jj,ii)
            proposal_phase.append((p,jj,self.lp[p,jj]+delta));dd+=delta
        if not np.any(dd):return False
        newloads=self.loads[ii]+dd;newvalues=newloads.max(1)
        change=int((newvalues-self.values[ii])@self.weights[ii]);self.valid+=1
        if greedy:self.greedy_valid+=1
        take=change<0 if greedy else change<=0 or (temp>0 and self.rng.random()<math.exp(-change/(self.weights.sum()*temp)))
        if take:
            mutation();self.loads[ii]=newloads;self.values[ii]=newvalues;self.total+=change
            for p,jj,vv in proposal_phase:self.lp[p,jj]=vv
            self.accepted+=1;self.remember()
        return take

    def global_move(self,p,a,b,temp,greedy=False):
        phases=[0,1] if self.kind=='coupled' else [p]
        ii=np.arange(len(self.routes),dtype=np.int32);changes=[]
        for phase in phases:
            dd=np.zeros_like(self.lp[phase]);dd[:,a]=self.lp[phase,:,b]-self.lp[phase,:,a];dd[:,b]=-dd[:,a]
            changes.append((phase,ii,dd))
        def mutation():
            for phase in phases:
                self.rows[:,phase,[a,b]]=self.rows[:,phase,[b,a]]
                self.cc[:,phase,[a,b]]=self.cc[:,phase,[b,a]]
                self.perm[phase,[a,b]]=self.perm[phase,[b,a]]
        return self.apply(changes,mutation,temp,greedy)

    def local_move(self,p,e,a,b,temp,greedy=False):
        ii=self.members[e]
        if not len(ii):return False
        phases=[0,1] if self.kind=='coupled' else [p]
        changes=[]
        for phase in phases:
            dd=np.zeros((len(ii),16),np.int64);dd[:,a]=self.cc[e,phase,b]-self.cc[e,phase,a];dd[:,b]=-dd[:,a]
            changes.append((phase,ii,dd))
        def mutation():
            for phase in phases:
                self.rows[e,phase,[a,b]]=self.rows[e,phase,[b,a]]
                self.cc[e,phase,[a,b]]=self.cc[e,phase,[b,a]]
        return self.apply(changes,mutation,temp,greedy)

    def label_move(self,p,e,newlabel,temp,greedy=False):
        ii=self.members[e]
        if not len(ii):return False
        rr=self.variants[p][newlabel,self.perm[p]];cc=cost(rr,self.columns[p],self.s)
        dd=np.broadcast_to(cc-self.cc[e,p],(len(ii),16))
        def mutation():self.rows[e,p]=rr;self.cc[e,p]=cc;self.labels[p][e]=newlabel
        return self.apply([(p,ii,dd)],mutation,temp,greedy)

    def run(self,iterations,greedy_limit):
        start=time.monotonic()
        for it in range(iterations):
            p=int(self.rng.integers(2));a,b=self.rng.choice(16,2,replace=False)
            temp=2000*(.0001**(it/max(1,iterations-1)))
            if self.rng.random()<.08:self.global_move(p,int(a),int(b),temp);continue
            e=int(self.rng.integers(64))
            if self.kind=='cyclic':self.label_move(p,e,int(self.rng.integers(len(self.variants[p]))),temp)
            else:self.local_move(p,e,int(a),int(b),temp)
        terminal=self.total/self.weights.sum()
        self.rows,self.perm,self.labels=self.snapshot;self.rebuild()
        assert self.total==self.best
        attempts=0
        # One bounded deterministic greedy pass, same evaluation allowance per
        # family/seed. Cyclic pool splits the allowance between its geometries.
        for e in self.order:
            if attempts>=greedy_limit:break
            if self.kind=='cyclic':
                candidates=[(p,j) for p in range(2) for j in range(len(self.variants[p]))]
                for p,j in candidates:
                    if attempts>=greedy_limit:break
                    attempts+=1;self.label_move(p,e,j,0,True)
            else:
                for p in range(1 if self.kind=='coupled' else 2):
                    for a,b in itertools.combinations(range(16),2):
                        if attempts>=greedy_limit:break
                        attempts+=1;self.local_move(p,e,a,b,0,True)
                    if attempts>=greedy_limit:break
        self.rows,self.perm,self.labels=self.snapshot;self.rebuild()
        direct=work(self.rows,self.routes,self.shared,self.columns,self.s)
        assert np.array_equal(direct,self.values)
        return self.rows.copy(),dict(kind=self.kind,proposals=iterations,valid=self.valid,accepted=self.accepted,
            greedy_attempts=attempts,greedy_valid=self.greedy_valid,initial_work_ns=float(self.initial_value),
            terminal_anneal_work_ns=float(terminal),final_work_ns=float(self.total/self.weights.sum()),
            initial_phase_work_ns=self.initial_phase,final_phase_work_ns=self.phase_bounds(),seconds=time.monotonic()-start,
            initial_and_final_exact=True,unrestricted_histogram=True,
            final_global_permutations=self.perm.tolist(),final_labels=[x.tolist() for x in self.labels])


def artifact(native,backend,rows,shared_rows,shape,routes,s,path):
    D,I,E,K,L,SI=shape;allrows=np.r_[rows,shared_rows[None]]
    plan=backend.lower_rows(allrows[:,0],allrows[:,1],D,[I]*E+[SI],backend.Service(**s))
    value,bound=native.run(plan,routes,s);physical=backend.physical_addresses(plan,backend.Service(**s))
    address=np.asarray([[a[k] for k in ['chip','block','wordline','reserved_wordlines','pages','first_plane']] for a in physical.pop('addresses')],np.int32)
    logical=np.asarray([[a[k] for k in ['expert','phase','chip','row_start','rows','columns']] for a in plan['logical']],np.int32)
    np.savez_compressed(path,rows=allrows,offsets=plan['offsets'],jobs=plan['jobs'],input_columns=plan['input_columns'],
                        total_rows=plan['total_rows'],addresses=address,logical=logical,routes=routes,cost=value,lower_bound=bound)
    return dict(arrays=str(path),sha256=sha(path),mean_ns=float(value.mean()),p99_ns=float(np.percentile(value,99)),
                mean_native_lower_bound_ns=float(bound.mean()),physical=physical),value


def run_case(args):
    started=time.monotonic();campaign='full-campaign-v5-W8-training' if args.copies==1 else 'full-campaign-v6-W8-input4'
    formal=(args.iterations,args.greedy)==(20000,4096)
    if formal:
        protocol=json.loads((OUT/'protocol.frozen.json').read_text())
        assert protocol['source_sha256']==sha(__file__)
        assert [args.model,args.layer,args.copies] in protocol['cases']
        for source_record in protocol['sources']:
            assert sha(source_record['path'])==source_record['sha256']
    source=SOURCE/campaign/'summary.json';summary=json.loads(source.read_text())
    rec=next(r for r in summary['records'] if r['model']==args.model and r['layer']==args.layer and r['command_ns']==250)
    s=rec['service'];columns=rec['shape'][:2]
    with np.load(rec['training']['path']) as z:
        assert z['split'].item()=='train' and z['completed'].all()
        routes=np.ascontiguousarray(z['routes'][:,:,args.layer].reshape(-1,6),np.int32)
    assert sha(rec['training']['path'])==rec['training']['sha256']
    unique,weights=np.unique(np.sort(routes,axis=1),axis=0,return_counts=True)
    native=Replay();backend=load_python()
    taskroot=OUT if formal else OUT/'smoke'
    dest=taskroot/f'{args.model}.L{args.layer}.input{args.copies}';dest.mkdir(parents=True,exist_ok=True)
    if formal:assert not (dest/'result.json').exists(),'Never overwrite a completed formal case.'
    initial_independent=independent_initial(rec,args.model,routes,s)
    source_rows={}
    for family in ['cyclic','whole_rows']:
        a=next(a for a in rec['artifacts'] if a['family']==family)
        with np.load(ROOT/a['arrays']) as z:pp={k:z[k] for k in z.files}
        assert sha(ROOT/a['arrays'])==a['arrays_sha256']
        source_rows[family]=rows_from_plan(pp)[:-1]
        check=native.run(pp,pp['selection_routes'],s)[0];assert np.array_equal(check,pp['cost'])
    shared_geometries={'source':np.asarray(rec['shared_rows'],np.int64)}
    if args.model=='deepseek':
        rr=shared_geometries['source'].copy();rr[1]=np.asarray([138]*9+[115]*6+[116]);assert rr[1].sum()==2048
        shared_geometries['balanced89']=rr
    records=[];baselines=[]
    for shared_name,shared0 in shared_geometries.items():
        for si,seed in enumerate(SEEDS):
            perm=np.arange(16) if si==0 else np.random.default_rng(20261103+173).permutation(16)
            shared_rows=shared0[:,perm]
            shared=sum(cost(shared_rows[p],C,s) for p,C in enumerate([rec['shape'][0],rec['shape'][5]]))
            tag=shared_name+'.seed'+str(si)
            for family,rr in source_rows.items():
                desc,_=artifact(native,backend,rr,shared_rows,rec['shape'],routes,s,dest/(tag+'.frozen_'+family+'.npz'))
                baselines.append(dict(shared_geometry=shared_name,seed=seed,shared_permutation=perm.tolist(),family=family,**desc))
            pool=cyclic_pool(rec,routes,shared,columns,s,source_rows['cyclic'])
            initialcoupled=coupled_random(args.model,seed) if si==0 else projected(args.model,initial_independent)
            families=[('joint_cyclic',[(c['label'],c['rows']) for c in pool]),
                      ('independent_permutation',[('same_wave_geometry',initialcoupled if si==0 else initial_independent)]),
                      ('counterbalanced',[('standard_3category',initialcoupled)])]
            for family,starts in families:
                for pi,(label,initial) in enumerate(starts):
                    it=args.iterations//len(starts);greedy=args.greedy//len(starts)
                    kind='cyclic' if family=='joint_cyclic' else 'independent' if family=='independent_permutation' else 'coupled'
                    fit=Fit(initial,kind,unique,weights,shared,columns,s,seed)
                    final,statistics=fit.run(it,greedy)
                    stem=tag+'.'+family+'.pool'+str(pi)
                    ini,initialcost=artifact(native,backend,initial,shared_rows,rec['shape'],routes,s,dest/(stem+'.initial.npz'))
                    fin,finalcost=artifact(native,backend,final,shared_rows,rec['shape'],routes,s,dest/(stem+'.final.npz'))
                    selected='final' if fin['mean_ns']<ini['mean_ns'] else 'initial'
                    records.append(dict(family=family,kind=kind,pool_label=label,pool_index=pi,shared_geometry=shared_name,
                        shared_rows=shared_rows.tolist(),shared_permutation=perm.tolist(),seed=seed,seed_index=si,
                        fit=statistics,initial=ini,final=fin,selected=selected))
                print(args.model,args.layer,args.copies,tag,family,'complete',flush=True)
    winners={}
    for family in ['joint_cyclic','independent_permutation','counterbalanced']:
        candidates=[(r[r['selected']]['mean_ns'],r) for r in records if r['family']==family]
        _,winner=min(candidates,key=lambda x:x[0]);winners[family]=winner
    result=dict(model=args.model,layer=args.layer,physical_layer=rec['training']['physical_model_layer'],copies=args.copies,
        shape=rec['shape'],service=s,training=rec['training'],source_summary=str(source),source_summary_sha256=sha(source),
        records=records,baselines=baselines,winners=winners,elapsed_seconds=time.monotonic()-started,
        script_sha256=sha(__file__),scope='All512 actualW8TRAIN queries; no heldout. Static onecopy complete-row plans, shared candidatepool matched. Mean-native selection, fixed greedy replay, finitepool/heuristic only.')
    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(case=str(dest),seconds=result['elapsed_seconds'],winners={k:v[v['selected']]['mean_ns'] for k,v in winners.items()})),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['granite4tiny','deepseek'],required=True)
    ap.add_argument('--layer',type=int,required=True);ap.add_argument('--copies',type=int,choices=[1,4],required=True)
    ap.add_argument('--iterations',type=int,default=20000);ap.add_argument('--greedy',type=int,default=4096)
    run_case(ap.parse_args())
