#!/usr/bin/env python3
"""Small fixed-map controls addressing group symmetry; no fullFFN replay."""
import json
from pathlib import Path
import numpy as np
import oracle as o


def secondary_refine(A,start,tables,seed,variable=False,budget=4096):
    rng=np.random.default_rng(seed);p=start.copy();W=o.load(A,p,tables)
    def score(x):return (float(x.max(axis=1).sum()),float(np.sort(x,axis=1)[:,-4:].sum()),float(x.sum()))
    current=score(W);accepted=valid=0
    active=[np.flatnonzero(A[:,e]) for e in range(65)]
    for _ in range(budget):
        e=int(rng.integers(65 if variable else 64));src=int(rng.choice(np.flatnonzero(p[e])));dst=int(rng.integers(16))
        if src==dst or (not variable and p[e,dst]):continue
        amount=int(rng.choice([1,4,int(p[e,src])])) if variable else 12
        if amount>p[e,src]:continue
        price=tables[int(e==64)];s,d=int(p[e,src]),int(p[e,dst]);ss,dd=s-amount,d+amount
        if dd>=97 or not np.isfinite(price[ss]) or not np.isfinite(price[dd]):continue
        rr=o.reserve(p)[[src,dst]]+np.array([(ss+3)//4-(s+3)//4,(dd+3)//4-(d+3)//4])
        if np.any(128+rr>192):continue
        ids=active[e];before=score(W[ids]);trial=W[ids].copy()
        trial[:,src]+=price[ss]-price[s];trial[:,dst]+=price[dd]-price[d]
        after=score(trial);candidate=tuple(a+b-c for a,b,c in zip(current,after,before));valid+=1
        if candidate<current:
            p[e,src]=ss;p[e,dst]=dd;W[ids]=trial;current=candidate;accepted+=1
    assert o.capacity(p) and np.array_equal(W,o.load(A,p,tables))
    return p,dict(critical_work_ns=o.stats(W.max(axis=1)),accepted=accepted,valid=valid,proposals=budget,
                  top4_sum_mean_ns=current[1]/len(A),resident_combined_max_WL=int((128+o.reserve(p)).max()))


def dedicated(A,tables,seed):
    counts=A[:,:64].sum(axis=0);cold=int(np.argmin(counts));remaining=np.array([e for e in range(64) if e!=cold]);best=None;receipts=[]
    for si in range(4):
        rng=np.random.default_rng(seed+si);perm=rng.permutation(remaining)
        labels=np.full(64,3,dtype=np.int64);labels[perm]=np.repeat(np.arange(3),21)
        group=A[:,:64]@np.eye(4,dtype=np.int64)[labels]
        def critical(g):return np.maximum((g[:,:3]*tables[0][12]).max(axis=1),g[:,3]*tables[0][12]+tables[1][24])
        values=critical(group);score=float(values.sum());initial=score/len(A);accepted=0
        for _ in range(4096):
            e,f=map(int,rng.choice(remaining,2,replace=False));a,b=int(labels[e]),int(labels[f])
            if a==b:continue
            change=A[:,e]-A[:,f];trial=group.copy();trial[:,a]-=change;trial[:,b]+=change
            new=critical(trial);value=float(new.sum())
            if value<score:
                labels[e],labels[f]=b,a;group=trial;values=new;score=value;accepted+=1
        receipt=dict(seed=seed+si,initial_mean_ns=initial,final=o.stats(values),accepted=accepted)
        receipts.append(receipt)
        if best is None or (score,labels.tolist())<(best[0],best[1].tolist()):best=(score,labels.copy(),values.copy())
    pages=np.zeros((65,16),dtype=np.int64)
    for e,g in enumerate(best[1]):pages[e,4*g:4*g+4]=12
    pages[64,12:16]=24
    assert o.capacity(pages) and np.array_equal(o.load(A,pages,tables).max(axis=1),best[2])
    return pages,dict(cold_expert=cold,cold_expert_TRAIN_frequency=int(counts[cold]),
                      critical_work_ns=o.stats(best[2]),resident_combined_max_WL=int((128+o.reserve(pages)).max()),starts=receipts)


def main():
    with np.load(o.TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3]
    with np.load(o.DEST/'oracle_arrays.npz') as z:initial={l:z[f'L{l}.shared_exact.pages'].copy() for l in range(3)}
    tables,_=o.price_tables(o.load_pipeline(),1);four,_=o.price_tables(o.load_pipeline(),4)
    records=[];arrays={};rows={}
    for layer in range(3):
        A=o.incidence(routes[layer]);free,fr=secondary_refine(A,initial[layer],tables,20270102+layer)
        variable,vr=secondary_refine(A,free,tables,20270202+layer,True)
        ded,dr=dedicated(A,tables,20270302+10*layer)
        for label,p,r in [('secondary_free4',free,fr),('secondary_variable',variable,vr),('dedicated_shared_g4',ded,dr)]:
            r=dict(layer=layer,label=label,**r,input4_same_map_critical_work_ns=o.stats(o.load(A,p,four).max(axis=1)),
                   shared_pages_per_chip=p[64].tolist())
            records.append(r);arrays[f'L{layer}.{label}.pages']=p;arrays[f'L{layer}.{label}.critical']=o.load(A,p,tables).max(axis=1)
            rows[f'L{layer}.{label}']=dict(routed_rows=(p[:64]*32).tolist(),shared_rows=(p[64]*16).tolist(),capacity_reserve_WL=o.reserve(p).tolist(),useful_bytes=int(p.sum()*16384),stored_copies=1)
    np.savez_compressed(o.DEST/'plateau_arrays.npz',**arrays)
    o.save('plateau_row_witnesses.json',rows)
    o.save('plateau_report.json',dict(status='PASS',source_sha256=o.sha(__file__),oracle_source_sha256=o.sha(Path(o.__file__)),
           protocol_sha256=o.sha(o.DEST/'plateau_addendum.frozen.json'),records=records,
           limits='TRAIN-only occupation objectives; no causal fullFFN latency, optimality or new-mapping-algorithm claims.'))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
