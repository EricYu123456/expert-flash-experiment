import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment')
sys.path.insert(0,str(ROOT/'scripts'))
from research_20261103_topology_placement_replay import load_python

OUT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


protocol=json.loads((OUT/'protocol.frozen.json').read_text());backend=load_python()
for s in protocol['sources']:assert sha(s['path'])==s['sha256']
aggregate=defaultdict(list);checks=[];artifact_count=0;work_count=0;physical_jobs=0
for model,layer,copies in protocol['cases']:
    path=OUT/f'{model}.L{layer}.input{copies}/result.json';d=json.loads(path.read_text())
    assert d['script_sha256']==protocol['source_sha256']
    budget=defaultdict(lambda:[0,0]);shared={};initials={}
    for r in d['records']:
        key=(r['shared_geometry'],r['seed_index'],r['family'])
        budget[key][0]+=r['fit']['proposals'];budget[key][1]+=r['fit']['greedy_attempts']
        common=(r['shared_geometry'],r['seed_index'])
        if common in shared:assert shared[common]==r['shared_rows']
        shared[common]=r['shared_rows']
        initial=None
        for stage in ['initial','final']:
            a=r[stage];assert sha(a['arrays'])==a['sha256'];z=np.load(a['arrays'])
            rr=z['rows'];routes=z['routes'];D,I,E,K,L,SI=d['shape']
            C=np.tile(np.array([D,I]),(E+1,1));C[-1,1]=SI
            pages=(rr*C[:,:,None]+16383)//16384;waves=(pages+3)//4
            cc=waves*28000+(rr>0)*(250+C[:,:,None]*copies//2)+2*(pages*16384//C[:,:,None])
            necessary=(cc[routes].sum((1,2))+cc[-1].sum(0)).max(1)
            target=r['fit']['initial_work_ns' if stage=='initial' else 'final_work_ns']
            assert abs(necessary.mean()-target)<1e-6
            assert np.all(z['cost']>=necessary+2000)
            assert np.all(rr[:E,0].sum(1)==2*I) and np.all(rr[:E,1].sum(1)==D)
            assert rr[-1,0].sum()==2*SI and rr[-1,1].sum()==D
            if initial is None:initial=rr.copy()
            else:assert np.array_equal(np.sort(initial,axis=2),np.sort(rr,axis=2))
            jobs=z['jobs'];addr=z['addresses'];logical=z['logical']
            assert np.all(jobs[:,3]==(jobs[:,1].astype('i8')*jobs[:,2]+16383)//16384)
            assert np.all(jobs[:,4]==(jobs[:,3]+3)//4) and jobs[:,4].max()<=192
            assert np.array_equal(addr[:,0],jobs[:,0]) and np.array_equal(addr[:,3],jobs[:,4])
            assert np.all(addr[:,2]+addr[:,3]<=192)
            for chip,block in np.unique(addr[:,:2],axis=0):
                aa=addr[(addr[:,0]==chip)&(addr[:,1]==block)];aa=aa[np.argsort(aa[:,2])]
                assert np.all(aa[:-1,2]+aa[:-1,3]<=aa[1:,2])
            for e in range(E+1):
                for p in range(2):
                    lo,hi=z['offsets'][e,p];ll=logical[lo:hi]
                    assert ll[0,3]==0 and ll[-1,3]+ll[-1,4]==rr[e,p].sum()
                    assert np.all(ll[:-1,3]+ll[:-1,4]==ll[1:,3])
            if r['family']=='counterbalanced':
                joined=waves[:E].sum(1);low=2 if model=='granite4tiny' else 8
                assert np.all((joined==low+1).sum(1)==5) and np.all((joined==low).sum(1)==11)
            artifact_count+=1;work_count+=len(routes);physical_jobs+=len(jobs)
        if r['seed_index']==0 and r['family'] in ['independent_permutation','counterbalanced']:
            compare=(r['shared_geometry'],r['seed_index'])
            if compare in initials:assert np.array_equal(initials[compare],initial)
            initials[compare]=initial
    assert all(a==20000 and b<=4096 for a,b in budget.values())
    for family,r in d['winners'].items():
        a=r[r['selected']];z=np.load(a['arrays']);aggregate[(model,copies,family)].append((layer,z['cost']))
        for q in [0,255,511]:
            plan={k:z[k] for k in ['jobs','offsets','input_columns','total_rows']}
            got=backend.ReferenceGraph.build(plan,np.r_[z['routes'][q],64],backend.Service(**d['service'])).run()
            assert got['cost']==z['cost'][q]
            checks.append(dict(model=model,layer=layer,copies=copies,family=family,query=q,cost_ns=int(got['cost'])))
totals={};arrays={}
for (model,copies,family),values in aggregate.items():
    values.sort();v=np.stack([a for _,a in values]).sum(0);key=f'{model}.input{copies}.{family}'
    arrays[key]=v;totals[key]=dict(mean_ns=float(v.mean()),p99_ns=float(np.percentile(v,99)),layers=3,queries=512)
for model in ['granite4tiny','deepseek']:
    for copies in [1,4]:
        prefix=f'{model}.input{copies}.';c=totals[prefix+'counterbalanced'];j=totals[prefix+'joint_cyclic']
        c['mean_gain_vs_jointcyclic_pct']=100*(1-c['mean_ns']/j['mean_ns'])
        c['p99_gain_vs_jointcyclic_pct']=100*(1-c['p99_ns']/j['p99_ns'])
np.savez_compressed(OUT/'paired_totals.v1.npz',**arrays)
summary=dict(status='v1_sealed_pending_stronger_matched_moves_and_full_simple_menu',
    protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=protocol['source_sha256'],totals=totals,
    validation=dict(artifacts=artifact_count,independently_recomputed_query_work=work_count,
                    physical_jobs=physical_jobs,python_native_checks=len(checks)),checks=checks,
    limits='v1 independent uses only singlephase moves; cyclic excludes balanced/depthtemplates. Do not treat gains as outperforming strongest generic/simplemenu control. V2 separate, no edits to v1.')
(OUT/'summary.v1.json').write_text(json.dumps(summary,indent=2)+'\n')
files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*'))
       if p.is_file() and 'smoke' not in p.parts and 'v2' not in p.parts and p.name!='completion.v1.json']
(OUT/'completion.v1.json').write_text(json.dumps(dict(status='sealed_v1',utc=datetime.now(timezone.utc).isoformat(),
    files=files,source_sha256=protocol['source_sha256'],protocol_sha256=sha(OUT/'protocol.frozen.json')),indent=2)+'\n')
print(json.dumps(dict(completion_sha256=sha(OUT/'completion.v1.json'),validation=summary['validation'],totals=totals),indent=2))
