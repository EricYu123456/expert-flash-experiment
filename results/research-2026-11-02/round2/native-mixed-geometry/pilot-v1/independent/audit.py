"""Read-only independent accounting of the bounded mixed-geometry pilot."""
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment')
OUT=Path(__file__).resolve().parent
PILOT=OUT.parent
sys.path.insert(0,str(ROOT/'scripts'))
from research_20261103_topology_placement_replay import Replay,load_python


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inventory(rows,shape):
    D,I,E,K,L,SI=shape
    columns=np.tile([D,I],(E+1,1));columns[-1,1]=SI
    coef=rows*columns[:,:,None]
    pages=(coef+16383)//16384;waves=(pages+3)//4
    return columns,coef,pages,waves


def audit_artifact(path,record):
    assert sha(path)==record['arrays_sha256']
    with np.load(path) as z:a={k:z[k] for k in z.files}
    rr=a['rows'];D,I,E,K,L,SI=record['shape'];s=record['service']
    assert rr.shape==(E+1,2,16) and np.all(rr>=0)
    assert np.all(rr[:E,0].sum(1)==2*I) and np.all(rr[:E,1].sum(1)==D)
    assert rr[-1,0].sum()==2*SI and rr[-1,1].sum()==D
    assert np.array_equal(rr[-1],a['initial_rows'][-1])
    assert np.array_equal(np.sort(a['initial_rows'],axis=2),a['initial_row_multisets'])
    assert np.array_equal(rr[-1],record['shared_rows'])
    with np.load(record['training']['path']) as z:
        assert z['split'].item()=='train' and z['completed'].all()
        assert np.array_equal(a['routes'],z['routes'][:,:,record['layer']].reshape(-1,K))
    assert a['cost'].shape==(512,) and a['initial_cost'].shape==(512,)
    assert record['mean_ns']==float(a['cost'].mean())
    assert record['p99_ns']==float(np.percentile(a['cost'],99))
    assert np.sum(a['cost'])<=np.sum(a['initial_cost'])
    for e in range(E):
        for p in range(2):
            candidate=np.asarray(record['menus'][p][a['profile_choices'][e,p]]['rows'])
            old=np.asarray(record['menus'][p][a['initial_profile_choices'][e,p]]['rows'])
            assert np.array_equal(np.sort(rr[e,p]),candidate)
            assert np.array_equal(np.sort(a['initial_rows'][e,p]),old)
    C,coef,pages,waves=inventory(rr,record['shape'])
    assert np.max(waves)<=192
    cc=waves*28000+(rr>0)*(250+C[:,:,None]*s.get('input_channel_copies',1)//2)+2*(pages*16384//C[:,:,None])
    joined_lb=(cc[a['routes']].sum((1,2))+cc[-1].sum(0)).max(1)+s['host_submit_ns']
    assert np.all(a['cost']>=joined_lb)
    jobs=a['jobs'];addresses=a['addresses'];logical=a['logical']
    assert jobs.shape[1]==5 and addresses.shape[1]==6
    assert np.all(jobs[:,1:]>0)
    assert np.all(jobs[:,3]==(jobs[:,1].astype('i8')*jobs[:,2]+16383)//16384)
    assert np.all(jobs[:,4]==(jobs[:,3]+3)//4)
    assert np.array_equal(addresses[:,0],jobs[:,0]) and np.array_equal(addresses[:,3],jobs[:,4])
    assert np.all(addresses[:,2]+addresses[:,3]<=192) and np.all(addresses[:,5]==0)
    blocks=[]
    for ch in range(16):
        aa=addresses[addresses[:,0]==ch];blocks.append(len(np.unique(aa[:,1])))
        for block in np.unique(aa[:,1]):
            bb=aa[aa[:,1]==block];bb=bb[np.argsort(bb[:,2])]
            assert np.all(bb[:-1,2]+bb[:-1,3]<=bb[1:,2])
    for e in range(E+1):
        for p in range(2):
            lo,hi=a['offsets'][e,p];jj=jobs[lo:hi];ll=logical[lo:hi]
            assert len(jj)==np.count_nonzero(rr[e,p])
            assert np.array_equal(jj[:,0],np.flatnonzero(rr[e,p]))
            assert np.array_equal(jj[:,1],rr[e,p,jj[:,0]])
            assert np.all(jj[:,2]==C[e,p])
            assert ll[0,3]==0 and ll[-1,3]+ll[-1,4]==rr[e,p].sum()
            assert np.all(ll[:-1,3]+ll[:-1,4]==ll[1:,3])
    ledger=record['per_chip_ledger'];physical=record['physical']
    assert ledger['useful_bytes_by_chip']==coef.sum((0,1)).tolist()
    assert ledger['programmed_page_bytes_by_chip']==(pages.sum((0,1))*16384).tolist()
    assert ledger['reserved_wave_bytes_by_chip']==(waves.sum((0,1))*65536).tolist()
    assert ledger['allocated_lsb_block_bytes_by_chip']==[n*192*65536 for n in blocks]
    assert physical['chip_blocks']==blocks
    assert physical['useful_bytes']==int(coef.sum())
    assert physical['programmed_page_bytes']==int(pages.sum()*16384)
    assert physical['reserved_wave_bytes']==int(waves.sum()*65536)
    assert physical['allocated_lsb_block_bytes']==sum(blocks)*192*65536
    plan=backend.lower_rows(rr[:,0],rr[:,1],D,[I]*E+[SI],backend.Service(**s))
    for k in ['offsets','jobs','input_columns','total_rows']:assert np.array_equal(plan[k],a[k])
    fresh,bounds=native.run(plan,a['routes'],s)
    assert np.array_equal(fresh,a['cost']) and np.array_equal(bounds,a['lower_bound'])
    python_checks=[]
    for q in [0,255,511]:
        got=backend.ReferenceGraph.build(plan,np.r_[a['routes'][q],E],backend.Service(**s)).run()
        assert got['cost']==fresh[q];python_checks.append(int(got['cost']))
    select=np.concatenate([a['routes'],np.full((512,1),E,dtype=a['routes'].dtype)],axis=1)
    # Query ledgers include all inactive fringe output rows actually emitted.
    qpages=pages[select].sum((1,2));qwaves=waves[select].sum((1,2))
    qcommands=(rr[select]>0).sum((1,2))
    qinputs=((rr>0)*C[:,:,None])[select].sum((1,2))
    qout=(pages*16384//C[:,:,None])[select].sum((1,2))
    queryledger=dict(ordinary_pages_mean=float(qpages.sum(1).mean()),ordinary_waves_mean=float(qwaves.sum(1).mean()),native_commands_mean=float(qcommands.sum(1).mean()),input_bytes_mean=float(qinputs.sum(1).mean()),emitted_output_bytes_mean=float((qout.sum(1)*4).mean()),joint_native_chip_bound_mean_ns=float(joined_lb.mean()))
    return a,dict(jobs=len(jobs),query_ledger=queryledger,physical=physical,per_chip_ledger=ledger,python_costs=python_checks)


protocol_path=PILOT/'protocol.frozen.json';protocol=json.loads(protocol_path.read_text())
for item in protocol['inputs']:assert sha(item['path'])==item['sha256']
launch=json.loads((PILOT/'launch_results.json').read_text());assert len(launch)==4 and all(x['returncode']==0 for x in launch)
native=Replay();backend=load_python();records=[];data={};comparisons=[]
baseline_launch=json.loads(Path(protocol['baseline_launch_path']).read_text())
for item in launch:
    path=Path(item['path'])/'result.json';r=json.loads(path.read_text())
    assert r['source_sha256']==protocol['source_sha256'] and r['protocol_sha256']==sha(protocol_path)
    assert r['seed'] in protocol['seeds'] and r['fit']['proposals']==2048
    assert sum(v['attempts'] for v in r['fit']['move_counts'].values())==2048
    assert len(r['fit']['checkpoints'])>=19
    a,check=audit_artifact(r['arrays'],r);data[(r['model'],r['seed'])]=(r,a)
    changed=sum(not np.array_equal(a['profile_choices'][e],a['initial_profile_choices'][e]) for e in range(r['shape'][2]))
    geom_attempts=sum(v['attempts'] for k,v in r['fit']['move_counts'].items() if k.startswith('geometry_'))
    geom_accepted=sum(v['accepted'] for k,v in r['fit']['move_counts'].items() if k.startswith('geometry_'))
    records.append(dict(path=str(path),sha256=sha(path),model=r['model'],seed=r['seed'],mean_ns=r['mean_ns'],p99_ns=r['p99_ns'],profile_changes=r['fit']['profile_changes'],experts_with_changed_profiles=changed,geometry_attempts=geom_attempts,geometry_accepted=geom_accepted,profile_histograms=r['fit']['profile_histograms'],fit_seconds=r['fit']['seconds'],cpu_seconds=r['fit']['cpu_seconds'],optimization_calls=r['fit']['optimization_native_calls'],optimization_query_evaluations=r['fit']['optimization_native_query_evaluations'],validation_calls=r['fit']['validation_native_calls'],validation_query_evaluations=r['fit']['validation_native_query_evaluations'],initial_native_calls=1,initial_native_query_evaluations=r['fit']['unique_ordered_routes'],audit=check))
for model in ['granite4tiny','deepseek']:
    best_r,best_a=min([data[(model,s)] for s in protocol['seeds']],key=lambda x:x[0]['mean_ns'])
    eligible=[x for x in baseline_launch if json.loads(Path(x['case']).read_text())['model']==model]
    best=min(eligible,key=lambda x:x['mean_ns']);same=min([x for x in eligible if x['family']=='independent_paired'],key=lambda x:x['mean_ns'])
    with np.load(Path(best['path'])/'arrays.npz') as z:baseline_cost=z['cost']
    gain=100*(1-best_r['mean_ns']/best['mean_ns']);tail=100*(1-best_r['p99_ns']/best['p99_ns'])
    delta=baseline_cost-best_a['cost']
    comparisons.append(dict(model=model,best_seed=best_r['seed'],mixed_mean_ns=best_r['mean_ns'],mixed_p99_ns=best_r['p99_ns'],fixed_best_path=best['path'],fixed_best_family=best['family'],fixed_best_seed=best['seed'],fixed_best_mean_ns=best['mean_ns'],fixed_best_p99_ns=best['p99_ns'],same_start_fixed_path=same['path'],mean_gain_pct=gain,p99_gain_pct=tail,paired_improved_queries=int((delta>0).sum()),paired_equal_queries=int((delta==0).sum()),paired_worse_queries=int((delta<0).sum()),paired_delta_ns_quantiles=np.percentile(delta,[0,1,50,95,99,100]).tolist(),expand_gate=bool(gain>=2 or tail>=5),new_geometry_at_best=best_r['fit']['profile_changes']))
summary=dict(status='independently_audited',utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(protocol_path),source_sha256=protocol['source_sha256'],validation=dict(fits=4,native_query_costs=2048,python_checks=12,jobs=sum(x['audit']['jobs'] for x in records)),records=records,comparisons=comparisons,limits=['TRAIN512 only, no heldout','finite random menu/proposal search, not an upper bound or global optimum','matched proposal count does not imply matched CPU/native query count; both reported','initial inherited v2 work is additional upstream search','no native hardware, full-model or new heterosharding algorithm claim'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(comparisons=comparisons,validation=summary['validation']),indent=2))
