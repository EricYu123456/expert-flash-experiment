"""Independent accounting of immutable v2 + explicit pre-fit G4 assertion fix.

Does not import the fitter or mutate its results. Recomputes mandatory work,
row/inventory legality, family membership, budget, and selected native costs.
"""
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/home/ericyu/expert_flash_experiment')
BASE = ROOT/'results/research-2026-11-02/round2/shared-counterbalanced'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts'))
from research_20261103_topology_placement_replay import load_python


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def orbit(a):
    return min(tuple(np.roll(a,k).tolist()) for k in range(16))


backend = load_python()
artifacts = {}; totals = defaultdict(list); cases = []; checks = []; ledgers=[]
query_work = 0; physical_jobs = 0
v1completion_sha = sha(BASE/'completion.v1.json')
for version, model in [('v2-fix1','granite4tiny'),('v2','deepseek')]:
    protocol_path = BASE/version/'protocol.frozen.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['v1_completion_sha256'] == v1completion_sha
    for src in protocol['sources']:
        assert sha(src['path']) == src['sha256']
    for layer in range(3):
        for copies in [1,4]:
            path = BASE/version/f'{model}.L{layer}.input{copies}/result.json'
            d = json.loads(path.read_text())
            assert d['script_sha256'] == protocol['source_sha256']
            assert sha(d['v1_case_path']) == d['v1_case_sha256']
            old = json.loads(Path(d['v1_case_path']).read_text())
            source = json.loads(Path(old['source_summary']).read_text())
            rec = next(r for r in source['records'] if r['model']==model and r['layer']==layer and r['command_ns']==250)
            assert sha(d['training']['path']) == d['training']['sha256']
            with np.load(d['training']['path']) as z:
                train_routes = z['routes'][:,:,layer].reshape(-1,6)
            budget=defaultdict(lambda:[0,0]); shared={}; family_inventory={}
            selections=defaultdict(list); menu_count=0; paired_attempts=defaultdict(int)
            for r in d['records']:
                family=r['family']; context=(r['shared_geometry'],r['seed_index'])
                key=(*context,family); budget[key][0]+=r['fit']['proposals']; budget[key][1]+=r['fit']['greedy_attempts']
                for k,v in r['fit']['move_attempts'].items():paired_attempts[(family,k)]+=v
                if context in shared:assert shared[context]==r['shared_rows']
                shared[context]=r['shared_rows']
                eligible=['joint_cyclic'] if family=='joint_simple_menu' else ['counterbalanced','independent_permutation']
                inherited=min((q[stage]['mean_ns'] for q in old['records'] if (q['shared_geometry'],q['seed_index'])==context and q['family'] in eligible for stage in ['initial','final']))
                assert r['inherited']['mean_ns']==inherited
                options=[(r[t]['mean_ns'],t) for t in ['initial','final','inherited']]
                assert r[r['selected']]['mean_ns']==min(options)[0]
                selections[family].append((r[r['selected']]['mean_ns'],r))
                ini=None
                for stage in ['initial','final','inherited']:
                    a=r[stage]; ap=Path(a['arrays']); assert sha(ap)==a['sha256']
                    with np.load(ap) as z: arr={k:z[k] for k in z.files}
                    rr=arr['rows']; routes=arr['routes']; costs=arr['cost']
                    assert np.array_equal(routes,train_routes)
                    assert costs.shape==(512,) and a['mean_ns']==float(costs.mean())
                    assert a['p99_ns']==float(np.percentile(costs,99))
                    D,I,E,K,L,SI=d['shape']; C=np.tile([D,I],(E+1,1)); C[-1,1]=SI
                    pages=(rr*C[:,:,None]+16383)//16384; waves=(pages+3)//4
                    cc=waves*28000+(rr>0)*(250+C[:,:,None]*copies//2)+2*(pages*16384//C[:,:,None])
                    necessary=(cc[routes].sum((1,2))+cc[-1].sum(0)).max(1)
                    assert np.all(costs>=necessary+2000)
                    assert np.all(rr[:E,0].sum(1)==2*I) and np.all(rr[:E,1].sum(1)==D)
                    assert rr[-1,0].sum()==2*SI and rr[-1,1].sum()==D
                    assert np.array_equal(rr[-1],r['shared_rows'])
                    if stage in ['initial','final']:
                        target=r['fit']['initial_work_ns' if stage=='initial' else 'final_work_ns']
                        assert abs(necessary.mean()-target)<1e-6
                        query_work+=512
                        if ini is None:ini=rr.copy()
                        else:assert np.array_equal(np.sort(ini,axis=2),np.sort(rr,axis=2))
                    if stage=='initial' and family=='independent_paired':
                        if r['seed_index']==0:
                            guide=next(q for q in old['records'] if q['family']=='counterbalanced' and (q['shared_geometry'],q['seed_index'])==context)
                            with np.load(guide['initial']['arrays']) as z:assert np.array_equal(rr,z['rows'])
                        else:
                            with np.load(r['inherited']['arrays']) as z:assert np.array_equal(rr,z['rows'])
                    if stage=='final' and family=='joint_simple_menu':
                        perm=np.asarray(r['fit']['final_global_permutations'])
                        for ph in range(2):
                            variants=np.stack([np.roll(ini[0,ph],k)[perm[ph]] for k in range(16)])
                            assert all(np.any(np.all(variants==row,axis=1)) for row in rr[:E,ph])
                    if ap not in artifacts:
                        jobs=arr['jobs']; aa=arr['addresses']; logical=arr['logical']
                        assert np.all(jobs[:,3]==(jobs[:,1].astype('i8')*jobs[:,2]+16383)//16384)
                        assert np.all(jobs[:,4]==(jobs[:,3]+3)//4) and np.max(jobs[:,4])<=192
                        assert np.array_equal(aa[:,0],jobs[:,0]) and np.array_equal(aa[:,3],jobs[:,4])
                        assert np.all(aa[:,2]+aa[:,3]<=192)
                        assert int((rr*C[:,:,None]).sum())==a['physical']['useful_bytes']
                        assert int(pages.sum()*16384)==a['physical']['programmed_page_bytes']
                        assert int(waves.sum()*65536)==a['physical']['reserved_wave_bytes']
                        blocks=[len(np.unique(aa[aa[:,0]==ch,1])) for ch in range(16)]
                        assert blocks==a['physical']['chip_blocks']
                        assert sum(blocks)*192*65536==a['physical']['allocated_lsb_block_bytes']
                        for ch,bl in np.unique(aa[:,:2],axis=0):
                            part=aa[(aa[:,0]==ch)&(aa[:,1]==bl)];part=part[np.argsort(part[:,2])]
                            assert np.all(part[:-1,2]+part[:-1,3]<=part[1:,2])
                        for e in range(E+1):
                            for ph in range(2):
                                lo,hi=arr['offsets'][e,ph];ll=logical[lo:hi]
                                assert ll[0,3]==0 and ll[-1,3]+ll[-1,4]==rr[e,ph].sum()
                                assert np.all(ll[:-1,3]+ll[:-1,4]==ll[1:,3])
                        physical_jobs+=len(jobs);artifacts[ap]=a['sha256']
                if family=='joint_simple_menu':
                    proof=r['menu_pool'];assert proof['phase_candidates']==[len(x) for x in rec['fitting']]
                    assert proof['contains_whole_rows_templates'];menu_count+=1
            assert all(n==20000 and g<=4096 for n,g in budget.values())
            assert all(sum(v for (f,k),v in paired_attempts.items() if f==family and k.startswith('paired_'))>0 for family in selections)
            for family,r in d['winners'].items():
                assert r[r['selected']]['mean_ns']==min(v for v,_ in selections[family])
                a=r[r['selected']]
                with np.load(a['arrays']) as z:arr={k:z[k] for k in z.files}
                totals[(model,copies,family)].append((layer,arr['cost']))
                rr=arr['rows'];D,I,E,K,L,SI=d['shape'];C=np.tile([D,I],(E+1,1));C[-1,1]=SI
                pp=(rr*C[:,:,None]+16383)//16384;ww=(pp+3)//4
                ledgers.append(dict(model=model,layer=layer,copies=copies,family=family,arrays=a['arrays'],sha256=a['sha256'],shared_geometry=r['shared_geometry'],seed=r['seed'],selected=r['selected'],useful_bytes_by_chip=(rr*C[:,:,None]).sum((0,1)).tolist(),programmed_page_bytes_by_chip=(pp.sum((0,1))*16384).tolist(),reserved_wave_bytes_by_chip=(ww.sum((0,1))*65536).tolist(),allocated_lsb_block_bytes_by_chip=[n*192*65536 for n in a['physical']['chip_blocks']]))
                for query in [0,255,511]:
                    plan={k:arr[k] for k in ['jobs','offsets','input_columns','total_rows']}
                    got=backend.ReferenceGraph.build(plan,np.r_[arr['routes'][query],64],backend.Service(**d['service'])).run()
                    assert got['cost']==arr['cost'][query]
                    checks.append(dict(model=model,layer=layer,copies=copies,family=family,query=query,cost_ns=int(got['cost'])))
            cases.append(dict(path=str(path),sha256=sha(path),families={f:dict(selected=r['selected'],arrays=r[r['selected']]['arrays'],mean_ns=r[r['selected']]['mean_ns'],shared=r['shared_geometry'],seed=r['seed'],proposal_budget=20000,greedy_budget=4096) for f,r in d['winners'].items()},budget_checks=len(budget),menu_pool_records=menu_count,move_attempts={f+'.'+k:v for (f,k),v in paired_attempts.items()}))

v1=json.loads((BASE/'summary.v1.json').read_text());summary={};npz={}
for (model,copies,family),vals in totals.items():
    vals.sort();a=np.stack([v for _,v in vals]).sum(0);key=f'{model}.input{copies}.{family}'
    npz[key]=a;summary[key]=dict(mean_ns=float(a.mean()),p99_ns=float(np.percentile(a,99)),queries=512,layers=3)
comparisons=[]
for model in ['granite4tiny','deepseek']:
    for copies in [1,4]:
        key=f'{model}.input{copies}.';simple=summary[key+'joint_simple_menu'];generic=summary[key+'independent_paired'];prior=v1['totals'][key+'counterbalanced']
        comparisons.append(dict(model=model,copies=copies,generic_mean_gain_vs_simple_pct=100*(1-generic['mean_ns']/simple['mean_ns']),generic_p99_gain_vs_simple_pct=100*(1-generic['p99_ns']/simple['p99_ns']),generic_incremental_mean_gain_vs_retained_v1coupled_pct=100*(1-generic['mean_ns']/prior['mean_ns']),generic_incremental_p99_gain_vs_retained_v1coupled_pct=100*(1-generic['p99_ns']/prior['p99_ns'])))
np.savez_compressed(OUT/'paired_totals.v2.npz',**npz)
result=dict(status='passed_v2_with_explicit_prefit_G4_fix',utc=datetime.now(timezone.utc).isoformat(),v1_completion_sha256=v1completion_sha,validation=dict(unique_artifacts=len(artifacts),independent_query_work=query_work,physical_jobs=physical_jobs,python_cpp_checks=len(checks),cases=len(cases)),totals=summary,comparisons=comparisons,cases=cases,checks=checks,limits=['TRAIN only, no heldout opened','Simple menu is a finite geometry/heuristic pool; no claim of all-permutation optimum','Generic retains prior coupled incumbent; inherited search effort is explicitly additional and not a fresh equal-total-compute comparison','Native values are synthetic timing replay, not measured full-model or hardware speed','V1 coupled and generic v2 are almost identical; no performance claim for constrained-guide algorithm over generic'])
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'winner_ledgers.json').write_text(json.dumps(ledgers,indent=2)+'\n')
files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(BASE.rglob('*')) if p.is_file() and ('v2' in p.parts or 'v2-fix1' in p.parts or 'independent-v2' in p.parts) and p.name!='completion.json']
(OUT/'completion.json').write_text(json.dumps(dict(status='sealed_v2_and_explicit_fix1',utc=datetime.now(timezone.utc).isoformat(),files=files),indent=2)+'\n')
print(json.dumps(dict(validation=result['validation'],comparisons=comparisons,summary_sha256=sha(OUT/'summary.json'),completion_sha256=sha(OUT/'completion.json')),indent=2))
