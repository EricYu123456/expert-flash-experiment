#!/usr/bin/env python3
"""Independent frozen replica address/identity and matched-anchor audit."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip, hashlib, json
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    protocol = read(OUT / 'protocol.frozen.json')
    for p,h in protocol['inputs_sha256'].items():assert sha(p)==h,p
    with gzip.open(OUT / 'physical_ledger.frozen.json.gz','rt') as z:records=json.load(z)
    totals=Counter();summary=[]
    for record in records:
        address_count=0;coefficient_bands=defaultdict(list)
        nonempty=0;max_by_plane=np.zeros(4,dtype=np.int64)
        for chip in record['chips']:
            occupied=set();counts=np.zeros((2,4),dtype=np.int64)
            for p in chip['GU']:
                key=(p['block'],p['plane'],p['wl'])
                assert key not in occupied;occupied.add(key);counts[key[:2]]+=1
            for job in chip['Down_jobs']:
                assert job['chip']==chip['chip'] and job['rows']*job['columns']==job['pages']*16384
                assert job['wl_start']+job['reserved_WL']<=192
                assert job['reserved_WL']==(job['pages']+3)//4
                coefficients=coefficient_bands[job['expert'],job['copy']]
                coefficients.append((job['row_start'],job['row_start']+job['rows'],job['columns']))
                # Complete C512/C1024 row bands; same expert and row identity
                # in each copy, so no mathematical coefficient is changed.
                for j in range(job['pages']):
                    key=(job['block'],j%4,job['wl_start']+j//4)
                    assert key not in occupied;occupied.add(key);counts[key[:2]]+=1
                totals['Down_jobs']+=1
            assert all(0<=b<2 and 0<=p<4 and 0<=w<192 for b,p,w in occupied)
            assert counts.tolist()==chip['actual_pages_per_block_plane']
            assert len(occupied)*16384==chip['programmed_bytes']
            assert (counts<=192).all()
            nonempty+=int(np.any(counts,axis=1).sum())
            max_by_plane=np.maximum(max_by_plane,counts.sum(axis=0))
            address_count+=len(occupied)
        assert nonempty==record['actual_nonempty_blocks']
        assert address_count*16384==record['programmed_bytes']
        for e in range(65):
            copies=3 if record['plan']=='triple_private_shared4' and e<64 else 1
            for copy in range(copies):
                bands=sorted(coefficient_bands[e,copy]);assert bands[0][0]==0 and bands[-1][1]==1536
                assert all(a[1]==b[0] for a,b in zip(bands,bands[1:]))
                assert all(c==(1024 if e==64 else 512) for _,_,c in bands)
        totals['plans']+=1;totals['physical_pages']+=address_count
        summary.append(dict(GU=record['GU'],plan=record['plan'],layer=record['layer'],
            programmed_bytes=record['programmed_bytes'],allocated_budget_blocks=32,
            actual_nonempty_blocks=nonempty,max_resident_pages_on_any_chip_by_plane=max_by_plane.tolist()))
    anchor_checks=[]
    child=BASE/'down-headroom/two-block-addendum'
    shared_winners=read(BASE/'down-headroom/final-run/selection.frozen.json')['winners']
    with np.load(OUT/'screen/service_arrays.npz') as own,np.load(child/'screen/service_arrays.npz') as other:
        for family in protocol['families']:
            for plan in ['original_g4_shared16','onecopy_shared4']:
                theirs='original_g4_shared16' if plan.startswith('original') else shared_winners[family]
                prefix=f'{family}.{plan}.input1.GF'
                target=f'{family}.{theirs}.input1.GF'
                for suffix in ['cost','fifo']:assert np.array_equal(own[prefix+'.'+suffix],other[target+'.'+suffix]),prefix
                anchor_checks.append(dict(our=prefix,existing=target,all192_cost_FIFO_exact=True))
    completed=OUT/'TRAIN512/report.json'
    full=[];full_anchor_checks=[];full_anchor_inputs={}
    if completed.exists():
        ownreport=read(completed)
        with np.load(OUT/'TRAIN512/service_arrays.npz') as arr:
            for r in ownreport['records']:
                assert r['completed']==1536 and r['valid_FIFO_capacities']['128']==1536
                if r['GU']=='candidate' and r['plan']=='triple_private_shared4':
                    cost=arr[r['name']+'.cost']
                    certificate=r['route_independence']
                    assert certificate['all_queries_same_DAG_and_cost'] and len(np.unique(cost))==1
                    assert sum(certificate['normalized_timing_DAG_counts'].values())==1536
                    full.append(dict(name=r['name'],formal_DAG_certificate=certificate,
                                     constant_model_cost_ns=int(cost.flat[0])))
            child_full=BASE/'down-headroom/final-run/TRAIN512'
            if (child_full/'report.json').exists():
                full_anchor_inputs={str(child_full/p):sha(child_full/p) for p in ['report.json','service_arrays.npz']}
                with np.load(child_full/'service_arrays.npz') as prior:
                    for family in protocol['families']:
                        for cp in [1,4]:
                            for priority in ['GF','DF']:
                                plans=['onecopy_shared4']+(['original_g4_shared16'] if family!='g8_fixed' else [])
                                for plan in plans:
                                    target_plan='original_g4_shared16' if plan.startswith('original') else shared_winners[family]
                                    own=f'{family}.{plan}.input{cp}.{priority}'
                                    target=f'{family}.{target_plan}.input{cp}.{priority}'
                                    for suffix in ['cost','fifo']:assert np.array_equal(arr[own+'.'+suffix],prior[target+'.'+suffix]),own
                                    full_anchor_checks.append(dict(our=own,existing=target,all1536_cost_FIFO_exact=True))
        # Validate every stored representative trace's exact private copy choice.
        train=Path(next(p for p in protocol['inputs_sha256'] if p.endswith('granite4tiny/decode.routes.npz')))
        with np.load(train) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3]
        for path in sorted((OUT/'TRAIN512').glob('trace.*.json')):
            trace=read(path);layer=int(path.name.split('.')[1][1:]);route=sorted(map(int,routes[layer,0]))+[64]
            private=[j for j in trace['jobs'] if j['phase']==1]
            by_expert=Counter()
            for j in private:
                physical=j['physical_down_job'];e=physical['expert'];rank=route.index(e)
                by_expert[e]+=physical['rows']
                command=trace['nodes'][j['command']]
                assert command['expert_rank']==rank
                assert physical['columns']==j['columns']==(1024 if e==64 else 512)
                if '.triple_private_shared4.' in path.name:
                    group=rank//2 if e<64 else 3
                    assert j['chip']//4==group
                    assert physical['copy']==(group if e<64 else 0)
                    assert physical['block']==(1 if e<64 else 0)
            assert by_expert==Counter({e:1536 for e in route})
            totals['full_representative_traces']+=1
    result=dict(status='PASS',source_sha256=sha(__file__),execution_source_sha256=protocol['inputs_sha256'][str(OUT/'replica_full_ffn.py')],
        checks=dict(totals),physical_summary=summary,matched_anchor_checks=anchor_checks,
        full_candidate_route_independence=full,full_anchor_checks=full_anchor_checks,full_anchor_inputs_sha256=full_anchor_inputs,
        scope='Complete-copy identity and physical-address audit. Formal service/cadence/copy/PB model only. No measuredhardware or numerical weight inference claim.')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',**dict(totals))),flush=True)

if __name__=='__main__':main()
