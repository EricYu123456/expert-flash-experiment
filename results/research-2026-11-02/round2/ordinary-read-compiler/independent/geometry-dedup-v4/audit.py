#!/usr/bin/env python3
"""Bounded geometry/search attribution audit; no fitting/replay/address scan."""
import ast
from collections import Counter, defaultdict, deque
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment')
BASE=ROOT/'results/research-2026-11-02/round2/ordinary-read-compiler'
V3=BASE/'full-campaign-v3'
V4=BASE/'full-campaign-v4-deduplicated'
OUT=Path(__file__).resolve().parent
RANK={'disjoint':0,'cyclic':1,'whole_rows':2}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')

def variants(record):
    b=np.asarray(record['base_rows'],dtype=np.int64)
    if record['family']=='disjoint':
        width=int(record['label'].split('_')[-1])
        shifts=[(v%(16//width))*width for v in range(16)]
    else:shifts=range(16)
    return np.asarray([np.roll(b,x) for x in shifts])

def initial(record,E):
    if record['family']=='disjoint':
        width=int(record['label'].split('_')[-1]);return np.arange(E)%(16//width)
    return np.arange(E)*max(1,16//E)%16

def signature(record,E):
    vv=[tuple(x.tolist()) for x in variants(record)]
    hist=Counter(vv[x] for x in initial(record,E))
    return tuple(sorted(vv)),tuple(sorted(hist.items()))

@lru_cache(None)
def reachable(initial_labels):
    todo=deque([initial_labels]);seen={initial_labels};E=len(initial_labels)
    while todo:
        state=todo.popleft()
        for i in range(E):
            for j in range(i+1,E):
                x=list(state);x[i],x[j]=x[j],x[i];x=tuple(x)
                if x not in seen:seen.add(x);todo.append(x)
    return frozenset(seen)

def small_exhaustive():
    checked=0;classes=0;images=0
    # Actual BFS over all expert-pair label swaps, allowing duplicate variants.
    for M in range(1,5):
        for E in range(1,5):
            groups={}
            for vv in itertools.product(range(3),repeat=M):
                for labels in itertools.product(range(M),repeat=E):
                    hist=tuple(sorted(Counter(vv[x] for x in labels).items()))
                    key=tuple(sorted(vv)),hist
                    physical=frozenset(tuple(vv[x] for x in state) for state in reachable(labels))
                    hist_permutations=frozenset(itertools.permutations(tuple(vv[x] for x in labels)))
                    assert physical==hist_permutations
                    if key in groups:assert groups[key]==physical
                    else:groups[key]=physical;classes+=1
                    checked+=1;images+=len(physical)
    # Omitting assigned histogram is unsound for E<M.
    vv=(0,1,2,3);a=(0,1);b=(0,2)
    pa={tuple(vv[x] for x in s) for s in reachable(a)}
    pb={tuple(vv[x] for x in s) for s in reachable(b)}
    assert pa!=pb
    return dict(cases=checked,signature_classes=classes,total_reachable_physical_images=images,
                dimensions='M=1..4,E=1..4,all3^Mvariant tables,allM^Einitial label vectors; duplicate ordinal variants included',
                omitted_histogram_counterexample=dict(variants=vv,initial_A=a,initial_B=b,physical_A=sorted(pa),physical_B=sorted(pb)),
                sufficiency='PASS',necessity='No: the variant multiset is conservative. Equal assigned physical histograms alone determine reachable assignments under arbitrary pair swaps.')

def snapshot_probe():
    p=V4/'source-snapshot/research_20261103_ordinary_read_compile.py'
    tree=ast.parse(p.read_text())
    only=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('initial_labels','deduplicate')]
    assert len(only)==2
    ns={'np':np,'hashlib':hashlib}
    exec(compile(ast.Module(body=only,type_ignores=[]),str(p),'exec'),ns)
    result=[]
    for E in (1,2,8,16,64,512):
        raw=[]
        for family in ('disjoint','cyclic'):
            for g in (1,16):
                base=np.zeros(16,dtype=np.int64);base[:g]=128//g
                shifts=[(v%(16//g))*g if family=='disjoint' else v for v in range(16)]
                raw.append(dict(label=f'{family}_{g}',family=family,variants=np.asarray([np.roll(base,s) for s in shifts])))
        out=ns['deduplicate'](raw,E)
        pairs={c['label']:c['aliases'] for c in out}
        # E=1 starts at chip0 in both policies; the E<M warning is not universal.
        assert ('cyclic_1' in pairs['disjoint_1'])==(E==1 or E>=16)
        assert 'cyclic_16' in pairs['disjoint_16']
        result.append(dict(experts=E,groups=pairs))
    return result

def phase_cost(vv,C,service):
    # All full matrices in this342-record campaign fit192WL; no split rule is
    # needed for these shapes. Independent arithmetic, no backend import.
    assert int(vv.max())*C<=service['block_wordlines']*65536
    pages=(vv*C+16383)//16384;waves=(pages+3)//4
    output=pages*16384//C
    work=waves*service['ordinary_read_ns']+service['command_ns']+(C+1)//2+2*output
    return np.where(vv>0,work,0)

def main():
    receipts={}
    for v in (V3,V4):
        for name in ('summary.json','protocol.frozen.json','model_totals.json'):
            receipts[str(v/name)]=sha(v/name)
    manifest=json.loads((V4/'source-snapshot/manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(V4/'protocol.frozen.json')
    for s in manifest['sources']:
        p=ROOT/s['snapshot'];assert sha(p)==s['sha256'];receipts[str(p)]=sha(p)
    p3=json.loads((V3/'protocol.frozen.json').read_text());p4=json.loads((V4/'protocol.frozen.json').read_text())
    assert p4['arguments']['reuse_fitting']==str(V3.relative_to(ROOT))
    for p,h in p4['reused_fitting_inputs_sha256'].items():assert sha(ROOT/p)==h
    a=json.loads((V3/'summary.json').read_text());b=json.loads((V4/'summary.json').read_text())
    key=lambda r:(r['model'],r['layer'],r['command_ns'])
    old={key(r):r for r in a['records']};assert len(b['records'])==342
    route_cache={};counts=Counter();cases=[];group_examples={};totals=defaultdict(lambda:np.zeros(64,dtype=np.int64));cert=Counter()
    for record in b['records']:
        kk=key(record);previous=old[kk];E=record['shape'][2]
        reuse=record['fitting_reused_from'];path=ROOT/reuse['path']
        assert sha(path)==reuse['sha256'];receipts[str(path)]=reuse['sha256']
        perfile=json.loads(path.read_text());assert perfile==previous
        for field in ('training','service','shape','selection_training_indices','shared_rows'):
            assert record[field]==previous[field],(kk,field)
        source=record['training']['path']
        if source not in route_cache:
            receipts[source]=sha(source)
            assert receipts[source]==record['training']['sha256']
            with np.load(source) as z:route_cache[source]=z['routes'].copy()
        full=route_cache[source]
        events=full[:,::4,record['layer']].reshape(-1,10) if record['model']=='qwen38' else full[:,:,record['layer']].reshape(-1,record['shape'][3])
        selected=events[record['training']['selected_indices']]
        assert len(selected)==1024
        signatures=[]
        for phase,(olds,news) in enumerate(zip(previous['fitting'],record['fitting'])):
            grouped={};ordered=[]
            for candidate in olds:
                sig=signature(candidate,E)
                if sig not in grouped:
                    grouped[sig]=dict(first=candidate,aliases=[],families=[]);ordered.append(sig)
                g=grouped[sig];g['aliases'].append(candidate['label'])
                if candidate['family'] not in g['families']:g['families'].append(candidate['family'])
            assert len(news)==len(ordered)
            actual_sigs=[]
            for candidate,sig in zip(news,ordered):
                group=grouped[sig];canonical=group['first']
                for field in ('label','family','labels','base_rows','mean_phase_chip_bound_ns','fit'):
                    assert candidate[field]==canonical[field],(kk,phase,candidate['label'],field)
                assert candidate['aliases']==group['aliases']
                assert candidate['families']==group['families']
                assert candidate['family']==min(group['families'],key=RANK.get)
                assert candidate['geometry_sha256']==hashlib.sha256(repr(sig).encode()).hexdigest()
                assert Counter(candidate['labels'])==Counter(initial(candidate,E).tolist())
                vv=variants(candidate);C=record['shape'][0 if phase==0 else 1]
                table=phase_cost(vv,C,record['service']);labels=np.asarray(candidate['labels'])
                obj=float(table[labels[selected]].sum(1).max(1).mean())
                assert obj==candidate['mean_phase_chip_bound_ns'],(kk,phase,candidate['label'],obj,candidate['mean_phase_chip_bound_ns'])
                actual_sigs.append(sig);counts['canonical_phase_fits']+=1
            signatures.append(actual_sigs)
            counts['removed_duplicate_phase_fits']+=len(olds)-len(news)
            group_examples[(record['model'],phase)]=[dict(canonical=grouped[s]['first']['label'],aliases=grouped[s]['aliases'],families=grouped[s]['families']) for s in ordered]
        assert sum(s['additional_replays'] for s in record['selection'])==record['total_distinct_replays']
        last=np.inf;fitted=record['fitting'];sel_by_fam={}
        for selection in record['selection']:
            family=selection['family'];rank=RANK[family]
            n=[sum(RANK[c['family']]<=rank for c in part) for part in fitted]
            assert selection['finite_pool_pairs']==n[0]*n[1]*16
            gi,di,shift=selection['key'];assert 0<=shift<16
            g,d=fitted[0][gi],fitted[1][di]
            assert g['label']==selection['gu_label'] and d['label']==selection['down_label']
            assert RANK[g['family']]<=rank and RANK[d['family']]<=rank
            assert selection['mean_ns']<=last;last=selection['mean_ns']
            assert selection['finite_pool_certified_best']==(not selection['replay_budget_exhausted'])
            assert 0<=selection['additional_replays']<=p4['arguments']['replay_budget']
            assert 0<=selection['lower_bound_pruned_pairs']<=selection['finite_pool_pairs']
            if selection['replay_budget_exhausted']:
                assert selection['additional_replays']==p4['arguments']['replay_budget']
                assert selection['lower_bound_pruned_pairs']==0
            artifact=next(x for x in record['artifacts'] if x['family']==family)
            # Read only cost/selection_routes; do not decompress jobs or scan PPAs.
            with np.load(ROOT/artifact['arrays']) as z:
                cost=z['cost'].copy();assert np.array_equal(z['selection_routes'],selected[record['selection_training_indices']])
            assert cost.shape==(64,)
            assert float(cost.mean())==selection['mean_ns']
            assert float(np.percentile(cost,99))==selection['p99_ns']
            totals[(record['model'],record['command_ns'],family)]+=cost
            cert[(record['model'],record['command_ns'],family)]+=selection['finite_pool_certified_best']
            sel_by_fam[family]=selection
        cyc,whole=sel_by_fam['cyclic'],sel_by_fam['whole_rows']
        if whole['mean_ns']<cyc['mean_ns']:
            gi,di,_=whole['key'];new=any(RANK[fitted[p][idx]['family']]==2 for p,idx in enumerate((gi,di)))
            assert new,(kk,'positive gain without newgeometry')
            counts['positive_layer_newgeometry']+=1
        else:new=False
        cases.append(dict(model=kk[0],layer=kk[1],command_ns=kk[2],cyclic_mean_ns=cyc['mean_ns'],whole_mean_ns=whole['mean_ns'],new_geometry=new,
                          cyclic_certified=cyc['finite_pool_certified_best'],whole_certified=whole['finite_pool_certified_best'],gu_label=whole['gu_label'],down_label=whole['down_label']))
    tm=json.loads((V4/'model_totals.json').read_text())['records'];assert len(tm)==27
    totals_index={}
    for r in tm:
        k=(r['model'],r['command_ns'],r['family']);v=totals[k]
        assert r['mean_ns']==float(v.mean()) and r['p99_ns']==float(np.percentile(v,99))
        assert r['finite_pool_mean_certified_layers']==cert[k]
        totals_index[k]=r
    attribution=[]
    old_totals={(r['model'],r['command_ns'],r['family']):r for r in a['model_totals']}
    for model in ('granite4tiny','deepseek','qwen38'):
        for command in (250,5000,10000):
            c,w=[totals_index[(model,command,f)] for f in ('cyclic','whole_rows')]
            oc,ow=[old_totals[(model,command,f)] for f in ('cyclic','whole_rows')]
            subset=[x for x in cases if x['model']==model and x['command_ns']==command]
            attribution.append(dict(model=model,command_ns=command,cyclic_mean_ns=c['mean_ns'],whole_mean_ns=w['mean_ns'],
                extra_mean_gain_pct=100*(1-w['mean_ns']/c['mean_ns']),v3_extra_mean_gain_pct=100*(1-ow['mean_ns']/oc['mean_ns']),
                positive_layers=sum(x['new_geometry'] for x in subset),cyclic_certified_layers=c['finite_pool_mean_certified_layers'],whole_certified_layers=w['finite_pool_mean_certified_layers']))
    counts['records']=len(cases);counts['global_totals']=len(tm);counts['address_jobs_scanned']=0;counts['fit_calls']=0;counts['schedule_replays']=0
    equivalence=small_exhaustive();equivalence['actual_snapshot_E_probes']=snapshot_probe()
    save('equivalence.json',equivalence)
    save('attribution.json',dict(totals=attribution,layers=cases,canonical_group_examples=[dict(model=k[0],phase=k[1],groups=v) for k,v in group_examples.items()]))
    assert all(sha(p)==h for p,h in receipts.items())
    result=dict(status='PASS bounded geometry deduplication and attribution audit',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),counts=dict(counts),
        exact_reuse='Every v4canonical labels/base_rows/objective/fitstats equals the earliest matching v3candidate exactly. Every saved objective independently recomputed on its1024training routes.',
        family_eligibility='Canonical family is minimum rank of its aliases. Because allowed sets are nested disjoint<cyclic<whole_rows, source family test equals alias-family-union eligibility in all records.',
        proof='Pair swaps generate every permutation of the initial ordinal-label multiset. Projecting labels through variants yields exactly every expert assignment with the initial physical row-vector histogram; duplicate ordinal labels can be distributed within each row class to meet their original counts. Thus equal assigned histograms suffice. Adding variant multiset to the key is conservative and cannot produce a false merge under these assumptions.',
        caveats=['Searchspace equality does not imply identical seeded stochastic trajectories or valid/accepted proposal counts. The fix reuses the first fit instead of choosing the best alias seed.',
                 'No per-expert allowed-owner constraints, ordinal-specific costs or restricted swap graph are present; such extensions would require a stronger key.',
                 'Global modelmaps were counted and their stored totals checked; no address/job scan was repeated. Architecture agent owns physical validation.',
                 'Finite pool counts/certification flags/budgets checked. No full finitepool LB recomputation or event replay was performed; prior backend/LB audits remain required.',
                 '342records cover3models/allMoElayers/3costs; E8 behavior separately tested with actual snapshot functions and exhaustive small analogs, not a new Mixtralcampaign.',
                 'Lowcostpositive winners use non-cyclic/disjoint geometry; this removes duplicate-seed attribution, but does not establish optimizer novelty/global label optimality/device speed or heldout generalization.'],
        input_sha256=receipts)
    save('audit.json',result)
    print(json.dumps(dict(status=result['status'],counts=dict(counts),small_exhaustive_cases=equivalence['cases'],attribution=attribution),indent=2))

if __name__=='__main__':main()
