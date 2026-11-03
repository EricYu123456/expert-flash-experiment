#!/usr/bin/env python3
"""Post-hoc development diagnosis; fixed maps, no fitting or native replay."""
import argparse
import gzip
import hashlib
import itertools
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/'results/research-2026-11-02/round2'
BASE=ROUND/'chip-count-regime-screen'
PARENT=BASE/'m32-prospective-tail-validation'
OUT=PARENT/'posthoc-generalization-diagnosis'
TRAIN=ROUND/'w8-format-training/granite4tiny/decode.routes.npz'
FRESH=PARENT/'replay-adapter-v2/prospective/L0.routes.npz'
SECOND=PARENT/'all-layer-fixed-map-transfer/execution'
NAMES=('free','common_cyclic','phase_independent','tail_cyclic')
ORIGINS={'free':('g4-M32-coupled-cp','incumbent4.GF.cost'),
 'common_cyclic':('g4-M32-cyclic-rotation-cp','incumbent0.GF.cost'),
 'phase_independent':('g4-M32-independent-rotation-cp','incumbent2.GF.cost'),
 'tail_cyclic':('m32-tail-directed-rotation-control','input1.independent_top6_work.GF.cost')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def correlation(a,b):
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def stats(a):return dict(mean=float(np.mean(a)),p99=float(np.percentile(a,99)),max=float(np.max(a)))


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    files=[Path(__file__),TRAIN,FRESH,PARENT/'analysis/report.json',
           PARENT/'replay-adapter-v2/prospective/costs.npz',PARENT/'replay-adapter-v2/prospective/completion.json',
           PARENT/'protocol.frozen.json',SECOND/'costs.npz',SECOND/'completion.frozen.json',
           BASE/'g4-M32-coupled-cp/protocol.frozen.json',ROOT/'scripts/research_20261103_g4_m32_coupled_cp.py']
    files += [PARENT/'plans'/(name+'.frozen.npz') for name in NAMES[:3]]
    files += [BASE/'m32-tail-directed-rotation-control/native/independent_top6_work.plan.npz']
    files += [BASE/folder/'native_arrays.npz' for folder,key in ORIGINS.values()]
    references=[]
    for path in files:
        item=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        if path.suffix=='.py':
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        references.append(item)
    dump(OUT/'protocol.frozen.json',dict(status='posthoc_development_diagnosis_after_primary_negative',utc=datetime.now(timezone.utc).isoformat(),
      source_sha256=sha(__file__),references=references,
      scope='This previously prospective batch is now DEVELOPMENT data. Original frozen results remain intact. No fitting, map selection, native replay, new data or method claim.',
      maps='Original three mean-selected GF maps plus TRAIN-selected supplementary tail_cyclic; tail fresh L0 costs only bridge already sealed Boundary secondary arrays.',
      cold_definitions={'unseen':'TRAIN count=0','very_low':'TRAIN count1..4','low':'TRAIN count1..8','other':'TRAIN count>=9'},
      work='Independent scalar billing: each nonzero full-row job250ns+C/2ns+ceil(pages/4)*28000ns+2*floor(pages*16384/C)ns. Input1 only; shared fixed. NAND readwaves separate from native critical path.',
      masks='G=GUpositive,D=Downpositive,X=G&D,O=notG. Report perexpert masks/degrees, perchip inventory and TRAIN/dev frequency-weighted degree.',
      correlation='Center each chip owned-work vector across512queries; correlation only nonconstant chips. Effective correlation dimension=(sum nonnegative correlation eigenvalues)^2/sum eigenvalues^2. It is NOT independent physical loads/PE count. Also positive-correlation>=0.90 connected components, descriptive chaining caveat.',
      regression='Primary native difference free minus phase_independent, positive=regression. Describe all512, cold/unseen and no-cold counterexamples, delta-work vs native regression and shared-tier hot chips.',
      witnesses='Free TRAIN peak10 certificate implies dev chipN>10 needs at least5-S X-experts for shared-wave S=1/2/3. Save minimal deterministic sorted X-subset, independently check its TRAIN joint occurrences. This is a trace-specific explanation, not a causal or population proof.',
      selection='At most8 deterministic witnesses: worstnative regression, max work increase, no-low regression, work-nonincreasing regression, cold-containing improvement, greatest improvement; deduplicate. All query counterexamples remain in arrays/JSONL.',
      limits='No claim fresh heldout after this analysis. Work-hot chip is a necessary-work descriptor, not proven native critical resource. Associations do not prove device behavior or causal quality/performance generalization.'))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__)),indent=2))


def own_table(rows):
    table=np.zeros((65,32),np.int64);waves=np.zeros_like(table)
    for e in range(65):
        for phase in (0,1):
            c=1536 if phase==0 else (1024 if e==64 else 512)
            for chip,n in enumerate(rows[e,phase]):
                if n:
                    pages=(int(n)*c+16383)//16384;w=(pages+3)//4
                    table[e,chip]+=250+c//2+w*28000+2*(pages*16384//c);waves[e,chip]+=w
    return table,waves


def load_correlations(values):
    variance=values.std(0);active=np.flatnonzero(variance>0);constant=np.flatnonzero(variance==0)
    corr=np.corrcoef(values[:,active],rowvar=False)
    ev=np.clip(np.linalg.eigvalsh(corr),0,None)
    dim=float(ev.sum()**2/np.square(ev).sum()) if np.square(ev).sum() else 0.
    components=[];remaining=set(range(len(active)))
    while remaining:
        reached={min(remaining)};todo=list(reached)
        while todo:
            i=todo.pop()
            for j in list(remaining-reached):
                if corr[i,j]>=.9:reached.add(j);todo.append(j)
        remaining-=reached;components.append([int(active[x]) for x in sorted(reached)])
    off=corr[~np.eye(len(active),dtype=bool)]
    return dict(nonconstant_chips=active.tolist(),constant_chips=constant.tolist(),matrix=corr.tolist(),
                effective_correlation_dimension=dim,eigenvalues=ev.tolist(),positive_ge_090_components=components,
                mean_offdiagonal=float(off.mean()),mean_abs_offdiagonal=float(np.abs(off).mean()),
                maximum_offdiagonal=float(off.max()),minimum_offdiagonal=float(off.min()),
                interpretation='Correlation rank/threshold groups only; neither statistical independence nor available native execution groups.')


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
    for item in p['references']:assert sha(item['path'])==item['sha256']
    train=load(TRAIN);fresh=load(FRESH)
    routes={'train':train['routes'][:,:,0].reshape(512,6),'dev':fresh['routes'].reshape(512,6)}
    counts={split:np.bincount(r.ravel(),minlength=64) for split,r in routes.items()}
    train_count,dev_count=counts['train'],counts['dev'];unseen=train_count==0;low=(train_count>=1)&(train_count<=8);verylow=(train_count>=1)&(train_count<=4)
    presence={split:np.zeros((512,64),bool) for split in routes}
    for split,r in routes.items():presence[split][np.arange(512)[:,None],r]=True
    coact={split:presence[split].astype(np.int64).T@presence[split].astype(np.int64) for split in routes}
    fresh_cost=load(PARENT/'replay-adapter-v2/prospective/costs.npz');secondary=load(SECOND/'costs.npz')
    native={'train':{},'dev':{}};plans={};arrays={};degrees={};correlations={};profiles={}
    for name in NAMES:
        plan_path=PARENT/'plans'/(name+'.frozen.npz') if name!='tail_cyclic' else BASE/'m32-tail-directed-rotation-control/native/independent_top6_work.plan.npz'
        rows=load(plan_path)['rows'];plans[name]=rows
        folder,key=ORIGINS[name];native['train'][name]=load(BASE/folder/'native_arrays.npz')[key]
        if name=='tail_cyclic':native['dev'][name]=secondary['input1.tail_cyclic.cost'][:,:,0].ravel()
        else:
            native['dev'][name]=fresh_cost['input1.'+name+'.cost'].ravel()
            assert np.array_equal(native['dev'][name],secondary['input1.'+name+'.cost'][:,:,0].ravel())
        table,waves=own_table(rows);G=rows[:64,0]>0;D=rows[:64,1]>0;X=G&D;O=~G
        shared=waves[64];assert np.array_equal(shared,own_table(plans['free'])[1][64])
        degrees[name]=dict(perexpert=[dict(expert=e,train_count=int(train_count[e]),dev_count=int(dev_count[e]),
             G=np.flatnonzero(G[e]).tolist(),D=np.flatnonzero(D[e]).tolist(),X=np.flatnonzero(X[e]).tolist(),O=np.flatnonzero(O[e]).tolist()) for e in range(64)],
             perchip={mask:matrix.sum(0).tolist() for mask,matrix in [('G',G),('D',D),('X',X),('O',O)]},
             shared_readwaves=shared.tolist(),shared_ownedwork_ns=table[64].tolist(),
             perchip_useful_bytes=(rows[:,0].sum(0)*1536+rows[:64,1].sum(0)*512+rows[64,1]*1024).tolist(),
             frequency_weighted={split:{mask:(matrix.T@counts[split]/512).tolist() for mask,matrix in [('G',G),('D',D),('X',X),('O',O)]} for split in routes})
        profiles[name]={};correlations[name]={}
        for split,r in routes.items():
            work=table[r].sum(1)+table[64];nand=waves[r].sum(1)+waves[64]
            assert np.array_equal(work.mean(0),table[:64].T@counts[split]/512+table[64])
            arrays[f'{split}.{name}.chipwork']=work;arrays[f'{split}.{name}.chipwaves']=nand;arrays[f'{split}.{name}.native_cost']=native[split][name]
            peaks=work.max(1);hot=work==peaks[:,None]
            profiles[name][split]=dict(native=stats(native[split][name]),owned_work_peak_ns=stats(peaks),nand_peak_readwaves=stats(nand.max(1)),
                nand_peak_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(nand.max(1),return_counts=True))},
                work_hot_chip_occurrences_by_shared_tier={str(s):int(hot[:,shared==s].sum()) for s in sorted(set(shared))},
                hot_occurrence_denominator=int(hot.sum()),mean_perchip_work_ns=work.mean(0).tolist())
            correlations[name][split]=load_correlations(work)
    # All original free TRAIN routes exactly satisfy its saved peak10 certificate.
    assert np.all(arrays['train.free.chipwaves'].max(1)==10)
    freeG=plans['free'][:64,0]>0;freeD=plans['free'][:64,1]>0;freeX=freeG&freeD;freeO=~freeG
    shared=own_table(plans['free'])[1][64];sharedwork=own_table(plans['free'])[0][64]
    assert np.all((freeG|freeD)) and np.all(freeX.sum(1)==5)
    r=routes['dev'];native_delta=native['dev']['free']-native['dev']['phase_independent'];reg=native_delta>0
    work_delta=arrays['dev.free.chipwork'].max(1)-arrays['dev.phase_independent.chipwork'].max(1)
    nand_delta=arrays['dev.free.chipwaves'].max(1)-arrays['dev.phase_independent.chipwaves'].max(1)
    has_unseen=unseen[r].any(1);has_low=low[r].any(1);has_verylow=verylow[r].any(1)
    def subset_stats(mask):
        return dict(queries=int(mask.sum()),unseen_expert_queries=int((mask&has_unseen).sum()),low_expert_queries=int((mask&has_low).sum()),
                    verylow_expert_queries=int((mask&has_verylow).sum()),work_peak_increases=int((mask&(work_delta>0)).sum()),
                    work_peak_equal=int((mask&(work_delta==0)).sum()),work_peak_decreases=int((mask&(work_delta<0)).sum()),
                    nand_peak_increases=int((mask&(nand_delta>0)).sum()),nand_peak_equal=int((mask&(nand_delta==0)).sum()),nand_peak_decreases=int((mask&(nand_delta<0)).sum()))
    violations=[];query_records=[]
    for q,route in enumerate(r):
        work=arrays['dev.free.chipwork'][q];nand=arrays['dev.free.chipwaves'][q];hot=np.flatnonzero(work==work.max()).tolist()
        witnesses=[]
        for chip in np.flatnonzero(nand>10):
            xs=sorted(int(e) for e in route if freeX[e,chip]);need=5-int(shared[chip]);assert len(xs)>=need
            small=xs[:need];train_joint=int(presence['train'][:,small].all(1).sum());dev_joint=int(presence['dev'][:,small].all(1).sum())
            assert train_joint==0
            entry=dict(query=q,chip=int(chip),shared_waves=int(shared[chip]),route_nand_waves=int(nand[chip]),
                       all_X_experts=xs,minimal_subset=small,train_joint_occurrences=train_joint,dev_joint_occurrences=dev_joint,
                       train_expert_counts=[int(train_count[e]) for e in small],dev_expert_counts=[int(dev_count[e]) for e in small])
            violations.append(entry);witnesses.append(entry)
        selected=[]
        representative=hot[0]
        for e in route:
            e=int(e);selected.append(dict(expert=e,train_count=int(train_count[e]),dev_count=int(dev_count[e]),
                unseen=bool(unseen[e]),low_train=bool(low[e]),G=bool(freeG[e,representative]),D=bool(freeD[e,representative]),X=bool(freeX[e,representative]),O=bool(freeO[e,representative])))
        query_records.append(dict(query=q,prompt_id=str(fresh['prompt_ids'][q//16]),source_key=str(fresh['source_keys'][q//16]),token_index=q%16,
          route=list(map(int,route)),native_free_minus_primary_ns=int(native_delta[q]),free_work_peak_minus_primary_ns=int(work_delta[q]),
          free_nand_peak_minus_primary=int(nand_delta[q]),native_costs={name:int(native['dev'][name][q]) for name in NAMES},
          ownedwork_peaks={name:int(arrays['dev.'+name+'.chipwork'][q].max()) for name in NAMES},
          nand_peaks={name:int(arrays['dev.'+name+'.chipwaves'][q].max()) for name in NAMES},
          free_work_hot_chips=hot,representative_hot_chip=representative,representative_shared_waves=int(shared[representative]),
          representative_shared_work_ns=int(sharedwork[representative]),selected_on_representative_chip=selected,
          unseen_joint_activation_witnesses=witnesses))
    picks={}
    def pick(label,mask,values,largest=True):
        indices=np.flatnonzero(mask)
        if len(indices):picks[label]=int(indices[np.argmax(values[indices]) if largest else np.argmin(values[indices])])
        else:picks[label]=None
    pick('worst_native_regression',reg,native_delta);pick('largest_ownedwork_increase',np.ones(512,bool),work_delta)
    pick('regression_without_low_experts',reg&~has_low,native_delta)
    pick('regression_without_unseen_experts',reg&~has_unseen,native_delta)
    pick('regression_with_nonincreasing_ownedwork',reg&(work_delta<=0),native_delta)
    pick('improvement_with_low_experts',(native_delta<0)&has_low,native_delta,False)
    pick('greatest_native_improvement',native_delta<0,native_delta,False)
    selected_indices=sorted({q for q in picks.values() if q is not None})
    witnesses=dict(selection_rules=picks,queries=[query_records[q] for q in selected_indices])
    train_sets={tuple(sorted(x)) for x in routes['train']};train_ordered={tuple(x) for x in routes['train']}
    p1=train_count/train_count.sum();p2=dev_count/dev_count.sum();mid=(p1+p2)/2
    kl=lambda a,b:float(np.sum(a[a>0]*np.log2(a[a>0]/b[a>0])))
    pairs=[tuple(pair) for route in r for pair in itertools.combinations(sorted(map(int,route)),2)]
    experts=[dict(expert=e,train_count=int(train_count[e]),dev_count=int(dev_count[e]),difference=int(dev_count[e]-train_count[e]),
                  train_category='unseen' if unseen[e] else 'very_low' if verylow[e] else 'low' if low[e] else 'other') for e in range(64)]
    diagnosis=dict(status='posthoc_dev_diagnosis_complete',protocol_sha256=sha(OUT/'protocol.frozen.json'),
      experts=dict(train_unseen=np.flatnonzero(unseen).tolist(),train_verylow=np.flatnonzero(verylow).tolist(),train_low=np.flatnonzero(low).tolist(),
                   activation_total_variation=float(np.abs(p1-p2).sum()/2),activation_Jensen_Shannon_bits=(kl(p1,mid)+kl(p2,mid))/2),
      route_shift=dict(dev_queries_exact_train_set=sum(tuple(sorted(x)) in train_sets for x in r),dev_queries_exact_ordered_train_route=sum(tuple(x) in train_ordered for x in r),
                       dev_pair_events=len(pairs),dev_pair_events_absent_train=sum(coact['train'][a,b]==0 for a,b in pairs),
                       dev_unique_pairs_absent_train=len({(a,b) for a,b in pairs if coact['train'][a,b]==0})),
      profiles=profiles,regression=dict(regressed=subset_stats(reg),improved=subset_stats(native_delta<0),tied=subset_stats(native_delta==0),
          native_delta_vs_owned_peak_delta_Pearson=correlation(native_delta,work_delta),native_delta_vs_nand_peak_delta_Pearson=correlation(native_delta,nand_delta)),
      unseen_joint_witness=dict(dev_query_chip_violations=len(violations),dev_queries_with_violation=len({x['query'] for x in violations}),
          minimal_subset_size_hist={str(k):sum(len(x['minimal_subset'])==k for x in violations) for k in (2,3,4)},
          all_minimal_subsets_joint_absent_train=True,violations_with_all_experts_train_count_ge9=sum(all(x>=9 for x in w['train_expert_counts']) for w in violations),
          explanation='At shared-tier3 a new co-selected pair of X experts creates11readwaves; tiers2/1 require new triples/quadruples. Every saved minimal subset is directly absent from all512TRAIN route sets. Individualexperts neednotbe unseen or cold.'),
      limits='Post-hoc DEVELOPMENT association only. Fixed maps, no fit or winner selection. Saved all512queries and allcounterexamples. Native regressions can occur without a larger owned-work peak due to DAG/control/channel ordering. Correlation dimension is not physical/statistical independence.')
    arrays.update(train_routes=routes['train'],dev_routes=r,train_expert_counts=train_count,dev_expert_counts=dev_count,native_free_minus_primary=native_delta,
                  owned_peak_free_minus_primary=work_delta,nand_peak_free_minus_primary=nand_delta)
    np.savez_compressed(OUT/'arrays.npz',**arrays);dump(OUT/'experts.json',experts);dump(OUT/'degrees.json',degrees)
    dump(OUT/'correlations.json',correlations);dump(OUT/'witnesses.json',witnesses);dump(OUT/'joint_activation_violations.json',violations)
    with gzip.open(OUT/'all_query_records.jsonl.gz','wt') as f:
        for record in query_records:f.write(json.dumps(record)+'\n')
    dump(OUT/'diagnosis.json',diagnosis)
    for item in p['references']:assert sha(item['path'])==item['sha256']
    dump(OUT/'completion.json',dict(status='sealed_posthoc_dev',utc=datetime.now(timezone.utc).isoformat(),
        files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(OUT.rglob('*')) if x.is_file() and not x.name.endswith('.log')]))
    print(json.dumps(dict(status=diagnosis['status'],experts=diagnosis['experts'],route_shift=diagnosis['route_shift'],
         regression=diagnosis['regression'],joint_witness=diagnosis['unseen_joint_witness'],
         completion_sha256=sha(OUT/'completion.json')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));a=parser.parse_args();{'freeze':freeze,'run':run}[a.mode]()
