#!/usr/bin/env python3
"""Independent sealed postcompute-engine replay; no inference or plan changes."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
SOURCE = BASE / 'ordinary-read-compiler/second-heldout-replay-v2'
DEST = BASE / 'ordinary-read-compiler/independent/second-heldout-v2'
OLD = BASE / 'wave-drain/snapshots'
KEYS = ('offsets', 'jobs', 'input_columns', 'total_rows')
SHAPES = {'deepseek': (2048,1408,26,2816), 'granite4tiny': (1536,512,40,1024)}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, data): Path(p).write_text(json.dumps(data, indent=2)+'\n')
def stamp(): return datetime.now(timezone.utc).isoformat()
def load(p):
    with np.load(p) as z: return {k:z[k].copy() for k in z.files}
def describe(a):
    a=np.asarray(a,dtype=np.int64)
    quantiles=np.percentile(a,[50,95,99])
    return dict(samples=a.size, mean_ns=float(np.sum(a,dtype=np.int64)/a.size),
                p50_ns=float(quantiles[0]),p95_ns=float(quantiles[1]),p99_ns=float(quantiles[2]),
                min_ns=int(np.min(a)),max_ns=int(np.max(a)))
def serial(x):
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x))
def same(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'/{i}')
    elif isinstance(a,(float,np.floating)):
        assert np.isclose(a,b,rtol=0,atol=1e-7 if abs(a)>1 else 1e-14),(path,a,b)
    else: assert a==b,(path,a,b)

def check_plan(plan,logical,service,model,prefix=False):
    j=plan['jobs'];o=plan['offsets'];C=plan['input_columns'];T=plan['total_rows']
    E=1 if prefix else 65
    assert o.shape==(E,2,2) and C.shape==T.shape==(E,2)
    assert j.dtype==o.dtype==C.dtype==np.int32 and j.ndim==2 and j.shape[1]==5
    assert len(j)==len(logical) and np.all(j[:,0]>=0) and np.all(j[:,0]<service['chips'])
    assert np.all(j[:,1:]>0)
    expectedpages=(j[:,1].astype(np.int64)*j[:,2]+service['page_bytes']-1)//service['page_bytes']
    assert np.array_equal(j[:,3],expectedpages)
    assert np.array_equal(j[:,4],(expectedpages+service['planes']-1)//service['planes'])
    assert np.all(j[:,4]<=service['block_wordlines'])
    if service['input_buffer_bytes']:assert np.all(C*service['input_bytes']<=service['input_buffer_bytes'])
    if prefix:
        assert np.array_equal(C,[[2048,10944]]) and np.array_equal(T,[[21888,2048]])
    else:
        D,I,L,SI=SHAPES[model]
        assert np.all(C[:64]==[D,I]) and np.array_equal(C[64],[D,SI])
        assert np.all(T[:64]==[2*I,D]) and np.array_equal(T[64],[2*SI,D])
    ptr=0
    for e in range(E):
        for p in range(2):
            lo,hi=map(int,o[e,p]);assert lo==ptr and lo<hi<=len(j)
            rows=j[lo:hi,1].astype(np.int64);cursor=np.r_[0,np.cumsum(rows)]
            assert cursor[-1]==T[e,p] and np.all(j[lo:hi,2]==C[e,p])
            for z in range(lo,hi):
                chip,r,c,pages,waves=map(int,j[z]);a=logical[z]
                want=dict(expert=e,phase=p,chip=chip,row_start=int(cursor[z-lo]),rows=r,
                          columns=c,pages=pages,waves=waves,useful_bytes=r*c)
                assert all(a[k]==v for k,v in want.items()),(z,want,a)
            ptr=hi
    assert ptr==len(j) and np.all(C[:,0]==C[0,0])
    # Our old engine serializes fanout*ceil(single-transfer time), whereas the
    # new engine uses ceil(fanout*bytes/rate). Check equality, do not assume it.
    rate=service['channel_bytes_per_second'];fanout=service['input_channel_copies']
    b=j[:,2].astype(np.int64)*service['input_bytes']
    assert np.array_equal(fanout*((b*10**9+rate-1)//rate),(fanout*b*10**9+rate-1)//rate)
    return len(j)

def main():
    assert not DEST.exists(),'Never overwrite an independent executed audit.'
    DEST.mkdir(parents=True);snap=DEST/'snapshots';snap.mkdir()
    summary=read(SOURCE/'summary.json');protocol=read(SOURCE/'protocol.frozen.json')
    assert summary['status']=='complete' and summary['costs']==202752
    assert sha(SOURCE/'protocol.frozen.json')==summary['protocol_sha256']=='cb3b22789217c6fc2b1658ac92537b9efe4f275cc347e213b6898ae6af764c96'
    assert protocol['plan_seal_sha256']=='00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00'
    sealed_sources={}
    for p in (SOURCE/'source-snapshot').iterdir():sealed_sources[str(p)]=sha(p)
    assert sorted(sealed_sources.values())==sorted(protocol['sources_sha256'].values())
    for name in ('research_20261103_wave_drain.py','research_20261103_wave_drain_native.cpp'):
        shutil.copyfile(OLD/name,snap/name)
    shutil.copyfile(Path(__file__),snap/Path(__file__).name)
    frozen=dict(created_utc=stamp(),root_protocol_sha256=summary['protocol_sha256'],
                root_summary_sha256=sha(SOURCE/'summary.json'),root_source_snapshots=sealed_sources,
                independent_sources_sha256={str(p):sha(p) for p in snap.iterdir()},
                scope='All792layercases x256 ordered routes; all792 first traces; four dense prefixes; every reported pooled/prompt/dataset statistic. No fit, inference or alteration of original files.',
                engine='Previously sealed independent wave-drain C++ engine, mode0 postcompute, fanout1/4. Python heap scheduler and interval/dependency/owner/barrier validator from the same sealed independent study. This is a second implementation, not device evidence.',
                planned_comparisons=dict(costs=202752,layer_first_traces=792,dense_prefixes=4,scalar_heap_traces=28),
                frozen_result_inputs_sha256={str(p):sha(p) for p in [SOURCE/'summary.json',SOURCE/'protocol.frozen.json',SOURCE/'numeric_inputs.bound.json',SOURCE/'per_prompt.json',SOURCE/'dataset_groups.json']})
    write(DEST/'protocol.frozen.json',frozen)
    t0=time.monotonic()
    for p,h in protocol['inputs_sha256'].items(): assert sha(p)==h,p
    for p,h in read(SOURCE/'numeric_inputs.bound.json')['sha256'].items():assert sha(p)==h,p
    routes={m:np.load(SOURCE/(m+'.routes.npy')) for m in SHAPES}
    for model,a in routes.items():
        assert a.shape==(16,16,SHAPES[model][2],6) and np.all((0<=a)&(a<64))
        assert np.all(np.diff(np.sort(a,axis=-1),axis=-1)>0)
    spec=importlib.util.spec_from_file_location('sealed_wave_independent',snap/'research_20261103_wave_drain.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.DEST=DEST
    native=mod.Native(snap/'research_20261103_wave_drain_native.cpp')
    totals=defaultdict(lambda:np.zeros((16,16),dtype=np.int64));ledger=[];scalar=[];trace_nodes=costs=jobs=0
    observations=defaultdict(set)
    print('START all792cases,202752costs',flush=True)
    for index,r in enumerate(summary['records']):
        for k in ('arrays','plan'):assert sha(r[k])==protocol['inputs_sha256'][r[k]]
        assert sha(r['artifact'])==r['artifact_sha256']
        z=load(r['arrays']);plan={k:z[k] for k in KEYS};p=read(r['plan']);s=r['service']
        jobs+=check_plan(plan,p['logical'],s,r['model'])
        D,I,L,SI=SHAPES[r['model']];layer=r['moe_layer_index']
        assert r['raw_layer']==layer+(r['model']=='deepseek')
        observations[(r['model'],s['input_channel_copies'],s['command_ns'],r['family'])].add(layer)
        route=np.ascontiguousarray(routes[r['model']][:,:,layer].reshape(256,6))
        actual=native.run(plan,route,64,s,0,s['input_channel_copies'],True)
        saved=load(r['artifact']);assert np.array_equal(saved['first_route'],route[0])
        assert np.array_equal(actual['cost'].reshape(16,16),saved['cost']),r['artifact']
        assert np.array_equal(actual['lower_bound'].reshape(16,16),saved['lower_bound']),r['artifact']
        assert np.array_equal(actual['trace'],saved['first_trace']),r['artifact']
        # Scalar graph is separately constructed and audits every real trace.
        g=mod.ScalarGraph.build(plan,list(map(int,route[0]))+[64],s,0,s['input_channel_copies'])
        check=g.audit(actual['trace'])
        if layer==0:
            scalarcost,scalartrace=g.run();assert scalarcost==actual['cost'][0] and np.array_equal(scalartrace,actual['trace'])
            scalar.append(dict(model=r['model'],copies=s['input_channel_copies'],command=s['command_ns'],family=r['family'],**check))
        selected=list(map(int,route[0]))+[64]
        selectedjobs=sum(int(plan['offsets'][e,p,1]-plan['offsets'][e,p,0]) for e in selected for p in range(2))
        assert check['jobs']==selectedjobs and check['full_GU_barriers_checked']==7
        # Explicit one shared endpoint and six unique routed endpoints per phase.
        for e in range(7):
            assert sum(n['stage']=='host_activation' and n['expert']==e for n in g.nodes)==1
            assert sum(n['stage']=='full_down_upload' and n['expert']==e for n in g.nodes)==1
        assert sum(n['stage']=='common_GU_upload' for n in g.nodes)==1
        same(describe(actual['cost'])['mean_ns'],r['mean_ns']);same(describe(actual['cost'])['p99_ns'],r['p99_ns'])
        key=(r['model'],s['input_channel_copies'],s['command_ns'],r['family'])
        totals[key]+=actual['cost'].reshape(16,16);trace_nodes+=len(actual['trace']);costs+=len(route)
        ledger.append(dict(model=r['model'],copies=s['input_channel_copies'],command_ns=s['command_ns'],family=r['family'],layer=layer,
                           jobs=len(plan['jobs']),selected_first_route_jobs=selectedjobs,trace_nodes=len(actual['trace']),costs=256,
                           independent_cost_sha256=hashlib.sha256(actual['cost'].tobytes()).hexdigest(),
                           independent_trace_sha256=hashlib.sha256(actual['trace'].tobytes()).hexdigest()))
        if (index+1)%132==0:print('PROGRESS',index+1,'elapsed',round(time.monotonic()-t0,1),flush=True)
    for key,seen in observations.items():assert seen==set(range(SHAPES[key[0]][2]))
    assert len(observations)==24 and costs==202752 and trace_nodes==675372
    prefix={};prefix_checks=[]
    for r in summary['dense_prefix']:
        fanout=r['input_channel_copies'];command=r['command_ns'];label=f'input{fanout}.cmd{command}'
        a=SOURCE/'dense-prefix'/(label+'.npz');p=SOURCE/'dense-prefix'/(label+'.physical.json')
        assert sha(a)==r['arrays_sha256'] and sha(p)==r['physical_sha256']
        z=load(a);plan={k:z[k] for k in KEYS};phy=read(p);s=phy['service']
        check_plan(plan,phy['logical'],s,'deepseek',True)
        out=native.run(plan,np.array([[0]],dtype=np.int32),1,s,0,fanout,True)
        assert int(out['cost'][0])==r['cost_ns'] and np.array_equal(out['trace'],z['trace'])
        g=mod.ScalarGraph.build(plan,[0],s,0,fanout);cost,tr=g.run()
        assert cost==r['cost_ns'] and np.array_equal(tr,z['trace']);checks=g.audit(tr)
        useful=int(np.sum(plan['jobs'][:,1].astype(np.int64)*plan['jobs'][:,2]));assert useful==3*2048*10944==r['useful_bytes']
        assert int(np.sum(plan['jobs'][:,3].astype(np.int64)*s['page_bytes']))==r['programmed_page_bytes']
        assert int(np.sum(plan['jobs'][:,4].astype(np.int64)*s['planes']*s['page_bytes']))==r['reserved_wave_bytes']
        assert checks['full_GU_barriers_checked']==1
        prefix[fanout,command]=cost;prefix_checks.append(dict(copies=fanout,command_ns=command,cost_ns=cost,useful_bytes=useful,**checks))
    # Regenerate every statistic from independent vectors, never root cost arrays.
    recomputed_totals=[];paired=[];perprompt=[];datasets=[]
    root_total={(r['model'],r['input_channel_copies'],r['command_ns'],r['family']):r for r in summary['totals']}
    allvectors={}
    for key,a in sorted(totals.items()):
        model,copies,command,family=key;p=prefix[copies,command] if model=='deepseek' else 0
        actual=dict(model=model,input_channel_copies=copies,command_ns=command,family=family,
                    moe_shared=describe(a),fixed_dense_prefix_ns=p,complete_ffn_service=describe(a+p))
        same(actual,root_total[key]);recomputed_totals.append(actual)
        saved=load(SOURCE/'totals'/f'input{copies}.{model}.cmd{command}.{family}.npz')
        assert np.array_equal(saved['moe_shared_ns'],a) and np.array_equal(saved['complete_ffn_ns'],a+p)
        assert np.all(saved['complete_ffn_ns']-saved['moe_shared_ns']==p)
        allvectors[f'input{copies}.{model}.cmd{command}.{family}.moe_shared_ns']=a
        allvectors[f'input{copies}.{model}.cmd{command}.{family}.complete_ffn_ns']=a+p
    def reduction(a,b,p):return 1-float(np.mean(b)+p)/float(np.mean(a)+p)
    for model in SHAPES:
        for copies in (1,4):
            for command in (250,5000):
                p=prefix[copies,command] if model=='deepseek' else 0
                b=totals[model,copies,command,'whole_rows']
                for base in ('disjoint','cyclic'):
                    a=totals[model,copies,command,base]
                    header=dict(model=model,input_channel_copies=copies,command_ns=command,baseline=base)
                    paired.append(dict(**header,moe_mean_reduction_fraction=reduction(a,b,0),complete_ffn_mean_reduction_fraction=reduction(a,b,p),
                                       paired_mean_delta_ns=float(np.mean(a-b)),faster_tokens=int((a>b).sum()),tied_tokens=int((a==b).sum()),slower_tokens=int((a<b).sum()),
                                       faster_prompt_means=int((a.mean(1)>b.mean(1)).sum()),tied_prompt_means=int((a.mean(1)==b.mean(1)).sum()),slower_prompt_means=int((a.mean(1)<b.mean(1)).sum())))
                    for i,item in enumerate(protocol['prompts']):
                        perprompt.append(dict(**header,prompt_id=item['id'],dataset=item['dataset'],baseline_moe=describe(a[i]),whole_rows_moe=describe(b[i]),
                                              moe_mean_reduction_fraction=reduction(a[i],b[i],0),complete_ffn_mean_reduction_fraction=reduction(a[i],b[i],p)))
                    for dataset in sorted(set(p['dataset'] for p in protocol['prompts'])):
                        indices=[i for i,prompt in enumerate(protocol['prompts']) if prompt['dataset']==dataset];aa=a[indices];bb=b[indices]
                        datasets.append(dict(**header,dataset=dataset,prompts=len(indices),baseline_moe=describe(aa),whole_rows_moe=describe(bb),
                                             moe_mean_reduction_fraction=reduction(aa,bb,0),complete_ffn_mean_reduction_fraction=reduction(aa,bb,p)))
    same(paired,summary['paired']);same(perprompt,read(SOURCE/'per_prompt.json'));same(datasets,read(SOURCE/'dataset_groups.json'))
    np.savez_compressed(DEST/'independent_total_vectors.npz',**allvectors)
    # json conversion here is explicit: protocols/results contain numpy counts.
    result=dict(status='PASS',completed_utc=stamp(),elapsed_seconds=time.monotonic()-t0,costs_checked=costs,
                layer_cases=792,full_first_traces_checked=792,first_trace_nodes_checked=trace_nodes,
                full_dependency_resource_owner_GU_barrier_audits=792,independent_python_heap_trace_replays=len(scalar)+4,
                all_plan_jobs_checked=jobs,root_protocol_input_hashes_checked=len(protocol['inputs_sha256']),
                independent_native_binary_sha256=sha(native.binary),prefix_checks=prefix_checks,
                totals=recomputed_totals,paired=paired,per_prompt_stat_records=len(perprompt),dataset_stat_records=len(datasets),
                all_root_statistics_equal=True,all_root_saved_total_vectors_equal=True,
                shared_and_prefix='Exactly six distinct routed experts plus shared expert64 once per layer. GU common PCIe input once; private Down input once per selected/shared expert. Every MoE layer included once; Deep raw0 dense fixed prefix added exactly once only after layer summation, identical across families. G4 prefix zero.',
                rounding='Independent fanout*ceil(inputbytes/rate) verified equal to root ceil(fanout*inputbytes/rate) for every actual job. No general equivalence asserted for odd/artificial sizes.',
                limitations=summary['limits']+['Reproduces the same declared postcompute/owner and scheduling policy, not device truth or optimal scheduling. Root and independent scheduler share a historical native scheduler ancestor; independent Python heap construction plus interval/dependency/owner checks reduce but cannot eliminate shared specification assumptions.',
                                               'Sixteen prompts (eight per dataset), sixteen correlated decode positions each; descriptive performance statistics do not prove task accuracy, independent-token confidence intervals or future-workload generalization.',
                                               'The separate route-blind swap8 capacity audit remains relevant: G4 low-command whole_rows uses6.0GiB versus improved cyclic5.8125GiB in that restricted symmetry family. Reproducing heldout costs does not establish equal-capacity superiority.'])
    for name,obj in [('results.json',result),('layer_ledger.json',ledger),('scalar_trace_checks.json',scalar),('per_prompt.recomputed.json',perprompt),('dataset_groups.recomputed.json',datasets)]:
        (DEST/name).write_text(json.dumps(obj,indent=2,default=serial)+'\n')
    for p,h in frozen['frozen_result_inputs_sha256'].items():assert sha(p)==h
    write(DEST/'completion.json',dict(status='complete',completed_utc=stamp(),outputs_sha256={str(p):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='completion.json' and '__pycache__' not in str(p)}))
    print(json.dumps(dict(status='PASS',costs=costs,traces=792,nodes=trace_nodes,scalar_traces=len(scalar)+4,results_sha256=sha(DEST/'results.json'),elapsed_seconds=result['elapsed_seconds']),indent=2))

if __name__=='__main__':main()
