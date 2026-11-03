#!/usr/bin/env python3
"""Independent read-only inventory, node-ledger and aggregate checks.

No fit/replay generator or scheduler module is imported.  Only sealed JSON,
NPZ and trace ledgers are consumed; external source tasks are never opened.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2/qwen38-placement'
FIT = BASE/'phase-fit3'
TRAIN = BASE.parent/'qwen38-trace-audit/train_decode.npz'
FIT_SHA = 'b9efbfdc784fc2772aace65701f8671af19c31a5a75352c4927902c4c75fad1c'
TRAIN_SHA = '953e8db87d4bc4a5b5b9a77f5aebae1578b1e522df5bf3ac6a32a591f9e8b145'
RESOURCES = ['host','controller','pcie']+[f'channel{i}' for i in range(8)]+[f'chip{i}' for i in range(16)]+['sync']
SHAPES = {'gate_up':(1280,2560),'down':(2560,640)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ceildiv(a,b):
    return (int(a)+int(b)-1)//int(b)


def read(p):
    return json.loads(Path(p).read_text())


def coverage(jobs,shape):
    rows,cols=shape
    re=sorted({0,rows}|{j['row_start'] for j in jobs}|{j['row_start']+j['rows'] for j in jobs})
    ce=sorted({0,cols}|{j['column_start'] for j in jobs}|{j['column_start']+j['columns'] for j in jobs})
    counts=np.zeros((len(re)-1,len(ce)-1),dtype=np.int16)
    for j in jobs:
        r,n,c,m=(j[k] for k in ('row_start','rows','column_start','columns'))
        assert 0<=r<r+n<=rows and 0<=c<c+m<=cols
        counts[re.index(r):re.index(r+n),ce.index(c):ce.index(c+m)]+=1
        assert 0<=j['chip']<16
        assert ceildiv(n*m,65536)<=192
    assert np.all(counts==1)
    assert len({j['chip'] for j in jobs})==len(jobs)
    assert sum(j['rows']*j['columns'] for j in jobs)==rows*cols


def ledger(jobs):
    # [useful I8 bytes, actual 16KiB pages, reserved 64KiB waves,
    #  command count, input bytes, output bytes, native chip occupancy ns]
    result=np.zeros((16,7),dtype=np.int64)
    for j in jobs:
        r,c=j['rows'],j['columns']; size=r*c; waves=ceildiv(size,65536)
        result[j['chip']]+=[size,ceildiv(size,16384),waves,1,c,4*r,
                            250+ceildiv(c,2)+28000*waves+2*r]
    return result


def templates():
    data=read(FIT/'templates.json')
    extra=read(BASE/'independent_rectangles.json')['controls']
    for name,key in [('rect_min_output','gate_up_two_stripes_min_output'),
                     ('rect_min_io4','gate_up_three_stripes_min_io4')]:
        variants=[]
        for color in range(16):
            variants.append([dict(chip=(i+color)%16,row_start=r,rows=n,column_start=c,columns=m)
                             for i,(r,n,c,m) in enumerate(extra[key]['rectangles'])])
        data['gate_up'].append(dict(label=name,family='rectangular',conditional_partial_sum=True,variants=variants))
    return {p:{c['label']:c for c in vv} for p,vv in data.items()}


def check_fit(inv,routes):
    doc=read(FIT/'fit.json'); assert sha(FIT/'fit.json')==FIT_SHA and sha(TRAIN)==TRAIN_SHA
    for p,h in doc['inputs_sha256'].items(): assert sha(p)==h
    for p,h in doc['source_sha256'].items(): assert sha(p)==h
    assert doc['arguments']['stride']==4
    assert sha(FIT/'templates.json')==doc['templates_sha256']
    reports=[];checked=0
    for p,candidates in inv.items():
        for name,candidate in candidates.items():
            for jobs in candidate['variants']:
                coverage(jobs,SHAPES[p]);checked+=1
    for result in doc['results']:
        layer,p=result['layer'],result['phase']
        selected=routes[:,::4,layer].reshape(-1,10)
        unique,counts=np.unique(np.sort(selected,axis=1),axis=0,return_counts=True)
        assert len(selected)==3200 and result['fit_tokens']==3200 and result['unique_routes']==len(unique)
        assert result['fit_stride']==4
        for record in result['records']:
            labels=np.asarray(record['labels']); hist=np.bincount(labels,minlength=16)
            assert labels.shape==(512,) and np.array_equal(hist,np.full(16,32))
            candidate=inv[p][record['label']]
            ll=np.stack([ledger(jobs) for jobs in candidate['variants']])
            base=ll[np.arange(512)%16].sum(axis=0); learned=ll[labels].sum(axis=0)
            assert np.array_equal(base,learned)
            chip_work=ll[labels, :,6][selected].sum(axis=1)
            before=ll[np.arange(512)%16,:,6][selected].sum(axis=1)
            assert abs(float(chip_work.max(axis=1).mean())-record['mean_bound_ns'])<1e-7
            assert abs(float(before.max(axis=1).mean())-record['initial_mean_bound_ns'])<1e-7
            reports.append(dict(layer=layer,phase=p,label=record['label'],histogram=hist.tolist(),
                shape=list(SHAPES[p]),native_jobs_per_expert=len(candidate['variants'][0]),
                full_onecopy_payload_bytes=int(learned[:,0].sum()),
                per_chip_useful_bytes=learned[:,0].tolist(),per_chip_actual_pages=learned[:,1].tolist(),
                per_chip_reserved_wave_bytes=(learned[:,2]*65536).tolist(),
                mean_bound_ns=float(chip_work.max(axis=1).mean()),
                inventory_unchanged_from_initial=True))
    return dict(fit_sha256=FIT_SHA,fit_tokens=3200,requests=16,steps_per_request=200,
        first_step=0,last_step=796,all_16_requests_used=True,unique_routes_by_layer={str(l):len(np.unique(np.sort(routes[:,::4,l].reshape(-1,10),axis=1),axis=0)) for l in range(3)},
        variants_exact_onecopy_checked=checked,candidate_records_checked=len(reports),records=reports,
        notes=['The fit protocol text ends ...798, but stride4 actually ends796; neither799 nor798 is sampled.',
               'Histograms preserve all per-chip ledger fields within each geometry. Different geometries need not have equal reserved pages/waves.',
               'rect_min_output/io4 GU uses labels fitted to the original exact_work_rectangular grid, not independently refitted stripe labels.',
               'Search is balanced-label heuristic fitting with identical attempted swap budgets, not a global optimum certificate.',
               'No physical block-address image is emitted here; onecopy coefficient cover and sufficient per-job <=192WL packing are checked.'])


def expected_nodes(items,reduction,activation):
    nodes=[];job_number=0
    def add(resource,duration,deps=(),phase=0,expert=-1,kind='',chip=-1,job=-1,acquire=False,release=False):
        i=len(nodes);nodes.append(dict(index=i,resource=resource,duration=duration,deps=list(deps),phase=phase,
            expert=expert,kind=kind,chip=chip,job=job,acquire=acquire,release=release));return i
    submit=add('host',2000,kind='initial_host_submission')
    upload=add('pcie',ceildiv(2560,8),[submit],kind='common_input_to_controller')
    def matrix(e,phase,j,ds,up):
        nonlocal job_number
        num=job_number;job_number+=1;chip=j['chip'];r=j['rows'];c=j['columns']
        cmd=add('controller',250,[ds],phase,e,'matrix_command',chip,num,True)
        inp=add(f'channel{chip%8}',ceildiv(c,2),[cmd,up],phase,e,'input_channel_reload',chip,num)
        comp=add(f'chip{chip}',ceildiv(r*c,65536)*28000,[inp],phase,e,'nonpreemptive_native_compute',chip,num)
        out=add(f'channel{chip%8}',2*r,[comp],phase,e,'output_channel',chip,num,release=True)
        return add('pcie',ceildiv(r,2),[out],phase,e,'output_pcie')
    ready=[]
    for e,item in enumerate(items):
        gu=[matrix(e,0,j,submit,upload) for j in item['gate_up']]
        node=add('sync',0,gu,expert=e,kind='whole_expert_gu_host_ready')
        if item['gu_partial']:
            node=add('host',reduction,[node],expert=e,kind='gu_partial_reconstruction')
        ready.append(node)
    terminal=[]
    for e,item in enumerate(items):
        act=add('host',activation,[ready[e]],1,e,'host_activation')
        ds=add('host',2000,[act],1,e,'expert_down_host_submission')
        up=add('pcie',ceildiv(640,8),[ds],1,e,'private_input_to_controller')
        down=[matrix(e,1,j,ds,up) for j in item['down']]
        if item['down_partial']:
            terminal.append(add('host',reduction,down,1,e,'down_partial_reconstruction'))
        else:terminal.extend(down)
    add('host',0,terminal,1,kind='host_combine')
    return nodes


def validate_trace(path,plan,inv,routes,record,cost,bound):
    trace=read(path); layer=record['layer'];selected=routes[0,0,layer]
    assert np.array_equal(trace['selected'],selected)
    items=[]
    for expert in selected:
        item={}
        for p in SHAPES:
            spec=plan['plans'][p];candidate=inv[p][spec['label']]
            item[p]=candidate['variants'][spec['labels'][int(expert)]]
            item['gu_partial' if p=='gate_up' else 'down_partial']=candidate['conditional_partial_sum']
        items.append(item)
    shared={p:inv[p][spec['label']]['variants'][spec['color']] for p,spec in plan['fixed_shared'].items()}
    shared.update(gu_partial=False,down_partial=False);items.append(shared)
    assert items==trace['items'] and len(items)==11
    assert trace['fixed_shared']==plan['fixed_shared']
    for item in items:
        for p in SHAPES:coverage(item[p],SHAPES[p])
    nodes=trace['trace']; expected=expected_nodes(items,record['reduction_ns'],record['activation_ns'])
    assert len(nodes)==len(expected)==trace['nodes']
    for n,x in zip(nodes,expected):
        for k,v in x.items():
            if k=='duration':assert n['end_ns']-n['start_ns']==v
            else:assert n[k]==v,(str(path),n['index'],k,n[k],v)
        assert n['start_ns']>=0
        assert all(nodes[d]['end_ns']<=n['start_ns'] for d in n['deps'])
    work=[]
    for resource in RESOURCES:
        active=sorted([(n['start_ns'],n['end_ns'],n['index']) for n in nodes if n['resource']==resource and n['end_ns']>n['start_ns']])
        assert all(a[1]<=b[0] for a,b in zip(active,active[1:]))
        work.append(sum(b-a for a,b,_ in active))
    assert work==trace['resource_work_ns']
    jobs={}
    for n in nodes:
        if n['job']>=0:jobs.setdefault(n['job'],{})[n['kind']]=n
    ownership=[[] for _ in range(16)]
    for job in jobs.values():
        assert set(job)=={'matrix_command','input_channel_reload','nonpreemptive_native_compute','output_channel'}
        a=job['matrix_command'];b=job['output_channel'];chip=a['chip']
        assert a['acquire'] and b['release']
        assert all(n['chip']==chip and n['expert']==a['expert'] and n['phase']==a['phase'] for n in job.values())
        ownership[chip].append((a['start_ns'],b['end_ns']))
    for active in ownership:
        active.sort();assert all(a[1]<=b[0] for a,b in zip(active,active[1:]))
    makespan=max(n['end_ns'] for n in nodes)
    assert makespan==trace['makespan_ns']==int(cost[0])
    ll=sum((ledger(item[p]) for item in items for p in SHAPES),np.zeros((16,7),dtype=np.int64))
    calculated_bound=2000+int(ll[:,6].max());assert calculated_bound==int(bound[0]) and makespan>=calculated_bound
    return dict(path=str(path),sha256=sha(path),nodes_checked=len(nodes),native_jobs_checked=len(jobs),
        shared_expert_index=10,shared_always_active=True,all_node_fields_costs_and_dependencies_exact=True,
        no_resource_or_context_overlap=True,makespan_ns=makespan,native_chip_bound_ns=calculated_bound,
        resource_work_ns=dict(zip(RESOURCES,work)))


def check_run(directory,inv,routes):
    screen=read(directory/'screen.json')
    for p,h in screen['inputs_sha256'].items():assert sha(p)==h
    for p,h in screen['source_sha256'].items():assert sha(p)==h
    totals={};traces=[];records=[];tokens={};boundtotals={};saved_inventory=[]
    for r in screen['records']:
        prefix=directory/r['label']
        for ext,h in r['artifact_sha256'].items():assert sha(str(prefix)+'.'+ext)==h
        with np.load(str(prefix)+'.npz') as a:
            cost=a['cost'].copy();bound=a['native_chip_bound'].copy();steps=a['token_steps'].copy()
        assert cost.shape==bound.shape==(16*len(steps),) and len(cost)==r['samples']
        if screen['arguments']['samples_per_request']==0: assert np.array_equal(steps,np.arange(799))
        else:assert np.array_equal(steps,np.unique(np.linspace(0,798,screen['arguments']['samples_per_request'],dtype=int)))
        assert abs(float(cost.mean())-r['mean_ns'])<1e-7
        assert abs(float(np.percentile(cost,99))-r['p99_ns'])<1e-7
        assert abs(float(bound.mean())-r['mean_bound_ns'])<1e-7
        assert np.all(cost>=bound)
        plan=read(str(prefix)+'.plan.json')
        assert plan['layer']==r['layer'] and plan['case']==r['case']
        assert {p:s['label'] for p,s in plan['plans'].items()}==r['routed_families']
        assert plan['fixed_shared']==r['fixed_shared']
        # Recompute every query's necessary occupancy bound from the saved
        # static plan, independently of the root table/generator.
        load=np.zeros((16*len(steps),16),dtype=np.int64)
        inventory=np.zeros((16,7),dtype=np.int64)
        selected=routes[:,steps,r['layer']].reshape(-1,10)
        for p,spec in plan['plans'].items():
            ll=np.stack([ledger(j) for j in inv[p][spec['label']]['variants']]);labels=np.asarray(spec['labels'])
            assert np.array_equal(np.bincount(labels,minlength=16),np.full(16,32))
            load+=ll[labels,:,6][selected].sum(axis=1)
            inventory+=ll[labels].sum(axis=0)
            s=plan['fixed_shared'][p]; sl=ledger(inv[p][s['label']]['variants'][s['color']])
            load+=sl[None,:,6];inventory+=sl
        assert np.array_equal(2000+load.max(axis=1),bound)
        assert inventory[:,0].sum()==513*(1280*2560+2560*640)
        saved_inventory.append(dict(layer=r['layer'],case=r['case'],
            per_chip_useful_bytes=inventory[:,0].tolist(),per_chip_actual_pages=inventory[:,1].tolist(),
            per_chip_reserved_wave_bytes=(inventory[:,2]*65536).tolist(),
            total_useful_bytes=int(inventory[:,0].sum()),
            capacity='Listed physical useful/page/wave inventories; no invented per-chip average or20GiB cap.'))
        traces.append(validate_trace(Path(str(prefix)+'.trace.json'),plan,inv,routes,r,cost,bound))
        totals.setdefault(r['case'],[]).append(cost);boundtotals.setdefault(r['case'],[]).append(bound)
        if r['case'] in tokens:assert np.array_equal(tokens[r['case']],steps)
        tokens[r['case']]=steps
        records.append(dict(layer=r['layer'],case=r['case'],samples=len(cost),mean_ns=float(cost.mean()),p99_ns=float(np.percentile(cost,99))))
    result=[]
    for expected in screen['totals']:
        case=expected['case'];cost=np.stack(totals[case]).sum(axis=0);bound=np.stack(boundtotals[case]).sum(axis=0)
        assert len(totals[case])==screen['arguments']['layers']
        mean=float(cost.mean());p99=float(np.percentile(cost,99))
        assert abs(mean-expected['mean_ns'])<1e-7 and abs(p99-expected['p99_ns'])<1e-7
        result.append(dict(case=case,samples=len(cost),paired_layer_sum_mean_ns=mean,paired_layer_sum_p99_ns=p99,
                           mean_native_chip_bound_ns=float(bound.mean())))
    return dict(directory=str(directory),screen_sha256=sha(directory/'screen.json'),
        arguments=screen['arguments'],aggregate_checks_exact=True,records=records,totals=result,
        representative_traces=traces,physical_inventory=saved_inventory,
        checked_scope='Every saved scalar/sample bound and representative trace node, not independent replay of all event schedules.',
        warnings=['Host activation, partial reconstruction, final weighted combine and shared-gate scalar numerical work are optimistic/default parameters, not measured latency.',
                  'RawI8/A8/4B-output service contract and partial-output scales are conditional; actual AiF output ABI is not established.',
                  'This is first3 FFN layers including one shared expert; not attention/SSM/MTP/embedding/full-model execution.',
                  'All samples remain the same exploratory training task; no external or heldout payload was read.'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',default=['native-smoke'])
    ap.add_argument('--output',type=Path,default=BASE/'independent-audit/audit-smoke.json');args=ap.parse_args()
    assert sha(TRAIN)==TRAIN_SHA
    with np.load(TRAIN) as a:routes=a['routes'].copy();req=a['req_ids'].copy()
    assert routes.shape==(16,799,48,10) and np.array_equal(req,np.arange(16))
    inv=templates();fit=check_fit(inv,routes)
    results=[check_run(BASE/name,inv,routes) for name in args.runs]
    out=dict(status='complete independent read-only audit under stated conditional service contract',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),fit=fit,runs=results,
        approach='No generator/scheduler imported; geometric coefficient sweep, independent byte/page/wave ledgers, expected dependency nodes, interval resource/context validation, NPZ paired aggregates.',
        external_task_payloads_opened=False,physical_mapping_fit_performed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),runs=[dict(directory=r['directory'],totals=r['totals']) for r in results]),indent=2))


if __name__=='__main__':main()
