#!/usr/bin/env python3
"""Independent scalar/trace audit of the expanded shared geometry controls."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
DEST=BASE/'shared-pipeline-more'
RES=['host','controller','pcie']+[f'channel{i}' for i in range(8)]+[f'chip{i}' for i in range(16)]+['sync']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ceildiv(x,y):return (x+y-1)//y


def service(r,c):
    return ceildiv(r*c,65536)*28000+np.where(r>0,250+ceildiv(c,2)+2*r,0)


def audit_trace(t,first_gu,first_down,sd,sg,cost):
    nodes=t['trace'];assert len(nodes)==t['nodes']
    assert t['makespan_ns']==int(cost)
    assert max(n['end_ns'] for n in nodes)==int(cost)
    intervals=defaultdict(list);work=Counter()
    for i,n in enumerate(nodes):
        assert n['index']==i and n['start_ns']>=0 and n['end_ns']>=n['start_ns']
        assert all(0<=j<i and nodes[j]['end_ns']<=n['start_ns'] for j in n['deps'])
        dur=n['end_ns']-n['start_ns'];work[n['resource']]+=dur
        if dur:intervals[n['resource']].append((n['start_ns'],n['end_ns']))
    for ts in intervals.values():
        ts.sort();assert all(b<=c for (_,b),(c,_) in zip(ts,ts[1:]))
    assert t['resource_work_ns']==[work[r] for r in RES]
    assert work['host']==16000
    shapes=[]
    for phase in (0,1):
        for e,item in enumerate(t['items']):
            gr=first_gu[e] if e<6 else sg
            dr=first_down[e] if e<6 else sd
            assert item['gu_rows']==gr.tolist() and item['down_rows']==dr.tolist()
            assert item['down_columns']==(512 if e<6 else 1024)
            rr=gr if phase==0 else dr;c=1536 if phase==0 else item['down_columns']
            shapes.extend((phase,e,chip,int(r),c) for chip,r in enumerate(rr) if r)
    assert work['pcie']==704+sum(ceildiv(4*r,8) for _,_,_,r,_ in shapes)
    stages=['matrix_command','input_channel_reload','nonpreemptive_native_compute','output_channel','output_pcie']
    jobs=defaultdict(list);pcie={};ready={};downhost={};guout=defaultdict(list);locks=defaultdict(list)
    for n in nodes:
        if n['kind'] in stages[:4]:jobs[n['job']].append(n)
        if n['kind']=='output_pcie':pcie[n['deps'][0]]=n
        if n['kind']=='whole_expert_gu_host_ready':ready[n['expert']]=n
        if n['kind']=='expert_down_host_submission':downhost[n['expert']]=n
    assert len(jobs)==len(shapes)
    for j,(phase,e,chip,r,c) in enumerate(shapes):
        ns=jobs[j];assert len(ns)==4;ns=ns+[pcie[ns[-1]['index']]]
        assert [n['kind'] for n in ns]==stages
        assert all(n['expert']==e and n['phase']==phase for n in ns)
        assert all(n['chip']==chip for n in ns[:4]) and ns[4]['chip']==-1
        assert [n['resource'] for n in ns]==['controller',f'channel{chip%8}',f'chip{chip}',f'channel{chip%8}','pcie']
        assert [n['end_ns']-n['start_ns'] for n in ns]==[250,ceildiv(c,2),28000*ceildiv(r*c,65536),2*r,ceildiv(4*r,8)]
        assert ns[0]['acquire'] and ns[3]['release']
        assert not any(n['acquire'] for n in ns[1:]) and not any(n['release'] for n in ns[:3]+ns[4:])
        assert all(a['end_ns']<=b['start_ns'] for a,b in zip(ns,ns[1:]))
        locks[chip].append((ns[0]['start_ns'],ns[3]['end_ns']))
        if phase==0:guout[e].append(ns[-1]['index'])
        else:
            ds=downhost[e];act=nodes[ds['deps'][0]]
            assert act['kind']=='host_activation' and act['deps']==[ready[e]['index']]
            assert ns[0]['deps']==[ds['index']]
    for ts in locks.values():
        ts.sort();assert all(b<=c for (_,b),(c,_) in zip(ts,ts[1:]))
    assert len(ready)==len(downhost)==7
    for e in range(7):assert set(ready[e]['deps'])==set(guout[e])
    coverage=np.zeros(2048,dtype=np.uint8);physical=np.zeros(16,dtype=np.int64)
    for chip,start,r in t['shared_gu_ranges']:
        assert chip%8==start//256 and r in (128,256)
        if r==128:assert start%256==(0 if chip<8 else 128)
        else:assert start%256==0 and t['replica']
        coverage[start:start+r]+=1;physical[chip]+=r
    assert np.all(coverage==1) and np.array_equal(physical,sg)
    return len(nodes),len(jobs)


def main():
    path=DEST/'screen.json';d=json.loads(path.read_text())
    source=BASE/'geometry-study/granite4tiny.frozen.json'
    jp=BASE/'pipeline-joint/joint_work_screen.json';dp=BASE/'pipeline-joint-depth/screen.json'
    src=ROOT/'scripts/research_20261102_shared_more_profiles.py'
    frozen=json.loads(source.read_text());joint=json.loads(jp.read_text());depth=json.loads(dp.read_text())
    route_path=Path(frozen['training_routes'])
    paths=[path,source,jp,dp,route_path,src]
    paths += [ROOT/'scripts'/n for n in d['imported_sources_sha256']]
    paths += sorted(DEST.glob('L*.npz'))+sorted(DEST.glob('L*.trace.json'))
    hashes={str(p):sha(p) for p in paths}
    assert sha(src)==d['source_sha256']
    assert all(sha(ROOT/'scripts'/n)==h for n,h in d['imported_sources_sha256'].items())
    assert all(sha(p)==h for p,h in d['inputs_sha256'].items())
    assert sha(route_path)==d['training_routes_sha256']
    with np.load(route_path) as f:
        routes=f['routes'][:32,:64].copy();layers=f['layers'].tolist()
        assert all(x.startswith('train_') for x in f['prompt_ids'].tolist())
    phases={(p['layer'],p['name']):p for p in frozen['phases']}
    results=[];costs={};nodes=0
    for rec in d['records']:
        layer,name=rec['layer'],rec['mapping']
        gu=np.asarray(phases[layer,'gate_up']['rows'],dtype=np.int64)
        down=np.asarray(phases[layer,'down']['rows'],dtype=np.int64)
        if name=='joint':
            labels=np.asarray(next(x for x in joint['results'] if x['layer']==layer)['joint_best']['labels'])
            down=np.where(np.arange(16)[None,:]//4==labels[:,None],0,128)
        elif name=='opposite':down=np.where(gu>0,0,192)
        elif name=='wave':gu=np.asarray(next(c for c in phases[layer,'gate_up']['candidates'] if c['label']=='wave_rotations')['rows'])
        elif name.startswith('depth_'):
            c=next(x for x in depth['results'] if x['layer']==layer)[name[6:]]
            gu=np.asarray(c['gu_rows']);down=np.asarray(c['down_rows'])
        else:assert name=='source'
        assert np.all(gu.sum(axis=1)==1024) and np.all(down.sum(axis=1)==1536)
        selected=routes[:,:,layers.index(layer)].reshape(2048,6)
        sd=np.full(16,64,dtype=np.int64);sd[rec['side']:rec['side']+8]=128
        occupied=(service(gu,1536)+service(down,512))[selected].sum(axis=1)+service(sd,1024)
        pn=DEST/(rec['label']+'.npz');pt=DEST/(rec['label']+'.trace.json')
        assert sha(pn)==rec['npz_sha256'] and sha(pt)==rec['trace_sha256']
        with np.load(pn) as f:cost=f['cost'].copy();lb=f['native_chip_bound'].copy();choices=f['choices'].copy()
        assert cost.shape==lb.shape==(2048,) and choices.shape==(2048,8)
        sg=np.zeros((2048,16),dtype=np.int64)
        for q in range(8):
            cs=[]
            for a in ((0,1,2) if rec['replica'] else (1,)):
                xa=128*a;xb=256-xa
                wa=int(service(np.asarray(xa),1536));wb=int(service(np.asarray(xb),1536))
                peaks=np.maximum(occupied[:,q]+wa,occupied[:,q+8]+wb)
                cs.append((peaks,wa+wb,abs(a-1),a))
            # Full tuple tie order is the declared policy, checked for all routes.
            winner=np.asarray([min((int(v[0][i]),*v[1:]) for v in cs)[-1] for i in range(2048)])
            assert np.array_equal(winner,choices[:,q])
            sg[:,q]=winner*128;sg[:,q+8]=(2-winner)*128
        expected_lb=2000+(occupied+service(sg,1536)).max(axis=1)
        assert np.array_equal(expected_lb,lb) and np.all(cost>=lb)
        assert cost.mean()==rec['mean_ns'] and np.percentile(cost,99)==rec['p99_ns'] and lb.mean()==rec['mean_bound_ns']
        t=json.loads(pt.read_text());assert t['selected']==selected[0].tolist()
        assert t['shared_gu_choices']==choices[0].tolist()
        nn,nj=audit_trace(t,gu[selected[0]],down[selected[0]],sd,sg[0],cost[0]);nodes+=nn
        results.append(dict(label=rec['label'],saved_trace_nodes=nn,native_jobs=nj,scalar_routes=2048))
        costs[rec['label']]=(cost,lb)
    for row in d['totals']:
        matched=[r for r in d['records'] if all(r[k]==row[k] for k in ('mapping','replica','side'))]
        assert len(matched)==3
        c=np.stack([costs[r['label']][0] for r in matched]).sum(axis=0)
        b=np.stack([costs[r['label']][1] for r in matched]).sum(axis=0)
        assert c.mean()==row['mean_ns'] and np.percentile(c,99)==row['p99_ns'] and b.mean()==row['mean_native_bound_ns']
    old=[r for r in d['totals'] if not r['replica']];new=[r for r in d['totals'] if r['replica']]
    bm=min(old,key=lambda r:r['mean_ns']);br=min(new,key=lambda r:r['mean_ns'])
    bt=min(old,key=lambda r:r['p99_ns']);rt=min(new,key=lambda r:r['p99_ns'])
    assert hashes=={str(p):sha(p) for p in paths}
    out=dict(status='PASS under stated scalar/native-output contract',created_utc=datetime.now(timezone.utc).isoformat(),
        auditor_source_sha256=sha(__file__),imported_author_cost_or_graph=False,inputs_unchanged=True,input_sha256=hashes,
        cases=len(results),independent_scalar_routes=len(results)*2048,saved_trace_nodes=nodes,cases_detail=results,
        fair_comparison=dict(best_onecopy_mean=bm,best_replica_mean=br,mean_gain_pct=100*(1-br['mean_ns']/bm['mean_ns']),
            best_onecopy_p99=bt,best_replica_p99=rt,p99_gain_pct=100*(1-rt['p99_ns']/bt['p99_ns'])),
        limitations=['Only one saved trace per case received full DAG/resource/native-lock audit; all route scalar choices/bounds/statistics checked.',
            'No physical allocator/native address execution or native numeric output dtype/scales proof.',
            'Activation, row reassembly and combine cost0 remain optimistic. Host submissions and all PCIe/ONFI input/output costs were verified.',
            'All data are fitting training routes; no new inference or heldout validation. Shared/routed/host-pin equal-capacity controls still required.',
            'Fixed priority feasible schedules, not global scheduling optima or measured inference; no novelty conclusion.'])
    target=DEST/'diagnostic_audit.json';target.write_text(json.dumps(out,indent=2)+'\n')
    (DEST/'diagnostic_audit_completion.json').write_text(json.dumps(dict(status='complete',
        artifacts_sha256={str(target):sha(target),str(Path(__file__)):sha(__file__)},inputs_unchanged=True),indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],cases=len(results),scalar_routes=len(results)*2048,
        saved_trace_nodes=nodes,fair_comparison=out['fair_comparison']),indent=2))


if __name__=='__main__':main()
