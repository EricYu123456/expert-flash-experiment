#!/usr/bin/env python3
"""Independent geometry, scalar-cost and saved-trace audit; never imports Graph."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
DEST = BASE/'shared-pipeline'
RESOURCES = ['host','controller','pcie']+[f'channel{i}' for i in range(8)]+[f'chip{i}' for i in range(16)]+['sync']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ceildiv(x, y):
    return (x+y-1)//y


def main():
    screen_path = DEST/'screen.json'
    d = json.loads(screen_path.read_text())
    source = BASE/'geometry-study/granite4tiny.frozen.json'
    joint_path = BASE/'pipeline-joint/joint_work_screen.json'
    frozen = json.loads(source.read_text())
    joint = json.loads(joint_path.read_text())
    route_path = Path(frozen['training_routes'])
    shared_src = ROOT/'scripts/research_20261102_shared_pipeline.py'
    graph_src = ROOT/'scripts/research_20261102_granite4_pipeline.py'
    metadata = Path('/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-metadata.json')
    inputs = [screen_path, source, joint_path, route_path, shared_src, graph_src, metadata]
    inputs += sorted(DEST.glob('L*.npz'))+sorted(DEST.glob('L*.trace.json'))
    before = {str(p):sha(p) for p in inputs}
    for p, key in [(shared_src,'source_sha256'),(graph_src,'strict_scheduler_source_sha256'),
                   (source,'frozen_plan_sha256'),(joint_path,'root_joint_sha256'),
                   (route_path,'training_routes_sha256')]:
        assert sha(p) == d[key]
    with np.load(route_path) as f:
        routes = f['routes'][:32,:64].copy()
        layers = f['layers'].tolist()
        prompt_ids = f['prompt_ids'][:32].tolist()
    assert routes.shape == (32,64,40,6)
    assert all(s.startswith('train_') for s in prompt_ids)
    phases = {(p['layer'],p['name']):p for p in frozen['phases']}
    tensors = {t['name']:t for t in json.loads(metadata.read_text())['tensors']}
    for layer in layers:
        assert tensors[f'blk.{layer}.ffn_gate_shexp.weight']['shape_ggml'] == [1536,1024]
        assert tensors[f'blk.{layer}.ffn_up_shexp.weight']['shape_ggml'] == [1536,1024]
        assert tensors[f'blk.{layer}.ffn_down_shexp.weight']['shape_ggml'] == [1024,1536]
    audited = []
    arrays = {}
    trace_nodes = 0
    scalar_rows = 0
    for rec in d['records']:
        label = rec['label']
        pnpz, pt = DEST/(label+'.npz'), DEST/(label+'.trace.json')
        assert sha(pnpz) == rec['numpy_sha256'] and sha(pt) == rec['trace_sha256']
        with np.load(pnpz) as f:
            ix, cost, chip_bound, stored = [f[k].copy() for k in ['indices','cost','native_chip_bound','total_chip_waves']]
        assert np.array_equal(ix, np.arange(2048))
        arrays[label] = cost
        li = layers.index(rec['layer'])
        selected = routes[:,:,li].reshape(2048,6)
        gu = np.asarray(phases[rec['layer'],'gate_up']['rows'],dtype=np.int64)
        down = np.asarray(phases[rec['layer'],'down']['rows'],dtype=np.int64)
        if rec['mapping'] == 'joint':
            labels = np.asarray(next(x for x in joint['results'] if x['layer']==rec['layer'])['joint_best']['labels'])
            down = np.where(np.arange(16)[None,:]//4 == labels[:,None],0,128)
        elif rec['mapping'] == 'opposite':
            down = np.where((gu[:,0]>0)[:,None] == (np.arange(16)[None,:]>=8),192,0)
        home = gu[:,0]>0
        assert np.array_equal(gu,np.where(home[:,None] == (np.arange(16)[None,:]<8),128,0))
        na = home[selected].sum(axis=1)
        shared_gu = np.full((2048,16),128,dtype=np.int64)
        if rec['shared_gu_replica']:
            shared_gu[na>3,:8] = 0
            shared_gu[na>3,8:] = 256
            shared_gu[na<3,:8] = 256
            shared_gu[na<3,8:] = 0
        sd = np.full(16,64,dtype=np.int64)
        sd[rec['shared_down_high_half']:rec['shared_down_high_half']+8] = 128
        assert np.all(shared_gu.sum(axis=1)==2048) and sd.sum()==1536
        gw, dw = gu[selected], down[selected]
        chip = np.zeros((2048,16),dtype=np.int64)
        total = np.zeros(2048,dtype=np.int64)
        for rr,cols in [(gw,1536),(dw,512),(shared_gu[:,None,:],1536),(np.broadcast_to(sd,(2048,1,16)),1024)]:
            waves=ceildiv(rr*cols,65536)
            total+=waves.sum(axis=(1,2))
            chip+=(waves*28000+np.where(rr>0,250+ceildiv(cols,2)+rr*2,0)).sum(axis=1)
        assert np.array_equal(chip_bound,2000+chip.max(axis=1))
        assert np.array_equal(stored,total)
        assert np.all(cost>=chip_bound)
        assert cost.mean()==rec['mean_ns'] and np.percentile(cost,99)==rec['p99_ns']
        scalar_rows += len(cost)

        t=json.loads(pt.read_text())
        assert t['selected']==selected[t['route_index']].tolist()
        assert t['route_index']==0 and len(t['items'])==7
        for e,item in enumerate(t['items']):
            expect_gu = gu[t['selected'][e]] if e<6 else shared_gu[0]
            expect_down = down[t['selected'][e]] if e<6 else sd
            assert item['gu_rows']==expect_gu.tolist()
            assert item['down_rows']==expect_down.tolist()
            assert item['down_columns']==(512 if e<6 else 1024)
        # The logical row map is checked independently of per-chip row counts.
        coverage=np.zeros(2048,dtype=np.uint8)
        for chip_id,start,count in t['shared_gu_ranges']:
            assert chip_id%8 == start//256
            assert count in [128,256]
            if count==256:
                assert rec['shared_gu_replica'] and start%256==0
            elif chip_id<8:
                assert start%256==0
            else:
                assert start%256==128
            coverage[start:start+count]+=1
        assert np.all(coverage==1)
        nodes=t['trace']
        assert len(nodes)==t['nodes']
        trace_nodes+=len(nodes)
        resources=defaultdict(list)
        work=Counter()
        for i,n in enumerate(nodes):
            assert n['index']==i
            assert n['start_ns']>=0 and n['end_ns']>=n['start_ns']
            assert all(nodes[j]['end_ns']<=n['start_ns'] for j in n['deps'])
            assert all(j<i for j in n['deps'])
            duration=n['end_ns']-n['start_ns']
            work[n['resource']]+=duration
            if duration:
                resources[n['resource']].append((n['start_ns'],n['end_ns']))
        for intervals in resources.values():
            intervals.sort()
            assert all(b<=c for (_,b),(c,_) in zip(intervals,intervals[1:]))
        assert [work[x] for x in RESOURCES] == t['resource_work_ns']
        assert max(n['end_ns'] for n in nodes)==t['makespan_ns']==int(cost[0])
        # Derive all matrix shapes and expected jobs without importing author code.
        shapes=[]
        for phase in [0,1]:
            for e,item in enumerate(t['items']):
                rows=item['gu_rows' if phase==0 else 'down_rows']
                cols=1536 if phase==0 else item['down_columns']
                shapes.extend((phase,e,c,r,cols) for c,r in enumerate(rows) if r)
        native=defaultdict(list)
        ready={}
        host_down={}
        outputs=defaultdict(list)
        chip_locks=defaultdict(list)
        stage_kinds=['matrix_command','input_channel_reload','nonpreemptive_native_compute','output_channel','output_pcie']
        host_output_by_parent={n['deps'][0]:n for n in nodes if n['kind']=='output_pcie'}
        for n in nodes:
            if n['kind'] in stage_kinds[:4]:
                native[n['job']].append(n)
            if n['kind']=='whole_expert_gu_host_ready':ready[n['expert']]=n
            if n['kind']=='expert_down_host_submission':host_down[n['expert']]=n
        assert len(native)==len(shapes)
        for job,(phase,e,c,r,cols) in enumerate(shapes):
            ns=native[job]
            assert len(ns)==4
            ns=ns+[host_output_by_parent[ns[-1]['index']]]
            assert [n['kind'] for n in ns]==stage_kinds
            assert all(n['phase']==phase and n['expert']==e for n in ns)
            assert all(n['chip']==c for n in ns[:4]) and ns[4]['chip']==-1
            assert [n['resource'] for n in ns]==['controller',f'channel{c%8}',f'chip{c}',f'channel{c%8}','pcie']
            durations=[250,ceildiv(cols,2),28000*ceildiv(r*cols,65536),ceildiv(4*r,2),ceildiv(4*r,8)]
            assert [n['end_ns']-n['start_ns'] for n in ns]==durations
            assert ns[0]['acquire'] and ns[3]['release']
            assert not any(n['acquire'] for n in ns[1:]) and not any(n['release'] for n in ns[:3]+ns[4:])
            assert all(a['end_ns']<=b['start_ns'] for a,b in zip(ns,ns[1:]))
            chip_locks[c].append((ns[0]['start_ns'],ns[3]['end_ns']))
            if phase==0:outputs[e].append(ns[4]['index'])
            else:
                assert nodes[host_down[e]['deps'][0]]['kind']=='host_activation'
                act=nodes[host_down[e]['deps'][0]]
                assert act['deps']==[ready[e]['index']]
                assert ns[0]['deps']==[host_down[e]['index']]
        for c,intervals in chip_locks.items():
            intervals.sort()
            assert all(b<=cc for (_,b),(cc,_) in zip(intervals,intervals[1:]))
        assert len(ready)==len(host_down)==7
        for e in range(7):assert set(ready[e]['deps'])==set(outputs[e])
        # Host submission slots, all private input bytes, and all returned rows.
        assert work['host']==16000
        assert work['pcie']==10176
        kind_counts=Counter(n['kind'] for n in nodes)
        assert kind_counts['private_input_to_controller']==7
        assert kind_counts['common_input_to_controller']==1
        assert kind_counts['host_activation']==7
        assert kind_counts['host_combine']==1
        audited.append(dict(label=label, scalar_routes=len(cost), saved_trace_nodes=len(nodes),
            native_jobs=len(shapes), executed_chip_waves=int(stored[0]),host_work_ns=work['host'],
            pcie_work_ns=work['pcie'], trace_makespan_ns=t['makespan_ns']))
    for total_rec in d['totals']:
        match=[r for r in d['records'] if all(r[k]==total_rec[k] for k in ['mapping','shared_gu_replica','shared_down_high_half'])]
        assert len(match)==3
        v=np.stack([arrays[r['label']] for r in match]).sum(axis=0)
        assert v.mean()==total_rec['mean_ns'] and np.percentile(v,99)==total_rec['p99_ns']
    onecopy=[r for r in d['totals'] if not r['shared_gu_replica']]
    replica=[r for r in d['totals'] if r['shared_gu_replica']]
    bm=min(onecopy,key=lambda r:r['mean_ns']);br=min(replica,key=lambda r:r['mean_ns'])
    bt=min(onecopy,key=lambda r:r['p99_ns']);rt=min(replica,key=lambda r:r['p99_ns'])
    after={str(p):sha(p) for p in inputs}
    assert before==after
    out=dict(status='PASS under stated scalar cost/output contract',created_utc=datetime.now(timezone.utc).isoformat(),
        auditor_source_sha256=sha(__file__),imported_author_graph_or_cost_functions=False,
        input_sha256=before,author_inputs_unchanged=True,cases=len(audited),
        independently_recomputed_scalar_routes=scalar_rows,independently_checked_saved_trace_nodes=trace_nodes,
        cases_detail=audited,
        fair_comparison=dict(best_onecopy_mean=bm,best_replica_mean=br,
            mean_reduction_pct=100*(1-br['mean_ns']/bm['mean_ns']),
            best_onecopy_p99=bt,best_replica_p99=rt,
            p99_reduction_pct=100*(1-rt['p99_ns']/bt['p99_ns'])),
        native_geometry=dict(shared_gu_rows=2048,shared_gu_columns=1536,shared_down_rows=1536,
            shared_down_columns=1024,shared_down_waves=24,extra_shared_gu_payload_full40_bytes=125829120,
            shared_gu_exactly_once=True,shared_gu_duplicate_contiguous_pair_storage=True,
            single_context_chip_lock=True,per_expert_complete_GU_before_private_down=True,
            host_submission_work_ns_per_layer=16000,pcie_work_ns_per_layer=10176),
        caveats=['36 saved first-route traces independently checked; all73728 scalar summaries/bounds checked. Other schedule traces were not saved, so their full DAG feasibility was not independently replayed here.',
            'Physical allocator block addresses/capacity reservations were not supplied. Full-wave tile useful bytes/pages match, but device command/address execution is not proven.',
            'Output4bytes is only the inherited timing contract; published/local AiF interface does not establish actual output dtype/scale/rounding. No native numeric-equivalence claim.',
            'Host activation, row reassembly/scatter, scaling and final combine duration0 are optimistic explicit assumptions. Host submissions and PCIe/channel resources are charged.',
            'Existing unmodified weights are abstract rawI8. Actual training routes came from Q4_K/Q6_K GGUF; native same-model W8 routing has not been established.',
            'Fixed gate-first scheduler and three fitted train layers are not global optimal scheduling or generalization evidence.',
            'Same-byte routed/shared-hot replication and host pinning remain necessary strong controls; no novelty claim follows from this audit.'])
    target=DEST/'diagnostic_audit.json'
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],cases=len(audited),scalar_routes=scalar_rows,
        saved_trace_nodes=trace_nodes,fair_comparison=out['fair_comparison']),indent=2))


if __name__=='__main__':main()
