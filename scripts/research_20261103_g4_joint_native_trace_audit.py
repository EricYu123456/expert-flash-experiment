#!/usr/bin/env python3
"""Read-only all-owner/row/work checks and six independent Python joint traces."""
import hashlib
import json
from pathlib import Path

import numpy as np
import research_20261103_ordinary_read_backend as backend

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
SOURCE = BASE / 'g4-joint-prefix-replica-screen'
NATIVE = SOURCE / 'native-v1'
OUT = SOURCE / 'independent-native'
GU = BASE / 'g4-elastic-prefix-screen/work-v3-independent-prefix'
INC = BASE / 'compact-coupled-solver/continuation/native/0107.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def load(p):
    with np.load(p) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    OUT.mkdir(exist_ok=False)
    files = [Path(__file__).resolve(), NATIVE / 'protocol.frozen.json', NATIVE / 'report.json', NATIVE / 'arrays.npz',
             NATIVE / 'row_owners.npz', SOURCE / 'work_arrays.npz', GU / 'rowid_ledgers.json', INC,
             ROOT / 'scripts/research_20261103_ordinary_read_backend.py']
    files += [NATIVE / f'q{q}.{case}.npz' for q in (0,255,511) for case in ('baseline','joint')]
    protocol = dict(scope='Read-only all512row/owner/work/aggregate check and six Python ReferenceGraph traces. No C++ rerun, other queries, fitting, weights or heldout.',
                    inputs=[dict(path=str(p), sha256=sha(p)) for p in files])
    dump(OUT / 'protocol.frozen.json', protocol)
    producer = json.loads((NATIVE / 'protocol.frozen.json').read_text())
    for item in producer['inputs']:
        assert sha(item['path']) == item['sha256']
    result = json.loads((NATIVE / 'report.json').read_text())
    assert result['protocol_sha256'] == sha(NATIVE / 'protocol.frozen.json')
    a, w, b, owners = [load(p) for p in (NATIVE/'arrays.npz',SOURCE/'work_arrays.npz',INC,NATIVE/'row_owners.npz')]
    entries = json.loads((GU / 'rowid_ledgers.json').read_text())[3]['experts']
    service = backend.Service(**producer['service'])
    assert np.array_equal(a['baseline'], b['cost']) and np.array_equal(a['baseline_LB'], b['lower_bound'])
    assert np.array_equal(a['routes'], w['routes']) and np.array_equal(owners['routes'], w['routes'])
    resident = {}
    for e in range(65):
        for phase in range(2):
            for j in range(*w['resident_offsets'][e,phase]):
                resident[e,phase,int(w['resident_jobs'][j,0])] = w['resident_addresses'][j]
    gu_images = [{int(c): np.asarray(ids,np.int32) for c,ids in entry['resident_prefix_image'].items()} for entry in entries]
    down_images = []
    for e in range(64):
        starts = np.r_[0,np.cumsum(w['rows_B'][e,1])]
        original = [np.arange(starts[c],starts[c+1],dtype=np.int32) for c in range(16)]
        image = [x.copy() for x in original]
        for src,dest in w['Down_edges'][e]:
            image[dest] = np.r_[original[dest],original[src]]
        down_images.append(image)
    owner_row_values = 0
    counter_checks = 0
    for q,route in enumerate(w['routes']):
        local = w['rows_B'][np.r_[route,64]].copy()
        for ordinal,e in enumerate(route):
            entry = entries[e];u=w['chosen_u'][q,ordinal];v=w['chosen_v'][q,ordinal]
            assert np.isin(u,[0,1]).all() and np.isin(v,[0,1]).all()
            for i,c in enumerate(entry['sources']):
                local[ordinal,0,c] = 128 if u[i] else w['rows_B'][e,0,c]
            for i,c in enumerate(entry['low_destinations']):
                local[ordinal,0,c] = 0 if u[i] else 40
            hd=entry['high_destination']
            local[ordinal,0,hd] = w['rows_B'][e,0,hd] if not u[7] else entry['high_prefix_rows'] if not u.all() else 0
            for i,(src,dest) in enumerate(w['Down_edges'][e]):
                if v[i]:local[ordinal,1,src]=0;local[ordinal,1,dest]=256
            for phase,key,image,R in [(0,'GU',gu_images[e],1024),(1,'Down',down_images[e],1536)]:
                mapping=owners[key][q,ordinal]
                assert mapping.shape==(R,2) and len(np.unique(mapping,axis=0))==R
                for chip in range(16):
                    ids=np.flatnonzero(mapping[:,0]==chip);positions=mapping[ids,1]
                    assert np.all((positions>=0)&(positions<local[ordinal,phase,chip]))
                    assert np.array_equal(np.asarray(image[chip])[positions],ids)
                owner_row_values += R
            assert local[ordinal,1].sum()==1536
        assert np.array_equal(local,a['compact_rows'][q]) and np.array_equal(local[6],b['rows'][64])
        plan=backend.lower_rows(local[:,0],local[:,1],1536,[512]*6+[1024],service)
        occupancy=backend.occupancy_table(plan,service)['chip_ns'].sum((0,1))
        assert occupancy.max()==a['actual_work'][q]==w['chosen_work'][q]
        counters=np.zeros((2,7),np.int64)
        for e in range(7):
            for phase in range(2):
                for c,nr,nc,pages,waves in plan['jobs'][slice(*plan['offsets'][e,phase])]:
                    emit=int(pages)*16384//int(nc)
                    counters[phase]+=[1,nr*nc,pages,waves,nc,emit*4,(nc+1)//2+emit*2]
        assert np.array_equal(counters,a['counters'][q]);counter_checks+=1
    trace_records=[]
    for q in (0,255,511):
        selected=np.r_[w['routes'][q],64]
        for case in ('baseline','joint'):
            saved=load(NATIVE/f'q{q}.{case}.npz')
            plan={k:saved[k] for k in ('offsets','jobs','input_columns','total_rows')}
            backend.validate_plan(plan,7,service)
            expected_rows=b['rows'][selected] if case=='baseline' else a['compact_rows'][q]
            assert np.array_equal(saved['rows'],expected_rows)
            assert np.array_equal(saved['total_rows'],expected_rows.sum(2))
            for ordinal,e in enumerate(selected):
                for phase in range(2):
                    for j in range(*plan['offsets'][ordinal,phase]):
                        c,nr,nc,pages,waves=map(int,plan['jobs'][j]);address=resident[int(e),phase,c]
                        assert np.array_equal(address,saved['aliases'][j])
                        assert pages<=address[4] and waves<=address[3] and address[2]+waves<=192
                        assert nc==(1536 if phase==0 else 512 if ordinal<6 else 1024)
            graph=backend.ReferenceGraph.build(plan,np.arange(7),service);reference=graph.run();tr=reference['trace']
            expected=a['baseline'][q] if case=='baseline' else a['cost'][q]
            assert reference['cost']==saved['cost'][0]==expected and np.array_equal(tr,saved['trace'])
            for i,node in enumerate(graph.nodes):
                assert tr[i,3]-tr[i,2]==node['duration']
                assert all(tr[d,3]<=tr[i,2] for d in node['deps'])
            # Every scheduled resource and every full chip-owner interval is exclusive.
            for r in range(service.resources):
                spans=sorted((int(tr[i,2]),int(tr[i,3])) for i,n in enumerate(graph.nodes) if n['resource']==r)
                assert all(x[1]<=y[0] for x,y in zip(spans,spans[1:]))
            intervals=[[] for _ in range(16)]
            for job in range(graph.job_count):
                indexes=[i for i,n in enumerate(graph.nodes) if n['job']==job]
                assert len(indexes)==4;cmd,inp,comp,out=indexes;n=graph.nodes[cmd];e=n['expert'];phase=n['phase'];chip=n['chip']
                assert n['acquire'] and graph.nodes[out]['release']
                width=int(plan['input_columns'][e,phase])
                assert graph.nodes[inp]['duration']==service.transfer(width,service.channel_bytes_per_second)
                intervals[chip].append((int(tr[cmd,2]),int(tr[out,3])))
                if phase==1:
                    ready=[i for i,x in enumerate(graph.nodes) if x['resource']==service.resources-1 and x['expert']==e and x['phase']==0]
                    assert len(ready)==1 and tr[cmd,2]>=tr[ready[0],3]+service.host_submit_ns+service.activation_ns
            for spans in intervals:
                spans.sort();assert all(x[1]<=y[0] for x,y in zip(spans,spans[1:]))
            trace_records.append(dict(query=q,case=case,cost_ns=int(expected),nodes=len(tr),jobs=graph.job_count,
                                      all_node_traces_dependencies_resources_owner_reload_barrier_exact=True))
    mean_gain=100*(1-a['cost'].mean()/a['baseline'].mean());p99_gain=100*(1-np.percentile(a['cost'],99)/np.percentile(a['baseline'],99))
    assert mean_gain==result['improvement']['mean_pct'] and p99_gain==result['improvement']['p99_pct']
    for item in protocol['inputs']+producer['inputs']:
        assert sha(item['path'])==item['sha256']
    report=dict(status='PASS read-only joint native trace/ownership audit',protocol_sha256=sha(OUT/'protocol.frozen.json'),
                producer_native_protocol_sha256=sha(NATIVE/'protocol.frozen.json'),baseline_cost_LB_exact_values=1024,
                all512computed_rows_work_counters_exact=True,all_GU_Down_owner_map_values_checked=owner_row_values,
                row_owner_views_each_phase=3072,trace_checks=trace_records,trace_nodes=sum(x['nodes'] for x in trace_records),
                mean_gain_pct=float(mean_gain),p99_gain_pct=float(p99_gain),native_mean_ns=float(a['cost'].mean()),
                native_p99_ns=float(np.percentile(a['cost'],99)),
                beyond_Down_single_mean_gain_pct=float(100*(1-a['cost'].mean()/a['Down_single'].mean())),
                gate_passed=bool(mean_gain>=2 or p99_gain>=5),
                scope='Saved all512cost statistics, ownership/work/counters checked; only six Python traces independently rescheduled. No C++rerun, selectorbenchmark, new fit, strongcontrols, weights or heldout.')
    dump(OUT/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
