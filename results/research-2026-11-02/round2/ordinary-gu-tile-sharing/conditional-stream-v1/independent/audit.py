"""Conditional-stream ownership/LB/buffer audit; no fitting or new payload."""
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment');OUT=Path(__file__).resolve().parent;PILOT=OUT.parent
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261103_conditional_gu_stream as stream
from research_20261103_topology_placement_replay import load_python


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def peak(events):
    live=maximum=0
    for t,delta in sorted(events):live+=delta;assert live>=0;maximum=max(maximum,live)
    assert live==0;return maximum


def valid_bound(graph,s):
    resource=np.zeros(s.resources,np.int64);context=np.zeros(16,np.int64);critical=[];local=[]
    for i,n in enumerate(graph.nodes):
        resource[n['resource']]+=n['duration']
        critical.append(n['duration']+max((critical[j] for j in n['deps']),default=0))
        local.append(n['duration']+max((local[j] for j in n['deps'] if n['job']>=0 and graph.nodes[j]['job']==n['job']),default=0))
        if n['release']:context[n['chip']]+=local[-1]
    return max(int(resource.max()),max(critical),s.host_submit_ns+int(context.max()),s.host_submit_ns+int(resource[1]))


protocol=json.loads((PILOT/'protocol.frozen.json').read_text())
for item in protocol['inputs']:assert sha(item['path'])==item['sha256']
assert sha(stream.__file__)==protocol['source_sha256']
backend=load_python();native=stream.Native();checks=[];comparisons=[];records=[];paired_arrays={};bridge=0
buffer_reservations=[];gatefirst_identical=0;downfirst_different=0
for copies in [1,4]:
    path=PILOT/f'input{copies}/stage0/screen.json';d=json.loads(path.read_text())
    assert d['protocol_sha256']==sha(PILOT/'protocol.frozen.json') and d['stage']==0
    preferred=next(c for c in d['comparisons'] if c['transport']==3);assert not preferred['expand_gate']
    for r in d['records']:
        with np.load(r['arrays']) as z:rr=z['rows'];ranges=z['ranges'];routes=z['routes'];cost=z['cost'];bound=z['lower_bound'];fifo=z['chip_FIFO_required_bytes'];buf=z['controller_GU_output_required_bytes']
        assert cost.shape==(512,) and np.all(cost>=bound)
        assert r['mean_ns']==float(cost.mean()) and r['p99_ns']==float(np.percentile(cost,99))
        assert np.all(rr[:,0]%32==0) and np.all(ranges%32==0)
        assert int(fifo.max())==r['FIFO_max_required_bytes'] and int(buf.max())==r['controller_GU_max_required_bytes']
        if r['transport']==0:
            assert np.all(fifo==-1) and np.all(buf==-1);bridge+=512
        else:assert np.all(fifo==128) and np.all(buf>=0)
        config={**protocol['services'][str(copies)],'down_first':r['priority']};s=backend.Service(**config)
        plan=backend.lower_rows(rr[:,0],rr[:,1],2048,[1408]*64+[2816],s)
        for q in [0,511]:
            graph,waves,vectors=stream.python_graph(backend,plan,ranges,np.r_[routes[q],64],s,r['mode'],r['transport'],0,0)
            direct=graph.run();v,b,f,cb,trace=native.run(plan,ranges,routes[q:q+1],s,r['mode'],r['transport'],0,0,True)
            assert v[0]==direct['cost']==cost[q] and b[0]==bound[q]==valid_bound(graph,s)
            assert np.array_equal(trace,direct['trace'])
            assert np.array_equal(f[0],fifo[q]) and cb[0]==buf[q]
            if r['transport']:
                assert len(waves)==704 # All complete GU rows, routed+shared.
                event=[[] for _ in range(16)];controller=[]
                for chip,read,out,pci in waves:
                    assert graph.nodes[read]['duration']==28000
                    event[chip]+=[(int(trace[read,3]),128),(int(trace[out,3]),-128)]
                    controller.append((int(trace[out,3]),128))
                    if pci is not None:controller.append((int(trace[pci,3]),-128))
                for pci,nbytes in vectors:controller.append((int(trace[pci,3]),-nbytes))
                assert np.array_equal(np.asarray([peak(e) for e in event]),fifo[q])
                assert peak(controller)==buf[q]
                # Every GU context owns PB until its final channel drain; all
                # intermediate wave nodes carry the same real job identity.
                jobs={}
                for i,n in enumerate(graph.nodes):
                    if n['job']>=0:jobs.setdefault(n['job'],[]).append(i)
                contexts=[[] for _ in range(16)]
                for ids in jobs.values():
                    acq=[i for i in ids if graph.nodes[i]['acquire']];rel=[i for i in ids if graph.nodes[i]['release']]
                    assert len(acq)==len(rel)==1
                    first,last=acq[0],rel[0];chip=graph.nodes[first]['chip']
                    contexts[chip].append((int(trace[first,2]),int(trace[last,3])))
                    assert all(trace[first,2]<=trace[i,2]<=trace[i,3]<=trace[last,3] for i in ids)
                    if graph.nodes[first]['phase']==0:
                        reads=[i for i in ids if graph.nodes[i]['resource']==3+s.channels+chip]
                        for a,bn in zip(reads,reads[1:]):
                            assert graph.nodes[bn]['deps']==[a] and trace[bn,2]==trace[a,3]
                        outputs=[i for i in ids if graph.nodes[i]['resource']==3+chip%s.channels]
                        # input is first channel node, followed by all outputs.
                        assert int(trace[last,3])==max(int(trace[i,3]) for i in outputs[1:])
                for intervals in contexts:
                    intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
                if r['transport']==3:
                    assert len(vectors)==7 and sum(n for _,n in vectors)==90112
                    reserve=[]
                    for ordinal,(pci,nbytes) in enumerate(vectors):
                        expected=22528 if ordinal==6 else 11264
                        assert nbytes==expected
                        deps=graph.nodes[pci]['deps'];assert len(deps)==expected//128 and len(set(deps))==len(deps)
                        assert trace[pci,2]>=max(trace[i,3] for i in deps)
                        reserve.extend([(int(min(trace[i,3] for i in deps)),expected),(int(trace[pci,3]),-expected)])
                    buffer_reservations.append(dict(id=r['id'],query=q,payload_peak_bytes=int(buf[q]),full_vector_buffers_allocated_on_first_arrival_peak_bytes=peak(reserve),all_CID_static_buffer_bytes=90112))
            checks.append(dict(id=r['id'],query=q,nodes=len(trace),lower_bound_ns=int(bound[q]),FIFO_bytes=int(fifo[q].max()),controller_GU_payload_bytes=int(buf[q])))
        records.append(dict(id=r['id'],arrays=r['arrays'],arrays_sha256=sha(r['arrays']),mean_ns=r['mean_ns'],p99_ns=r['p99_ns'],FIFO_required_bytes=r['FIFO_max_required_bytes'],controller_GU_payload_peak_bytes=r['controller_GU_max_required_bytes']))
    for r in [r for r in d['records'] if r['transport']==2]:
        full=next(x for x in d['records'] if x['candidate']==r['candidate'] and x['priority']==r['priority'] and x['mode']==r['mode'] and x['transport']==3)
        with np.load(r['arrays']) as a,np.load(full['arrays']) as z:delta=z['cost']-a['cost']
        if r['priority']==0:assert np.all(delta==0);gatefirst_identical+=len(delta)
        else:downfirst_different+=int(np.count_nonzero(delta))
    fused=preferred['fused_best']
    with np.load(fused['arrays']) as z:new=z['cost']
    for name in ['strongest_control','original_best','same_layout_singleton']:
        base=preferred[name]
        with np.load(base['arrays']) as z:old=z['cost']
        delta=old-new;paired_arrays[f'input{copies}.{name}']=delta
        comparisons.append(dict(copies=copies,baseline=name,mean_gain_pct=100*(1-new.mean()/old.mean()),p99_gain_pct=100*(1-np.percentile(new,99)/np.percentile(old,99)),improved=int((delta>0).sum()),equal=int((delta==0).sum()),worse=int((delta<0).sum()),delta_ns_quantiles=np.percentile(delta,[0,1,50,95,99,100]).tolist()))
np.savez_compressed(OUT/'paired_deltas.npz',**paired_arrays)
result=dict(status='passed_conditional_only_no_expansion',utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PILOT/'protocol.frozen.json'),source_sha256=protocol['source_sha256'],validation=dict(records=len(records),query_costs=len(records)*512,legacy_bridge_queries=bridge,python_cpp_traces=len(checks),python_cpp_nodes=sum(c['nodes'] for c in checks),gatefirst_waveDMA_vs_fullvector_identical_queries=gatefirst_identical,downfirst_waveDMA_vs_fullvector_different_queries=downfirst_different),preferred_full_expert_DMA_paired=comparisons,records=records,checks=checks,buffer_accounting=dict(frozen_controller_GU_output_required_bytes_means='Resident payload occupancy atchannelarrival/DMAdone, NOT sufficient reservedcapacity forfixedcontiguous fullvectors.',static_seven_vector_reservation_bytes=90112,sampled_first_arrival_fullvector_reservation_peak_bytes=max(x['full_vector_buffers_allocated_on_first_arrival_peak_bytes'] for x in buffer_reservations),sampled_reservations=buffer_reservations,capacity_status='NeitherchipFIFO norcontrollerRAMverified;prefixfinalECC andDMA/CID contractsunverified.'),decision='Stage0 preferredcontract fails frozenmean/tail gate evenwithzeroextraCQE/DMA overhead. Stage1nonzero sensitivity notrun. Noalllayer/heldout/placementfit.',conditions=protocol['conditions'])
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
(PILOT/'decision.json').write_text(json.dumps(dict(status='sealed_pending_completion_no_expansion',protocol_sha256=result['protocol_sha256'],stage0_passed=False,stage1_run=False,reason=result['decision'],classification='conditional sensitivity only,notvalidated existinghardware method'),indent=2)+'\n')
print(json.dumps(dict(validation=result['validation'],paired=comparisons,reserved_controller_static_bytes=90112,sampled_reservation_peak_bytes=result['buffer_accounting']['sampled_first_arrival_fullvector_reservation_peak_bytes']),indent=2))
