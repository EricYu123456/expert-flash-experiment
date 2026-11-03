#!/usr/bin/env python3
"""Granite4 train-only same-scheduler placement audit with single chip context.

Unlike the older study, each matrix job reloads its input and owns its chip
from command start through output channel drain. No parallel chip contexts,
column split, new dtype or projection composition is used. Whole-expert GU
host receipt precedes activation and any down job for that expert.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import time

import numpy as np

from research_20261102_single_request import RESOURCE_NAMES, profiles
from run_corpus_traces import ROOT, sha256

DEST=ROOT/'results/research-2026-11-02/round2/granite4-pipeline'
SOURCE=ROOT/'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'
EXPECTED='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'


def ceildiv(a,b):return (a+b-1)//b


class Graph:
    def __init__(self):self.nodes=[];self.jobs=[]
    def add(self,r,d,deps=(),phase=0,expert=-1,kind='',chip=-1,job=-1,acquire=False,release=False):
        i=len(self.nodes);assert d>=0
        self.nodes.append(dict(resource=r,duration=int(d),deps=list(deps),phase=phase,expert=expert,
            kind=kind,chip=chip,job=job,acquire=acquire,release=release))
        return i
    def matrix(self,expert,chip,phase,rows,columns,waves,submit,upload):
        job=len(self.jobs)
        cmd=self.add(1,250,[submit],phase,expert,'matrix_command',chip,job,acquire=True)
        inp=self.add(3+chip%8,ceildiv(columns,2),[cmd,upload],phase,expert,'input_channel_reload',chip,job)
        comp=self.add(11+chip,28000*waves,[inp],phase,expert,'nonpreemptive_native_compute',chip,job)
        out=self.add(3+chip%8,ceildiv(rows*4,2),[comp],phase,expert,'output_channel',chip,job,release=True)
        host=self.add(2,ceildiv(rows*4,8),[out],phase,expert,'output_pcie')
        self.jobs.append(dict(expert=expert,chip=chip,phase=phase,command=cmd,input=inp,compute=comp,release=out,host=host))
        return host
    def run(self,down_first,trace=False):
        nodes=self.nodes;successors=[[] for _ in nodes];remaining=[];ready=[[] for _ in RESOURCE_NAMES]
        busy=[False]*len(RESOURCE_NAMES);owner=[-1]*16;work=np.zeros(len(RESOURCE_NAMES),dtype=np.int64)
        starts=np.full(len(nodes),-1,dtype=np.int64);ends=starts.copy();events=[]
        def enqueue(i):
            n=nodes[i];phase=1-n['phase'] if down_first else n['phase']
            heapq.heappush(ready[n['resource']],((phase,n['expert'],i),i))
        for i,n in enumerate(nodes):
            remaining.append(len(n['deps']))
            for p in n['deps']:assert p<i;successors[p].append(i)
            if not n['deps']:enqueue(i)
        now=done=0
        while done<len(nodes):
            for r,queue in enumerate(ready):
                if busy[r] or not queue:continue
                held=[];chosen=None
                while queue:
                    item=heapq.heappop(queue);i=item[1];n=nodes[i]
                    eligible=n['job']<0 or (owner[n['chip']]==-1 if n['acquire'] else owner[n['chip']]==n['job'])
                    if eligible:chosen=i;break
                    held.append(item)
                for item in held:heapq.heappush(queue,item)
                if chosen is None:continue
                i=chosen;n=nodes[i]
                if n['acquire']:assert owner[n['chip']]==-1;owner[n['chip']]=n['job']
                starts[i]=now;busy[r]=True;work[r]+=n['duration'];heapq.heappush(events,(now+n['duration'],i))
            assert events,'Dependency/native-context deadlock'
            now=events[0][0]
            while events and events[0][0]==now:
                _,i=heapq.heappop(events);n=nodes[i];ends[i]=now;busy[n['resource']]=False
                if n['release']:assert owner[n['chip']]==n['job'];owner[n['chip']]=-1
                done+=1
                for q in successors[i]:
                    remaining[q]-=1
                    if remaining[q]==0:enqueue(q)
        assert owner==[-1]*16
        last=[0]*len(RESOURCE_NAMES)
        for i in np.argsort(starts,kind='stable'):
            n=nodes[i];assert ends[i]-starts[i]==n['duration']
            assert all(ends[p]<=starts[i] for p in n['deps'])
            if n['duration']:assert starts[i]>=last[n['resource']];last[n['resource']]=int(ends[i])
        chip_end=[0]*16
        for job in sorted(self.jobs,key=lambda j:starts[j['command']]):
            a=int(starts[job['command']]);b=int(ends[job['release']]);c=job['chip']
            assert a>=chip_end[c];chip_end[c]=b
            assert ends[job['command']]<=starts[job['input']]<=ends[job['input']]<=starts[job['compute']]
            assert ends[job['compute']]<=starts[job['release']]
        out=dict(makespan_ns=int(ends.max()),resource_work_ns=work.tolist(),nodes=len(nodes),
            dependency_resource_and_single_context_audit=True)
        if trace:out['trace']=[dict(index=i,start_ns=int(starts[i]),end_ns=int(ends[i]),resource=RESOURCE_NAMES[n['resource']],**{k:v for k,v in n.items() if k not in ['resource','duration']}) for i,n in enumerate(nodes)]
        return out


def build(selected,gu,down,barrier,activation_ns,combine_ns):
    g=Graph();submit=g.add(0,2000,kind='initial_host_submission')
    upload=g.add(2,ceildiv(gu['columns'],8),[submit],kind='common_input_to_controller')
    gu_ready=[]
    for ei,e in enumerate(selected):
        outputs=[]
        for chip in range(16):
            r=int(gu['rows'][e,chip]);w=int(gu['waves'][e,chip])
            if r:assert w<=192;outputs.append(g.matrix(ei,chip,0,r,gu['columns'],w,submit,upload))
        gu_ready.append(g.add(27,0,outputs,expert=ei,kind='whole_expert_gu_host_ready'))
    if barrier:
        barrier_node=g.add(27,0,gu_ready,kind='global_gu_barrier')
        activations=[g.add(0,activation_ns,[barrier_node],1,ei,'host_activation') for ei in range(len(selected))]
        batch_submit=g.add(0,2000,activations,1,kind='batch_down_host_submission')
    terminal=[]
    for ei,e in enumerate(selected):
        if barrier:ds=batch_submit
        else:
            act=g.add(0,activation_ns,[gu_ready[ei]],1,ei,'host_activation')
            ds=g.add(0,2000,[act],1,ei,'expert_down_host_submission')
        up=g.add(2,ceildiv(down['columns'],8),[ds],1,ei,'private_input_to_controller')
        for chip in range(16):
            r=int(down['rows'][e,chip]);w=int(down['waves'][e,chip])
            if r:assert w<=192;terminal.append(g.matrix(ei,chip,1,r,down['columns'],w,ds,up))
    g.add(0,combine_ns,terminal,1,kind='host_combine')
    return g


def stats(x):
    return dict(mean_ns=float(x.mean()),p95_ns=float(np.percentile(x,95)),p99_ns=float(np.percentile(x,99)),min_ns=int(x.min()),max_ns=int(x.max()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=0,help='0 means all32x64training routes')
    ap.add_argument('--activation-ns',type=int,default=0);ap.add_argument('--combine-ns',type=int,default=0)
    ap.add_argument('--output',type=Path,default=DEST);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert sha256(SOURCE)==EXPECTED;doc=json.loads(SOURCE.read_text());route=Path(doc['training_routes'])
    assert sha256(route)==doc['training_routes_sha256']
    with np.load(route) as f:
        routes=f['routes'][:32,:64].copy();ids=f['prompt_ids'][:32].tolist();layers=f['layers'].tolist()
    assert len(ids)==32 and all(s.startswith('train_') for s in ids)
    indices=np.arange(2048) if args.samples==0 else np.unique(np.linspace(0,2047,args.samples,dtype=int))
    records=[];costs={};began=time.monotonic();resource=Counter()
    for layer in layers[:3]:
        ps=[p for p in doc['phases'] if p['layer']==layer];assert [p['name'] for p in ps]==['gate_up','down']
        selected=routes[:,:,layers.index(layer)].reshape(-1,doc['active']);bounds={};layer_costs={}
        for placement,key in [('frozen_phase_group','group_rows'),('frozen_g12','rows')]:
            gu=profiles(ps[0][key],ps[0]['columns']);down=profiles(ps[1][key],ps[1]['columns'])
            gw=gu['waves'][selected].sum(axis=1);dw=down['waves'][selected].sum(axis=1)
            bounds[placement]=dict(serial_nand=stats((gw.max(axis=1)+dw.max(axis=1))*28000),
                joined_chip_work_lower_bound=stats((gw+dw).max(axis=1)*28000),
                integer_pooled_stored_work_lower_bound=stats(np.asarray([ceildiv(int(s),16) for s in (gw+dw).sum(axis=1)])*28000))
            for barrier in [True,False]:
                graph_cache={}
                for priority in ['down_first','gate_first']:
                    name=placement+('__barrier' if barrier else '__ready')+'__'+priority
                    values=[];cache={};work=np.zeros(len(RESOURCE_NAMES),dtype=np.int64)
                    for ix,idx in enumerate(indices):
                        s=selected[idx];t=tuple(s.tolist())
                        if t not in cache:
                            if t not in graph_cache:graph_cache[t]=build(s,gu,down,barrier,args.activation_ns,args.combine_ns)
                            result=graph_cache[t].run(priority=='down_first',trace=(ix==0))
                            cache[t]=result
                            if ix==0:
                                (args.output/f'layer{layer}.{name}.trace.json').write_text(json.dumps(dict(selected=s.tolist(),**result),indent=2)+'\n')
                        result=cache[t];values.append(result['makespan_ns']);work+=result['resource_work_ns']
                    values=np.asarray(values);layer_costs[name]=stats(values)
                    costs.setdefault(name,[]).append(values)
                    layer_costs[name]['resource_work_mean_ns']=dict(zip(RESOURCE_NAMES,(work/len(indices)).tolist()))
                    print('layer',layer,name,'mean',float(values.mean()),'unique',len(cache),'elapsed',round(time.monotonic()-began,1),flush=True)
        records.append(dict(layer=layer,bounds_all32x64=bounds,schedules=layer_costs))
        partial=dict(layer_results=records,elapsed_seconds=time.monotonic()-began)
        (args.output/'partial.json').write_text(json.dumps(partial,indent=2)+'\n')
    totals={name:stats(np.stack(values).sum(axis=0)) for name,values in costs.items()}
    comparisons={}
    for priority in ['down_first','gate_first']:
        g=totals['frozen_phase_group__ready__'+priority]['mean_ns'];n=totals['frozen_g12__ready__'+priority]['mean_ns']
        comparisons[priority]=dict(group_ready_ns=g,g12_ready_ns=n,placement_gain_same_pipeline_pct=100*(1-n/g),
            group_barrier_ns=totals['frozen_phase_group__barrier__'+priority]['mean_ns'],
            g12_barrier_ns=totals['frozen_g12__barrier__'+priority]['mean_ns'])
    assert sha256(SOURCE)==EXPECTED
    out=dict(status='train-only first3layer Granite4 existing-full-row native-job pipeline screen',created_utc=datetime.now(timezone.utc).isoformat(),
        source_plan=str(SOURCE),source_plan_sha256=EXPECTED,training_routes=str(route),training_routes_sha256=sha256(route),
        training_ids=ids,training_slice='[:32,:64]',scheduled_flat_indices=indices.tolist(),all_bounds_use_2048_routes=True,
        arguments={**vars(args),'output':str(args.output)},layer_results=records,total_schedules=totals,comparisons=comparisons,
        source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('research_20261102_single_request.py')]},
        hardware_contract=dict(chips=16,ordinary_wave_ns=28000,wordlines_per_block=192,controller_command_ns=250,
            host_submission_ns=2000,pcie_bytes_per_ns=8,channel_bytes_per_ns=2,input_element_bytes=1,output_element_bytes=4,
            chip_ownership='Each job owns its chip from command start until output drained over channel. Every job reloads input; exactly one native context/input/output resident per chip.',
            buffers='Controller retains selected vectors and drained outputs; no cached weights or parallel native contexts.'),
        limitations=['Event schedules are feasible under the stated resource/output contract, not globally optimal schedules or physical measurements.',
            'Existing service model A8 input/FP32 output byte costs are inherited; actual PE throughput and native output ABI remain uncalibrated.',
            'Host activation and final combine are exposed sensitivity parameters;0default is an optimistic bound, not a measurement.',
            'Same dispatch priority, native job locking and full input-reload costs apply to both layouts. No mapping optimized here.',
            'Global barrier batches its down host submission once; expert-ready mode pays2us for each selected expert down submission.',
            'Only routed FFN for three layers; shared FFN,SSM,attention and full-model wall time excluded. No heldout opened.'])
    target=args.output/'screen.json';target.write_text(json.dumps(out,indent=2)+'\n')
    (args.output/'completion.json').write_text(json.dumps(dict(status='complete',screen_sha256=sha256(target),
        source_sha256=sha256(Path(__file__)),comparisons=comparisons),indent=2)+'\n')
    np.savez_compressed(args.output/'schedule_costs.npz',indices=indices,**{name:np.stack(values) for name,values in costs.items()})
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
