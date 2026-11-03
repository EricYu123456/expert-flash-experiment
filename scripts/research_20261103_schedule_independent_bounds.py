#!/usr/bin/env python3
"""Vectorized valid resource/causal bounds for existing frozen Q38 jobs.

This audits scheduling headroom of specific layouts under explicit service
contracts. It does not fit labels, retile matrices, or rerun native campaigns.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
QWEN=BASE/'qwen38-placement'
DEST=QWEN/'schedule-independent-bounds'
CASES=('group_g16','group_best','overlap_best','whole_row')
CONTROLS=CASES[:3]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ceildiv(a,b):return (a+b-1)//b
def write(name,d):
    p=DEST/name;p.write_text(json.dumps(d,indent=2)+'\n');return p


def descriptor_plan(plan,inventory):
    offsets=np.zeros((513,2,2),dtype=np.int32);inputs=np.zeros((513,2),dtype=np.int32)
    jobs=[];semantic=[]
    for e in range(513):
        for phase,name in enumerate(('gate_up','down')):
            p=plan['plans'][name] if e<512 else plan['fixed_shared'][name]
            color=p['labels'][e] if e<512 else p['color']
            c=inventory[name][p['label']];assert not c['conditional_partial_sum']
            bands=c['variants'][color];offsets[e,phase,0]=len(jobs)
            coverage=[]
            for j in bands:
                chip,r,cols=map(int,(j['chip'],j['rows'],j['columns']))
                assert j['column_start']==0 and r>0
                assert inputs[e,phase] in (0,cols);inputs[e,phase]=cols
                pages=ceildiv(r*cols,16384);waves=ceildiv(pages,4);assert waves<=192
                jobs.append((chip,r,cols,pages,waves));semantic.append((e,phase))
                coverage.extend(range(j['row_start'],j['row_start']+r))
            assert sorted(coverage)==list(range(len(coverage)))
            offsets[e,phase,1]=len(jobs)
    assert np.all(inputs[:,0]==inputs[0,0])
    return dict(offsets=offsets,jobs=np.asarray(jobs,dtype=np.int32),input_columns=inputs),semantic


def ledgers(plan,semantic,s):
    chips=s.chips;channels=s.channels
    owned=np.zeros((513,chips),dtype=np.int64);stream_owned=owned.copy();nand=owned.copy()
    channel=np.zeros((513,channels),dtype=np.int64)
    command=np.zeros(513,dtype=np.int64);outpcie=command.copy()
    tails=np.zeros((513,2),dtype=np.int64)
    for job,(expert,phase) in zip(plan['jobs'],semantic):
        chip,r,c,pages,waves=map(int,job)
        emitted=pages*s.page_bytes//c
        inp=ceildiv(c*s.input_bytes*10**9,s.channel_bytes_per_second)
        out=ceildiv(emitted*s.output_bytes*10**9,s.channel_bytes_per_second)
        pci=ceildiv(emitted*s.output_bytes*10**9,s.pcie_bytes_per_second)
        read=waves*s.ordinary_read_ns
        owned[expert,chip]+=s.command_ns+inp+read+out
        stream_owned[expert,chip]+=s.command_ns+inp+max(read,out)
        nand[expert,chip]+=read
        channel[expert,chip%channels]+=inp+out
        command[expert]+=s.command_ns;outpcie[expert]+=pci
        tails[expert,phase]=max(tails[expert,phase],inp+read+out+pci)
    downupload=(plan['input_columns'][:,1].astype(np.int64)*s.input_bytes*10**9+s.pcie_bytes_per_second-1)//s.pcie_bytes_per_second
    common=ceildiv(int(plan['input_columns'][0,0])*s.input_bytes*10**9,s.pcie_bytes_per_second)
    chain=s.host_submit_ns+max(s.command_ns,common)+tails[:,0]+s.activation_ns+s.host_submit_ns+np.maximum(s.command_ns,downupload)+tails[:,1]+s.combine_ns
    return dict(owned=owned,stream_owned=stream_owned,nand=nand,channel=channel,
                command=command,outpcie=outpcie,downupload=downupload,common=common,chain=chain)


def vector_bounds(table,routes,s):
    def selected(a):return a[routes].sum(axis=1)+a[512]
    host=np.full(len(routes),(1+routes.shape[1]+1)*s.host_submit_ns+(routes.shape[1]+1)*s.activation_ns+s.combine_ns,dtype=np.int64)
    owned=selected(table['owned']).max(1);stream_owned=selected(table['stream_owned']).max(1)
    nand=selected(table['nand']).max(1);channel=selected(table['channel']).max(1)
    command=selected(table['command']);output=selected(table['outpcie'])
    input_=table['common']+selected(table['downupload']);pcie=input_+output
    critical=np.maximum(table['chain'][routes].max(1),table['chain'][512])
    resource=np.maximum.reduce((host,nand,channel,command,pcie))
    # Exactly matches the current C++ fixed-DAG lower_bound expression.
    cpp=np.maximum.reduce((resource,critical,s.host_submit_ns+owned,s.host_submit_ns+command))
    # Every channel/PCIe operation is after the initial submission as well.
    strict=np.maximum.reduce((cpp,s.host_submit_ns+channel,s.host_submit_ns+pcie))
    # Broader relaxation: allow read/output overlap inside a job and full-duplex
    # PCIe. Drop the postcompute-drain DAG critical path. Full-input reload and
    # one held chip context per job are still retained explicitly.
    streaming=np.maximum.reduce((host,s.host_submit_ns+stream_owned,s.host_submit_ns+command,
                                 s.host_submit_ns+channel,s.host_submit_ns+np.maximum(input_,output)))
    return dict(cpp=cpp,strict=strict,streaming=streaming,nand_only=nand,
                strict_chip=s.host_submit_ns+owned,controller=s.host_submit_ns+command,
                channel=s.host_submit_ns+channel,pcie=s.host_submit_ns+pcie,critical=critical,host=host)


def comparison(whole,per_layer,whole_layers):
    total=per_layer.sum(axis=0);mean_lb=float(total.mean());mean_whole=float(whole.mean())
    return dict(mean_control_LB_ns=mean_lb,whole_feasible_mean_ns=mean_whole,
       whole_as_fraction_of_LB=mean_whole/mean_lb,
       signed_reduction_vs_LB=1-mean_whole/mean_lb,
       positive_mean_certified_margin_ns=max(0.,mean_lb-mean_whole),
       per_token_whole_below_LB=int(np.count_nonzero(whole<total)),
       per_token_equal=int(np.count_nonzero(whole==total)),tokens=len(whole),
       per_token_whole_below_LB_fraction=float(np.mean(whole<total)),
       per_layer_query_whole_below_LB=int(np.count_nonzero(whole_layers<per_layer)),
       layer_queries=int(per_layer.size),
       per_layer_query_whole_below_LB_fraction=float(np.mean(whole_layers<per_layer)))


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    protocol_path=DEST/'protocol.frozen.json';assert not protocol_path.exists(),'Use a fresh output directory rather than overwrite a frozen study.'
    snapshot=DEST/'snapshots';snapshot.mkdir()
    backend_names=['research_20261103_ordinary_read_backend.py','research_20261103_ordinary_read_native.cpp','research_20261102_whole_row_compiler.py']
    originals={}
    for name in backend_names:
        p=ROOT/'scripts'/name;raw=p.read_bytes();(snapshot/name).write_bytes(raw);originals[str(p)]=hashlib.sha256(raw).hexdigest()
    sys.path.insert(0,str(snapshot))
    from research_20261103_ordinary_read_backend import Service,NativeReplay
    service=Service(output_fringe=1,input_buffer_bytes=32768)
    frozen_path=QWEN/'native-full48/plans.frozen.json';frozen=json.loads(frozen_path.read_text())
    templates_path=QWEN/'phase-fit3/templates.json';templates=json.loads(templates_path.read_text())
    inventory={phase:{v['label']:v for v in variants} for phase,variants in templates.items()}
    route_path=BASE/'qwen38-trace-audit/external-doc-grounded-stride1-short/external_decode.npz'
    inputs={str(p):sha(p) for p in [Path(__file__),frozen_path,templates_path,route_path,QWEN/'output-fringe/summary.json',BASE/'host-row-gather/summary.json']}
    records={}
    for rec in frozen['records']:
        if rec['case'] not in CASES:continue
        path=ROOT/rec['plan'];assert sha(path)==rec['plan_sha256'];inputs[str(path)]=rec['plan_sha256']
        costpath=QWEN/'output-fringe'/f"{rec['label']}.mode1.npz";inputs[str(costpath)]=sha(costpath)
        records[rec['layer'],rec['case']]=(path,costpath)
    assert len(records)==48*len(CASES)
    diagnostic_layers=[0,16,32,47]
    protocol=write('protocol.frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=inputs,
       original_backend_sources=originals,service=service.__dict__,controls=list(CONTROLS),whole='whole_row',
       diagnostics=dict(layers=diagnostic_layers,uniform_tokens=16,expected_native_costs=256),
       scope='All5519already-used external decode tokens×48frozen layers; no refit or new external task. Bulk uses vectorized independent job ledgers, not native replay.',
       contracts=['Strict lower bounds permit arbitrary legal scheduling of the same frozen native jobs, full-input reloads, chip ownership through output-channel drain, and postcompute full-output drain. They do not certify all command coalescing/retilings/labels/native ABIs.',
                  'Streaming relaxation drops the strict DAG path and permits job-internal read/output overlap plus duplex PCIe, while retaining one chip context and full-input reload per frozen command.',
                  'NAND-only bound sums ordinary wave read service on each chip and imposes no command/input/output costs. It still assumes the same frozen wave reads and no cache/reuse/cofetch change.',
                  'Free control relaxation picks the minimum control LB perlayer/actualtoken, then sums48layers. It grants unphysical free layout selection but is only a bound over these three frozen controls, not all cyclic geometries/all labels.',
                  'Optional+1.243ms host-copy allowance is charged once per full48-layer token to whole only. It is a separate EPYC benchmark scenario, not a hardware worst-case guarantee.']))
    with np.load(route_path) as data:routes=data['routes'].astype(np.int32)
    assert routes.shape==(5519,48,10) and np.all((routes>=0)&(routes<512))
    assert np.all(np.diff(np.sort(routes,axis=2),axis=2)>0)
    steps=np.unique(np.linspace(0,len(routes)-1,16,dtype=np.int64))
    native=NativeReplay(DEST/'build');diagnostics=[]
    bounds={name:np.zeros((len(CASES),48,len(routes)),dtype=np.int64) for name in ['strict','streaming','nand_only']}
    gf=np.zeros((len(CASES),48,len(routes)),dtype=np.int64)
    components=[]
    for layer in range(48):
        for ci,case in enumerate(CASES):
            pp,cp=records[layer,case];plan,sem=descriptor_plan(json.loads(pp.read_text()),inventory)
            with np.load(cp) as data:cost=data['cost'].copy()
            assert cost.shape==(len(routes),)
            gf[ci,layer]=cost;table=ledgers(plan,sem,service);b=vector_bounds(table,routes[:,layer],service)
            assert np.all(b['strict']<=cost) and np.all(b['streaming']<=cost) and np.all(b['nand_only']<=cost)
            assert np.all(b['streaming']<=b['strict'])
            for name in bounds:bounds[name][ci,layer]=b[name]
            components.append(dict(layer=layer,case=case,**{k:float(v.mean()) for k,v in b.items()}))
            if layer in diagnostic_layers:
                got=native.run(plan,routes[steps,layer],512,1,service)
                assert np.array_equal(got['lower_bound'],b['cpp'][steps])
                assert np.array_equal(got['cost'],cost[steps])
                diagnostics.append(dict(layer=layer,case=case,queries=len(steps),CPP_bound_max_difference_ns=0,frozen_cost_max_difference_ns=0))
        if layer%8==0:print('completed bounds layer',layer,flush=True)
    whole_layers=gf[3];whole=whole_layers.sum(axis=0)
    results=[];relaxations=[];arrays=dict(whole_feasible_token_ns=whole,whole_feasible_layers_ns=whole_layers)
    for name,values in bounds.items():
        arrays[name+'_control_layers_ns']=values[:3]
        for ci,case in enumerate(CONTROLS):
            results.append(dict(contract=name,case=case,mean_control_GF_ns=float(gf[ci].sum(0).mean()),**comparison(whole,values[ci],whole_layers)))
        relaxed=values[:3].min(axis=0)
        record=dict(contract=name,case='free_per_layer_query_min_of_three_controls',**comparison(whole,relaxed,whole_layers))
        extra=1_243_000;relaxed_total=relaxed.sum(0)
        record['whole_plus_optional_EPYC_copy_1243000ns']=dict(mean_whole_ns=float(whole.mean()+extra),signed_reduction_vs_LB=1-float(whole.mean()+extra)/float(relaxed_total.mean()),positive_mean_certified_margin_ns=max(0.,float(relaxed_total.mean()-whole.mean()-extra)),tokens_whole_below_LB=int(np.count_nonzero(whole+extra<relaxed_total)),token_fraction=float(np.mean(whole+extra<relaxed_total)))
        relaxations.append(record);arrays[name+'_free_min_token_ns']=relaxed_total
        arrays[name+'_free_min_control_index']=values[:3].argmin(axis=0).astype(np.int8)
    np.savez_compressed(DEST/'bounds.npz',**arrays)
    assert all(sha(p)==h for p,h in inputs.items())
    summary=write('summary.json',dict(status='complete frozen-control schedule-independent lower-bound study',protocol_sha256=sha(protocol),
       native_diagnostic_build=native.provenance,diagnostics=diagnostics,native_diagnostic_costs=sum(x['queries'] for x in diagnostics),
       tokens=len(routes),layers=48,per_control_layer_queries=len(routes)*48,
       whole_feasible_mean_ns=float(whole.mean()),whole_feasible_p99_ns=float(np.percentile(whole,99)),
       comparisons=results,free_control_relaxations=relaxations,layer_component_means=components,
       arrays_sha256=sha(DEST/'bounds.npz'),
       interpretation='Positive margins are certified only for the named contract and frozen controls/jobs. Zero positive margin means no schedule-independent gain established by that bound. No all-geometry/global-label/hardware-optimal claim.'))
    print(json.dumps(dict(summary=str(summary),whole_mean_ms=whole.mean()/1e6,comparisons=results,free_relaxations=relaxations),indent=2),flush=True)


if __name__=='__main__':main()
