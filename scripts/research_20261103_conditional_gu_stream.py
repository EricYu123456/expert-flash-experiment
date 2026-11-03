#!/usr/bin/env python3
"""Explicitly conditional FIFO/early-CID sensitivity, not validated firmware."""
import argparse
import ctypes
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

import research_20261103_ordinary_gu_tile_dag as drain
from research_20261103_topology_placement_replay import load_python,CPP,PYTHON

ROOT=drain.ROOT;BASE=drain.BASE
PREVIOUS=BASE/'ordinary-gu-tile-sharing/native-dag-v1'
OUT=BASE/'ordinary-gu-tile-sharing/conditional-stream-v1'
CAPABILITY=BASE/'ordinary-gu-tile-sharing/streaming-capability-audit'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    assert not OUT.exists();OUT.mkdir();(OUT/'source-snapshot').mkdir()
    old=json.loads((PREVIOUS/'protocol.frozen.json').read_text());known=json.loads((PREVIOUS/'screen.json').read_text())
    candidates=[]
    for c in old['candidates']:
        with np.load(c['arrays']) as z:rows=z['rows'];ranges=z['ranges']
        if np.all(rows[:,0]%32==0) and np.all(ranges%32==0):candidates.append(c)
    assert all(0<len([c for c in candidates if c['copies']==v])<=6 for v in [1,4])
    inputs={x['path']:x['sha256'] for x in old['inputs']}
    for path in [Path(__file__),Path(__file__).with_suffix('.cpp'),Path(drain.__file__),Path(drain.__file__).with_suffix('.cpp'),CPP,PYTHON,PREVIOUS/'protocol.frozen.json',PREVIOUS/'screen.json',CAPABILITY/'conditional_contract.json',CAPABILITY/'notes.json',CAPABILITY/'completion.json',ROOT/'paper/aif.txt']:
        inputs[str(path)]=sha(path)
    # Preserve all old nonaligned (and aligned) whole-view costs as strong
    # known implementations; do not retrofit them with unproven partial rows.
    for r in known['records']:inputs[r['arrays']]=sha(r['arrays'])
    protocol=dict(status='frozen_before_conditional_replay',utc=datetime.now(timezone.utc).isoformat(),classification='Optimistic conditional software/ABI sensitivity, not validatedexistinghardwaremethod',source_sha256=sha(__file__),cpp_sha256=sha(Path(__file__).with_suffix('.cpp')),candidates=candidates,services=old['services'],schedulers=[0,1],modes=[0,1,2],transports={"0":"legacy whole-view drain","1":"wave FIFO+wave hostDMA but whole-view ready","2":"wave FIFO+wave hostDMA and per-owner fullGU ready (more optimistic)","3":"wave FIFO; collect full expert across chips; one full-vectorDMA/finalCID CQE (preferred conditional contract)"},stages={"0":dict(cqe_ns=0,dma_setup_ns=0,transports=[0,1,2,3],scope='Zero extra-overhead optimistic upper sensitivity; not calibrated callbacks/DMA'),"1":dict(cqe_ns=250,dma_setup_ns=50,transports=[0,3],scope='Uncalibrated nonzero-overhead sensitivity; not a mainhardwareprofile')},stage1_gate='Only after stage0 preferred full-expertDMA passes >=2% mean or >=5% P99 against strongest same-capability original/retainedlegacy andsame-layoutsingleton; no all-layer or heldout expansion evenifpasses',submit_accounting='Existing2us common batchedGU submission is unchanged; no extra perCIDsubmit to avoid doublecount. Every logicalexpert receives exactlyone final CQE afterfullGU,charged cqe_ns in eachfamily. Existing privateDownsubmit unchanged.',DMA_accounting='dma_setup_ns serializes on modeledPCIe/DMA engine for each outputDMA; allfamilies charged. Wave-hostDMA perwave; preferredcontract oneDMA/expertafterallneededchanneloutputs. Down remains fullmatrix/private and pays oneDMA/job.',conditions=['Prefix rows musthave finalECC/committedsuccess, withoutnativecommand-endlateerror invalidatingcompletedCID; unpublished','Controller canjoin same-x logicalCIDs, demux rowranges, keepinput/PB/groupreference lifetime and independentlyfinalize originalCIDs; unpublished','FinitechipFIFO lossless ordereddrain andcontrollerbuffercapacity mustmeet measuredtrace requirements; neither capacity is published','Producer produces32I32-like4Brows aftereach28us readwave with noaddedPEdelay/pollcost; optimistictiming assumption, not6.4GB/sPEclaim','Full-expertGU activation/globalA8scale barrier retained; no partialDown/nonlinear fusion','Chip/inputlatch remainslocked untilallviewrows channel-drain; no pause/backpressure/newcontext','Onlyfully32rowaligned GU maps included. Allchannel128B/wave emitted. In theseplans everyGUwave hasone selectedlogicalowner; nofringe/auxiliaryrows quietlyremoved. Nonaligned oldwholeview timings retainedasstrongcontrol'],known_legacy_path=str(PREVIOUS/'screen.json'),inputs=[dict(path=k,sha256=v) for k,v in sorted(inputs.items())],validation='Legacy zeroextra bridge all512/modes/priorities exact oldcost/bounds; independentPython graph everymode/transport/priority q0/q511 fulltrace; FIFO/controller occupancy derivedfromrealtrace events, notrequiredcapacity claim.',FIFO_model='Discrete128Bburst atwavecomputeend, remove128B atwholewave channelend; drainsbeforearrivals on same timestamp. Nextread depends onlypreviousread, neverdrain. ControllerGUbuffer arrivesatchannelend/removesatDMAdone. Peakrequirementsaremodelconditions, notverifiedexistingcapacities.')
    for item in protocol['inputs']:
        path=Path(item['path'])
        if path.suffix in ['.py','.cpp']:
            dest=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,dest);item['snapshot']=str(dest)
    (OUT/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),candidates=len(candidates))))


def python_graph(backend,plan,ranges,selected,s,mode,transport,cqe,dma):
    g=backend.ReferenceGraph(s);submit=g.add(0,s.host_submit_ns)
    upload=g.add(2,s.transfer(2048,s.pcie_bytes_per_second),[submit]);outputs=[[] for _ in selected];ready=[];waves_log=[];vectors=[]
    def whole(expert,phase,d,sub,up):
        done=g.matrix(expert,phase,d,sub,up);g.nodes[done]['duration']+=dma;return done
    def view(priority,chip,waves,owners,perwave):
        if transport==0:
            done=whole(priority,0,[chip,waves*32,2048,waves*4,waves],submit,upload)
            for owner in owners:outputs[owner].append(done)
            return
        job=g.job_count;g.job_count+=1
        cmd=g.add(1,s.command_ns,[submit],0,priority,chip,job,True)
        inp=g.add(3+chip%s.channels,s.transfer(2048*s.input_channel_copies,s.channel_bytes_per_second),[cmd,upload],0,priority,chip,job)
        prev_read=inp;prev_out=None;pieces=[]
        for w in range(waves):
            read=g.add(3+s.channels+chip,s.ordinary_read_ns,[prev_read],0,priority,chip,job);prev_read=read
            deps=[read]+([] if prev_out is None else [prev_out])
            out=g.add(3+chip%s.channels,s.transfer(128,s.channel_bytes_per_second),deps,0,priority,chip,job,False,w+1==waves);prev_out=out
            pci=None if transport==3 else g.add(2,dma+s.transfer(128,s.pcie_bytes_per_second),[out],0,priority)
            waves_log.append((chip,read,out,pci));pieces.append((out if transport==3 else pci,perwave[w]))
        for owner in owners:
            outputs[owner].extend(node for node,needed in pieces if transport==1 or owner in needed)
    if mode==0:
        for k,e in enumerate(selected):
            lo,hi=plan['offsets'][e,0]
            for d in plan['jobs'][lo:hi]:
                chip,r,C,p,w=map(int,d);assert r==w*32 and p==w*4
                view(k,chip,w,[k],[[k]]*w)
            if transport!=3:ready.append(g.add(s.resources-1,0,outputs[k],expert=k))
    else:
        for v in drain.python_views(ranges,selected,mode,s):
            perwave=[]
            for w in range(v['waves']):
                row=v['block']*6144+(v['start']+w)*32
                owners=[k for k,e in enumerate(selected) if ranges[e,v['chip'],0]<=row and row+32<=ranges[e,v['chip'],1]]
                assert len(owners)==1;perwave.append(owners)
            view(v['priority'],v['chip'],v['waves'],v['owners'],perwave)
        if transport!=3:ready=[g.add(s.resources-1,0,deps,expert=k) for k,deps in enumerate(outputs)]
    if transport==3:
        for k,e in enumerate(selected):
            lo,hi=plan['offsets'][e,0];nbytes=int(plan['jobs'][lo:hi,1].sum())*4
            pci=g.add(2,dma+s.transfer(nbytes,s.pcie_bytes_per_second),outputs[k],0,k);vectors.append((pci,nbytes))
            ready.append(g.add(s.resources-1,0,[pci],expert=k))
    complete=[]
    for k,e in enumerate(selected):
        done=ready[k]
        if cqe:done=g.add(0,cqe,[done],expert=k)
        act=g.add(0,s.activation_ns,[done],1,k);ds=g.add(0,s.host_submit_ns,[act],1,k)
        up=g.add(2,s.transfer(int(plan['input_columns'][e,1]),s.pcie_bytes_per_second),[ds],1,k)
        lo,hi=plan['offsets'][e,1];complete.extend(whole(k,1,d,ds,up) for d in plan['jobs'][lo:hi])
    g.add(0,s.combine_ns,complete,1)
    return g,waves_log,vectors


def occupancy(trace,waves,vectors,transport):
    if not transport:return np.full(16,-1,np.int64),-1
    chips=[[] for _ in range(16)];controller=[]
    for chip,read,out,pci in waves:
        chips[chip].extend([(int(trace[read,3]),128),(int(trace[out,3]),-128)])
        controller.append((int(trace[out,3]),128))
        if pci is not None:controller.append((int(trace[pci,3]),-128))
    controller.extend((int(trace[pci,3]),-nbytes) for pci,nbytes in vectors)
    def peak(events):
        current=maximum=0
        for t,delta in sorted(events):current+=delta;assert current>=0;maximum=max(maximum,current)
        assert current==0;return maximum
    return np.asarray([peak(x) for x in chips],np.int64),peak(controller)


class Native:
    def __init__(self):
        self.binary=OUT/'conditional_stream.so';self.lib=ctypes.CDLL(str(self.binary));self.fn=self.lib.eval_conditional_stream
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.fn.argtypes=[i32]*5+[ctypes.c_int]*6+[ctypes.c_int64]*3+[i64]*5+[ctypes.c_void_p,i64]
    def run(self,plan,ranges,routes,s,mode,transport,cqe,dma,trace=False):
        values=np.zeros(len(routes),np.int64);bounds=values.copy();fifo=np.zeros((len(routes),16),np.int64);controller=values.copy();n=np.zeros(1,np.int64);log=np.zeros((20000 if trace else 1,8),np.int64)
        self.fn(plan['offsets'],plan['jobs'],plan['input_columns'],ranges,np.ascontiguousarray(routes,np.int32),64,len(plan['jobs']),len(routes),6,mode,transport,0,cqe,dma,s.array(),values,bounds,fifo,controller,ctypes.c_void_p(log.ctypes.data) if trace else None,n)
        return values,bounds,fifo,controller,log[:n[0]].copy() if trace else None


def run(copies,stage):
    p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
    for x in p['inputs']:assert sha(x['path'])==x['sha256']
    dest=OUT/f'input{copies}'/f'stage{stage}';dest.mkdir(parents=True,exist_ok=False)
    native=Native();backend=load_python();cost=p['stages'][str(stage)];records=[];checks=[]
    previous=json.loads((PREVIOUS/'screen.json').read_text())
    for c in [c for c in p['candidates'] if c['copies']==copies]:
        with np.load(c['arrays']) as z:rr=z['rows'];ranges=z['ranges'];routes=z['routes']
        for priority in p['schedulers']:
            config={**p['services'][str(copies)],'down_first':priority};s=backend.Service(**config)
            plan=backend.lower_rows(rr[:,0],rr[:,1],2048,[1408]*64+[2816],s)
            for mode in p['modes']:
                for transport in cost['transports']:
                    values,bounds,fifo,controller,_=native.run(plan,ranges,routes,s,mode,transport,cost['cqe_ns'],cost['dma_setup_ns'])
                    if stage==0 and transport==0:
                        old=next(r for r in previous['records'] if r['candidate']==c['id'] and r['priority']==priority and r['mode']==mode)
                        with np.load(old['arrays']) as z:assert np.array_equal(values,z['cost']) and np.array_equal(bounds,z['lower_bound'])
                    for q in [0,511]:
                        graph,waves,vectors=python_graph(backend,plan,ranges,np.r_[routes[q],64],s,mode,transport,cost['cqe_ns'],cost['dma_setup_ns']);direct=graph.run()
                        v,b,f,buf,trace=native.run(plan,ranges,routes[q:q+1],s,mode,transport,cost['cqe_ns'],cost['dma_setup_ns'],True)
                        assert v[0]==direct['cost']==values[q] and np.array_equal(trace,direct['trace'])
                        pf,pb=occupancy(trace,waves,vectors,transport);assert np.array_equal(pf,f[0]) and pb==buf[0]
                        checks.append(dict(candidate=c['id'],priority=priority,mode=mode,transport=transport,query=q,nodes=len(trace),FIFO_max=int(f.max()),controller_GU_bytes=int(buf[0])))
                    name=c['id']+f'.p{priority}.m{mode}.t{transport}'
                    ap=dest/(name+'.npz');np.savez_compressed(ap,rows=rr,ranges=ranges,routes=routes,cost=values,lower_bound=bounds,chip_FIFO_required_bytes=fifo,controller_GU_output_required_bytes=controller)
                    records.append(dict(id=name,candidate=c['id'],profile=c['profile'],ordering=c['ordering'],labels=c['labels'],copies=copies,stage=stage,priority=priority,mode=mode,transport=transport,mean_ns=float(values.mean()),p99_ns=float(np.percentile(values,99)),FIFO_max_required_bytes=int(fifo.max()),controller_GU_max_required_bytes=int(controller.max()),arrays=str(ap)))
            print(c['id'],priority,'complete',flush=True)
    comparisons=[]
    legacy=[r for r in previous['records'] if r['copies']==copies];legacybest=min(legacy,key=lambda r:r['mean_ns'])
    for transport in cost['transports']:
        eligible=[r for r in records if r['transport']==transport]
        original=min([r for r in eligible if r['mode']==0],key=lambda r:r['mean_ns'])
        singleton=min([r for r in eligible if r['mode']==1],key=lambda r:r['mean_ns'])
        fused=min([r for r in eligible if r['mode']==2],key=lambda r:r['mean_ns'])
        same=min([r for r in eligible if r['mode']==1 and r['candidate']==fused['candidate']],key=lambda r:r['mean_ns'])
        strongest=min([original,singleton,legacybest],key=lambda r:r['mean_ns'])
        def gain(base):return dict(mean_pct=100*(1-fused['mean_ns']/base['mean_ns']),p99_pct=100*(1-fused['p99_ns']/base['p99_ns']))
        a=gain(strongest);b=gain(same)
        comparisons.append(dict(transport=transport,original_best=original,singleton_best=singleton,fused_best=fused,same_layout_singleton=same,retained_known_legacy_best=legacybest,strongest_control=strongest,gain_vs_original=gain(original),gain_vs_strongest=a,gain_vs_same_layout_singleton=b,expand_gate=bool((a['mean_pct']>=2 or a['p99_pct']>=5) and (b['mean_pct']>=2 or b['p99_pct']>=5))))
    result=dict(status='conditional_only_not_validated_hardware',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),copies=copies,stage=stage,cost_parameters=cost,records=records,comparisons=comparisons,checks=checks,validation=dict(queries_per_record=512,records=len(records),bridge_queries=sum(512 for r in records if stage==0 and r['transport']==0),python_traces=len(checks),python_nodes=sum(x['nodes'] for x in checks)),conditions=p['conditions'],limits='No hardwareperformance,capacity/ECC/nativeABI claim. Zeroextras areoptimisticupper sensitivity; nonzero uncalibratedonly. No heldout/alllayer/placementrefit.')
    (dest/'screen.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(copies=copies,stage=stage,comparisons=comparisons,validation=result['validation']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);ap.add_argument('--copies',type=int,choices=[1,4]);ap.add_argument('--stage',type=int,choices=[0,1],default=0);a=ap.parse_args()
    freeze() if a.mode=='freeze' else run(a.copies,a.stage)
