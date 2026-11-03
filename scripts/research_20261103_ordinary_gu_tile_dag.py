#!/usr/bin/env python3
"""Bounded command5us ordinary GU sharing sensitivity, no new hardware."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from research_20261103_topology_placement_replay import Replay,load_python,CPP,PYTHON

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
WORK=BASE/'ordinary-gu-tile-sharing/work-screen-v1';OUT=BASE/'ordinary-gu-tile-sharing/native-dag-v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    assert not OUT.exists();OUT.mkdir();(OUT/'source-snapshot').mkdir()
    s=json.loads((WORK/'screen.json').read_text());p=json.loads((WORK/'protocol.frozen.json').read_text());candidates=[]
    for copies in [1,4]:
        eligible=[r for r in s['records'] if r['copies']==copies and r['command_ns']==5000]
        want=[]
        for label in ['v2_independent_paired','v2_joint_simple_menu','original_cmd5000_disjoint','original_cmd5000_cyclic','original_cmd5000_whole_rows','equal_gu176_on_v2free']:
            choices=[r for r in eligible if label in r['labels']]
            assert choices,label
            want.append(min(choices,key=lambda r:r['coalesced_joint_bound_mean_ns']))
        free=[r for r in eligible if 'v2_independent_paired' in r['labels']]
        for name in ['identity_dense','matching_blockslots']:
            want.append(next(r for r in free if r['ordering']==name))
        seen=set()
        for r in want:
            if r['id'] in seen:continue
            seen.add(r['id']);candidates.append(dict(id=r['id'],profile=r['profile'],copies=copies,labels=r['labels'],ordering=r['ordering'],arrays=str(WORK/(r['id']+'.npz'))))
        assert len(seen)<=8
    inputs={x['path']:x['sha256'] for x in p['inputs']}
    for path in [Path(__file__),Path(__file__).with_suffix('.cpp'),CPP,PYTHON,WORK/'screen.json',WORK/'protocol.frozen.json',WORK/'ordering.fit.json']:
        inputs[str(path)]=sha(path)
    for c in candidates:inputs[c['arrays']]=sha(c['arrays'])
    protocol=dict(status='frozen_before_native_DAG_replay',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),cpp_sha256=sha(Path(__file__).with_suffix('.cpp')),candidates=candidates,services={str(c):p['services'][f'{c}.5000'] for c in [1,4]},schedulers=[0,1],modes=[0,1,2],mode_names=['original_bridge','tight_singleton','DP_coalesced'],inputs=[dict(path=k,sha256=v) for k,v in sorted(inputs.items())],candidate_rule='Perinput <=8 static pairs: each v2free/simple,source5k3families,uniform176 best interval-DP ordering,plus freeidentity/matchingblock. No additionalfit/heldout.',contract='Same sealed singlecontext scheduler, native fullwaveGU32row outputs,common fullH PCIe upload once,ONFI inputcopies1/4 everyjob. DPview completion after entire outputPCIe drain is dependency of everyowner expert. No early outputs/context/columnpartials. Down unchanged/private/full-GU A8barrier.',priority='FusedGU priority minselectedordinal,then chip/block/wordline for insertion. Gate/down priority both offered to everymode/family,TRAINmean selects uniformly.',validation='Original bridge all512 equals sealednative; representative independentPython node-by-node fulltrace vsCPP for all3modes,schedulers,candidates. No hidden device/quality assertion.',gate='Only expand if mean>=2% or P99>=5% vsstrongestoriginalnative andsame-layoutsingleton; otherwiseseal. Allresults are command5us sensitivity, notdefault250ns/device measurement.')
    for x in protocol['inputs']:
        path=Path(x['path'])
        if path.suffix in ['.py','.cpp']:
            dest=OUT/'source-snapshot'/(x['sha256'][:16]+'-'+path.name);shutil.copy2(path,dest);x['snapshot']=str(dest)
    (OUT/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),candidates=len(candidates))))


def python_views(ranges,selected,mode,service):
    result=[]
    for chip in range(16):
        parts=[]
        for ordinal,e in enumerate(selected):
            lo,hi=map(int,ranges[e,chip])
            while lo<hi:
                b=lo//6144;end=min(hi,(b+1)*6144);parts.append((b,lo-b*6144,end-b*6144,ordinal));lo=end
        parts.sort();groups=[]
        if mode==1:groups=[(i,i+1) for i in range(len(parts))]
        else:
            dp=[(0,0,0)]+[(10**30,0,0)]*len(parts);previous=[None]*(len(parts)+1)
            for i in range(len(parts)):
                for j in range(i,len(parts)):
                    if parts[j][0]!=parts[i][0]:break
                    w=(parts[j][2]+31)//32-parts[i][1]//32
                    ns=w*service.ordinary_read_ns+service.command_ns+service.transfer(2048*service.input_bytes*service.input_channel_copies,service.channel_bytes_per_second)+service.transfer(w*32*service.output_bytes,service.channel_bytes_per_second)
                    value=(dp[i][0]+ns,dp[i][1]+w,dp[i][2]+1)
                    if value<dp[j+1]:dp[j+1]=value;previous[j+1]=i
            j=len(parts)
            while j:
                i=previous[j];groups.append((i,j));j=i
            groups.reverse()
        for i,j in groups:
            owners=sorted(set(p[3] for p in parts[i:j]));start=parts[i][1]//32;waves=(parts[j-1][2]+31)//32-start
            result.append(dict(chip=chip,block=parts[i][0],start=start,waves=waves,owners=owners,priority=min(owners)))
    result.sort(key=lambda v:(v['priority'],v['chip'],v['block'],v['start'],v['waves']))
    return result


def python_graph(backend,plan,ranges,selected,service,mode):
    if mode==0:return backend.ReferenceGraph.build(plan,selected,service)
    g=backend.ReferenceGraph(service);submit=g.add(0,service.host_submit_ns)
    upload=g.add(2,service.transfer(2048*service.input_bytes,service.pcie_bytes_per_second),[submit])
    outputs=[[] for _ in selected]
    for v in python_views(ranges,selected,mode,service):
        d=[v['chip'],v['waves']*32,2048,v['waves']*4,v['waves']]
        done=g.matrix(v['priority'],0,d,submit,upload)
        for ordinal in v['owners']:outputs[ordinal].append(done)
    ready=[g.add(service.resources-1,0,outs,expert=ordinal) for ordinal,outs in enumerate(outputs)]
    complete=[]
    for ordinal,e in enumerate(selected):
        act=g.add(0,service.activation_ns,[ready[ordinal]],1,ordinal)
        ds=g.add(0,service.host_submit_ns,[act],1,ordinal)
        up=g.add(2,service.transfer(int(plan['input_columns'][e,1])*service.input_bytes,service.pcie_bytes_per_second),[ds],1,ordinal)
        lo,hi=plan['offsets'][e,1]
        complete.extend(g.matrix(ordinal,1,d,ds,up) for d in plan['jobs'][lo:hi])
    g.add(0,service.combine_ns,complete,1)
    return g


class Native:
    def __init__(self):
        source=Path(__file__).with_suffix('.cpp');self.binary=OUT/'tile_dag.so'
        assert not self.binary.exists()
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(self.binary)],check=True)
        self.lib=ctypes.CDLL(str(self.binary));self.fn=self.lib.eval_gu_tile_dag
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.fn.argtypes=[i32,i32,i32,i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64,i64,i64,ctypes.c_void_p,i64]

    def run(self,plan,ranges,routes,s,mode,trace=False):
        values=np.zeros(len(routes),np.int64);bounds=values.copy();n=np.zeros(1,np.int64)
        log=np.zeros((20000,8),np.int64) if trace else np.zeros((1,8),np.int64)
        self.fn(plan['offsets'],plan['jobs'],plan['input_columns'],ranges,np.ascontiguousarray(routes,np.int32),64,len(plan['jobs']),len(routes),6,mode,s.array(),values,bounds,ctypes.c_void_p(log.ctypes.data) if trace else None,n)
        assert n[0]<=len(log) if trace else True
        return values,bounds,log[:n[0]].copy() if trace else None


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
    for x in p['inputs']:assert sha(x['path'])==x['sha256']
    native=Native();old=Replay();backend=load_python();records=[];checks=[]
    for c in p['candidates']:
        with np.load(c['arrays']) as z:rr=z['rows'];routes=z['routes'];ranges=z['ranges']
        for priority in p['schedulers']:
            config={**p['services'][str(c['copies'])],'down_first':priority};s=backend.Service(**config)
            plan=backend.lower_rows(rr[:,0],rr[:,1],2048,[1408]*64+[2816],s)
            original,_=old.run(plan,routes,config)
            for mode in p['modes']:
                values,bounds,_=native.run(plan,ranges,routes,s,mode)
                if mode==0:assert np.array_equal(values,original)
                for q in [0,511]:
                    got=python_graph(backend,plan,ranges,np.r_[routes[q],64],s,mode).run()
                    v,b,trace=native.run(plan,ranges,routes[q:q+1],s,mode,True)
                    assert v[0]==got['cost']==values[q]
                    assert np.array_equal(trace,got['trace']),(c['id'],priority,mode,q)
                    checks.append(dict(id=c['id'],priority=priority,mode=mode,query=q,nodes=len(trace),cost_ns=int(v[0])))
                ident=c['id']+'.priority'+str(priority)+'.mode'+str(mode)
                np.savez_compressed(OUT/(ident+'.npz'),rows=rr,ranges=ranges,routes=routes,cost=values,lower_bound=bounds)
                records.append(dict(id=ident,candidate=c['id'],labels=c['labels'],profile=c['profile'],ordering=c['ordering'],copies=c['copies'],priority=priority,mode=mode,mean_ns=float(values.mean()),p99_ns=float(np.percentile(values,99)),arrays=str(OUT/(ident+'.npz'))))
            print(c['id'],'priority',priority,'complete',flush=True)
    comparisons=[]
    for copies in [1,4]:
        eligible=[r for r in records if r['copies']==copies]
        original=min([r for r in eligible if r['mode']==0],key=lambda r:r['mean_ns'])
        singleton=min([r for r in eligible if r['mode']==1],key=lambda r:r['mean_ns'])
        fused=min([r for r in eligible if r['mode']==2],key=lambda r:r['mean_ns'])
        same=min([r for r in eligible if r['mode']==1 and r['candidate']==fused['candidate']],key=lambda r:r['mean_ns'])
        samerule=next(r for r in eligible if r['mode']==1 and r['candidate']==fused['candidate'] and r['priority']==fused['priority'])
        def gain(base):return dict(mean_pct=100*(1-fused['mean_ns']/base['mean_ns']),p99_pct=100*(1-fused['p99_ns']/base['p99_ns']))
        a=gain(original);b=gain(same)
        comparisons.append(dict(copies=copies,original_best=original,singleton_best=singleton,fused_best=fused,same_layout_singleton=same,same_scheduler_singleton=samerule,gain_vs_original=a,gain_vs_same_layout_singleton=b,expand_gate=bool((a['mean_pct']>=2 or a['p99_pct']>=5) and (b['mean_pct']>=2 or b['p99_pct']>=5))))
    result=dict(status='bounded_native_DAG_sensitivity',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),binary_sha256=sha(native.binary),records=records,comparisons=comparisons,checks=checks,validation=dict(original_bridge_queries=len(p['candidates'])*2*512,python_exact_traces=len(checks),python_exact_nodes=sum(x['nodes'] for x in checks)),limits='TRAIN512 only,cmd5000 sensitivity; whole-view PCIe barrier and fullGU A8 barrier retained; no new native numerical ABI or device measurement;allfamilies share scheduler selection.')
    (OUT/'screen.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(comparisons=comparisons,validation=result['validation']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
