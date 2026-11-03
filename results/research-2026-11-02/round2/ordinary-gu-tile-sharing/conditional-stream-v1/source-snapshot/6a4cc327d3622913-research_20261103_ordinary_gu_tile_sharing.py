#!/usr/bin/env python3
"""TRAIN-only strict whole-wave GU tile-sharing work screen; no native-DAG claim."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path

import networkx as nx
import numpy as np

from research_20261103_topology_placement_replay import Replay,load_python,CPP,PYTHON

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'ordinary-gu-tile-sharing/work-screen-v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def planrows(a):
    rr=np.zeros((len(a['offsets']),2,16),np.int64)
    for e in range(len(rr)):
        for p in range(2):
            lo,hi=a['offsets'][e,p]
            for c,n,*_ in a['jobs'][lo:hi]:rr[e,p,c]+=n
    return rr


def freeze():
    assert not OUT.exists();OUT.mkdir(parents=True);(OUT/'source-snapshot').mkdir()
    inputs={};profiles=[];services={}
    def pin(p):inputs[str(p)]=sha(p)
    for copies,campaign in [(1,'full-campaign-v5-W8-training'),(4,'full-campaign-v6-W8-input4')]:
        path=BASE/'ordinary-read-compiler'/campaign/'summary.json';pin(path)
        records=[r for r in json.loads(path.read_text())['records'] if r['model']=='deepseek' and r['layer']==0]
        for r in records:
            if r['command_ns'] not in [250,5000]:continue
            services[f'{copies}.{r["command_ns"]}']=r['service']
            for art in r['artifacts']:
                p=ROOT/art['arrays'];pin(p)
                with np.load(p) as z:rr=planrows(z)
                profiles.append(dict(copies=copies,label=f'original_cmd{r["command_ns"]}_{art["family"]}',rows=rr.tolist(),shape=r['shape'],source=str(p)))
        p=BASE/'shared-counterbalanced/v2'/f'deepseek.L0.input{copies}/result.json';pin(p)
        rec=json.loads(p.read_text())
        for family in ['joint_simple_menu','independent_paired']:
            w=rec['winners'][family];a=w[w['selected']];ap=Path(a['arrays']);pin(ap)
            with np.load(ap) as z:rr=z['rows']
            profiles.append(dict(copies=copies,label='v2_'+family,rows=rr.tolist(),shape=rec['shape'],source=str(ap)))
            if family=='independent_paired':
                rr=rr.copy();rr[:-1,0]=176
                profiles.append(dict(copies=copies,label='equal_gu176_on_v2free',rows=rr.tolist(),shape=rec['shape'],source=str(ap)))
    # Preserve source aliases while avoiding repeated evaluation of identical maps.
    unique={}
    for p in profiles:
        key=(p['copies'],np.asarray(p['rows'],np.int64).tobytes())
        if key not in unique:unique[key]={**p,'aliases':[p['label']]}
        else:unique[key]['aliases'].append(p['label'])
    profiles=list(unique.values())
    for i,p in enumerate(profiles):p['id']=f'input{p["copies"]}.profile{i}'
    train=BASE/'w8-format-training/deepseek/decode.routes.npz';pin(train)
    for p in [Path(__file__),Path(__file__).with_suffix('.cpp'),CPP,PYTHON,ROOT/'scripts/research_20261103_topology_placement_replay.py',BASE/'gu-coalescing/research_note.md',BASE/'gu-coalescing/prior_and_legality_audit.json']:pin(p)
    protocol=dict(status='frozen_before_ordering_fit',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),cpp_sha256=sha(Path(__file__).with_suffix('.cpp')),train_path=str(train),train_sha256=sha(train),profiles=profiles,services=services,inputs=[dict(path=k,sha256=v) for k,v in sorted(inputs.items())],ordering_layouts=['identity_dense','frequency_dense','coactivation_path_dense','matching_dense','matching_blockslots'],selection='TRAIN512 only; no annealing, mask/row-label fit or heldout; deterministic max-weight perfect matching/coactivation path are standard controls',native_contract='C2048 fullrow/32rows per wholewave;6144rows/block; singlecopy stored intervals; fullwave readviews may overlap; singleton rereads charged; fixed logical ownership scatters each needed row once; all inactive/padding/fringe rows drain and drop',cost='Exact independent perchip intervalDP additive native context work, including cmd250/5000, all fullH input copies and all fullwave outputs. Down remains private/fullGU barrier. Not fullFFN DAG latency.',budget='Only Deep L0 W8TRAIN512, input1/4,2cmdprofiles,all frozen baseline geometries same5ordering/layout controls. No nativeDAG expansion until root reviews >=2% possible headroom.',capacity='GU wholewave read regions protected; stored coefficients unique. Program padding rows explicitly and charge pages. Report GU-dedicated blocks and safe GU-tail-filled Down blocks; Down never lies in any possible GU readrange.')
    for item in protocol['inputs']:
        p=Path(item['path'])
        if p.suffix in ['.py','.cpp']:
            dest=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+p.name);shutil.copy2(p,dest);item['snapshot']=str(dest)
    (OUT/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(protocol=str(OUT/'protocol.frozen.json'),sha256=sha(OUT/'protocol.frozen.json'),profiles=len(profiles),source_sha256=sha(__file__))))


def orders(routes):
    chosen=np.concatenate([routes,np.full((len(routes),1),64)],axis=1)
    inc=np.zeros((len(routes),65),np.int64);inc[np.arange(len(routes))[:,None],chosen]=1
    freq=inc.sum(0);co=inc.T@inc
    frequency=sorted(range(65),key=lambda e:(-freq[e],e))
    path=[frequency[0]];left=set(range(65))-set(path)
    while left:
        _,_,e,side=max((int(co[v,path[0] if side==0 else path[-1]]),int(freq[v]),-v,side) for v in left for side in [0,1])
        e=-e;path.insert(0,e) if side==0 else path.append(e);left.remove(e)
    g=nx.Graph();g.add_nodes_from(range(64))
    for i in range(64):
        for j in range(i+1,64):g.add_edge(i,j,weight=int(co[i,j])*1000000+4096-(i*64+j))
    matched=nx.max_weight_matching(g,maxcardinality=True,weight='weight')
    pairs=sorted([tuple(sorted(p)) for p in matched],key=lambda p:(-int(co[p[0],p[1]]),-int(freq[list(p)].sum()),p))
    assert len(pairs)==32 and sorted(e for pair in pairs for e in pair)==list(range(64))
    matching=[e for pair in pairs for e in pair]+[64]
    return dict(identity_dense=list(range(65)),frequency_dense=frequency,coactivation_path_dense=path,matching_dense=matching,matching_blockslots=matching),pairs,co


def layout(rows,order,blockslots=False):
    ranges=np.zeros((65,16,2),np.int32);ends=[]
    for chip in range(16):
        at=0
        groups=[order[i:i+2] for i in range(0,64,2)]+[[64]] if blockslots else [[e] for e in order]
        for group in groups:
            n=sum(int(rows[e,0,chip]) for e in group)
            if blockslots:
                at=(at+31)//32*32
                if at%6144+(n+31)//32*32>6144:at=(at+6143)//6144*6144
            for e in group:
                r=int(rows[e,0,chip]);ranges[e,chip]=[at,at+r];at+=r
        ends.append(at)
    # Dedicated zone and tail-fill ledgers include explicit fullwave padding.
    protected=[];tailfilled=[];dedicated=[];guwaves=[]
    for chip in range(16):
        used={}
        for lo,hi in ranges[:,chip]:
            for at in range(int(lo),int(hi),1):
                b=at//6144;used[b]=max(used.get(b,0),(at-b*6144)//32+1)
        guwords=sum(used.values());guwaves.append(guwords)
        free=[192-used[b] for b in sorted(used)];downwaves=[]
        for e in range(65):
            C=1408 if e<64 else 2816;r=int(rows[e,1,chip])
            if r:downwaves.append((r*C+65535)//65536)
        bins=[]
        for w in downwaves:
            if not bins or bins[-1]<w:bins.append(192)
            bins[-1]-=w
        dedicated.append(len(used)+len(bins))
        bins=free.copy()
        for w in downwaves:
            where=next((i for i,space in enumerate(bins) if space>=w),None)
            if where is None:bins.append(192);where=len(bins)-1
            bins[where]-=w
        tailfilled.append(len(bins));protected.append([used.get(b,0) for b in range(max(used,default=-1)+1)])
    return ranges,dict(gu_protected_wordlines_by_block=protected,gu_programmed_bytes_by_chip=[w*65536 for w in guwaves],dedicated_blocks_by_chip=dedicated,tailfilled_blocks_by_chip=tailfilled,tailfilled_lsb_bytes=sum(tailfilled)*192*65536,useful_GU_bytes=int(rows[:,0].sum()*2048),stored_coefficient_copies=1)


def py_dp(ranges,selected,chip,cmd,copies):
    seg=[]
    for e in selected:
        lo,hi=map(int,ranges[e,chip])
        while lo<hi:
            b=lo//6144;end=min(hi,(b+1)*6144);seg.append((b,lo-b*6144,end-b*6144,int(e)));lo=end
    seg.sort();dispatch=cmd+1024*copies
    single=[0,0,0,0]
    for b,lo,hi,e in seg:
        w=(hi+31)//32-lo//32
        for k,v in enumerate([w*28064+dispatch,w,1,w*32]):single[k]+=v
    dp=[(0,0,0,0)]+[(10**30,0,0,0)]*len(seg);back=[None]*(len(seg)+1)
    for i in range(len(seg)):
        for j in range(i,len(seg)):
            if seg[j][0]!=seg[i][0]:break
            w=(seg[j][2]+31)//32-seg[i][1]//32
            v=tuple(dp[i][k]+x for k,x in enumerate([w*28064+dispatch,w,1,w*32]))
            if v[:3]<dp[j+1][:3]:dp[j+1]=v;back[j+1]=i
    # Fixed ownership: only segments in a DP group scatter, even if fringe
    # covers selected rows belonging to another view; all physical rows drain.
    views=[];j=len(seg);owned=[]
    while j:
        i=back[j];group=seg[i:j];views.append(dict(block=group[0][0],first_wordline=group[0][1]//32,wordlines=(group[-1][2]+31)//32-group[0][1]//32,owners=group));owned.extend(group);j=i
    assert sorted(owned)==seg
    return np.asarray(single+list(dp[-1]),np.int64),views[::-1]


def run():
    protocol=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==protocol['source_sha256']
    for x in protocol['inputs']:assert sha(x['path'])==x['sha256']
    cpp=Path(__file__).with_suffix('.cpp');binary=OUT/'tile_sharing.so';assert not binary.exists()
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));fn=lib.tile_sharing
    i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    fn.argtypes=[i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64]
    with np.load(protocol['train_path']) as z:
        assert z['split'].item()=='train' and z['completed'].all();routes=np.ascontiguousarray(z['routes'][:,:,0].reshape(-1,6),np.int32)
    assert routes.shape==(512,6)
    ordering,pairs,co=orders(routes)
    (OUT/'ordering.fit.json').write_text(json.dumps(dict(orders=ordering,matching_pairs=pairs,matching_unperturbed_edge_weight=sum(int(co[a,b]) for a,b in pairs),networkx_version=nx.__version__,scope='TRAIN only; deterministic standard ordering controls'),indent=2)+'\n')
    native=Replay();backend=load_python();records=[];checks=[];witnesses=[]
    for profile in protocol['profiles']:
        rr=np.asarray(profile['rows'],np.int64);D,I,E,K,L,SI=profile['shape'];copies=profile['copies'];assert (D,I,E,K,SI)==(2048,1408,64,6,2816)
        # Unique expert coefficients retain their original logical row range
        # on each chip, independent of physical expert ordering.
        assert np.all(rr[:-1,0].sum(1)==2816) and rr[-1,0].sum()==5632
        for name,order in ordering.items():
            ranges,capacity=layout(rr,order,name=='matching_blockslots')
            np.savez_compressed(OUT/(profile['id']+'.'+name+'.layout.npz'),ranges=ranges,rows=rr)
            for command in [250,5000]:
                s=protocol['services'][f'{copies}.{command}'];plan=backend.lower_rows(rr[:,0],rr[:,1],D,[I]*E+[SI],backend.Service(**s))
                original,bounds=native.run(plan,routes,s)
                C=np.full(65,I,np.int64);C[-1]=SI;down=rr[:,1];pages=(down*C[:,None]+16383)//16384;waves=(pages+3)//4
                dcc=waves*28000+(down>0)*(command+C[:,None]*copies//2)+2*(pages*16384//C[:,None])
                dload=dcc[routes].sum(1)+dcc[-1]
                a=np.zeros((512,16,8),np.int64);fn(ranges,routes,512,64,6,command,copies,a)
                singleton=(a[:,:,0]+dload).max(1)+s['host_submit_ns'];coal=(a[:,:,4]+dload).max(1)+s['host_submit_ns']
                assert np.all(coal<=singleton)
                ident=profile['id']+'.'+name+'.cmd'+str(command)
                np.savez_compressed(OUT/(ident+'.npz'),routes=routes,ranges=ranges,rows=rr,gu_controls=a,down_context=dload,original_native=original,original_bound=bounds,singleton_joint_bound=singleton,coalesced_joint_bound=coal)
                for q in [0,255,511]:
                    for chip in [0,7,8,15]:
                        expected,views=py_dp(ranges,np.r_[routes[q],64],chip,command,copies);assert np.array_equal(expected,a[q,chip])
                        checks.append(dict(id=ident,query=q,chip=chip))
                        if q==0 and chip==0 and command==250:witnesses.append(dict(id=ident,query=0,chip=0,views=views))
                records.append(dict(id=ident,profile=profile['id'],labels=profile['aliases'],copies=copies,command_ns=command,ordering=name,original_native_mean_ns=float(original.mean()),original_native_p99_ns=float(np.percentile(original,99)),original_bound_mean_ns=float(bounds.mean()),singleton_joint_bound_mean_ns=float(singleton.mean()),coalesced_joint_bound_mean_ns=float(coal.mean()),coalesced_joint_bound_p99_ns=float(np.percentile(coal,99)),same_layout_bound_reduction_pct=100*(1-coal.mean()/singleton.mean()),original_to_coalesced_bound_possible_pct=100*(1-coal.mean()/original.mean()),singleton_GU_waves_mean=float(a[:,:,1].sum(1).mean()),coalesced_GU_waves_mean=float(a[:,:,5].sum(1).mean()),singleton_GU_commands_mean=float(a[:,:,2].sum(1).mean()),coalesced_GU_commands_mean=float(a[:,:,6].sum(1).mean()),coalesced_output_rows_mean=float(a[:,:,7].sum(1).mean()),capacity=capacity))
    comparisons=[]
    for copies in [1,4]:
        for command in [250,5000]:
            eligible=[r for r in records if r['copies']==copies and r['command_ns']==command]
            incumbent=min(eligible,key=lambda r:r['original_native_mean_ns']);best=min(eligible,key=lambda r:r['coalesced_joint_bound_mean_ns'])
            comparisons.append(dict(copies=copies,command_ns=command,strongest_native_incumbent=incumbent['id'],incumbent_mean_ns=incumbent['original_native_mean_ns'],best_DP_bound=best['id'],best_bound_mean_ns=best['coalesced_joint_bound_mean_ns'],possible_headroom_pct=100*(1-best['coalesced_joint_bound_mean_ns']/incumbent['original_native_mean_ns']),not_a_latency_gain=True))
    result=dict(status='bounded_interval_work_screen',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),binary_sha256=sha(binary),queries=512,records=records,comparisons=comparisons,python_DP_checks=len(checks),scope='TRAIN only; perchip additive intervalDP necessary work. Full native resource/dependency DAG not yet implemented; no measured speed or novelty claim.')
    (OUT/'screen.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'DP_witnesses.json').write_text(json.dumps(witnesses,indent=2)+'\n')
    print(json.dumps(dict(comparisons=comparisons,records=len(records),python_DP_checks=len(checks)),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
