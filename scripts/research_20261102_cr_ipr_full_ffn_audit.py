#!/usr/bin/env python3
"""Independent scalar full-FFN event replay, semantics, bytes and FIFO audit.

Saved nodes/edges are the input. The target Python audit/run/executor are not
called. Scheduling uses a global pending set plus timed completions, and FIFO
occupancy uses the integral of each transfer at every arrival timestamp.
"""
from collections import Counter,defaultdict
import copy
import ctypes
import datetime
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py'
CPP=SOURCE.with_suffix('.cpp')
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr/full-ffn'
OUT=ROOT/'results/research-2026-11-02/round3-cr-ipr/full-ffn-independent'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def scalar_execute(nodes,down_first):
    """Global pending vertices, resource leases and timed completion heap."""
    N=len(nodes)
    waiting=set()
    unfinished=[]
    children=[[] for _ in nodes]
    degree=[len(n['deps']) for n in nodes]
    start=[-1]*N
    end=[-1]*N
    busy={}
    owner={}
    now=0
    completed=0
    def activate(i):
        if nodes[i]['resource']<0:
            start[i]=now
            heapq.heappush(unfinished,(now+nodes[i]['duration'],i))
        else:
            waiting.add(i)
    for i,n in enumerate(nodes):
        require(n['index']==i and n['duration']>=0,'node geometry')
        for dep in n['deps']:
            require(0<=dep<i,'non-topological dependency')
            children[dep].append(i)
    for i,d in enumerate(degree):
        if d==0:
            activate(i)
    while completed<N:
        # Choose one eligible vertex for every currently free scalar resource.
        # Candidate selection is global; no target per-resource queues are used.
        candidates={}
        for i in waiting:
            n=nodes[i]
            r=n['resource']
            if r in busy:
                continue
            if n['acquire'] and n['chip'] in owner:
                continue
            key=(1-n['phase'] if down_first else n['phase'],n['expert_rank'],i)
            if r not in candidates or key<candidates[r][0]:
                candidates[r]=(key,i)
        for r,(key,i) in sorted(candidates.items()):
            n=nodes[i]
            waiting.remove(i)
            if n['job']>=0:
                if n['acquire']:
                    require(n['chip'] not in owner,'chip acquired twice')
                    owner[n['chip']]=n['job']
                else:
                    require(owner.get(n['chip'])==n['job'],'job used chip before acquisition')
            busy[r]=i
            start[i]=now
            heapq.heappush(unfinished,(now+n['duration'],i))
        require(bool(unfinished),'scheduler deadlock')
        now=unfinished[0][0]
        # Process all same-time completions, including zero-duration causal
        # timers activated by a completion, before dispatching the next epoch.
        while unfinished and unfinished[0][0]==now:
            _,i=heapq.heappop(unfinished)
            n=nodes[i]
            end[i]=now
            if n['resource']>=0:
                require(busy.pop(n['resource'])==i,'wrong resource completion')
            if n['release']:
                require(owner.pop(n['chip'])==n['job'],'wrong chip owner release')
            completed+=1
            for j in children[i]:
                degree[j]-=1
                if degree[j]==0:
                    activate(j)
    require(not waiting and not busy and not owner,'unfinished resource/owner state')
    return np.column_stack([start,end]).astype(np.int64),now


def overlap_free(rr):
    v=sorted((a,b) for a,b in rr if b>a)
    return all(b<=c for (a,b),(c,d) in zip(v,v[1:]))


def fifo_integral(arrivals,transfers):
    """Exact continuous 2B/ns drain, evaluated at all production instants."""
    def occupancy(t):
        produced=sum(n for at,n in arrivals if at<=t)
        drained=sum(min(n,max(0,2*(t-a))) for a,b,n in transfers)
        return produced-drained
    points={t for t,n in arrivals}|{a for a,b,n in transfers}|{b for a,b,n in transfers}
    values=[(t,occupancy(t)) for t in sorted(points)]
    require(all(v>=0 for t,v in values),'FIFO underflow / drain before production')
    require(values[-1][1]==0,'FIFO final byte conservation')
    peak=max(v for t,v in values)
    return peak,values


def semantic_audit(record,trace):
    ns=record['nodes']
    jobs=record['jobs']
    bykind=defaultdict(list)
    resources=defaultdict(list)
    byjob=defaultdict(list)
    for i,n in enumerate(ns):
        bykind[n['kind']].append(i)
        require(trace[i,1]-trace[i,0]==n['duration'],'duration mismatch')
        require(all(trace[d,1]<=trace[i,0] for d in n['deps']),'dependency violation')
        if n['resource']>=0:
            resources[n['resource']].append(tuple(map(int,trace[i])))
        if n['job']>=0:
            byjob[n['job']].append(i)
    require(all(overlap_free(rr) for rr in resources.values()),'exclusive resource overlap')
    require(len(bykind['GU_submit'])==len(bykind['GU_input_PCIe'])==len(bykind['weighted_combine'])==1,'global GU/combine count')
    require(len(bykind['full_GU_at_host'])==len(bykind['full_vector_activation_A8'])==len(bykind['Down_submit'])==len(bykind['Down_input_PCIe'])==7,'seven full private vectors')
    service=record['service']
    gu_submit=bykind['GU_submit'][0]
    gu_upload=bykind['GU_input_PCIe'][0]
    require(ns[gu_upload]['deps']==[gu_submit] and ns[gu_upload]['bytes']==1536 and ns[gu_upload]['duration']==192,'one complete shared GU PCIe input')
    gu_outputs=defaultdict(set)
    down_outputs=defaultdict(set)
    rank_gu_bytes=Counter()
    rank_down_bytes=Counter()
    arrivals=defaultdict(list)
    transfers=defaultdict(list)
    owner_intervals=defaultdict(list)
    post_release_pcie=0
    pipeline_counts=Counter()
    input_onfi_bytes=0
    emitted_bytes=0
    for j,job in enumerate(jobs):
        cmd,inp,comp,release=(job[k] for k in ['command','input','compute','release'])
        chip,phase,columns=(job[k] for k in ['chip','phase','columns'])
        require(ns[cmd]['acquire'] and ns[cmd]['resource']==1 and ns[cmd]['duration']==service['command_ns'],'command ownership/service')
        require(ns[release]['release'] and ns[release]['resource']==-1,'resource-free ownership release')
        require(ns[inp]['resource']==3+chip%8 and ns[comp]['resource']==11+chip,'chip/channel resource mapping')
        require(ns[inp]['bytes']==columns*service['input_copies'] and ns[inp]['duration']==(ns[inp]['bytes']+1)//2,'full input ONFI byte/service count')
        require(ns[comp]['deps']==[inp] and trace[comp,0]==trace[inp,1],'compute waits for full input')
        require(set(ns[release]['deps'])=={comp,*job['output_ONFI']},'release excludes compute or ONFI output')
        require(len(job['output_ONFI'])==len(set(job['output_ONFI'])),'duplicated job output')
        begin,finish=int(trace[cmd,0]),int(trace[release,1])
        owner_intervals[chip].append((begin,finish))
        for i in byjob[j]:
            require(begin<=trace[i,0]<=trace[i,1]<=finish,'node outside chip owner lifetime')
        if phase==0:
            require(set(ns[cmd]['deps'])=={gu_submit} and set(ns[inp]['deps'])=={cmd,gu_upload},'GU command/full-upload dependency')
        else:
            rank=ns[cmd]['expert_rank']
            require(columns==(1024 if rank==6 else 512),'private Down input dimension')
            ds=next(i for i in bykind['Down_submit'] if ns[i]['expert_rank']==rank)
            upload=next(i for i in bykind['Down_input_PCIe'] if ns[i]['expert_rank']==rank)
            require(ns[cmd]['deps']==[ds] and set(ns[inp]['deps'])=={cmd,upload},'Down command/full-private-upload dependency')
        job_bytes=0
        for onfi in job['output_ONFI']:
            o=ns[onfi]
            require(o['kind']=='output_ONFI' and o['job']==j and o['resource']==3+chip%8,'job ONFI ownership')
            require(len(o['deps'])==1,'output production dependency')
            produce=o['deps'][0]
            pn=ns[produce]
            require(pn['kind']=='row_produced' and pn['resource']==-1 and pn['deps']==[inp],'nonblocking row completion timer')
            require(pn['duration']<=ns[comp]['duration'],'output produced after recipe completion')
            n=o['bytes']
            require(n==pn['bytes'] and n>0 and n%4==0 and o['duration']==n//2,'row bytes/ONFI service')
            pcs=[i for i in bykind['output_PCIe'] if ns[i]['deps']==[onfi]]
            require(len(pcs)==1,'exactly one PCIe transfer per ONFI output')
            pc=pcs[0]
            require(ns[pc]['resource']==2 and ns[pc]['job']==-1 and ns[pc]['chip']==-1,'PCIe incorrectly retains chip owner')
            require(ns[pc]['bytes']==n and ns[pc]['duration']==(n+7)//8,'PCIe output byte/service count')
            post_release_pcie+=int(trace[pc,1]>finish)
            rank=o['expert_rank']
            if phase==0:
                gu_outputs[rank].add(pc)
                rank_gu_bytes[rank]+=n
            else:
                down_outputs[rank].add(pc)
                rank_down_bytes[rank]+=n
            arrivals[chip].append((int(trace[produce,1]),n))
            transfers[chip].append((int(trace[onfi,0]),int(trace[onfi,1]),n))
            job_bytes+=n
        require(job_bytes==(job['pipeline']['bytes']//columns)*4,'native emitted-row/fringe byte conservation')
        emitted_bytes+=job_bytes
        input_onfi_bytes+=ns[inp]['bytes']
        for k in ['waves','pages','bytes','physical_read_pages','physical_read_bytes','ecc_pages','ecc_bytes','cold_waves','cr_waves']:
            pipeline_counts[k]+=job['pipeline'][k]
    require(all(overlap_free(rr) for rr in owner_intervals.values()),'overlapping chip job owners')
    for rank in range(7):
        ready=next(i for i in bykind['full_GU_at_host'] if ns[i]['expert_rank']==rank)
        act=next(i for i in bykind['full_vector_activation_A8'] if ns[i]['expert_rank']==rank)
        ds=next(i for i in bykind['Down_submit'] if ns[i]['expert_rank']==rank)
        up=next(i for i in bykind['Down_input_PCIe'] if ns[i]['expert_rank']==rank)
        require(set(ns[ready]['deps'])==gu_outputs[rank] and gu_outputs[rank],'full GU missing chip-shard PCIe output')
        require(ns[act]['deps']==[ready] and ns[act]['resource']==0 and ns[act]['duration']==service['activation_ns'],'whole-vector host activation/A8 barrier')
        require(ns[ds]['deps']==[act] and ns[ds]['resource']==0 and ns[ds]['duration']==service['submit_ns'],'private Down submission')
        expected=1024 if rank==6 else 512
        require(ns[up]['deps']==[ds] and ns[up]['bytes']==expected and ns[up]['duration']==expected//8 and ns[up]['resource']==2,'complete private Down PCIe vector')
        require(rank_gu_bytes[rank]>=(2048 if rank==6 else 1024)*4,'logical full GU rows missing')
        require(rank_down_bytes[rank]>=1536*4,'logical full Down rows missing')
    combine=bykind['weighted_combine'][0]
    require(set(ns[combine]['deps'])==set().union(*down_outputs.values()) and set(down_outputs)==set(range(7)),'combine missing one of seven full Down outputs')
    require(ns[combine]['resource']==0 and ns[combine]['duration']==service['combine_ns'],'combine host service')
    require(sum(ns[i]['bytes'] for i in bykind['Down_input_PCIe'])==4096,'private A8 PCIe byte total')
    require(sum(ns[i]['bytes'] for i in bykind['input_ONFI'])==input_onfi_bytes,'ONFI input byte total')
    require(sum(ns[i]['bytes'] for i in bykind['row_produced'])==sum(ns[i]['bytes'] for i in bykind['output_ONFI'])==sum(ns[i]['bytes'] for i in bykind['output_PCIe'])==emitted_bytes,'produce/ONFI/PCIe output conservation')
    peaks=[]
    for chip in range(16):
        v,_=fifo_integral(arrivals[chip],transfers[chip])
        peaks.append(v)
    require(peaks==record['audit']['peak_output_fifo_bytes_per_chip'],'independent continuous-drain FIFO peak')
    require(record['audit']['valid_FIFO_capacities']=={str(k):max(peaks)<=k for k in [64,128,256,2048]},'FIFO budget validity flags')
    require(dict(pipeline_counts)==record['audit']['pipeline_counts'],'job/pipeline counter conservation')
    counts=Counter()
    for n in ns:
        counts[n['kind']]+=1
        counts[n['kind']+'_bytes']+=n['bytes']
    require(dict(counts)==record['audit']['counts'],'saved event counts and bytes')
    return dict(nodes=len(ns),edges=sum(len(n['deps']) for n in ns),jobs=len(jobs),
                peak_FIFO_bytes=max(peaks),FIFO128_valid=max(peaks)<=128,output_bytes=emitted_bytes,
                input_ONFI_bytes=input_onfi_bytes,input_PCIe_bytes=1536+4096,
                GU_emitted_rows=sum(rank_gu_bytes.values())//4,Down_emitted_rows=sum(rank_down_bytes.values())//4,
                logical_GU_rows=8192,logical_Down_rows=10752,
                post_owner_release_PCIe_completions=post_release_pcie,
                timer_nodes=len([n for n in ns if n['resource']==-1]),
                pipeline_counts=dict(pipeline_counts))


def load_target_for_cache_probe():
    # This import is confined to testing the target cache as an object under
    # test. Neither its audit nor its scheduler/executor is used by the oracle.
    spec=importlib.util.spec_from_file_location('full_ffn_cache_object_under_test',SOURCE)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m


def block_cache_probe():
    m=load_target_for_cache_probe()
    pipeline=m.load_pipeline()
    waves=[]
    for wi in range(2):
        pp=[dict(page_id=f'x.p{wi*4+j}',owner='x',page_index=wi*4+j,plane=j,block=0,wl=wi,bytes=16384) for j in range(4)]
        waves.append(dict(pages=pp,source_order=list(range(4))))
    changed=copy.deepcopy(waves)
    for p in changed[1]['pages']:
        p['block']=1
    cache=m.Recipes(pipeline)
    ordinary=cache.make(waves,'strict100',1536,{'x':0},validate=False)
    stale=cache.make(changed,'strict100',1536,{'x':0},validate=False)
    fresh=m.Recipes(pipeline).make(changed,'strict100',1536,{'x':0},validate=False)
    require(ordinary['duration']==48580 and fresh['duration']==66440,'block probe manual cold/CR time oracle')
    require(stale['duration']!=fresh['duration'] and stale is ordinary,'expected block-key collision no longer reproduces; review erratum')
    return dict(kind='recipe_cache_omits_block_transition',severity='general_API_correctness_main_single_block_unaffected',
                source=str(SOURCE),same_block_duration_ns=ordinary['duration'],cross_block_cached_ns=stale['duration'],
                cross_block_fresh_ns=fresh['duration'],colliding_signature=ordinary['signature'],waves_same_block=waves,waves_cross_block=changed,
                main_scope='Main G4 templates use block0 throughout. This falsification does not establish bias in the saved single-block campaign.',
                proposed_followup='A future version should include block-transition/physical-read identity in its recipe cache key. Frozen source is not edited.')


def micro_oracles(binary):
    lib=ctypes.CDLL(str(binary))
    cpp=lib.run_graph
    a64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    a32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    cpp.argtypes=[ctypes.c_int,a64,a32,a32,ctypes.c_int,a64]
    cpp.restype=ctypes.c_int64
    def node(i,r,d,deps=(),chip=-1,job=-1,acquire=False,release=False,expert=-1):
        return dict(index=i,resource=r,duration=d,deps=list(deps),phase=0,expert_rank=expert,
                    chip=chip,job=job,acquire=acquire,release=release,kind='manual',bytes=0)
    timers=[node(0,-1,5),node(1,-1,7),node(2,0,3,[0]),node(3,0,2,[1])]
    owner=[node(0,1,1,chip=0,job=0,acquire=True,expert=0),
           node(1,3,1,[0],chip=0,job=0),node(2,11,5,[1],chip=0,job=0),
           node(3,-1,2,[1],chip=0,job=0),node(4,3,2,[3],chip=0,job=0),
           node(5,2,100,[4]),node(6,-1,0,[2,4],chip=0,job=0,release=True),
           node(7,1,1,chip=0,job=1,acquire=True,expert=1),
           node(8,11,1,[7],chip=0,job=1),node(9,-1,0,[8],chip=0,job=1,release=True)]
    results=[]
    for name,ns,expected_cost in [('resource_free_timers',timers,10),('owner_release_excludes_PCIe',owner,106)]:
        own,cost=scalar_execute(ns,0)
        n=np.ascontiguousarray([[x['resource'],x['duration'],x['phase'],x['expert_rank'],x['chip'],x['job'],x['acquire'],x['release']] for x in ns],dtype=np.int64)
        offsets=np.r_[0,np.cumsum([len(x['deps']) for x in ns])].astype(np.int32)
        deps=np.ascontiguousarray([d for x in ns for d in x['deps']],dtype=np.int32)
        trace=np.zeros((len(ns),2),dtype=np.int64)
        cpp_cost=int(cpp(len(ns),n,offsets,deps,0,trace))
        require(cost==cpp_cost==expected_cost and np.array_equal(own,trace),'manual executor oracle '+name)
        if name=='resource_free_timers':
            require(own.tolist()==[[0,5],[0,7],[5,8],[8,10]],'timers were serialized')
        else:
            require(own[7,0]==7 and own[5,1]==106,'PCIe incorrectly holds chip owner')
        results.append(dict(name=name,cost_ns=cost,trace=own.tolist(),cpp_exact=True))
    v,history=fifo_integral([(0,64),(16,64)],[(0,32,64),(32,64,64)])
    require(v==96,'continuous FIFO manual integral')
    results.append(dict(name='continuous_FIFO_instead_of_instant_start_clear',peak_bytes=v,
                        instant_clear_would_report_bytes=64,FIFO64_valid=False,FIFO128_valid=True,
                        history=history))
    return results


def corruption_controls(baseline):
    controls=[]
    def omit_shard(r):
        next(n for n in r['nodes'] if n['kind']=='full_GU_at_host')['deps'].pop()
    def short_private(r):
        n=next(n for n in r['nodes'] if n['kind']=='Down_input_PCIe')
        n['bytes']//=2
    def early_owner(r):
        n=next(n for n in r['nodes'] if n['kind']=='owner_release')
        n['deps'].pop()
    def omit_shared(r):
        n=next(n for n in r['nodes'] if n['kind']=='weighted_combine')
        dep=next(i for i in n['deps'] if r['nodes'][i]['expert_rank']==6)
        n['deps'].remove(dep)
    def timer_busy(r):
        n=next(n for n in r['nodes'] if n['kind']=='row_produced')
        n['resource']=11+n['chip']
    for name,mutate in [('missing_GU_chip_shard',omit_shard),('short_private_Down_input',short_private),
                        ('owner_release_without_all_ONFI',early_owner),('combine_missing_shared_Down',omit_shared),
                        ('timer_claims_compute_resource',timer_busy)]:
        r=copy.deepcopy(baseline)
        mutate(r)
        trace=np.array([(n['start_ns'],n['end_ns']) for n in r['nodes']],dtype=np.int64)
        try:
            semantic_audit(r,trace)
        except RuntimeError as e:
            controls.append(dict(name=name,detected=True,diagnostic=str(e)))
        else:
            raise RuntimeError('undetected corruption '+name)
    return controls


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    files=sorted((BASE/'smoke').glob('trace.*json'))
    require(len(files)==88,'expected all 88 frozen primary smoke configurations')
    binary=Path(json.loads((BASE/'smoke/run.frozen.json').read_text())['executor']['binary'])
    paths=[SOURCE,CPP,binary,BASE/'execution_plan.frozen.json',BASE/'smoke/run.frozen.json',BASE/'smoke/report.json',*files]
    hashes={str(p):sha(p) for p in paths}
    plan=json.loads((BASE/'execution_plan.frozen.json').read_text())
    require(plan['inputs_sha256'][str(SOURCE)]==sha(SOURCE) and plan['inputs_sha256'][str(CPP)]==sha(CPP),'source freeze mismatch')
    (OUT/'source-snapshot.py').write_bytes(SOURCE.read_bytes())
    (OUT/'source-snapshot.cpp').write_bytes(CPP.read_bytes())
    erratum=block_cache_probe()
    write(OUT/'erratum.recipe-cache-block.json',erratum)
    records=[]
    replay_arrays={}
    for k,p in enumerate(files):
        record=json.loads(p.read_text())
        independent,cost=scalar_execute(record['nodes'],record['service']['down_first'])
        expected=np.array([(n['start_ns'],n['end_ns']) for n in record['nodes']],dtype=np.int64)
        require(np.array_equal(independent,expected),'independent node start/end mismatch in '+p.name)
        require(cost==record['cost_ns'],'independent makespan mismatch')
        summary=semantic_audit(record,independent)
        records.append(dict(path=str(p),sha256=hashes[str(p)],cost_ns=cost,**summary))
        replay_arrays[p.name]=independent
        if k%16==0:
            print('AUDITED',k+1,'/',len(files),flush=True)
    np.savez_compressed(OUT/'independent_node_times.npz',**replay_arrays)
    manual=micro_oracles(binary)
    corruptions=corruption_controls(json.loads(files[0].read_text()))
    require(hashes=={p:sha(p) for p in hashes},'frozen inputs changed during audit')
    report=dict(status='pass_with_scoped_API_erratum',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                audit_source_sha256=sha(__file__),inputs_sha256=hashes,records=records,errata=[erratum],
                total_nodes=sum(r['nodes'] for r in records),total_edges=sum(r['edges'] for r in records),
                total_jobs=sum(r['jobs'] for r in records),all_node_start_end_times_exact=True,
                all_dependency_resource_owner_semantics_checked=True,
                FIFO128_valid_cases=sum(r['FIFO128_valid'] for r in records),
                independent_node_times_sha256=sha(OUT/'independent_node_times.npz'),
                manual_oracles=manual,corruption_controls=corruptions,
                actual_command=['python',str(Path(__file__).resolve())],elapsed_seconds=time.monotonic()-began,
                limitations=[
                    'The trace audit verifies the saved conditional DAG and scheduler; it does not establish measured native hardware timing or arithmetic/task quality.',
                    '6.4GB/s is an explicit ECC service cap and an implicit assumption that the PE can sustain the matching W8 MAC stream; no multiplier-lane or MAC-throughput hardware proof is supplied.',
                    '28us cold first availability is service accounting for published tPRE+tEVAL+tDISCH; the numerical first-phase/final-discharge split is unknown.',
                    'Skip-WL CR and IPR are user premises. Programmable source order, per-plane cache copies and before-ECC dropping of keep-hot overfetch remain conditional.',
                    'Host row scales, positional scatter/fringe removal, full-vector A8 activation at declared0/2us, and zero-cost weighted combine are model assumptions.',
                    'Down and private GU recipes reuse block0/WL templates for timing. They do not form a unique-address resident physical-layout certificate across jobs/experts.',
                    'Graph caching preserves deterministic node/edge service identity, not arbitrary physical-address identity. A block-omission recipe-cache counterexample is retained; current single-block main inputs are unaffected.',
                    'Sorted active IDs set this service model scheduling order. They can differ from original router rank/fit order; this replay performs no arithmetic and does not prove numerical routing-order invariance.',
                    'FIFO validity is posthoc. Above a stated budget the unconstrained trace is invalid; no hidden backpressure or larger FIFO is granted.',
                    'Only 88 saved primary smoke traces are replayed here, rather than every query of the running TRAIN512 campaign. No new heldout/quality data is opened.'
                ])
    write(OUT/'report.json',report)
    (OUT/'report.md').write_text('Independent full-FFN replay: all88 saved smoke configurations match every C++ node start/end and makespan. '+
        f"{report['total_nodes']:,} nodes, {report['total_edges']:,} edges and {report['total_jobs']:,} chip jobs were checked. "+
        f"FIFO128 is valid for {report['FIFO128_valid_cases']}/88 traces under exact continuous2B/ns ONFI drain.\n\n"+
        'A separate scalar pending-vertex/time-heap executor checks causal timers without busy resources and holds each chip owner from command acquisition until compute plus all ONFI results finish. PCIe does not retain chip ownership. Full GU shards must arrive at the host before whole-vector activation/A8 and full private Down upload; all seven Down outputs precede combine. Produce/ONFI/PCIe bytes and padded native emitted rows are conserved.\n\n'+
        'Scoped API erratum: the recipe key omits block transitions. Two full waves in one block cost48580ns at strict100; changing the second to block1 returns stale48580ns from the populated cache, while a fresh cache costs66440ns. Current main G4 templates all use block0, so this counterexample does not change the audited primary campaign. No sealed source was edited.\n\n'+
        'Three manual oracles confirm concurrent resource-free timers, chip release before delayed PCIe completes, and a continuous-drain FIFO peak96B where an incorrect instant-start-clear model would report64B. Five semantic corruptions (missing GU shard, shortened private Down vector, early ownership release, missing shared Down at combine, and a timer occupying compute) are all detected.\n\n'+
        '\n'.join(report['limitations'])+'\n\nReproduce: `python '+str(Path(__file__).resolve())+'`\n')
    print(json.dumps(dict(status=report['status'],traces=len(records),nodes=report['total_nodes'],edges=report['total_edges'],jobs=report['total_jobs'],FIFO128_valid=report['FIFO128_valid_cases'],elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':
    main()
