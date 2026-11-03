#!/usr/bin/env python3
"""Fixed0123 phase copies versus equal-byte standard whole-GU group copies."""
from pathlib import Path
from collections import Counter,defaultdict
from copy import deepcopy
import argparse,gzip,hashlib,importlib.util,json,sys,time
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'scripts/research_20261102_cr_ipr_full_ffn.py').exists())
BASE=OUT.parent
REPLICA=BASE/'down-replica-full-ffn'
BUILDERS=BASE/'gu-phase-replica'
sys.path.insert(0,str(BUILDERS))
sys.path.insert(0,str(REPLICA))
import replica_full_ffn as previous
import phase_builder as phase
import group_builder as group
f=previous.f
CASES={'phase_triples':('phase','triple_private_shared4'),
       'phase_onecopy':('phase','onecopy_shared4'),
       'group_triples':('group','triple_private_shared4')}
REFERENCE=REPLICA/'TRAIN512'


def get_inputs():
    routes,_,_,_=previous.inputs()
    old=f.read(REPLICA/'Down_job_maps.frozen.json');maps={}
    for case,(_,down) in CASES.items():
        maps[case]=deepcopy(old['candidate.'+down])
        for jobs in maps[case]:
            for job in jobs:
                if job['expert']==64:
                    job.update(block=0,wl_start=186)
    return routes,maps


def make_ledger(maps):
    records=[]
    for case,(kind,down) in CASES.items():
        for layer in range(3):
            chips=[];total=0
            for chip in range(16):
                layout=phase.build_layout() if kind=='phase' else group.build_layout(chip//8)
                gu=layout['resident_pages'];assert len(gu)==744
                jobs=[j for j in maps[case][layer] if j['chip']==chip]
                occupied={(p['block'],p['plane'],p['wl']) for p in gu};assert len(occupied)==744
                for job in jobs:
                    for j in range(job['pages']):
                        key=(job['block'],j%4,job['wl_start']+j//4)
                        assert key not in occupied;occupied.add(key)
                assert all(0<=b<2 and 0<=p<4 and 0<=w<192 for b,p,w in occupied)
                total+=len(occupied)*16384
                chips.append(dict(chip=chip,GU=gu,Down_jobs=jobs,
                    GU_row_identity='Each expert page source is original W8 GU tensor, identical copies.',
                    routed_GU_chip_span=16 if kind=='phase' else 8,
                    routed_rows_per_chip=64 if kind=='phase' else 128,
                    routed_row_start=(chip if kind=='phase' else chip%8)*(64 if kind=='phase' else 128),
                    shared_GU_chip_span=16,shared_rows_per_chip=128,shared_row_start=chip*128,
                    nonempty_blocks=len({b for b,p,w in occupied}),programmed_bytes=len(occupied)*16384))
            expected=155713536+91226112+(100663296 if down=='triple_private_shared4' else 0)
            assert total==expected
            records.append(dict(case=case,layer=layer,programmed_bytes=total,
                extra_GU_bytes=91226112,extra_Down_bytes=100663296 if down=='triple_private_shared4' else 0,
                allocated_budget_blocks=32,nonempty_blocks=sum(c['nonempty_blocks'] for c in chips),chips=chips))
    return records


def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.frozen.json').exists()
    _,maps=get_inputs();ledger=make_ledger(maps)
    f.dump(OUT/'Down_maps.frozen.json',maps)
    with gzip.open(OUT/'physical_ledger.frozen.json.gz','wt') as z:json.dump(ledger,z,separators=(',',':'))
    paths=[Path(__file__),Path(previous.__file__),BUILDERS/'phase_builder.py',BUILDERS/'group_builder.py',
        BUILDERS/'protocol.frozen.json',BUILDERS/'group.protocol.frozen.json',BUILDERS/'resident_ledger.json',
        BUILDERS/'group.resident_ledger.json',BUILDERS/'phase_smoke.json',BUILDERS/'group_smoke.json',
        REPLICA/'protocol.frozen.json',REPLICA/'Down_job_maps.frozen.json',REFERENCE/'report.json',REFERENCE/'service_arrays.npz',
        OUT/'Down_maps.frozen.json',OUT/'physical_ledger.frozen.json.gz',f.TRAIN,f.PIPE,f.CPP,
        ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py',previous.BASE/'asymmetric_down.py']
    f.dump(OUT/'protocol.frozen.json',dict(status='frozen_before_screen',frozen_utc=f.utc(),
        inputs_sha256={str(p):f.sha(p) for p in paths},cases=CASES,
        split='All new GU kernels use fixed source0123, uniform originalwave10/prefix1 split.52physicalpages/48ECCpages, strictcopy500. Noall-dummywave extension or PBsourcepermutation.',
        standard_control='Equal-byte wholeGU g8group copies:3expertsgroup0only,3group1only,58dual; route3/group using same fixed capacity3 assignment; eachchip3E*12pages+shared12=48usefulpages. SameIPRsplit capability asphase. Plain/whole-dummy group kernel can be sameWL, outsideprimaryFFN.',
        phase='3Ephase0only[0,1,2],3Ephase2only[3,4,5],58dual; lexfill3/phase thenalternate6pageexperts,sharedRRlast. NoTRAINcopyfit or newCBCalgorithm.',
        GU_native_stream='Common fullH1536 input once perchip job. EveryselectedE uses sixorderedpages/64rows on16chips forphase or12orderedpages/128rows on8chips forgroup. EachE boundaryiscompleteC1536dot; pagefringes continue in fixed0123order. Shared12pages/128rows on16chips. Host scatter uses originalexpert,row/shard identity, not a PE/PBorderselector.',
        common_GU_issue='Singlecommonhostsubmit. Canonicalstaticchiporder: command/input/compute/release scheduling rank0 onallGU jobs; output owner ranks unchanged. Avoid accidental minowner-priority weakening ofgroupcopy strongcontrol. NoFFNcostfit.',
        Down='Unchanged frozen candidateonecopyshared4map or standardtriples sortedrank2/2/2. FullprivateC512 andsharedC1024 inputs, freshcold fixed0123 strict500, no private rescue. Shared384rows/chip onchips12..15 placedWL186..191; privatejobsblock1.',
        storage='AllnewGU744pages/chip,186WL/plane, +87MiB/layer vsoldGU. Totalonecopy235.5MiB ortriples331.5MiB; oldtriples244.5MiB. Budget32blocks/layer; newcases28nonemptyblocks. Actualsourceidentity/addressledger stored beforeexecution.',
        reference='Three old sealedtriples controls candidatepermutation/RRpermutation/g8fixed use sameTRAINinputs/service/downcopies. Reuse immutableperqueryarrays with hashes, do notrefit or silentlyrewrite.',
        screen_indices=np.linspace(0,511,64,dtype=int).tolist(),screen_axes=dict(input_copies=[1],priority=['GF']),
        full_gate='Screenall192 PB/FIFO128valid and fixed0123withoutprogrammablePBsource. Capability/storagetradeoff can pass evenwhen permutationcaseisfaster. Alwaysreportequal-bytegroupcopycontrol; ifittiesphase, noexclusivephase-layout claim. Parentdecidescontinuation afterscreen; anyfullplan is separatelyfrozen beforeexecution.',
        full_axes_if_continued=dict(input_copies=[1,4],priority=['GF','DF']),
        resources='Same causal fullFFN graph/resource/FIFO contract as frozen Downreplica experiment: hostsubmit2us,cmd250ns,PCIe8GB/s,8channels2GB/s,oneactiveGEMV/chip throughlastONFIoutput,fullGUhostbarrier/A8/fullprivateupload.',
        scope='Previouslyobserved W8/A8TRAIN first3layers; nosplitfit,heldout,fullweights,inference,hardwaremeasurement,newCBC/placementalgorithmclaim. Fixed9.7us/copy500/source/drop/PB model conditional.'))
    print(json.dumps(dict(status='frozen',source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT/'protocol.frozen.json'))),flush=True)


def graph_for(route,layer,case,service,recipes,maps):
    kind,down=CASES[case];route=sorted(map(int,route));chosen=route+[64]
    g=f.Graph(service);submit=g.add(0,service.submit_ns,kind='GU_submit')
    upload=g.add(2,192,[submit],kind='GU_input_PCIe',nbytes=1536);outputs=defaultdict(list)
    assignment=None
    for which in ([None] if kind=='phase' else [0,1]):
        kernel=phase.build_fixedphase_split(route) if kind=='phase' else group.build_fixedgroup_split(route,which)
        assert kernel['metadata']['fixed_source_0123']
        recipe=recipes.make(kernel['waves'],1536,kernel['owner_ranks'],'fixed')
        if recipe is None:return None,None
        assert recipe['duration']==155140 and recipe['counts']['physical_read_pages']==52 and recipe['counts']['ecc_pages']==48
        recipe=dict(recipe,summary=dict(recipe['summary'],policy='fixed0123_uniform_split10',native_source_0123=True))
        assignment=kernel['assignment']
        for chip in (range(16) if which is None else range(8*which,8*which+8)):
            jobid=len(g.jobs)
            for owner,nodes in g.matrix(chip,0,1536,recipe,submit,upload).items():outputs[owner]+=nodes
            job=g.jobs[jobid]
            for nodeid in [job['command'],job['input'],job['compute'],job['release']]:
                n=list(g.nodes[nodeid]);n[3]=0;g.nodes[nodeid]=tuple(n)
            job['GU_copy_metadata']=dict(kind=kind,group=which,assignment=assignment,
                native_input='full commonH1536',fixed_source_0123=True,
                routed_rows_per_chip=64 if kind=='phase' else 128,shared_rows_per_chip=128,
                routed_row_start=(chip if kind=='phase' else chip%8)*(64 if kind=='phase' else 128),
                shared_row_start=chip*128)
    complete=[]
    for rank,e in enumerate(chosen):
        ready=g.add(-1,0,outputs[rank],expert=rank,kind='full_GU_at_host')
        act=g.add(0,service.activation_ns,[ready],1,rank,kind='full_vector_activation_A8')
        ds=g.add(0,service.submit_ns,[act],1,rank,kind='Down_submit');C=1024 if e==64 else 512
        up=g.add(2,(C+7)//8,[ds],1,rank,kind='Down_input_PCIe',nbytes=C)
        copy=rank//2 if down=='triple_private_shared4' and e<64 else 0
        jobs=[j for j in maps[case][layer] if j['expert']==e and j['copy']==copy]
        assert sum(j['rows'] for j in jobs)==1536
        for job in jobs:
            recipe=recipes.private_at(job,rank)
            if recipe is None:return None,None
            jobid=len(g.jobs);complete+=g.matrix(job['chip'],1,C,recipe,ds,up)[rank]
            g.jobs[jobid]['physical_down_job']=job
    g.add(0,service.combine_ns,complete,1,kind='weighted_combine')
    return g,assignment


def checked():
    spec=f.read(OUT/'protocol.frozen.json')
    for p,h in spec['inputs_sha256'].items():assert f.sha(p)==h,p
    routes,_=get_inputs();return spec,routes,f.read(OUT/'Down_maps.frozen.json')


def full_freeze():
    spec,_,_=checked();gate=f.read(OUT/'screen/gate.json');assert gate['PB_FIFO128_and_fixed_source_pass']
    assert not (OUT/'full_plan.frozen.json').exists()
    f.dump(OUT/'full_plan.frozen.json',dict(frozen_utc=f.utc(),source_sha256=f.sha(__file__),
        protocol_sha256=f.sha(OUT/'protocol.frozen.json'),screen_report_sha256=f.sha(OUT/'screen/report.json'),
        cases=list(CASES),axes=spec['full_axes_if_continued'],
        scope='Parentauthorizedcontinuationofsamefrozenkernels/maps; all512areexistingTRAIN, nofit/heldout. Equalbytegroupcontrolmandatory.'))


def execute(mode):
    spec,routes,maps=checked();screen=mode=='screen'
    if not screen:
        full=f.read(OUT/'full_plan.frozen.json');assert full['screen_report_sha256']==f.sha(OUT/'screen/report.json')
    indices=np.asarray(spec['screen_indices']) if screen else np.arange(512)
    dest=OUT/('screen' if screen else 'TRAIN512');dest.mkdir(exist_ok=True);assert not (dest/'report.json').exists()
    configs=[(case,cp,df) for case in CASES for cp in ([1] if screen else [1,4]) for df in ([0] if screen else [0,1])]
    f.dump(dest/'run.frozen.json',dict(source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT/'protocol.frozen.json'),
        configurations=configs,indices=indices.tolist(),full_plan_sha256=None if screen else f.sha(OUT/'full_plan.frozen.json')))
    recipes=previous.Recipes(f.load_pipeline(),previous.a.load_shaper());executor=f.Executor(OUT)
    arrays={};records=[];cache={};unique=0;start=time.monotonic()
    for ci,(case,cp,df) in enumerate(configs):
        name=f'{case}.input{cp}.{"DF" if df else "GF"}';cost=np.full((3,len(indices)),-1,dtype=np.int64);fifo=cost.copy()
        dags=Counter();aggregate=Counter()
        for layer in range(3):
            for q,index in enumerate(indices):
                graph,assignment=graph_for(routes[layer,index],layer,case,f.Service(cp,df,0),recipes,maps)
                if graph is None:continue
                key,data=graph.signature();dags[key]+=1;cachekey=(key,df,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                if cachekey in cache:value,audit=cache[cachekey]
                else:
                    result=executor.run(graph,data);audit=f.audit(graph,result,full=q in [0,len(indices)-1]);value=result['cost_ns'];cache[cachekey]=value,audit;unique+=1
                if q==0:
                    result=executor.run(graph,data);audit=f.audit(graph,result,full=True)
                    f.dump_trace(dest/f'trace.L{layer}.{name}.json',graph,result,audit)
                cost[layer,q]=value;fifo[layer,q]=audit['peak_output_fifo_bytes']
                for k,v in audit['counts'].items():aggregate[k]+=v
                for k,v in audit['pipeline_counts'].items():aggregate['pipeline_'+k]+=v
                for k,v in audit['resource_work_ns'].items():aggregate[k]+=v
        good=cost>=0;arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo
        records.append(dict(name=name,case=case,input_copies=cp,priority='DF' if df else 'GF',completed=int(good.sum()),requested=int(cost.size),
            cost_ns=f.stats(cost[good]) if good.any() else None,three_layer_sum_ns=f.stats(cost.sum(axis=0)) if good.all() else None,
            FIFO_peak_bytes=f.stats(fifo[good]) if good.any() else None,
            valid_FIFO_capacities={str(k):int(np.count_nonzero(good&(fifo<=k))) for k in [64,128,256,2048]},
            normalized_DAG_counts=dict(dags),distinct_costs=int(len(np.unique(cost[good]))),
            counts_and_work_per_valid_query={k:v/int(good.sum()) for k,v in aggregate.items()} if good.any() else {},
            fixed_source_0123=True,requires_programmable_PBsource=False))
        progress=dict(finished=ci+1,total=len(configs),unique_graphs=unique,elapsed_seconds=time.monotonic()-start,last=name)
        f.dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    # Retain exact sealed oldtriples references on the same indices/axes.
    with np.load(REFERENCE/'service_arrays.npz') as old:
        for gu in ['candidate','RR','g8_fixed']:
            for cp in ([1] if screen else [1,4]):
                for df in ([0] if screen else [0,1]):
                    oldname=f'{gu}.triple_private_shared4.input{cp}.{"DF" if df else "GF"}'
                    for suffix in ['cost','fifo']:arrays['reference.'+oldname+'.'+suffix]=old[oldname+'.'+suffix][:,indices]
    np.savez_compressed(dest/'service_arrays.npz',**arrays)
    report=dict(status='complete_conditional_fixed0123_GU_copies',mode=mode,records=records,
        source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT/'protocol.frozen.json'),indices=indices.tolist(),
        unique_graphs=unique,elapsed_seconds=time.monotonic()-start,executor=executor.receipt,
        scope='KnownTRAIN, extra87MiBGUcopies, strongsamebyteswholegroupcopycontrol. Samefixed0123split/drop/IPR capabilities. Noalgorithm/hardwaremeasurementclaim.')
    f.dump(dest/'report.json',report)
    if screen:
        passed=all(r['completed']==192 and r['valid_FIFO_capacities']['128']==192 for r in records)
        phase_cost=arrays['phase_triples.input1.GF.cost'];group_cost=arrays['group_triples.input1.GF.cost']
        f.dump(dest/'gate.json',dict(PB_FIFO128_and_fixed_source_pass=passed,phase_group_cost_exact=bool(np.array_equal(phase_cost,group_cost)),
            gate=spec['full_gate'],source_sha256=f.sha(__file__),screen_report_sha256=f.sha(dest/'report.json')))
        print(json.dumps(dict(status='screen_complete',gate_passed=passed,phase_group_exact=bool(np.array_equal(phase_cost,group_cost)))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','screen','full_freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'full_freeze':full_freeze}.get(args.mode,lambda:execute(args.mode))()
