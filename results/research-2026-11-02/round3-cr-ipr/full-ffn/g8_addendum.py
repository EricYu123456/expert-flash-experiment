#!/usr/bin/env python3
"""Separately frozen fixed-plane-order g8 control, no changes to primary run."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
from collections import Counter,defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'scripts/research_20261102_cr_ipr_full_ffn.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261102_cr_ipr_full_ffn as f
DEST=OUT/'g8-addendum'

def fit_one(routes):
    x=np.zeros((512,64),dtype=np.int16);x[np.arange(512)[:,None],routes]=1
    rng=np.random.default_rng(20261102)
    starts=[np.arange(64)%2,np.arange(64)//32]
    for _ in range(2):
        row=np.r_[np.zeros(32,dtype=int),np.ones(32,dtype=int)];rng.shuffle(row);starts.append(row)
    runs=[];best=None
    for index,start in enumerate(starts):
        groups=np.asarray(start,dtype=np.int16).copy();count=x@groups
        score=int(np.abs(count-3).sum());initial=score;accepted=0
        for _ in range(64):
            zero=np.flatnonzero(groups==0);one=np.flatnonzero(groups==1)
            # Swap0->1 and1->0 while retaining exactly32experts/group.
            trial=count[:,None,None]+x[:,zero,None]-x[:,None,one]
            scores=np.abs(trial-3).sum(axis=0)
            a,b=np.unravel_index(np.argmin(scores),scores.shape)
            if int(scores[a,b])>=score:break
            count=trial[:,a,b];groups[zero[a]]=1;groups[one[b]]=0
            score=int(scores[a,b]);accepted+=1
        runs.append(dict(start=index,initial_absolute_imbalance=initial,final_absolute_imbalance=score,accepted_swaps=accepted))
        if best is None or score<best[0]:best=(score,groups.copy())
    score,groups=best
    assert int(groups.sum())==32
    return groups.astype(np.int32),dict(runs=runs,objective='TRAIN sum max(selected group0,selected group1);32experts/group',
                                      final_mean_peak=float(3+score/512),bounded_standard_swap_control=True,global_optimum_claim=False)


def freeze():
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'protocol.frozen.json').exists()
    with np.load(f.TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)
    maps=[];fits=[]
    for layer,route in enumerate(routes):
        groups,record=fit_one(route);maps.append(groups);fits.append(dict(layer=layer,**record))
    maps=np.asarray(maps)
    np.savez_compressed(DEST/'group_maps.frozen.npz',learned=maps,identity=np.arange(64,dtype=np.int32)%2,layers=np.arange(40))
    plan=dict(frozen_utc=f.utc(),source_sha256=f.sha(__file__),
              primary_source_sha256=f.sha(ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py'),
              primary_CPP_sha256=f.sha(f.CPP),pipeline_sha256=f.sha(f.PIPE),TRAIN_sha256=f.sha(f.TRAIN),
              group_maps_sha256=f.sha(DEST/'group_maps.frozen.npz'),
              primary_plan_sha256=f.sha(OUT/'execution_plan.frozen.json'),
              data='Previously seen TRAIN40layers512 for fit; replay first3layers512. No heldout.',
              fit=fits,group_size=32,expert_chips=8,
              groups='chips0..7 and8..15; one fixed static owner group/expert',
              GU='Perexpert128complete rows/chip=12pages=3fullwaves. Shared128rows/chip=12pages3waves onall16chips. One common-input chain/chip, fixed0123 plane source order. Resident99WL/plane.',
              capabilities='Skip arbitrary WL within one block; fixed0123 sources; same finitePB read/cache timing as primary; no new source selector needed by GU.',
              profiles=['strict100','cold_reset500'],copies=[1,4],priority=['GF','DF'],Down=['g16_equal','g12_cyclic'],activation_ns=0,
              comparison='Reuse immutable primary TRAIN512 candidate arrays after completed-report receipt verified; no fit depends on primary costs.',
              FIFO='Same posthoc64/128/256/2048B exact produced-minus-continuous-drained audit. Strict infeasible query remains -1; no pace rescue.',
              limits='Standard group swaps are a control, not a new algorithm. Perexpert GU inputs common; Down remain separate full private vectors.')
    f.dump(DEST/'protocol.frozen.json',plan)
    print(json.dumps(dict(status='frozen',maps_sha256=plan['group_maps_sha256'],first3_fits=[x['final_mean_peak'] for x in fits[:3]])),flush=True)


class GroupRecipes(f.Recipes):
    def group(self,selected,groups,ranks,profile):
        ids=sorted(e for e in range(64) if groups[e]==groups[selected[0]]) if selected else []
        positions={e:i for i,e in enumerate(ids)}
        stream=[]
        for e in selected:
            for j in range(12):stream.append(dict(page_id=f'e{e}.p{j}',owner=f'e{e}',page_index=j,plane=j%4,block=0,wl=3*positions[e]+j//4,bytes=f.PAGE))
        for j in range(12):stream.append(dict(page_id=f's.p{j}',owner='s',page_index=j,plane=j%4,block=0,wl=96+j//4,bytes=f.PAGE))
        waves=[dict(pages=stream[j:j+4],source_order=[0,1,2,3]) for j in range(0,len(stream),4)]
        key=(profile,tuple(ranks[x['owner']] for x in stream))
        if key in self.cache:return self.cache[key]
        settings=dict(cache_copy_ns=100 if profile=='strict100' else 500,source_policy='fixed',
                      issue_policy='strict_no_interval' if profile=='strict100' else 'cold_reset',row_bytes=1536,validate_row_stream=True)
        result=self.p.simulate(waves,settings)
        if not result['feasible']:
            if len(self.failures)<10:self.failures.append(dict(profile=profile,selected=selected,groups=groups.tolist(),result={k:result[k] for k in ['makespan_ns','violations','stream_validation']}))
            self.cache[key]=None;return None
        counts={k:result[k] for k in ['waves','pages','bytes','physical_read_pages','physical_read_bytes','ecc_pages','ecc_bytes','cold_waves','cr_waves']}
        recipe=dict(duration=result['makespan_ns'],
                    milestones=[(ranks[x['owner']],x['end_ns'],(x['row_end']-x['row_start'])*4) for x in result['row_completions']],
                    counts=counts,signature=hashlib.sha256(repr(key).encode()).hexdigest(),
                    summary=dict(settings=settings,profile=profile,**counts,finite_PB_feasible=True,
                                 no_programmable_source_selector_required=True))
        self.cache[key]=recipe;return recipe


def graph_for(route,groups,down,profile,service,recipes,strong):
    route=sorted(map(int,route));chosen=route+[64];ranks={f'e{e}':i for i,e in enumerate(route)};ranks['s']=6
    g=f.Graph(service);submit=g.add(0,service.submit_ns,kind='GU_submit');up=g.add(2,192,[submit],kind='GU_input_PCIe',nbytes=1536)
    outputs=defaultdict(list)
    for group in [0,1]:
        selected=[e for e in route if groups[e]==group]
        recipe=recipes.group(selected,groups,ranks,profile)
        if recipe is None:return None
        for chip in range(group*8,group*8+8):
            for owner,nodes in g.matrix(chip,0,1536,recipe,submit,up).items():outputs[owner]+=nodes
    rows=f.down_rows(chosen,down,strong);complete=[]
    for rank,e in enumerate(chosen):
        ready=g.add(-1,0,outputs[rank],expert=rank,kind='full_GU_at_host')
        act=g.add(0,service.activation_ns,[ready],1,rank,kind='full_vector_activation_A8')
        ds=g.add(0,service.submit_ns,[act],1,rank,kind='Down_submit');columns=1024 if e==64 else 512
        upload=g.add(2,(columns+7)//8,[ds],1,rank,kind='Down_input_PCIe',nbytes=columns)
        for chip,n in enumerate(rows[rank]):
            if n:
                recipe=recipes.private(int(n),columns,rank,profile)
                if recipe is None:return None
                complete+=g.matrix(chip,1,columns,recipe,ds,upload)[rank]
    g.add(0,0,complete,1,kind='weighted_combine')
    return g


def execute(limit):
    spec=f.read(DEST/'protocol.frozen.json')
    for path,key in [(Path(__file__),'source_sha256'),(ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py','primary_source_sha256'),(f.CPP,'primary_CPP_sha256'),(f.PIPE,'pipeline_sha256'),(f.TRAIN,'TRAIN_sha256'),(DEST/'group_maps.frozen.npz','group_maps_sha256')]:assert f.sha(path)==spec[key]
    dest=DEST/('smoke' if limit==8 else 'TRAIN512');dest.mkdir(exist_ok=True);assert not (dest/'report.json').exists()
    f.dump(dest/'run.frozen.json',dict(utc=f.utc(),queries=limit,layers=[0,1,2],protocol_sha256=f.sha(DEST/'protocol.frozen.json')))
    with np.load(f.TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3,:limit]
    with np.load(DEST/'group_maps.frozen.npz') as z:learned=z['learned'];identity=z['identity']
    strong,_=f.load_strong();recipes=GroupRecipes(f.load_pipeline());executor=f.Executor(OUT)
    records=[];arrays={};cache={};started=time.monotonic();unique=0
    configs=[(profile,down,copies,df,kind) for profile in spec['profiles'] for down in spec['Down'] for copies in spec['copies'] for df in [0,1] for kind in ['identity','learned']]
    for ci,(profile,down,copies,df,kind) in enumerate(configs):
        name=f'{profile}.{down}.input{copies}.{("DF" if df else "GF")}.a0.g8_{kind}'
        cost=np.full((3,limit),-1,dtype=np.int64);fifo=cost.copy();aggregate=Counter();failed=0
        for layer in range(3):
            groups=identity if kind=='identity' else learned[layer]
            for q,route in enumerate(routes[layer]):
                graph=graph_for(route,groups,down,profile,f.Service(copies,df,0),recipes,strong[layer])
                if graph is None:failed+=1;continue
                key,data=graph.signature();key=(key,df)
                if key in cache:value,audit=cache[key]
                else:
                    result=executor.run(graph,data);audit=f.audit(graph,result,full=(q in [0,1,limit-1]));value=result['cost_ns'];unique+=1;cache[key]=(value,audit)
                    if q==0 and layer==0 and ci==0:f.dump_trace(dest/'small_full_trace.json',graph,result,audit)
                cost[layer,q]=value;fifo[layer,q]=audit['peak_output_fifo_bytes']
                for k,v in audit['counts'].items():aggregate[k]+=v
                for k,v in audit['pipeline_counts'].items():aggregate['pipeline_'+k]+=v
        good=cost>=0;arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo
        records.append(dict(name=name,profile=profile,down=down,copies=copies,priority='DF' if df else 'GF',group_map=kind,
                            completed_queries=int(good.sum()),requested_queries=int(cost.size),strict_pipeline_failed_queries=failed,
                            cost_ns=f.stats(cost[good]) if good.any() else None,FIFO_peak_bytes=f.stats(fifo[good]) if good.any() else None,
                            valid_FIFO_capacities={str(k):int(np.count_nonzero(good&(fifo<=k))) for k in [64,128,256,2048]},
                            event_counts_per_valid_query={k:v/int(good.sum()) for k,v in aggregate.items()} if good.any() else {}))
        progress=dict(finished=ci+1,total=len(configs),unique_graphs=unique,elapsed_seconds=time.monotonic()-started,last=name)
        f.dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez_compressed(dest/'service_arrays.npz',**arrays);f.dump(dest/'pipeline_failures.json',recipes.failures)
    f.dump(dest/'report.json',dict(status='complete_conditional_g8_TRAIN_control',limit=limit,layers=[0,1,2],records=records,
                                  unique_graphs=unique,elapsed_seconds=time.monotonic()-started,protocol_sha256=f.sha(DEST/'protocol.frozen.json'),
                                  note='Fixed source0123; one common-input g8chain perchip; infeasible strict cases not paced or omitted from requested counts. Compare only jointly feasible queries and retain all-query invalidity.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','smoke','run']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:execute(8 if args.action=='smoke' else 512)
