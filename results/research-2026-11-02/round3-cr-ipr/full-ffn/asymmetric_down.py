#!/usr/bin/env python3
"""Separately frozen GU-width/private-Down-width screen and TRAIN replay."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,sys,time
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import policy_addendum as policy
f=policy.f;a=policy.a
DEST=OUT/'asymmetric-down'
OLD=ROOT_SOURCE=f.ROOT/'results/research-2026-11-02/round2/ordinary-read-compiler/full-campaign-v5-W8-training/summary.json'
LAYOUTS=['candidate','RR','g8_fixed']


def menus():
    old=f.read(OLD);strong,inputs=f.load_strong();result={}
    for g in [1,2,4,8,16]:
        names=['disjoint']+(['cyclic'] if g in [2,4,8] else [])
        for kind in names:
            rows=np.zeros((3,65,16),dtype=np.int32)
            for e in range(64):
                start=(e%(16//g))*g if kind=='disjoint' else e%16
                rows[:,e,(np.arange(g)+start)%16]=1536//g
            rows[:,64]=96;result[f'g{g}_{kind}']=rows
    rows=np.zeros((3,65,16),dtype=np.int32)
    for e in range(64):rows[:,e,(np.arange(12)+e)%16]=128
    rows[:,64]=96;result['g12_cyclic']=rows
    for g in [2,4,8]:
        rows=np.zeros((3,65,16),dtype=np.int32)
        for layer in range(3):
            record=next(r for r in old['records'] if r['model']=='granite4tiny' and r['layer']==layer and r['command_ns']==250)
            fit=next(r for r in record['fitting'][1] if r['label']==f'disjoint_{g}')
            for e,label in enumerate(fit['labels']):rows[layer,e]=np.roll(fit['base_rows'],int(label)*g)
        rows[:,64]=96;result[f'g{g}_reused_TRAIN_labels']=rows
    rows=np.zeros((3,65,16),dtype=np.int32);rows[:,:64]=strong[:,:64,1];rows[:,64]=96
    result['g12_reused_strong_mask']=rows
    for name,rows in result.items():
        assert rows.shape==(3,65,16) and np.all(rows.sum(axis=2)==1536)
        pages=(rows.astype(np.int64)*np.asarray([512]*64+[1024])[None,:,None]+16383)//16384
        reserve=((pages+3)//4).sum(axis=1)
        assert np.all(reserve<=192),(name,reserve)
    return result,inputs


def freeze():
    DEST.mkdir(exist_ok=True);assert not (DEST/'protocol.frozen.json').exists()
    mm,strong_inputs=menus()
    np.savez_compressed(DEST/'Down_menus.frozen.npz',**mm)
    paths=[Path(__file__),OUT/'policy_addendum.py',OUT/'shaping_addendum.py',a.SHAPER,f.PIPE,f.CPP,
           f.ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py',f.TRAIN,OLD,
           OUT/'g8-addendum/group_maps.frozen.npz',DEST/'Down_menus.frozen.npz']
    receipt=dict(frozen_utc=f.utc(),inputs_sha256={str(p):f.sha(p) for p in paths},strong_inputs=strong_inputs,
                 GU=LAYOUTS,Down_menu=list(mm),
                 screen_indices=np.linspace(0,511,64,dtype=int).tolist(),screen_layers=[0,1,2],
                 screen='Same entireDownmenu for every GUfamily. ExistingTRAIN64 perlayer, previouslyobservedroutes, noheldout.',
                 selection='Eachfamily independently retains onlyall192valid queries and128BFIFOvalid; minimize arithmeticmean fullFFNns over64aligned3layersums; deterministicmenu-name tie break. Freezewinner before512evaluation. This is TRAINselection, not unseenvalidation.',
                 full_axes=dict(input_copies=[1,4],priority=['GF','DF'],activation_ns=0),
                 pipeline='copy500ns; candidate andRR commonGU use prior deterministicfirst-feasiblePBpolicy, permutation/dropconditional. G8GUI samefirst-feasiblepolicy but fixed0123sources; alltruthpagesordered. PrivateDown unshapedstrict500fixed0123, with initialcold perjob. g1N12fail is retained, neverrescued.',
                 shared='Always full sharedGU128rows/chip16chips; sharedDown96rows/chip16chips,C1024,unchangedforall menus.',
                 native_contract='PrivateDown fullC512inputvector for everyselectedroutedexpert; ownactivation/A8 andupload afterfullhostGUbarrier. No sharedx orprivatecontext multiplexing. Everymatrixownsonechip untilcomputeandlastONFIoutput.',
                 capacity='Onecopy usefulcoeff155713536B/layer. Allcontrols common2blocks/chip budget; GUblock0/privateDownblock1; privateDown fullmodel reservedWL<=192 onallchips. CommonGU may need less. No claimallFFNnecessarilyoneblock.',
                 interpretation='IfcompactingDownbenefitsallGUfamilies, attribute tocommonversusprivateoperand domains/sharding/startupcost, notsharedlayoutnovelty. Standardgroupmenus/oldlabels are strongcontrols, notnewalgorithms.',
                 limitations='Conditional9.7us producer/copy/PB/source/FIFO model; cadence sensitivity unmeasured. No fullinference,holdout,hardwaremeasurement orpublicationclaim.')
    f.dump(DEST/'protocol.frozen.json',receipt)
    print(json.dumps(dict(status='frozen',menus=list(mm),screen_queries_per_family=192)),flush=True)


class KernelRecipes:
    def __init__(self,pipeline,shaper):self.p=pipeline;self.s=shaper;self.cache={};self.failure_samples=[];self.decisions=Counter()
    def make(self,waves,columns,ranks,source='fixed',pool=None):
        settings=dict(cache_copy_ns=500,source_policy=source,issue_policy='strict_no_interval',row_bytes=columns,
                      validate_row_stream=True,unselected_policy='drop_before_ecc_conditional')
        payload=dict(waves=waves,settings=settings,ranks=ranks,pool=pool)
        key=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key in self.cache:
            rec=self.cache[key];self.decisions[rec['policy'] if rec else 'private_strict_infeasible']+=1;return rec
        if pool is None:
            selected=waves;result=self.p.simulate(waves,settings);choice='private_strict'
        else:
            fill=self.s.shape_kernel(waves,(),pool);selected=fill;result=self.p.simulate(fill,settings);choice='GU_fill_only'
            if not result['feasible']:
                cuts=[(10,1)]+[(wi,k) for wi,w in enumerate(waves) for k in range(1,len(w['pages'])) if (wi,k)!=(10,1)]
                for wi,k in cuts:
                    if wi>=len(waves) or k>=len(waves[wi]['pages']):continue
                    trial=self.s.shape_kernel(waves,((wi,k),),pool);rr=self.p.simulate(trial,settings)
                    if rr['feasible']:selected=trial;result=rr;choice=f'GU_split_{wi}_{k}';break
                else:
                    settings=dict(settings,issue_policy='cold_reset');selected=fill;result=self.p.simulate(fill,settings);choice='GU_explicit_cold_reset'
        if not result['feasible']:
            if len(self.failure_samples)<8:self.failure_samples.append(dict(payload=payload,result={k:result[k] for k in ['feasible','makespan_ns','violations','stream_validation']}))
            self.cache[key]=None;self.decisions['private_strict_infeasible']+=1;return None
        rows=Counter();milestones=[]
        for x in result['page_completions']:
            e=x['owner'];old=rows[e]//columns;rows[e]+=x['bytes'];new=rows[e]//columns
            if new>old:milestones.append((ranks[e],x['end_ns'],(new-old)*4))
        counts={k:result[k] for k in ['waves','pages','bytes','physical_read_pages','physical_read_bytes','ecc_pages','ecc_bytes','cold_waves','cr_waves']}
        recipe=dict(duration=result['makespan_ns'],milestones=milestones,signature=key,counts=counts,policy=choice,
                    summary=dict(settings=settings,policy=choice,**counts,finite_PB_feasible=True,source_policy=source,no_FFN_cost_oracle=True))
        self.cache[key]=recipe;self.decisions[choice]+=1;return recipe
    def private(self,rows,columns,owner):
        n=(rows*columns+16383)//16384;waves=[]
        for j in range(0,n,4):
            pp=[dict(page_id=f'v.p{k}',owner='v',page_index=k,plane=k%4,block=0,wl=k//4,bytes=16384) for k in range(j,min(j+4,n))]
            waves.append(dict(pages=pp,source_order=[p['plane'] for p in pp]))
        rec=self.make(waves,columns,{'v':0})
        if rec is None:return None
        return dict(rec,milestones=[(owner,t,b) for _,t,b in rec['milestones']])


def group_waves(route,groups,group):
    resident=np.flatnonzero(groups==group);position={int(e):i for i,e in enumerate(resident)};selected=[e for e in route if groups[e]==group]
    stream=[];pool=[]
    for e in resident:
        for j in range(12):
            page=dict(page_id=f'e{e}.p{j}',owner=f'e{e}',page_index=j,plane=j%4,block=0,wl=3*position[int(e)]+j//4,bytes=16384)
            if int(e) in selected:stream.append(page)
            elif int(e) not in route:pool.append(dict(page,unselected=True))
    for j in range(12):stream.append(dict(page_id=f's.p{j}',owner='s',page_index=j,plane=j%4,block=0,wl=96+j//4,bytes=16384))
    waves=[dict(pages=stream[j:j+4],source_order=[0,1,2,3]) for j in range(0,len(stream),4)]
    return waves,pool


def graph_for(route,layer,GU,downmap,service,recipes,groups):
    route=sorted(map(int,route));chosen=route+[64];ranks={f'e{e}':i for i,e in enumerate(route)};ranks['s']=6
    # Retain explicit invalid privatekernel before allocating an unused graph.
    for rank,e in enumerate(chosen):
        columns=1024 if e==64 else 512
        for rows in set(downmap[e].tolist()):
            if rows and recipes.private(int(rows),columns,rank) is None:return None
    g=f.Graph(service);submit=g.add(0,service.submit_ns,kind='GU_submit');upload=g.add(2,192,[submit],kind='GU_input_PCIe',nbytes=1536);outputs=defaultdict(list)
    if GU=='g8_fixed':
        for group in [0,1]:
            waves,pool=group_waves(route,groups[layer],group)
            recipe=recipes.make(waves,1536,ranks,'fixed',pool)
            if recipe is None:return None
            for chip in range(group*8,group*8+8):
                for owner,ids in g.matrix(chip,0,1536,recipe,submit,upload).items():outputs[owner]+=ids
    else:
        waves=recipes.p.build_g4_candidate(route) if GU=='candidate' else f.rr_greedy_waves(route)
        pool=a.resident_pool(route,GU,np.arange(64));recipe=recipes.make(waves,1536,ranks,'permutation',pool)
        if recipe is None:return None
        for chip in range(16):
            for owner,ids in g.matrix(chip,0,1536,recipe,submit,upload).items():outputs[owner]+=ids
    complete=[]
    for rank,e in enumerate(chosen):
        ready=g.add(-1,0,outputs[rank],expert=rank,kind='full_GU_at_host')
        act=g.add(0,0,[ready],1,rank,kind='full_vector_activation_A8');ds=g.add(0,service.submit_ns,[act],1,rank,kind='Down_submit')
        columns=1024 if e==64 else 512;up=g.add(2,(columns+7)//8,[ds],1,rank,kind='Down_input_PCIe',nbytes=columns)
        for chip,rows in enumerate(downmap[e]):
            if rows:complete+=g.matrix(chip,1,columns,recipes.private(int(rows),columns,rank),ds,up)[rank]
    g.add(0,0,complete,1,kind='weighted_combine');return g


def checked_inputs():
    spec=f.read(DEST/'protocol.frozen.json')
    for p,h in spec['inputs_sha256'].items():assert f.sha(p)==h,p
    with np.load(f.TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3]
    with np.load(DEST/'Down_menus.frozen.npz') as z:mm={k:z[k].copy() for k in z.files}
    with np.load(OUT/'g8-addendum/group_maps.frozen.npz') as z:groups=z['learned'].copy()
    return spec,routes,mm,groups


def execute(action):
    spec,routes,mm,groups=checked_inputs();screen=action=='screen'
    indices=np.asarray(spec['screen_indices']) if screen else np.arange(512)
    selection=None if screen else f.read(DEST/'selection.frozen.json')
    if not screen:
        assert selection['screen_report_sha256']==f.sha(DEST/'screen/report.json')
        assert selection['source_sha256']==f.sha(__file__)
    configurations=[(GU,menu,1,0) for GU in LAYOUTS for menu in mm] if screen else [(GU,selection['winners'][GU],copies,df) for GU in LAYOUTS for copies in [1,4] for df in [0,1]]
    dest=DEST/('screen' if screen else 'TRAIN512');dest.mkdir(exist_ok=True);assert not (dest/'report.json').exists()
    f.dump(dest/'run.frozen.json',dict(frozen_utc=f.utc(),protocol_sha256=f.sha(DEST/'protocol.frozen.json'),action=action,indices=indices.tolist(),selection_sha256=None if screen else f.sha(DEST/'selection.frozen.json')))
    recipes=KernelRecipes(f.load_pipeline(),a.load_shaper());executor=f.Executor(OUT);records=[];arrays={};cache={};started=time.monotonic();unique=0
    for ci,(GU,menu,copies,df) in enumerate(configurations):
        name=f'{GU}.{menu}.input{copies}.{("DF" if df else "GF")}'
        cost=np.full((3,len(indices)),-1,dtype=np.int64);fifo=cost.copy();aggregate=Counter();before=recipes.decisions.copy()
        for layer in range(3):
            for q,idx in enumerate(indices):
                graph=graph_for(routes[layer,idx],layer,GU,mm[menu][layer],f.Service(copies,df,0),recipes,groups)
                if graph is None:continue
                key,data=graph.signature();key=(key,df,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                if key in cache:value,audit=cache[key]
                else:
                    result=executor.run(graph,data);audit=f.audit(graph,result,full=(q in [0,len(indices)-1]));value=result['cost_ns'];cache[key]=(value,audit);unique+=1
                    if q==0 and layer==0:f.dump_trace(dest/('trace.'+name+'.json'),graph,result,audit)
                cost[layer,q]=value;fifo[layer,q]=audit['peak_output_fifo_bytes']
                for k,v in audit['counts'].items():aggregate[k]+=v
                for k,v in audit['pipeline_counts'].items():aggregate['pipeline_'+k]+=v
                for k,v in audit['resource_work_ns'].items():aggregate[k]+=v
        good=cost>=0;arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo
        rec=dict(name=name,GU=GU,Down_menu=menu,input_copies=copies,priority='DF' if df else 'GF',
                 requested_queries=int(cost.size),completed_queries=int(good.sum()),
                 cost_ns=f.stats(cost[good]) if good.any() else None,three_layer_sum_ns=f.stats(cost.sum(axis=0)) if good.all() else None,
                 FIFO_peak_bytes=f.stats(fifo[good]) if good.any() else None,
                 valid_FIFO_capacities={str(k):int(np.count_nonzero(good&(fifo<=k))) for k in [64,128,256,2048]},
                 counts_and_work_per_valid_query={k:v/int(good.sum()) for k,v in aggregate.items()} if good.any() else {},
                 policy_calls=dict(recipes.decisions-before))
        records.append(rec)
        progress=dict(finished=ci+1,total=len(configurations),unique_graphs=unique,elapsed_seconds=time.monotonic()-started,last=name)
        f.dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez_compressed(dest/'service_arrays.npz',**arrays);f.dump(dest/'negative_private_kernel_samples.json',recipes.failure_samples)
    report=dict(status='complete_conditional_TRAIN_asymmetric_Down',action=action,indices=indices.tolist(),layers=[0,1,2],records=records,
                source_sha256=f.sha(__file__),protocol_sha256=f.sha(DEST/'protocol.frozen.json'),unique_graphs=unique,elapsed_seconds=time.monotonic()-started,
                scope='AllGUfamilies receive identicalDownmenu. Fullprivatevectors andlogicalcoeffbytesconstant; conditionalresource/PBmodel, no hardwaremeasurement/newheldout/optimizerclaim.')
    f.dump(dest/'report.json',report)
    if screen:
        winners={}
        for GU in LAYOUTS:
            valid=[r for r in records if r['GU']==GU and r['completed_queries']==192 and r['valid_FIFO_capacities']['128']==192]
            assert valid,GU
            winners[GU]=min(valid,key=lambda r:(r['three_layer_sum_ns']['mean'],r['Down_menu']))['Down_menu']
        f.dump(DEST/'selection.frozen.json',dict(frozen_utc=f.utc(),source_sha256=f.sha(__file__),screen_report_sha256=f.sha(dest/'report.json'),screen_arrays_sha256=f.sha(dest/'service_arrays.npz'),winners=winners,
                                                rule=spec['selection'],scope='TRAINselected on192events; full512is sameTRAINdescriptive replay, nottest.'))
        print(json.dumps(dict(status='selected_and_frozen',winners=winners)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','screen','run']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:execute(args.action)
