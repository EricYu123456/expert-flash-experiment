#!/usr/bin/env python3
"""Uniform read-shaping control; separate source/config freeze from v1."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'scripts/research_20261102_cr_ipr_full_ffn.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261102_cr_ipr_full_ffn as f
SHAPER=f.BASE/'pipeline/shaping/shaping.py'
SHAPER_SHA='19d5ff3555b46b8992a2fd3a978a817bd827744b0b0245fd4f3727202cdb5337'
DEST=OUT/'shaping-addendum'
VARIANTS={
    'candidate_IPR_uniform_split10':('candidate',True),
    'matching_layout_IPR_uniform_split10':('sameWL_matching_keep_hot',True),
    'RR_IPR_uniform_split10':('IPR_standard_RR_row_safe',True),
    'matching_layout_IPR_fill_only':('sameWL_matching_keep_hot',False),
}


def load_shaper():
    assert f.sha(SHAPER)==SHAPER_SHA
    spec=importlib.util.spec_from_file_location('generic_read_shaping_frozen',SHAPER)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def freeze():
    DEST.mkdir(exist_ok=True);assert not (DEST/'protocol.frozen.json').exists()
    inputs=[Path(__file__),SHAPER,f.PIPE,ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py',f.CPP,
            f.TRAIN,f.BASE/'baselines/frozen_TRAIN_packed_expert_order.npz',OUT/'execution_plan.frozen.json']
    f.dump(DEST/'protocol.frozen.json',dict(
        frozen_utc=f.utc(),inputs_sha256={str(p):f.sha(p) for p in inputs},
        data='Previously observed TRAIN first3layers512; no heldout. Layouts unchanged, no fit.',
        uniform_rule='For all3kernels split original wave10 after first useful page, preserve every true page/order/address, fill everypartialread with real unselected resident pages on missing planes. One extra epoch; not per-query cost search.',
        disclosure='Uniform wave10/prefix1 chosen after completed canonical candidate sensitivity. Same frozen structural rule granted to standardRR and matching-layout IPR baselines; no per-query oracle selection.',
        profiles='Strict no-interval, copy500ns, permutation,2PBslots, evalstartoverwrite, perplane concurrentcopy, conditional inactive discard-beforeECC. Failure staysNone, no pace/cold rescue.',
        variants=VARIANTS,copies=[1,4],priority=['GF','DF'],Down=['g16_equal','g12_cyclic'],activation_ns=0,
        Down_contract='Same unshaped private full-vector Down jobs as primary. Copy500ns and strict issue;1/2wave Down fits existing finitePB. Full GU→hostactivation/A8 barrier retained.',
        physical='Candidate pool uses candidate staticpage→plane map; matching pool uses same frozen TRAIN Eorder; RR pool uses identityglobalpage map. All pool pages belong to unselected routed experts, no copies or vacant placeholders.',
        labels='Both matching-layout shaped/fill-only variants invoke granted IPR; they are not labeledsameWL. Prior packed sameWL keep-hot cold500 remains separate comparator.',
        cache='Complete JSON identity of settings,waves includingblock/WL/read_pages/masks/force_cold,andowner mapping; no v1 crossblock cache-key reuse.',
        cadence_limitation='Fixed9.7us producer cadence is one conditional model. If a device/controller can extend recycle/evaluation to10.24us without breakingCR, that may outperform dummy-read shaping; controllable timing/analog feasibility is unverified. Do not callshaping universallybest or use9.7model as sole actualhardware model.',
        comparisons='After primaryreport complete, verify exactsource/input receipts then reuse its candidatecold500 andmatching_sameWL_cold500 arrays. Genericcompiler same candidate exactalias. FIFO64/128/256/2048 remains posthoc valid/invalid.'))


def resident_pool(selected,kind,order):
    pool=[];inverse=np.argsort(order)
    for e in range(64):
        if e in selected:continue
        for j in range(6):
            if kind=='candidate':
                plane=[0,1,2,0,1,3][j];wl=2*e+j//3 if plane<2 else e
            else:
                page=(int(inverse[e]) if kind=='matching' else e)*6+j;plane=page%4;wl=page//4
            pool.append(dict(page_id=f'e{e}.p{j}',owner=f'e{e}',page_index=j,plane=plane,block=0,wl=wl,
                             bytes=16384,unselected=True,dummy_unselected_expert=e))
    assert len({(p['plane'],p['block'],p['wl']) for p in pool})==len(pool)
    return pool


class ShapedRecipes(f.Recipes):
    def __init__(self,pipeline,shaper,variant,order):
        super().__init__(pipeline);self.shaper=shaper;self.variant=variant;self.order=order;self.certificates=[]
    def make(self,waves,profile,columns,owner_ranks,validate=True):
        base,do_split=VARIANTS[self.variant]
        settings=dict(cache_copy_ns=500,source_policy='permutation',issue_policy='strict_no_interval',
                      row_bytes=columns,validate_row_stream=validate,unselected_policy='drop_before_ecc_conditional')
        if validate:
            selected={int(p['owner'][1:]) for w in waves for p in w['pages'] if p['owner'].startswith('e')}
            kind='candidate' if base=='candidate' else 'matching' if 'matching' in base else 'RR'
            pool=resident_pool(selected,kind,self.order)
            before=[(p['page_id'],p['plane'],p['block'],p['wl']) for w in waves for plane in w['source_order'] for p in w['pages'] if p['plane']==plane]
            shaped=self.shaper.shape_kernel(waves,((10,1),) if do_split else (),pool)
            after=[(p['page_id'],p['plane'],p['block'],p['wl']) for w in shaped for plane in w['source_order'] for p in w['pages'] if p['plane']==plane]
            assert before==after and len(before)==48
            allowed={(p['page_id'],p['plane'],p['block'],p['wl']) for p in pool}
            physical=set();dummy=0
            for w in shaped:
                true={p['page_id'] for p in w['pages']}
                for p in w['read_pages']:
                    key=(p['plane'],p['block'],p['wl']);assert key not in physical;physical.add(key)
                    if p['page_id'] not in true:
                        assert (p['page_id'],*key) in allowed;dummy+=1
            assert len(physical)==4*len(shaped) and dummy==4*len(shaped)-48
            waves=shaped
        payload=dict(settings=settings,waves=waves,owner_ranks=owner_ranks)
        key=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key in self.cache:return self.cache[key]
        result=self.p.simulate(waves,settings)
        if not result['feasible']:
            if len(self.failures)<10:self.failures.append(dict(variant=self.variant,payload=payload,violations=result['violations'],stream=result['stream_validation']))
            self.cache[key]=None;return None
        rows=Counter();milestones=[]
        for x in result['page_completions']:
            e=x['owner'];old=rows[e]//columns;rows[e]+=x['bytes'];new=rows[e]//columns
            if new>old:milestones.append((owner_ranks[e],x['end_ns'],(new-old)*4))
        counts={k:result[k] for k in ['waves','pages','bytes','physical_read_pages','physical_read_bytes','ecc_pages','ecc_bytes','cold_waves','cr_waves']}
        recipe=dict(duration=result['makespan_ns'],milestones=milestones,signature=key,counts=counts,
                    summary=dict(settings=settings,variant=self.variant,**counts,complete_row_stream=validate,finite_PB_feasible=True))
        if validate and len(self.certificates)<3:self.certificates.append(dict(variant=self.variant,true_pages=48,physical_pages=result['physical_read_pages'],dummy_pages=result['physical_read_pages']-48,read_stream_preserved=True,existing_unselected_address_pool=True,waves=waves))
        self.cache[key]=recipe;return recipe


def execute(limit):
    spec=f.read(DEST/'protocol.frozen.json')
    for p,h in spec['inputs_sha256'].items():assert f.sha(p)==h,p
    dest=DEST/('smoke' if limit==8 else 'TRAIN512');dest.mkdir(exist_ok=True);assert not (dest/'report.json').exists()
    f.dump(dest/'run.frozen.json',dict(frozen_utc=f.utc(),protocol_sha256=f.sha(DEST/'protocol.frozen.json'),limit=limit,layers=[0,1,2]))
    with np.load(f.TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3,:limit]
    with np.load(f.BASE/'baselines/frozen_TRAIN_packed_expert_order.npz') as z:orders=z['slot_to_expert']
    strong,_=f.load_strong();pipeline=f.load_pipeline();shaper=load_shaper();executor=f.Executor(OUT)
    recipes={(variant,layer):ShapedRecipes(pipeline,shaper,variant,orders[layer]) for variant in VARIANTS for layer in range(3)}
    records=[];arrays={};cache={};started=time.monotonic();unique=0
    configs=[(down,copies,df,variant) for down in spec['Down'] for copies in spec['copies'] for df in [0,1] for variant in VARIANTS]
    for ci,(down,copies,df,variant) in enumerate(configs):
        name=f'strict500.{down}.input{copies}.{("DF" if df else "GF")}.a0.{variant}'
        cost=np.full((3,limit),-1,dtype=np.int64);fifo=cost.copy();aggregate=Counter()
        for layer in range(3):
            rr=recipes[variant,layer];gu,_=VARIANTS[variant]
            for q,route in enumerate(routes[layer]):
                graph=f.build_graph(route,layer,gu,down,'strict500',f.Service(copies,df,0),rr,orders,strong[layer])
                if graph is None:continue
                key,data=graph.signature()
                # Include metadata in cache identity, unlike the sealed v1 API.
                key=(key,df,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                if key in cache:value,audit=cache[key]
                else:
                    result=executor.run(graph,data);audit=f.audit(graph,result,full=(q in [0,1,limit-1]));value=result['cost_ns'];unique+=1;cache[key]=(value,audit)
                    if ci==0 and q==0 and layer==0:f.dump_trace(dest/'small_full_trace.json',graph,result,audit)
                cost[layer,q]=value;fifo[layer,q]=audit['peak_output_fifo_bytes']
                for k,v in audit['counts'].items():aggregate[k]+=v
                for k,v in audit['pipeline_counts'].items():aggregate['pipeline_'+k]+=v
                for k,v in audit['resource_work_ns'].items():aggregate[k]+=v
        good=cost>=0;arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo
        records.append(dict(name=name,variant=variant,Down=down,copies=copies,priority='DF' if df else 'GF',
                            requested_queries=int(cost.size),completed_queries=int(good.sum()),strict_failed_queries=int((~good).sum()),
                            cost_ns=f.stats(cost[good]) if good.any() else None,FIFO_peak_bytes=f.stats(fifo[good]) if good.any() else None,
                            valid_FIFO_capacities={str(k):int(np.count_nonzero(good&(fifo<=k))) for k in [64,128,256,2048]},
                            counts_and_work_per_valid_query={k:v/int(good.sum()) for k,v in aggregate.items()} if good.any() else {}))
        progress=dict(finished=ci+1,total=len(configs),unique_graphs=unique,elapsed_seconds=time.monotonic()-started,last=name)
        f.dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez_compressed(dest/'service_arrays.npz',**arrays)
    f.dump(dest/'pipeline_failures.json',[fail for rr in recipes.values() for fail in rr.failures])
    f.dump(dest/'sample_address_certificates.json',[c for rr in recipes.values() for c in rr.certificates])
    f.dump(dest/'report.json',dict(status='complete_conditional_uniform_shaping_TRAIN',limit=limit,layers=[0,1,2],records=records,
                                  unique_graphs=unique,elapsed_seconds=time.monotonic()-started,
                                  protocol_sha256=f.sha(DEST/'protocol.frozen.json'),
                                  genericcompiler='Exact candidate graph/layout alias;0%gain; no noveloptimizer attribution.',
                                  note='Same uniform split10/prefix1 policy granted to all3kernels. Matching-layout variants explicitly useIPR, never labeledsameWL. Real inactive-page array bytes charged;ECC onlytrue48pages. Strict failure neverpaced.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','smoke','run']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:execute(8 if args.action=='smoke' else 512)
