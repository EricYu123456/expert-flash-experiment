#!/usr/bin/env python3
"""First-feasible PB-contract policy for every layout, never FFN-cost oracle."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import shaping_addendum as a
f=a.f
DEST=OUT/'policy-addendum'
LAYOUTS={'candidate':'candidate','RR':'IPR_standard_RR_row_safe','matching':'sameWL_matching_keep_hot'}


def freeze():
    DEST.mkdir(exist_ok=True);assert not (DEST/'protocol.frozen.json').exists()
    inputs=[Path(__file__),OUT/'shaping_addendum.py',a.SHAPER,f.PIPE,f.CPP,
            a.ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py',f.TRAIN,
            f.BASE/'baselines/frozen_TRAIN_packed_expert_order.npz',OUT/'execution_plan.frozen.json']
    f.dump(DEST/'protocol.frozen.json',dict(
        frozen_utc=f.utc(),inputs_sha256={str(p):f.sha(p) for p in inputs},
        data='Existing W8 TRAIN first3layers512, no heldout.',layouts=LAYOUTS,
        selection='Same deterministic runtime compiler for all layouts: strict500 fill-only ifPB/rowfeasible; otherwise split10/prefix1; otherwise ascendingwave/index,prefix firstPBfeasible; otherwise explicitcoldreset fill-only. No fullFFNcost read by policy and no latency minimization.',
        configuration='copy500ns,sourcepermutation,twoPBslots,evaluationstartoverwrite,perplane copyconcurrency,discardinactive-beforeECCconditional. No retained-bias pacing.',
        axes=dict(input_copies=[1],priority=['GF'],Down=['g16_equal','g12_cyclic'],activation_ns=0),
        nature='Router IDs and current ready stream shape are available to a sparse compiler; contract feasibility decisions are runtime scheduling, not heldout/fullFFNcost oracle. Static layout never changes.',
        scope_caveat='Measuredhardwaretiming is unavailable; fixed9.7us cadence is a conditional micro-model. Rounding/cadencestretch controls can eliminate buffering failure or beat dummy-readshaping. Genericcompiler exactlytiescandidate.'))


class PolicyRecipes(a.ShapedRecipes):
    def __init__(self,pipeline,shaper,layout,order):
        variant={'candidate':'candidate_IPR_uniform_split10','RR':'RR_IPR_uniform_split10','matching':'matching_layout_IPR_uniform_split10'}[layout]
        super().__init__(pipeline,shaper,variant,order);self.layout=layout;self.decisions={};self.decision_counts=Counter()
    def make(self,waves,profile,columns,owner_ranks,validate=True):
        if not validate:return super().make(waves,profile,columns,owner_ranks,validate)
        settings=dict(cache_copy_ns=500,source_policy='permutation',issue_policy='strict_no_interval',
                      row_bytes=columns,validate_row_stream=True,unselected_policy='drop_before_ecc_conditional')
        original=dict(waves=waves,settings=settings,ranks=owner_ranks,layout=self.layout)
        key=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key in self.decisions:
            recipe=self.decisions[key];self.decision_counts[recipe['policy'] if recipe else 'infeasible']+=1;return recipe
        selected={int(p['owner'][1:]) for w in waves for p in w['pages'] if p['owner'].startswith('e')}
        pool=a.resident_pool(selected,self.layout,self.order)
        fill=self.shaper.shape_kernel(waves,(),pool)
        result=self.p.simulate(fill,settings);selected_waves=fill;policy='strict_fill_only';attempts=1
        if not result['feasible']:
            tries=[(10,1)]
            tries += [(wi,prefix) for wi,w in enumerate(waves) for prefix in range(1,len(w['pages'])) if (wi,prefix)!=(10,1)]
            for wi,prefix in tries:
                if wi>=len(waves) or prefix>=len(waves[wi]['pages']):continue
                shaped=self.shaper.shape_kernel(waves,((wi,prefix),),pool)
                trial=self.p.simulate(shaped,settings);attempts+=1
                if trial['feasible']:
                    result=trial;selected_waves=shaped;policy=f'strict_first_feasible_split_{wi}_{prefix}';break
            else:
                settings=dict(settings,issue_policy='cold_reset')
                result=self.p.simulate(fill,settings);selected_waves=fill;policy='explicit_cold_reset_fill_only';attempts+=1
        if not result['feasible']:
            self.decisions[key]=None;self.decision_counts['infeasible']+=1;return None
        before=[(p['page_id'],p['plane'],p['block'],p['wl']) for w in waves for plane in w['source_order'] for p in w['pages'] if p['plane']==plane]
        after=[(p['page_id'],p['plane'],p['block'],p['wl']) for w in selected_waves for plane in w['source_order'] for p in w['pages'] if p['plane']==plane]
        assert before==after and len(before)==48
        allowed={(p['page_id'],p['plane'],p['block'],p['wl']) for p in pool};seen=set()
        for w in selected_waves:
            true={p['page_id'] for p in w['pages']}
            for p in w['read_pages']:
                physical=(p['plane'],p['block'],p['wl']);assert physical not in seen;seen.add(physical)
                if p['page_id'] not in true:assert (p['page_id'],*physical) in allowed
        rows=Counter();milestones=[]
        for x in result['page_completions']:
            e=x['owner'];old=rows[e]//columns;rows[e]+=x['bytes'];new=rows[e]//columns
            if new>old:milestones.append((owner_ranks[e],x['end_ns'],(new-old)*4))
        counts={k:result[k] for k in ['waves','pages','bytes','physical_read_pages','physical_read_bytes','ecc_pages','ecc_bytes','cold_waves','cr_waves']}
        recipe=dict(duration=result['makespan_ns'],milestones=milestones,signature=key,counts=counts,policy=policy,
                    summary=dict(settings=settings,layout=self.layout,policy=policy,feasibility_attempts=attempts,**counts,
                                 no_full_FFN_cost_used=True,finite_PB_feasible=True,complete_row_stream=True))
        self.decisions[key]=recipe;self.decision_counts[policy]+=1
        if len(self.certificates)<3:self.certificates.append(dict(layout=self.layout,policy=policy,attempts=attempts,summary=recipe['summary'],true_stream_preserved=True,waves=selected_waves))
        return recipe


def execute(limit):
    spec=f.read(DEST/'protocol.frozen.json')
    for p,h in spec['inputs_sha256'].items():assert f.sha(p)==h,p
    dest=DEST/('smoke' if limit==8 else 'TRAIN512');dest.mkdir(exist_ok=True);assert not (dest/'report.json').exists()
    f.dump(dest/'run.frozen.json',dict(frozen_utc=f.utc(),protocol_sha256=f.sha(DEST/'protocol.frozen.json'),limit=limit,layers=[0,1,2]))
    with np.load(f.TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3,:limit]
    with np.load(f.BASE/'baselines/frozen_TRAIN_packed_expert_order.npz') as z:orders=z['slot_to_expert']
    pipeline=f.load_pipeline();shaper=a.load_shaper();executor=f.Executor(OUT);strong,_=f.load_strong()
    recipes={(layout,layer):PolicyRecipes(pipeline,shaper,layout,orders[layer]) for layout in LAYOUTS for layer in range(3)}
    arrays={};records=[];cache={};started=time.monotonic();unique=0
    configs=[(down,layout) for down in ['g16_equal','g12_cyclic'] for layout in LAYOUTS]
    for ci,(down,layout) in enumerate(configs):
        name=f'first_feasible500.{down}.input1.GF.a0.{layout}'
        cost=np.full((3,limit),-1,dtype=np.int64);fifo=cost.copy();aggregate=Counter();decisions=Counter()
        for layer in range(3):
            rr=recipes[layout,layer];before=rr.decision_counts.copy()
            for q,route in enumerate(routes[layer]):
                graph=f.build_graph(route,layer,LAYOUTS[layout],down,'strict500',f.Service(1,0,0),rr,orders,strong[layer])
                if graph is None:continue
                key,data=graph.signature();key=(key,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                if key in cache:value,audit=cache[key]
                else:
                    result=executor.run(graph,data);audit=f.audit(graph,result,full=(q in [0,1,limit-1]));value=result['cost_ns'];cache[key]=(value,audit);unique+=1
                    if ci==0 and q==0 and layer==0:f.dump_trace(dest/'small_full_trace.json',graph,result,audit)
                cost[layer,q]=value;fifo[layer,q]=audit['peak_output_fifo_bytes']
                for k,v in audit['counts'].items():aggregate[k]+=v
                for k,v in audit['pipeline_counts'].items():aggregate['pipeline_'+k]+=v
                for k,v in audit['resource_work_ns'].items():aggregate[k]+=v
            decisions.update(rr.decision_counts-before)
        good=cost>=0;arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo
        records.append(dict(name=name,layout=layout,Down=down,requested_queries=int(cost.size),completed_queries=int(good.sum()),
                            cost_ns=f.stats(cost[good]) if good.any() else None,FIFO_peak_bytes=f.stats(fifo[good]) if good.any() else None,
                            policy_decision_counts=dict(decisions),
                            valid_FIFO_capacities={str(k):int(np.count_nonzero(good&(fifo<=k))) for k in [64,128,256,2048]},
                            counts_and_work_per_valid_query={k:v/int(good.sum()) for k,v in aggregate.items()} if good.any() else {}))
        progress=dict(finished=ci+1,total=len(configs),unique_graphs=unique,elapsed_seconds=time.monotonic()-started,last=name)
        f.dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    comparisons=[]
    for down in ['g16_equal','g12_cyclic']:
        prefix=f'first_feasible500.{down}.input1.GF.a0.';ca=arrays[prefix+'candidate.cost']
        for layout in ['RR','matching']:
            other=arrays[prefix+layout+'.cost'];valid=(ca>=0)&(other>=0)
            c,b=ca[valid],other[valid]
            comparisons.append(dict(candidate=prefix+'candidate',baseline=prefix+layout,joint_queries=int(valid.sum()),
                                    mean_gain_pct=100*(1-float(c.mean()/b.mean())),p99_gain_pct=100*(1-float(np.percentile(c,99)/np.percentile(b,99))),
                                    slower_queries=int(np.count_nonzero(c>b)),equal_queries=int(np.count_nonzero(c==b)),
                                    three_layer_sum_p99_gain_pct=100*(1-float(np.percentile(ca.sum(axis=0),99)/np.percentile(other.sum(axis=0),99)))))
    np.savez_compressed(dest/'service_arrays.npz',**arrays)
    f.dump(dest/'sample_policy_certificates.json',[c for rr in recipes.values() for c in rr.certificates])
    report=dict(status='complete_conditional_first_feasible_TRAIN',limit=limit,layers=[0,1,2],records=records,comparisons=comparisons,
                protocol_sha256=f.sha(DEST/'protocol.frozen.json'),unique_graphs=unique,elapsed_seconds=time.monotonic()-started,
                generic_compiler='Exact candidate graph/layout alias;0%gain.',
                limits='No fullFFN-cost oracle,heldout,measuredhardware,voltagehold,globaloptimality. Conditional read cadence/copy/source/PB timing. FIFO128B underpage-outputburst;64Binvalid.')
    f.dump(dest/'report.json',report)
    lines=['# First-feasible PB-contract control','','Samepolicy for every staticlayout:fill-only iflegal,thenfixedsplit,thenfirstlegalcut,thenexplicitcoldreset.No fullFFNcost lookup.','','| Down | Baseline | Joint queries | Mean gain | P99 gain | Slower | Equal |','|---|---|---:|---:|---:|---:|---:|']
    for c in comparisons:lines.append(f"| {c['candidate'].split('.')[1]} | {c['baseline'].split('.')[-1]} | {c['joint_queries']} | {c['mean_gain_pct']:.5f}% | {c['p99_gain_pct']:.5f}% | {c['slower_queries']} | {c['equal_queries']} |")
    lines+=['','Genericcompilerexactlytiescandidate. Alltimingsremainconditional. Fixed9.7us timerassumption is fragile:slightlyextendedreadphase/cadence mayeliminatePBfailureorbeatshaping; hardwaretimingunverified.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','smoke','run']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:execute(8 if args.action=='smoke' else 512)
