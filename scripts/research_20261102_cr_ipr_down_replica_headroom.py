#!/usr/bin/env python3
"""Low-cost fixed-GU g4-Down replica assignment headroom screen; TRAIN only.

No replica/full-FFN implementation, weight write, arithmetic change or heldout
execution. The oracle retains one384-row native job per routed Down band and
one96-row shared Down job on every chip. It relaxes other resource contention.
"""
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr'
ASYM=BASE/'full-ffn/asymmetric-down'
OUT=BASE/'down-replica-headroom'
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def need(x,s):
    if not x:raise RuntimeError(s)


def pipeline():
    spec=importlib.util.spec_from_file_location('replica_headroom_frozen_pipeline',PIPE)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def native_cost(p,pages,planes,columns):
    waves=[]
    width=len(planes)
    for lo in range(0,pages,width):
        pp=[dict(page_id=f'x.p{j}',owner='x',page_index=j,plane=planes[j%width],block=0,
                 wl=j//width,bytes=16384) for j in range(lo,min(lo+width,pages))]
        waves.append(dict(pages=pp,source_order=[x['plane'] for x in pp]))
    r=p.simulate(waves,cache_copy_ns=500,source_policy='fixed',issue_policy='strict_no_interval',row_bytes=columns)
    need(r['feasible'],'native kernel must be feasible')
    return dict(duration_ns=r['makespan_ns'],waves=r['waves'],pages=r['pages'],columns=columns,
                planes=planes,rows=pages*16384//columns,settings=r['config'],
                first_page_ns=r['first_page_ready_ns'],output_last_burst_bytes=r['row_completions'][-1]['row_end']*4-r['row_completions'][-1]['row_start']*4)


def calendar_lb(available,B,S):
    # Shared g16 contributes S to every chip; route has24 equal routed bands.
    # This is a lower bound for this fixed job geometry, not a global FFN LB.
    lo=max(available)+S
    hi=lo+24*B
    while lo<hi:
        mid=(lo+hi)//2
        slots=sum(max(0,(mid-a-S)//B) for a in available)
        if slots>=24:hi=mid
        else:lo=mid+1
    return int(lo)


def group_witness(available,releases,B,S):
    # Standard whole-Down-expert copies in all four disjoint g4 groups.
    # Coupled expert assignment is stricter than choosing its four bands apart.
    ranks=[[0,1],[2,3],[4],[5]]
    jobs=[]
    for chip,a in enumerate(available):
        t=int(a)
        for rank in ranks[chip//4]:
            t=max(t,int(releases[rank]))
            jobs.append(dict(chip=chip,expert_rank=rank,band=chip%4,row_start=(chip%4)*384,
                             rows=384,columns=512,start_ns=t,end_ns=t+B))
            t+=B
        t=max(t,int(releases[6]))
        jobs.append(dict(chip=chip,expert_rank=6,rows=96,columns=1024,start_ns=t,end_ns=t+S))
    need(len([x for x in jobs if x['expert_rank']<6])==24,'all24 bands assigned once')
    need(len({(x['expert_rank'],x['band']) for x in jobs if x['expert_rank']<6})==24,'band coverage')
    return dict(chip_calendar_end_ns=max(x['end_ns'] for x in jobs),expert_destination_groups=ranks,jobs=jobs,
                scope='Chip-local release/nonoverlap witness only. Global command/channel/PCIe priorities and FIFO are relaxed; no full-FFN schedule is claimed.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    priors=[ROOT/'results/research-2026-11-02/round2/g4-joint-prefix-replica-screen/native-v1/report.json',
            ROOT/'results/research-2026-11-02/round2/g4-joint-prefix-replica-screen/same-allocated-witness/audit.json',
            ROOT/'results/research-2026-11-02/round2/deep-gu-paired-replica-screen/independent-contract/contract_notes.txt',
            ROOT/'scripts/research_20261102_clean_replica.py']
    paths=[Path(__file__).resolve(),PIPE,TRAIN,ASYM/'Down_menus.frozen.npz',ASYM/'selection.frozen.json',
           ASYM/'protocol.frozen.json',ASYM/'selected_oneblock_capacity_audit.json',
           ASYM/'TRAIN512/service_arrays.npz',ASYM/'TRAIN512/report.json',*priors]
    traces={cp:ASYM/f'TRAIN512/trace.candidate.g4_reused_TRAIN_labels.input{cp}.GF.json' for cp in [1,4]}
    paths+=list(traces.values())
    hashes={str(p):sha(p) for p in paths}
    protocol=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=hashes,
                  decision='At most a low-cost optimistic band-assignment/capacity screen; stop below5% full-FFN headroom.',
                  scope='Fixed current candidate GU owner-release calendars, g4_reused_TRAIN_labels384-row native routed Down jobs, sharedg16 Down unchanged.',
                  data='Existing TRAIN32prompts×16decode×first3MoElayers; noheldout/noquality/noinference.')
    frozen=OUT/'protocol.frozen.json'
    if frozen.exists():need(json.loads(frozen.read_text())['inputs_sha256']==hashes,'frozen inputs changed; create explicit new revision')
    else:write(frozen,protocol)
    p=pipeline()
    kernels=dict(four_plane_band=native_cost(p,12,[0,1,2,3],512),
                 two_light_plane_band=native_cost(p,12,[2,3],512),
                 shared_g16=native_cost(p,6,[0,1,2,3],1024),
                 equal_GU_half=native_cost(p,24,[0,1,2,3],1536))
    need([kernels[k]['duration_ns'] for k in kernels]==[59220,82120,43860,89940],'independent expected kernel times')
    with np.load(TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3].copy()
    with np.load(ASYM/'Down_menus.frozen.npz') as z:
        inventory=list(z.files);menu=z['g4_reused_TRAIN_labels'].copy()
        need(all(z[k].shape==(3,65,16) for k in inventory),'frozen menu inventory')
    need(np.all(menu[:,:64].sum(axis=2)==1536) and np.all((menu[:,:64]==0)|(menu[:,:64]==384)),'g4band geometry')
    need(np.all(menu[:,64]==96),'fixed sharedg16 geometry')
    load=np.asarray([[np.count_nonzero(menu[l,route],axis=0) for route in routes[l]] for l in range(3)])
    need(np.all(load.sum(axis=2)==24),'24 routed bands/event')
    maxload=load.max(axis=2)
    distribution={str(k):int(v) for k,v in Counter(maxload.ravel()).items()}
    arrays=np.load(ASYM/'TRAIN512/service_arrays.npz')
    cases=[]
    npout=dict(routes=routes,original_routed_band_load=load,original_max_bands=maxload)
    for cp,path in traces.items():
        trace=json.loads(path.read_text());ns=trace['nodes'];service=trace['service']
        available=np.zeros(16,dtype=np.int64)
        for j in trace['jobs']:
            if j['phase']==0:
                available[j['chip']]=ns[j['release']]['end_ns']
                # No existing Down input delays the observed GU release: it is
                # the intrinsic compute end plus this job's final ONFI burst.
                last=ns[j['output_ONFI'][-1]]
                need(ns[j['release']]['end_ns']==ns[j['compute']]['end_ns']+last['duration'],'GU calendar contains downstream interference')
        releases=np.asarray([next(n['end_ns'] for n in ns if n['kind']=='Down_submit' and n['expert_rank']==rank) for rank in range(7)])
        B=service['command_ns']+512*cp//2+kernels['four_plane_band']['duration_ns']+64
        S=service['command_ns']+1024*cp//2+kernels['shared_g16']['duration_ns']+32
        bound=calendar_lb(available.tolist(),B,S)
        witness=group_witness(available,releases,B,S)
        need(witness['chip_calendar_end_ns']==bound,'wholeexpert group control does not match oracle')
        baseline=arrays[f'candidate.g4_reused_TRAIN_labels.input{cp}.GF.cost'].copy()
        rr=arrays[f'RR.g4_reused_TRAIN_labels.input{cp}.GF.cost'].copy()
        fixed=(available+S+load*B).max(axis=2)
        # Exact8ns residual is the final shared output PCIe packet. This checks
        # all1536 events against the saved campaign without rerunning its DAG.
        need(np.all(baseline-fixed==8),'fixed calendar does not explain every current TRAIN cost')
        per_event_upper=baseline-bound
        npout[f'input{cp}.baseline']=baseline;npout[f'input{cp}.calendar_LB']=np.full_like(baseline,bound)
        npout[f'input{cp}.fixed_band_load_LB']=fixed;npout[f'input{cp}.saving_upper_bound_ns']=per_event_upper
        rec=dict(input_copies=cp,events=1536,GU_available_ns=available.tolist(),Down_submit_release_ns=releases.tolist(),
                 minimum_band_owner_ns=B,minimum_shared_owner_ns=S,unlimited_fast_copy_calendar_LB_ns=bound,
                 coarser_min_GU_release_LB_ns=int(available.min()+2*B+S),
                 baseline_mean_ns=float(baseline.mean()),RR_reoptimized_mean_ns=float(rr.mean()),
                 candidate_gain_over_RR_pct=float(100*(1-baseline.mean()/rr.mean())),
                 headroom_upper_bound_pct=float(100*(1-bound/baseline.mean())),
                 mean_saving_upper_bound_ns=float(per_event_upper.mean()),
                 fixed_original_calendar_LB_mean_ns=float(fixed.mean()),fixed_current_cost_residual_ns=8,
                 ordinary_wholeexpert_group_copy_witness=witness,
                 limitations='Oracle relaxes global contention/priority, selector cost and FIFO; treats allcopies as fast4plane with unlimited destinations. It retains fixed currentGU and sharedg16 and one384-row native job/band.')
        need(rec['headroom_upper_bound_pct']<5,'unexpected expansion gate; do not start implementation automatically')
        cases.append(rec)
    capacity=json.loads((ASYM/'selected_oneblock_capacity_audit.json').read_text())
    ledger=[]
    for family in ['candidate','RR']:
        row=next(x for x in capacity['plans'] if x['GU']==family and x['layer']==0)
        used=np.array(row['resident_pages_per_chip_plane'],dtype=np.int64)
        need(np.all(used==used[0]),'symmetric capacity required')
        for blocks in [1,2]:
            free=blocks*192-used[0]
            fast=int(free.min()//3)
            slow_after_fast=int(min(free[2]-3*fast,free[3]-3*fast)//6)
            # All4fixed g4groups perexpert: original+3extra completeDowncopies.
            additional=np.full(4,144)
            fit_all_four=bool(np.all(used[0]+additional<=blocks*192))
            ledger.append(dict(GU=family,allocated_blocks_per_chip=blocks,
                               allocated_LSB_bytes=16*blocks*4*192*16384,
                               base_pages_per_chip_plane=used[0].tolist(),free_pages_per_chip_plane=free.tolist(),
                               fast384row_copy_slots_per_chip=fast,fast_band_copy_slots_allchips=fast*16,
                               slow_plane23_copy_slots_after_max_fast_per_chip=slow_after_fast,
                               standard_all4group_completeDowncopies_fit=fit_all_four,
                               all4group_extra_pages_per_chip_plane=additional.tolist(),
                               total_pages_if_all4group_copies=(used[0]+additional).tolist()))
    extra_bytes=64*3*1536*512
    need(extra_bytes==64*(2*512*1536+1536*512),'fullFFN-copy budget equivalence')
    equal_cut_lb=2*kernels['equal_GU_half']['duration_ns']+maxload*kernels['four_plane_band']['duration_ns']+kernels['shared_g16']['duration_ns']
    equal_cut=dict(two_GU_jobs_compute_ns_per_chip=2*kernels['equal_GU_half']['duration_ns'],
                   fixed_original_max_load_mean=float(maxload.mean()),
                   compute_only_exclusive_chip_LB_mean_ns=float(equal_cut_lb.mean()),
                   input1_current_mean_ns=cases[0]['baseline_mean_ns'],
                   excess_over_input1_current_ns=float(equal_cut_lb.mean()-cases[0]['baseline_mean_ns']),
                   omitted='All new command/fullinput/output costs, which cannot improve this lower bound.',
                   conclusion='Reject the simple twoequal commonGU g16 cut with unchanged currentg4 Down supports; this does not reject heterogeneous GUcuts/supports.')
    npout['equal_two_GU_cut_compute_only_LB']=equal_cut_lb
    np.savez_compressed(OUT/'query_oracle_arrays.npz',**npout)
    need(hashes=={p:sha(p) for p in hashes},'frozen inputs changed during screen')
    report=dict(status='negative_stop_no_copy_or_fullFFN_implementation',utc=protocol['utc'],inputs_sha256=hashes,
                data_scope=protocol['data'],scope=protocol['scope'],frozen_Down_menu_inventory=inventory,
                kernels=kernels,events=1536,original_max_routed_band_distribution=distribution,
                original_mean_max_routed_bands=float(maxload.mean()),assignment_cases=cases,capacity_ledger=ledger,
                standard_controls=dict(complete_Down_expert_replication='Replicate all4contiguous384-row bands together into each of fourdisjoint4chipgroups; routechooses a wholeexpert group. Its chip-calendar witness matches the unlimited independent-band oracle.',
                                       replicated_declustering='Independent band destination selection is ordinary replicateddeclustering; it is a control, not a new placement primitive.',
                                       all4group_extra_useful_bytes=extra_bytes,
                                       base_onecopy_useful_bytes=155713536,total_with_3extraDowncopies_bytes=155713536+extra_bytes,
                                       full_routed_FFN_expert_copy_same_extra_byte_budget=True,
                                       qualification='Capacity/page accounting and chip-local oracle only; no fullFFN speed or actual programmed-copy certificate is claimed.'),
                prior_findings=dict(g4_joint_native_gain_pct=2.1524268753560616,
                                    equal_allocated_whole_GU_witness='Previous completeGU replicas reproduced joint prefix service inside the same32allocatedblocks, so prefix programmed-page savings were not an allocated-capacity advantage.',
                                    adjacent_row_coalescing='Oldclean-replica explicitly shows input/command savings from adjacent sameexpert row coalescing. This screen fixes384-row jobgranularity and does not bound that changed geometry.',
                                    paths=[str(x) for x in priors]),
                equal_two_GU_cut=equal_cut,
                next_headroom='Heterogeneous GUjob cuts or different GU/Down supports might change release calendars. A new nativejob pays freshcold/command/fullinput. The simple equal twojob g16 cut is already rejected by the retained compute-only bound; no new experiment is started.',
                limitations=['Fixed currentGU release calendars, current384-row Downnativejobs and sharedg16 only; not a global lower bound over everyGU/Down configuration.',
                             'Unlimitedcopies and uncontented resources are optimistic. Finitecopy budgets, slowerplane23copies and selector costs cannot beat this bound within this fixedjob model.',
                             'No merging adjacentbands, newMAC, largerFIFO, multiinput context, partialsum, changedquantization or heldoutdata is assumed.',
                             'Strict500ns copy, sourceorder/drop, skipWL CR and IPR follow the previously declared conditional pipeline model; no measuredhardware timing claim.',
                             'Oneblock and twoblock budgets are both shown. Standard wholeexpert/replicateddeclustering controls receive the same allocatedcapacity.'],
                source_sha256=sha(__file__),arrays_sha256=sha(OUT/'query_oracle_arrays.npz'),actual_command=['python',str(Path(__file__).resolve())])
    write(OUT/'report.json',report)
    (OUT/'report.md').write_text(f'''Stop: unlimited fast private-Down band copies leave less than5% optimistic full-FFN headroom for the fixedcurrentGU + g4 Down jobs + sharedg16 configuration. No copies or new full-FFN replay were implemented.

The frozen TRAIN archive supplies512ordered top6routes ×3layers. Actual g4_reused_TRAIN_labels menus give24routed384-row bands/event. Maximum routed bands on onechip:2 in1202events,3 in328,4 in6; exactmean{maxload.mean():.12f}. Existing candidate versus equallyreoptimizedRR is only0.4016%input1 /0.3942%input4.

The finitePB pipeline gives59,220ns for a12page/3wave fourplane band; a plane2/3only copy needs6waves and82,120ns. Sharedg16 Down costs43,860ns. Everyjob still uploads the fullprivateC512vector, has onechip owner, and preserves384complete outputrows and their fixedbytes/scales.

With unlimited fastcopies, fixedGU ownerrelease calendars and minimumcommand/input/lastONFI owner time, the exactchip-calendar assignment bound is324,164ns forinput1 and329,540ns forinput4. Against currentmeans338,596.218750ns /344,970.645833ns, optimistic headroom is{cases[0]['headroom_upper_bound_pct']:.6f}% /{cases[1]['headroom_upper_bound_pct']:.6f}%. Every oneof1536currentcosts is independently explained by its frozenGU calendar plus nativeDown ownerload and an8ns finalPCIe packet. Otherresources/priority/FIFO/selector costs are relaxed in the oracle.

The common two-block budget can store all64experts' completeDown matrices in all4disjoint4-chip groups. This standardwholeexpert-copy control matches the independent-band oracle at chip-calendar level. Threeextra completeDown copies cost150,994,944usefulbytes, the same extra budget as onecompleteGU+DownFFNcopy ofall64routedexperts. No novelreplica/declustering primitive or allocatedcapacity advantage is claimed. One-block capacities are also disclosed: candidatehas only14/14/73/73freepages perchipplane versusRR43/43/44/44; the morelimited plane2/3copies are slower and cannot improve the unlimitedfast-copy bound.

The oldsame-allocated wholeGU replica witness and priorbounded replica no-gos are retained as context; their service regimes differ from this CRmodel. Adjacent sameexpert-band coalescing is outside the fixed384-row joboracle, as is a differentGU release calendar.

A simple equalGU split is separately rejected: two6wave nativejobs cost179,880ns/chip in compute alone. Adding actualfixedDown maxloads and sharedg16 gives meanexclusive-chip lowerbound{equal_cut_lb.mean():.6f}ns, above the achieved338,596.218750ns before any freshinput/command/output costs. HeterogeneousGUcuts or differentGU/Downsupports remain an untested headroom direction; no newexperiment is launched.

This is TRAINonly conditionalsoftware accounting, not hardwaremeasurement, taskquality, numericalABI evidence or a globaloptimum over changedjobgeometry. Reproduce: `python {Path(__file__).resolve()}`. Fullcommands, SHA256inputs, perqueryarrays and localassignment witnesses are retained in report.json/query_oracle_arrays.npz.
''')
    print(json.dumps(dict(status=report['status'],events=1536,copy_headroom_pct=[x['headroom_upper_bound_pct'] for x in cases],equal_cut_compute_LB_mean_ns=float(equal_cut_lb.mean())),indent=2))


if __name__=='__main__':main()
