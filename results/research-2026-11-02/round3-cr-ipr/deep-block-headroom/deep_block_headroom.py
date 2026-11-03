#!/usr/bin/env python3
"""Two-block Deep GU geometry controls, not a native completion-time model."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,gzip,hashlib,itertools,json,time
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'scripts/plane_wave_model.py').exists())
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/deepseek/decode.routes.npz'
EXPECTED_TRAIN='11168c1e899c8ce0bf23e719e908f243fa09d30980a65488a3f92aa448344fed'
PAGE,COLUMNS,EXPERTS,K,BLOCK=16384,2048,64,6,192

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def stats(a):
    a=np.asarray(a);u,n=np.unique(a,return_counts=True)
    return dict(mean=float(a.mean()),p99=float(np.percentile(a,99)),minimum=int(a.min()),maximum=int(a.max()),histogram={str(int(k)):int(v) for k,v in zip(u,n)})
def prof(n,phase=0):return np.bincount((np.arange(n)+phase)%4,minlength=4).astype(np.int16)


def freeze():
    assert not (OUT/'protocol.frozen.json').exists()
    assert sha(TRAIN)==EXPECTED_TRAIN
    protocol=dict(frozen_utc=utc(),source_sha256=sha(__file__),TRAIN=dict(path=str(TRAIN),sha256=sha(TRAIN)),
                  config=dict(H=2048,I=1408,E=64,K=6,shared_I=2816,GU_chips=16,planes=4,page_bytes=16384,block_WLs=192,routed_pages_per_chip=22,shared_pages_per_chip=44,coefficient_bits=8),
                  scope='Existing W8/A8 TRAIN26layers512inputs only; common-input GU geometry/cold-cost screening, no fullFFN/heldout/inference.',
                  capacity='Matchedcontrols:2four-plane192WLblocks/chip,1452programmedpages/chip,176selectedpages/query/chip,onecopy,nooverfetch. Legacycompensator uses3blocks and is only extra-capacityreference.',
                  execution='Synchronous sameblock-index epochs. Within eachblock take one requestedpage from eachnonemptyplane in fixed0123order; tails shrinkmonotonically, so noidleplane resumes in thatblock. Everyactiveblock starts cold28us, later epochs9.7us under skip-WL premise. No independently advancingplanes/crossblockIPR assumed.',
                  lower_bounds='Global44pagewaves isonlyworkLB. Per-layout sumceil(blockpages/4) isintegerblockLB; summax(perblockplane) isconstructible NANDpage-wave count. Independentperplane blockwork is optimistic only. ECC6.4GB/s +first28us release gives anothernecessarybound, not completiontime.',
                  universal44='Anecessaryalltop6/globalprofilecapacitybound is44+(64-6)*ceil(22/4)=392>384pages/plane; hence2blocks cannot guarantee44waves foralltop6 underowner-homogeneouswholepageplacement. Not a scheduling/PB sufficientcondition ornewtheorem.',
                  static_controls=['flatglobalRR sharedlast','wholeexperts32/32 RRphases sharedblock0','wholeexperts32/32 uniform4phases sharedblock0','blockstriped11/11 withglobalphasecontinuation','blockstriped11/11 uniform4phases'],
                  TRAIN_controls=['frequencyorderedflatRR sharedfirst','frequent32wholeexperts block0+shared','capacityrepair ofaligned11/11 compensation byleastfrequencywholepageevacuation','TRAIN perblockphasecoordinatecontrol','generic8bin one-pagemoves withsharedjointlymovable'],
                  learned_phase=dict(starts=['bothstaticstripedmaps'],passes=4,expert_order='ascendingID',objective='lexicographic(sumNANDsimplecost,sumwaves,sumglobalplaneLB)',hard_capacity=True),
                  generic=dict(start='bestTRAINmean incumbent among all matchedcontrols including learnedphase andcapacityrepair',passes=4,expert_order='0..64,sharedlast',proposal='Everypositive sourcebin to everyother destinationbin, onepage, destinationresident<192; choosebeststrictlexicographicdelta once/expert/pass; stop ifnosuccess.',objective='sum(9700*actualblockwaves+18300*activeblocks),thenwaves,thenglobalplaneLB; no newalgorithm/globaloptimumclaim'),
                  frequency='TRAINperlayer selectioncounts; tiesexpertID. Hotgroup isfirst32frequencyrankedexperts. Capacityrepair choosesleastfrequencyexpertthenID, movesmaxneeded wholepages fromoverflowplanes tothatblocklowplane.',
                  page_rows='C2048 divides16KiB:8fullrows/page. Everypagekeepsfullinput andoriginalrowidentity; fixed0123pageorder maypermute rows/owners withscatter, neverpartialcolumn/dot ornewaccumulators.',
                  omitted='FinitePB/ECCtimeline/sourcecommand/nativeABI/FIFO/fullFFN/Down/control/vectorcosts. NANDsimplecosts andresourcelowerbounds are not achievednativecompletionlatency.',
                  decision='If standardstriping/compiler controls closeheadroom, reportnegativefornewalgorithm and stop; do not expand fullFFN automatically.')
    dump(OUT/'protocol.frozen.json',protocol)
    print(json.dumps(dict(status='frozen',source_sha256=protocol['source_sha256'])),flush=True)


def valid(c,blocks=2):
    assert c.shape==(65,blocks,4) and np.all(c>=0)
    assert np.all(c[:64].sum(axis=(1,2))==22) and c[64].sum()==44
    assert np.all(c.sum(axis=0)<=192)


def flat(order,shared_first=False):
    c=np.zeros((65,2,4),dtype=np.int16);index=0
    for e in ([64]+list(order) if shared_first else list(order)+[64]):
        for _ in range(44 if e==64 else 22):
            c[e,index//768,index%4]+=1;index+=1
    valid(c);return c


def whole(order,uniform_four=False):
    c=np.zeros((65,2,4),dtype=np.int16)
    for pos,e in enumerate(order):
        phase=pos%4 if uniform_four else (pos%32)*22%4
        c[e,pos//32]=prof(22,phase)
    c[64,0]=prof(44);valid(c);return c


def striped(uniform_four=False):
    c=np.zeros((65,2,4),dtype=np.int16)
    for e in range(64):
        phase=e%4 if uniform_four else e*11%4
        c[e,0]=prof(11,phase);c[e,1]=prof(11,(phase+11)%4)
    c[64,0]=prof(22,0);c[64,1]=prof(22,2)
    valid(c);return c


def rare_repair(frequency):
    c=np.zeros((65,2,4),dtype=np.int16)
    for b,low in [(0,3),(1,2)]:
        c[:64,b]=3;c[:64,b,low]=2;c[64,b]=4;c[64,b,low]=10
        resident=c[:,b].sum(axis=0)
        for p in range(4):
            overflow=max(0,int(resident[p]-192))
            for e in sorted(range(64),key=lambda e:(frequency[e],e)):
                if not overflow:break
                move=min(overflow,int(c[e,b,p]));c[e,b,p]-=move;c[e,b,low]+=move
                resident[p]-=move;resident[low]+=move;overflow-=move
            assert not overflow
    valid(c);return c


def legacy_compensator():
    c=np.zeros((65,3,4),dtype=np.int16);cursor=np.zeros(4,dtype=int)
    for e in range(65):
        for p,n in enumerate([8,8,14,14] if e==64 else [6,6,5,5]):
            for _ in range(n):c[e,cursor[p]//192,p]+=1;cursor[p]+=1
    valid(c,3);return c


def loads(c,routes):return (c[routes].sum(axis=1)+c[64]).astype(np.int16)
def evaluate_load(x):
    w=x.max(axis=2).sum(axis=1).astype(np.int64)
    blocks=(x.sum(axis=2)>0).sum(axis=1).astype(np.int64)
    global_plane=x.sum(axis=1).max(axis=1).astype(np.int64)
    cost=w*9700+blocks*18300
    return cost,w,blocks,global_plane
def score(x):
    cost,w,_,gp=evaluate_load(x);return (int(cost.sum()),int(w.sum()),int(gp.sum()))


def phase_fit(start,routes):
    c=start.copy();x=loads(c,routes);resident=c.sum(axis=0);accepted=attempted=0
    for _ in range(4):
        changed=0
        for e in range(64):
            ii=np.flatnonzero((routes==e).any(axis=1))
            if not len(ii):continue
            for b in range(2):
                before=score(x[ii]);best=(0,0,0);bestdata=None
                for phase in range(4):
                    pp=prof(11,phase);delta=pp-c[e,b]
                    if not np.any(delta) or np.any(resident[b]+delta>192):continue
                    trial=x[ii].copy();trial[:,b]+=delta;after=score(trial)
                    diff=tuple(a-z for a,z in zip(after,before));attempted+=1
                    if diff<best:best=diff;bestdata=(pp,delta,trial)
                if bestdata is not None:
                    pp,delta,trial=bestdata;c[e,b]=pp;resident[b]+=delta;x[ii]=trial;accepted+=1;changed+=1
        if not changed:break
    valid(c);return c,dict(accepted=accepted,attempted=attempted,final_score=score(x))


def generic_fit(start,routes):
    c=start.copy();x=loads(c,routes);resident=c.sum(axis=0);accepted=attempted=0
    original=score(x)
    for _ in range(4):
        changed=0
        for e in range(65):
            ii=np.arange(len(routes)) if e==64 else np.flatnonzero((routes==e).any(axis=1))
            if not len(ii):continue
            old=score(x[ii]);best=(0,0,0);bestdata=None
            for source in range(8):
                sb,sp=divmod(source,4)
                if not c[e,sb,sp]:continue
                for dest in range(8):
                    db,dp=divmod(dest,4)
                    if dest==source or resident[db,dp]>=192:continue
                    trial=x[ii].copy();trial[:,sb,sp]-=1;trial[:,db,dp]+=1
                    after=score(trial);diff=tuple(a-z for a,z in zip(after,old));attempted+=1
                    if diff<best:best=diff;bestdata=(sb,sp,db,dp,trial)
            if bestdata is not None:
                sb,sp,db,dp,trial=bestdata;c[e,sb,sp]-=1;c[e,db,dp]+=1
                resident[sb,sp]-=1;resident[db,dp]+=1;x[ii]=trial;accepted+=1;changed+=1
        if not changed:break
    valid(c);assert score(x)<=original
    return c,dict(accepted=accepted,attempted=attempted,initial_score=original,final_score=score(x))


def address_ledger(c):
    cursor=np.zeros(c.shape[1:],dtype=np.int16);ledger=[]
    for e in range(65):
        local=0
        for b in range(c.shape[1]):
            for p in range(4):
                for _ in range(int(c[e,b,p])):
                    w=int(cursor[b,p]);cursor[b,p]+=1
                    ledger.append([e,local,p,b,w]);local+=1
        assert local==(44 if e==64 else 22)
    a=np.asarray(ledger,dtype=np.int16)
    assert len(a)==1452 and len(set(map(tuple,a[:,2:])))==1452
    assert np.all(a[:,4]<192) and np.array_equal(cursor,c.sum(axis=0))
    return a


def actual_schedule(ledger,route):
    selected=set(map(int,route));selected.add(64);bybin=defaultdict(list)
    for e,j,p,b,w in ledger:
        if int(e) in selected:bybin[int(b),int(p)].append([int(e),int(j),int(p),int(b),int(w)])
    waves=[];now=0;prior=set();priorblock=None;seen=set();cold_planes=Counter()
    for b in sorted({key[0] for key in bybin}):
        for p in range(4):bybin[b,p].sort(key=lambda page:page[4])
        n=max(len(bybin[b,p]) for p in range(4))
        for i in range(n):
            pages=[bybin[b,p][i] for p in range(4) if i<len(bybin[b,p])]
            active={page[2] for page in pages};cold=priorblock!=b or not active<=prior
            if priorblock==b:assert active<=prior,'No plane can idle then resume in a block.'
            duration=28000 if cold else 9700
            for e,j,p,block,w in pages:
                assert block==b and (e,j) not in seen;seen.add((e,j))
                if cold:cold_planes[p]+=1
            waves.append(dict(block=b,kind='cold' if cold else 'cr',start_ns=now,end_ns=now+duration,pages=pages,source_order=[page[2] for page in pages]))
            now+=duration;prior=active;priorblock=b
    assert len(seen)==176
    assert all(sum(e==owner for e,j in seen)==(44 if owner==64 else 22) for owner in selected)
    return dict(waves=waves,nand_simple_ns=now,selected_pages=176,cold_plane_starts=dict(cold_planes),
                full_rows_per_page=8,row_identity='row8*local_page throughrow8*local_page+7 within originalchipshard; scatterpreservescomplete rowdot.',native_PB_ECC_feasibility_proven=False)


def run():
    protocol=read(OUT/'protocol.frozen.json');assert protocol['source_sha256']==sha(__file__)
    assert sha(TRAIN)==protocol['TRAIN']['sha256'] and not (OUT/'report.json').exists()
    with np.load(TRAIN) as z:
        assert z['split'].item()=='train' and z['completed'].all()
        routes=z['routes'].transpose(2,0,1,3).reshape(26,512,6);layers=z['layers'];ids=z['prompt_ids']
    report=dict(status='complete_TRAIN_two_block_GU_geometry',source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.frozen.json'),
                TRAIN_shape=[32,16,26,6],physical_layers=layers.tolist(),ordered_route_sha256=digest(routes),previously_observed=True,
                selected_pages_per_chip=176,selected_bytes_per_chip=176*PAGE,programmed_pages_per_chip=1452,
                controls={},generic_search=[],phase_search=[],limits=protocol['omitted'])
    plans=defaultdict(list);values=defaultdict(list);ledgers=[];traces=[];started=time.monotonic()
    static={'flat_global_RR':flat(np.arange(64)),
            'whole32_RRphase_shared0':whole(np.arange(64)),
            'whole32_uniform4phase_shared0':whole(np.arange(64),True),
            'striped11_global_phase_continuation':striped(),
            'striped11_uniform4phase':striped(True)}
    for layer,r in enumerate(routes):
        frequency=np.bincount(r.reshape(-1),minlength=64)
        hot=np.asarray(sorted(range(64),key=lambda e:(-frequency[e],e)),dtype=int)
        candidates={name:c.copy() for name,c in static.items()}
        candidates.update(flat_RR_TRAIN_frequency_sharedfirst=flat(hot,True),
                          whole32_TRAIN_frequency_shared0=whole(hot),
                          striped11_TRAIN_rare_capacity_repair=rare_repair(frequency))
        fitted=[]
        for initial in ['striped11_global_phase_continuation','striped11_uniform4phase']:
            c,rec=phase_fit(candidates[initial],r);fitted.append((score(loads(c,r)),initial,c,rec))
        _,initial,c,rec=min(fitted,key=lambda item:(item[0],item[1]));candidates['striped11_TRAIN_phase_coordinate']=c
        report['phase_search'].append(dict(layer=layer,initial=initial,**rec))
        incumbent=min(candidates,key=lambda name:(score(loads(candidates[name],r)),name))
        c,rec=generic_fit(candidates[incumbent],r);candidates['generic_TRAIN_block_plane_onepage_moves']=c
        report['generic_search'].append(dict(layer=layer,initial=incumbent,**rec))
        candidates['legacy_compensation_3blocks_reference']=legacy_compensator()
        for name,c in candidates.items():
            matched=c.shape[1]==2;valid(c,c.shape[1]);x=loads(c,r);cost,w,blocks,gp=evaluate_load(x)
            block_integer=((x.sum(axis=2).astype(np.int64)+3)//4).sum(axis=1)
            perplane_runs=(x>0).sum(axis=1)
            optimistic_plane=(x.sum(axis=1).astype(np.int64)*9700+perplane_runs*18300).max(axis=1)
            resource=np.maximum(cost,28000+450560)
            assert np.all(w>=block_integer) and np.all(block_integer>=44) and np.all(cost>=optimistic_plane)
            plans[name].append(c);values[name].append(np.column_stack((w,blocks,cost,gp,block_integer,optimistic_plane,resource)))
            ledger=address_ledger(c);ledgers.append(dict(control=name,layer=layer,matched_two_blocks=matched,columns=['expert','local_page','plane','block','WL'],addresses=ledger.tolist(),address_sha256=digest(ledger)))
            # Real address/consume construction for four fixed TRAIN inputs perlayer.
            for q in [0,171,341,511]:
                schedule=actual_schedule(ledger,r[q]);assert schedule['nand_simple_ns']==cost[q]
                assert len(schedule['waves'])==w[q]
                if layer==0 and q in [0,511]:traces.append(dict(control=name,layer=layer,query=q,route=r[q].tolist(),schedule=schedule))
        print(json.dumps(dict(finished_layer=layer+1,total_layers=26,elapsed_seconds=time.monotonic()-started,initial_generic=incumbent)),flush=True)
    arrays={}
    for name,xx in values.items():
        xx=np.asarray(xx,dtype=np.int64);plan=np.asarray(plans[name],dtype=np.int16)
        arrays[name+'.metrics']=xx;arrays[name+'.counts']=plan
        report['controls'][name]=dict(matched_two_blocks=plan.shape[2]==2,physical_blocks_per_chip=plan.shape[2],
                                      programmed_pages_per_chip=1452,allocated_pages_per_chip=plan.shape[2]*768,
                                      waves=stats(xx[:,:,0]),active_block_cold_waves=stats(xx[:,:,1]),
                                      NAND_simple_ns=stats(xx[:,:,2]),global_plane_wave_LB=stats(xx[:,:,3]),
                                      integer_block_wave_LB=stats(xx[:,:,4]),optimistic_independent_plane_blockwork_LB_ns=stats(xx[:,:,5]),
                                      NAND_ECC_necessary_resource_LB_ns=stats(xx[:,:,6]),
                                      per_layer_mean_waves=xx[:,:,0].mean(axis=1).tolist(),
                                      resident_pages_per_block_plane_per_layer=plan.sum(axis=1).tolist(),
                                      exact_44wave_queries=int(np.count_nonzero(xx[:,:,0]==44)),total_queries=26*512,
                                      native_completion_latency_measured=False)
    np.savez_compressed(OUT/'TRAIN_maps_and_metrics.frozen.npz',**arrays)
    with gzip.open(OUT/'actual_page_address_ledgers.json.gz','wt') as out:json.dump(ledgers,out,separators=(',',':'))
    dump(OUT/'representative_NAND_only_traces.json',traces)
    dump(OUT/'map_receipt.frozen.json',dict(frozen_utc=utc(),protocol_sha256=sha(OUT/'protocol.frozen.json'),
                                          source_sha256=sha(__file__),maps_sha256=sha(OUT/'TRAIN_maps_and_metrics.frozen.npz'),
                                          ledgers_sha256=sha(OUT/'actual_page_address_ledgers.json.gz'),heldout_opened=False,
                                          scope='Allmaps fitted/evaluated onthe samepreviouslyobservedTRAIN; no prospectiveclaim. Onechipaddress template replicated across16distinctchips withoriginalrows164? fullrowbands176routed/352sharedperchip.'))
    report['verification']=dict(unique_pages_per_plan_per_chip=1452,plan_layers=len(ledgers),physical_addresses_checked=1452*len(ledgers),
                                actual_route_schedules_checked=len(ledgers)*4,page_rows=8,selected_pages_every_query=176,
                                monotone_active_planes_no_idle_resume=True,source_order='fixed0123',copies=1,overfetch=0)
    report['elapsed_seconds']=time.monotonic()-started
    report['universal44_capacity_bound']=dict(global_work_waves=44,minimal_max_resident_plane_pages=392,two_block_capacity=384,
                                             implication='Cannotguarantee44waves forallpossibletop6 at2blocks underwholepage ownerhomogeneous placement.',not_native_sufficient=True,not_algorithm_novelty=True)
    report['scope_notice']='NANDepoch/simplecoldcost isconstructible geometry only. FinitePB/ECC/analogsource/timer/input/outputnativecommandfeasibility isNOT demonstrated. Crossblockperplane work is onlyoptimisticLB.'
    dump(OUT/'report.json',report)
    lines=['# Deep two-block common-input GU headroom','','ExistingW8/A8TRAIN26layers ×512inputs. Geometry andNAND-onlycold-costscreening; notnativecompletionlatency. Allmatchedcontrols have2blocks,1452programmedpages/chip,176selectedpages/query/chip,onecopyandnooverfetch.','','| Control | Meanwaves | P99waves | Meanactiveblocks | NAND-onlymeanus | NAND+ECCnecessaryLBmeanus |','|---|---:|---:|---:|---:|---:|']
    for name,rec in report['controls'].items():lines.append(f"| {name} | {rec['waves']['mean']:.6f} | {rec['waves']['p99']:.1f} | {rec['active_block_cold_waves']['mean']:.6f} | {rec['NAND_simple_ns']['mean']/1000:.6f} | {rec['NAND_ECC_necessary_resource_LB_ns']['mean']/1000:.6f} |")
    lines+=['','Eachselectedpage contains8completeC2048rows, so theexplicitfixed0123pageconsumeorderpreservesfulldots withrow/expert scatter. Aplane thatrunsoutstaysidleuntilthenextblock; allactiveplanesstartcoldinthenewblock. No idleplane is silentlykeptCRhot.','','Global44waves alone isonlyworkLB. Universal44foralltop6 wouldrequireminimalmaxresident392pages/plane, exceeding384pagesavailablein2blocks. Thisis anecessarycapacityargument, notnativeexecution/PBtheorem ornewalgorithm.','','Frequencyordering,hotblockplacement,waterfillcapacityrepair,phasecoordinatefit andonepage8binmoves arestandardstrongcontrols. AllTRAINfitsreuseobservedroutes. The3blockcompensator is anextra-capacityreference, notcapacitymatched.','','FinitePB/ECCpipeline,command/input/outputcosts andfullFFN areoutside thisscreen. The6.4GB/s bound is necessaryonly andcanmasksmallNANDwave improvements. No measuredhardwareperformance ornewlayoutalgorithm claim.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',elapsed_seconds=report['elapsed_seconds'],controls={k:dict(waves=v['waves']['mean'],NAND_ns=v['NAND_simple_ns']['mean'],resource_LB_ns=v['NAND_ECC_necessary_resource_LB_ns']['mean']) for k,v in report['controls'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:run()
