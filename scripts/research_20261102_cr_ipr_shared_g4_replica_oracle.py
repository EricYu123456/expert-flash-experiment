#!/usr/bin/env python3
"""Independent low-cost standard whole-Down replica oracle, sharedg4 geometry.

No full-FFN executor or new weight/numeric experiment is run. Native durations
come from the frozen finite-PB pipeline; static addresses and chip calendars
are independently reconstructed. Group replication is a standard control.
"""
from collections import Counter
import datetime,hashlib,importlib.util,itertools,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr'
OUT=BASE/'down-replica-headroom/shared-g4-standard-triples'
OLD=BASE/'full-ffn/asymmetric-down'
NEW=BASE/'down-headroom/final-run'
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def need(x,s):
    if not x:raise RuntimeError(s)

def native(p,pages,C):
    ws=[]
    for k in range(0,pages,4):
        pp=[dict(page_id=f'x.p{j}',owner='x',page_index=j,plane=j%4,block=0,wl=j//4,bytes=16384) for j in range(k,min(k+4,pages))]
        ws.append(dict(pages=pp,source_order=[q['plane'] for q in pp]))
    r=p.simulate(ws,cache_copy_ns=500,source_policy='fixed',issue_policy='strict_no_interval',row_bytes=C)
    need(r['feasible'],'kernel must fit original two-slot PB contract')
    return r['makespan_ns']

def GU_addresses():
    for c in range(16):
        for e in range(64):
            for j,pl in enumerate([0,1,2,0,1,3]):
                yield c,0,pl,(2*e+j//3 if pl<2 else e)
        for j in range(12):yield c,0,[3,2][j%2],64+j//2

def capacity(menu,mode):
    own=set(GU_addresses());need(len(own)==6336,'GU unique resident pages')
    jobs=[];rows=Counter();fill=[[128,0] for c in range(16)]
    def put(e,c,logical_band,C,block,wl,waves):
        for k in range(waves*4):
            key=(c,block,k%4,wl+k//4)
            need(key not in own and 0<=wl+k//4<192,'physical replica collision/block crossing')
            own.add(key)
        rows[e,logical_band]+=1
        jobs.append(dict(expert=e,chip=c,logical_band=logical_band,columns=C,
                         rows=384,block=block,WL_start=wl,waves=waves,pages=waves*4))
    if mode=='triples':
        for c in range(12):
            for e in range(64):put(e,c,c%4,512,1,3*e,3)
            fill[c][1]=192
        for c in range(12,16):put(64,c,c%4,1024,0,128,6);fill[c][0]=134
        need(all(rows[e,b]==3 for e in range(64) for b in range(4)),'all64experts have complete threegroup copies')
    else:
        for c in range(16):
            for e in range(65):
                if menu[e,c]==0:continue
                waves=6 if e==64 else 3;C=1024 if e==64 else 512
                b=next(b for b in range(2) if fill[c][b]+waves<=192)
                put(e,c,c%4,C,b,fill[c][b],waves);fill[c][b]+=waves
            if mode=='spill_all64' and c>=12:
                for e in range(64):
                    b=next(b for b in range(2) if fill[c][b]+3<=192)
                    put(e,c,c%4,512,b,fill[c][b],3);fill[c][b]+=3
    needed=sum(sum(v>0 for v in rr) for rr in fill)
    programmed=len(own)*16384
    return dict(mode=mode,unique_pages=len(own),programmed_coefficient_bytes=programmed,
                max_two_blocks_per_chip=True,allocated_budget_LSB_bytes=16*2*4*192*16384,
                blocks_used_by_this_fixed_layout=needed,block_WL_fill=fill,
                address_sha256=hashlib.sha256(np.asarray(sorted(own),dtype=np.int32).tobytes()).hexdigest(),
                symbolic_Down_jobs=jobs,scope='Unique coefficient-page coordinates and logical whole-row-copy multiplicities only; no coefficient bytes programmed or native numeric ABI asserted.')

def spill_oracle(counts,A,B,S,force_cap2=False):
    # Whole expert moves to the sharedgroup; each move places one band on each
    # of its four chips. All64static backup copies are allowed for this bound.
    groupA=[max(A[4*g:4*g+4]) for g in range(3)];sharedA=max(A[12:16])
    best=None
    for moves in itertools.product(*(range(int(n)+1) for n in counts)):
        if force_cap2 and any(counts[g]-moves[g]>2 for g in range(3)):continue
        end=max([groupA[g]+(counts[g]-moves[g])*B for g in range(3)]+[sharedA+S+sum(moves)*B])
        key=(end,sum(moves),moves)
        if best is None or key<best:best=key
    return int(best[0]),best[2]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    newmenu=NEW/'menus.frozen.npz';selection=NEW/'selection.frozen.json';newreport=NEW/'TRAIN512/report.json'
    tracepaths={cp:OLD/f'TRAIN512/trace.candidate.g4_reused_TRAIN_labels.input{cp}.GF.json' for cp in [1,4]}
    paths=[Path(__file__).resolve(),PIPE,TRAIN,newmenu,selection,newreport,NEW/'TRAIN512/service_arrays.npz',OLD/'TRAIN512/service_arrays.npz',*tracepaths.values(),BASE/'down-replica-headroom/report.json']
    inputs={str(x):sha(x) for x in paths}
    write(OUT/'protocol.frozen.json',dict(inputs_sha256=inputs,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Standard threegroup wholeDown replicas and sharedgroup overflow oracle; fixed candidateGU, sharedg4, current384row native jobs. No fullFFN execution/heldout/numeric change.'))
    spec=importlib.util.spec_from_file_location('shared_g4_oracle_pipeline',PIPE);p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
    routed=native(p,12,512);shared=native(p,24,1024)
    need((routed,shared)==(59220,89940),'native expected times')
    with np.load(TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3].copy()
    chosen=json.loads(selection.read_text())['winners']['candidate']
    with np.load(newmenu) as z:menus=z[chosen].copy()
    need(np.all(menus[:,64,:12]==0) and np.all(menus[:,64,12:]==384),'dedicated sharedgroup3')
    need(np.all(menus[:,:64,12:]==0) and np.all(menus[:,:64].sum(axis=2)==1536),'routed only first3groups')
    counts=np.array([[[np.count_nonzero(menus[l,rr,4*g]) for g in range(3)] for rr in routes[l]] for l in range(3)])
    need(np.all(counts.sum(axis=2)==6),'complete six private experts')
    caps={mode:capacity(menus[0],mode) for mode in ['onecopy','triples','spill_all64']}
    need(caps['triples']['programmed_coefficient_bytes']-155713536==100663296,'96MiB extra triples')
    need(caps['spill_all64']['programmed_coefficient_bytes']-155713536==50331648,'48MiB allshared spill backups')
    old=np.load(OLD/'TRAIN512/service_arrays.npz');new=np.load(NEW/'TRAIN512/service_arrays.npz')
    cases=[];saved=dict(routes=routes,routed_group_counts=counts)
    for cp,fp in tracepaths.items():
        t=json.loads(fp.read_text());ns=t['nodes'];A=[0]*16
        for j in t['jobs']:
            if j['phase']==0:A[j['chip']]=ns[j['release']]['end_ns']
        B=routed+250+256*cp+64;S=shared+250+512*cp+32
        triple=max(max(A[:12])+2*B,max(A[12:])+S)
        # Allsixexpertshave allthreegroup copies; fixed2/2/2 assignment with
        # oneearly and onelateexpert pergroup gives chip-local release witness.
        releases=[next(n['end_ns'] for n in ns if n['kind']=='Down_submit' and n['expert_rank']==r) for r in range(7)]
        need(max(releases[:6])<min(A[:12])+B and releases[6]<min(A[12:]),'releases fit twojob local witness')
        before=[];spill=[];cap2=[];moves=[]
        for n in counts.reshape(-1,3):
            before.append(max([max(A[4*g:4*g+4])+int(n[g])*B for g in range(3)]+[max(A[12:])+S]))
            val,mm=spill_oracle(n,A,B,S);spill.append(val);moves.append(mm)
            val,_=spill_oracle(n,A,B,S,True);cap2.append(val)
        before=np.array(before).reshape(3,512);spill=np.array(spill).reshape(3,512);cap2=np.array(cap2).reshape(3,512)
        baseline_old=old[f'candidate.g4_reused_TRAIN_labels.input{cp}.GF.cost']
        baseline_new=new[f'candidate.{chosen}.input{cp}.DF.cost']
        saved[f'input{cp}.onecopy_calendar_LB']=before;saved[f'input{cp}.spill_LB']=spill;saved[f'input{cp}.forced_cap2_spill_LB']=cap2
        cases.append(dict(input_copies=cp,events=1536,GU_calendar_ns=A,band_minimum_owner_ns=B,shared_minimum_owner_ns=S,
                          triple_chip_calendar_LB_ns=int(triple),old_sharedg16_actual_mean_ns=float(baseline_old.mean()),new_sharedg4_actual_mean_ns=float(baseline_new.mean()),
                          triple_headroom_vs_old_actual_upper_pct=float(100*(1-triple/baseline_old.mean())),
                          triple_headroom_vs_new_actual_upper_pct=float(100*(1-triple/baseline_new.mean())),
                          new_onecopy_calendar_LB_mean_ns=float(before.mean()),all64_shared_backup_spill_LB_mean_ns=float(spill.mean()),
                          forced_cap2_spill_LB_mean_ns=float(cap2.mean()),cap2_regressions_vs_onecopy_calendar=int(np.count_nonzero(cap2>before)),
                          unconstrained_spill_never_worse=bool(np.all(spill<=before)),
                          spill_headroom_vs_new_actual_upper_pct=float(100*(1-spill.mean()/baseline_new.mean()))))
    examples=[]
    for n in [(3,2,1),(3,3,0),(4,1,1),(4,2,0)]:
        raw=[0]*16;v,m=spill_oracle(n,raw,59220,89940);v2,m2=spill_oracle(n,raw,59220,89940,True)
        examples.append(dict(group_counts=n,no_copy_compute_ns=max(max(n)*59220,89940),best_spill_compute_ns=v,best_moves=m,cap2_compute_ns=v2,cap2_moves=m2))
    np.savez_compressed(OUT/'oracle_arrays.npz',**saved)
    need(inputs=={x:sha(x) for x in inputs},'inputs changed during oracle')
    report=dict(status='potential_above5pct_actuals_require_root_decision',source_sha256=sha(__file__),inputs_sha256=inputs,menu=chosen,
                raw_native_kernel_ns=dict(routed384row_3wave=routed,shared384row_6wave=shared),
                max_routed_experts_per_original_group={str(k):int(v) for k,v in Counter(counts.max(axis=2).ravel()).items()},
                cases=cases,capacity=caps,manual_overflow_examples=examples,
                representation='Standard wholeexpert group replication/replicateddeclustering. All64routedexperts have completeDown copies in all3routedgroups;6chosenexperts may be assigned2/2/2. The overfloworacle instead gives all64a complete backup in the sharedgroup; subset-fitting is not performed.',
                scope=['Fixed current candidateGU release calendar and384-row native Down jobs; sharedDown is changed tog4. The previous fixedsharedg16<5% screen does not bound this geometry.',
                       'Lowerbounds relax globalcommand/channel/PCIe priority/contention, selection cost andFIFO; they are not executed fullFFN costs. Wholeprivate vectors, cold starts, chip-local ownership and jobgranularity are retained.',
                       'Triples cost96MiB additionalcoefficient storage/layer. Allshared backup inventory costs48MiB. A hypothetical smaller selectiveinventory is not claimed to achieve the oracle.',
                       'Budget is at most2blocks/chip, not identical actualallocated use: thisfixedcandidate layout uses20/28/24 chipblocks for onecopy/triples/allsharedbackup respectively.',
                       'Scales/scatter remainhost metadata; physicalproof covers coefficient addresses/logical copies, not bit-level copiedweights or native ABI.',
                       'CBC116 and Bujtas–Tuza/batch-code constructions are knownprior and outside thisscreen; no newreplicationrouting theorem/solver/priorityclaim is made.',
                       'No newfullFFN execution, heldout, quality orhardware capability. Native timing uses the existing conditionalstrict500ns PB/source model.'],
                arrays_sha256=sha(OUT/'oracle_arrays.npz'),actual_command=['python',str(Path(__file__).resolve())])
    write(OUT/'report.json',report)
    (OUT/'report.md').write_text(f'''New sharedg4 geometry has substantial standard-replication headroom; the old fixedsharedg16 bound does not apply.

With all64completeDown experts resident in each of3routedg4groups, everytop6 can use2/2/2. The nativekernel work is2×59.220=118.440us on routedchips versus89.940us on dedicatedsharedchips. FixedcurrentGU calendars plus minimumcommand/fullinput/lastONFI ownercost give optimisticfullFFN lowerbounds{cases[0]['triple_chip_calendar_LB_ns']/1000:.3f}/{cases[1]['triple_chip_calendar_LB_ns']/1000:.3f}us forinput1/4. Relative to the completednewsharedg4 means{cases[0]['new_sharedg4_actual_mean_ns']/1000:.6f}/{cases[1]['new_sharedg4_actual_mean_ns']/1000:.6f}us, remainingoptimisticheadroom is{cases[0]['triple_headroom_vs_new_actual_upper_pct']:.4f}%/{cases[1]['triple_headroom_vs_new_actual_upper_pct']:.4f}%. These arebounds, notachievedspeedups; actualfullFFN is anotheragent's standardcontrol.

Triplesadd96MiB/layer (twoextra completeDowncopies/expert), fitwithin2blocks/chip, and use28chipblocks in the fixedcandidate layout. RoutedDownoccupies192WL/plane inblock1; GUusesblock0max128WL. Sharedg4 needs6WL atblock0offset128. Uniquephysicalpagecoordinates andthreecompletecopies perlogicalband are checked. Onecopynewgeometry uses20blocks, so no equalactualallocation or capacityefficiency advantage is claimed.

The cheaperoverflow oracle gives everyexpert a completebackup in thesharedgroup:48MiB/layer and24chipblocks; it does not optimize a smallerstaticcopy inventory. Its bestchip-calendar means are{cases[0]['all64_shared_backup_spill_LB_mean_ns']/1000:.6f}/{cases[1]['all64_shared_backup_spill_LB_mean_ns']/1000:.6f}us. Forcing routedcap2 is not always latencyoptimal: (3,3,0) needs twospills andregresses, while (4,1,1) prefersonespill/threejobs overtwospills/twojobs. Exactenumeration andhandcases are retained.

This is standardwholeexpert replication/declustering, not a newrouting or CBCresult. The scopefixescurrentGU/nativejobgeometry, completeprivateinputs, row/scales semantics and thedeclaredconditionalhardwarecontract; otherresources/FIFO/selection remain relaxed. No newactualfullFFN, heldout orqualityrun was launched.
''')
    print(json.dumps(dict(status=report['status'],cases=cases,capacity_summary={k:{x:v[x] for x in ['programmed_coefficient_bytes','blocks_used_by_this_fixed_layout']} for k,v in caps.items()}),indent=2))

if __name__=='__main__':main()
