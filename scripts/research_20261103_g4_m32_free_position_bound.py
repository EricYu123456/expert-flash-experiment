#!/usr/bin/env python3
"""Exact fixed-geometry free-position work DP and one-route row witness.

This is a relaxation, not a deployable per-query address-remapping algorithm.
It uses standard bipartite multigraph perfect-matching decomposition. No native
simulator, solver fit, weights, model inference or heldout is run.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02/round2/chip-count-regime-screen"
OUT = BASE / "free-position-bound"
PLAN = BASE / "stage-b/granite4tiny.M32.minimum_depth.plan.npz"
HINT = BASE / "g4-M32-coupled-cp/hint.frozen.npz"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump_new(p, x):
    assert not p.exists(), p
    p.write_text(json.dumps(x, indent=2) + "\n")


def dependencies():
    stage = json.loads((BASE/"stage-b/protocol.frozen.json").read_text())
    backend = next(Path(x["snapshot"]) for x in stage["inputs"] if x["path"].endswith("ordinary_read_backend.py"))
    paths = [Path(__file__), PLAN, HINT, BASE/"stage-b/protocol.frozen.json", backend,
             BASE/"g4-M32-coupled-cp/protocol.frozen.json",
             BASE/"g4-M32-coupled-cp/producer-seal/report.json"]
    return paths, backend


def freeze():
    OUT.mkdir(exist_ok=False)
    paths, backend = dependencies()
    dump_new(OUT/"protocol.frozen.json", {
        "status": "frozen_before_bound_execution", "utc": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": {str(p): sha(p) for p in paths}, "backend": str(backend),
        "geometry": "32chips,64experts,K6. Each expert X5/O7 disjoint, GU25bands24x41+1x40, Down12x128. Shared exactly source stageB minimum_depth.",
        "state": "Per chip0<=x,o,x+o<=6;sumx30,sumo42. Compute exact integer service from frozen Service, with copies1/4 separately.",
        "bound": "Bitset DP tracks totalX0..30 and totalO0..42 across32chips. Binary-search sorted exact statecosts, independently compare set-DP atT andT-1. Also use7heavy first-acquire rank acrossglobal250ns command resource: host+min10wavework+6cmd ifmaxwaves<=10, versushost+11read ifanychip>=11. No chiplabel-specific dispatch or selector penalty.",
        "construction": "Split X30/O42 incidences among5/7 leftvertices degree6, add20 degree6 dummyvertices to obtain32by32 degree6 bipartite multigraph, remove6perfectmatchings. Each color is one expert X5/O7.",
        "physical_witness": "Use q0 orderedTRAIN expertIDs for6colors; remaining58experts use frozen coupled hint. One fixed image witnesses that route only, not512simultaneously. Repack all64plus shared in oneblock/chip, whole logical rows/scales, no copies, all read fringe charged.",
        "forbidden": "No native replay, weights, model inference, heldout, newfit, pagebufferreuse, newMAC, earlypartialA8, validR/reset capability, real-time relocation or deployable selector claim."
    })
    print(json.dumps({"protocol_sha256": sha(OUT/"protocol.frozen.json"), "source_sha256": sha(__file__)}))


def ceildiv(n, d):
    return (int(n)+int(d)-1)//int(d)


def service_one(rows, cols, s):
    if rows == 0:
        return {"work":0,"pages":0,"waves":0,"emitted_rows":0,"partial_row_bytes":0,
                "command":0,"input":0,"read":0,"output":0}
    pages = ceildiv(rows*cols, s["page_bytes"])
    waves = ceildiv(pages, s["planes"])
    emitted = pages*s["page_bytes"]//cols
    inp = ceildiv(cols*s["input_bytes"]*s["input_channel_copies"]*10**9, s["channel_bytes_per_second"])
    output = ceildiv(emitted*s["output_bytes"]*10**9, s["channel_bytes_per_second"])
    cmd, read = s["command_ns"], waves*s["ordinary_read_ns"]
    return {"work":cmd+inp+read+output,"pages":pages,"waves":waves,"emitted_rows":emitted,
            "partial_row_bytes":pages*s["page_bytes"]%cols,"command":cmd,"input":inp,"read":read,"output":output}


def allowed_states(sharedwork, gu, down, T):
    return [[(x,o) for x in range(7) for o in range(7-x)
             if 6*gu+int(w)+down*x+(down-gu)*o <= T] for w in sharedwork]


def bitset_dp(allowed, keep=False):
    mask = (1<<43)-1
    d = [0]*31
    d[0] = 1
    layers = [d] if keep else None
    for options in allowed:
        nxt = [0]*31
        for xprev, bits in enumerate(d):
            if bits:
                for x,o in options:
                    if xprev+x <= 30:
                        nxt[xprev+x] |= (bits<<o)&mask
        d = nxt
        if keep:
            layers.append(d)
    return bool(d[30] & (1<<42)), layers


def set_dp(allowed):
    reachable = {(0,0)}
    for options in allowed:
        reachable = {(a+x,b+o) for a,b in reachable for x,o in options if a+x<=30 and b+o<=42}
    return reachable


def solve_dp(sw, gu, down):
    values = sorted({6*gu+int(w)+down*x+(down-gu)*o
                     for w in sw for x in range(7) for o in range(7-x)})
    low, high, trials = 0, len(values)-1, []
    while low < high:
        mid = (low+high)//2
        ok,_ = bitset_dp(allowed_states(sw,gu,down,values[mid]))
        trials.append({"T":values[mid],"feasible":ok})
        if ok:
            high = mid
        else:
            low = mid+1
    T = values[low]
    allowed = allowed_states(sw,gu,down,T)
    yes,layers = bitset_dp(allowed,True)
    no,below = bitset_dp(allowed_states(sw,gu,down,T-1),True)
    assert yes and not no
    reachable = set_dp(allowed)
    unreachable = set_dp(allowed_states(sw,gu,down,T-1))
    for pairs,final in [(reachable,layers[-1]),(unreachable,below[-1])]:
        bits_pairs = {(x,o) for x,bits in enumerate(final) for o in range(43) if bits & (1<<o)}
        assert pairs == bits_pairs
    assert (30,42) in reachable and (30,42) not in unreachable
    x,o = 30,42
    choices = []
    for c in range(31,-1,-1):
        for dx,do in allowed[c]:
            if x>=dx and o>=do and layers[c][x-dx] & (1<<(o-do)):
                choices.append((dx,do));x-=dx;o-=do;break
        else:
            raise AssertionError("no predecessor")
    assert (x,o) == (0,0)
    choices = np.array(choices[::-1],np.int64)
    return T,choices,{"binary_search_trials":trials,"candidate_costs":len(values),
                       "T_feasible":True,"Tminus1_infeasible":True,"independent_set_DP_all_final_states_equal":True,
                       "final_T_bitsets_hex":[hex(v) for v in layers[-1]],
                       "final_Tminus1_bitsets_hex":[hex(v) for v in below[-1]],
                       "witness_histogram":[{"x":x,"o":o,"chips":n} for (x,o),n in sorted(Counter(map(tuple,choices.tolist())).items())]}


def manual_columns(sharedwaves):
    hi = np.flatnonzero(sharedwaves==3)
    mid = np.flatnonzero(sharedwaves==2)
    lo = np.flatnonzero(sharedwaves==1)
    assert (len(hi),len(mid),len(lo)) == (12,17,3)
    v = np.zeros((32,2),np.int64)
    v[hi[:7]] = [1,5]
    v[mid,0] = 1
    v[lo,0] = 2
    remaining = 42-int(v[:,1].sum())
    for c in np.r_[mid,lo]:
        n = min(remaining,6-int(v[c].sum()))
        v[c,1] += n;remaining -= n
        if not remaining:
            break
    assert not remaining and np.array_equal(v.sum(0),[30,42]) and np.all(v.sum(1)<=6)
    return v


def edge_coloring(columns):
    x,o = columns.T
    N = np.zeros((32,32),np.int64)
    for start, count, degrees in [(0,5,x),(5,7,o),(12,20,6-x-o)]:
        endpoints = [c for c,d in enumerate(degrees) for _ in range(int(d))]
        assert len(endpoints) == count*6
        for k,c in enumerate(endpoints):
            N[start+k//6,c] += 1
    assert np.all(N.sum(0)==6) and np.all(N.sum(1)==6)
    original=N.copy();colors=[]
    for color in range(6):
        match_r=np.full(32,-1,np.int64)
        def dfs(left,seen):
            for right in np.flatnonzero(N[left]>0):
                if not seen[right]:
                    seen[right]=True
                    if match_r[right]<0 or dfs(int(match_r[right]),seen):
                        match_r[right]=left
                        return True
            return False
        for left in range(32):
            assert dfs(left,np.zeros(32,bool))
        match_l=np.full(32,-1,np.int64)
        for right,left in enumerate(match_r):
            match_l[left]=right
        assert set(match_l.tolist())==set(range(32))
        N[np.arange(32),match_l]-=1
        assert N.min()>=0 and np.all(N.sum(0)==5-color) and np.all(N.sum(1)==5-color)
        colors.append(match_l)
    assert not N.any()
    colors=np.array(colors,np.int64)
    X=np.zeros((6,32),bool);O=np.zeros_like(X)
    for e,matching in enumerate(colors):
        X[e,matching[:5]]=True;O[e,matching[5:12]]=True
    assert np.all(X.sum(1)==5) and np.all(O.sum(1)==7) and not np.any(X&O)
    assert np.array_equal(X.sum(0),x) and np.array_equal(O.sum(0),o)
    return X,O,original,colors


def rows_for(X,O):
    rows=np.zeros((6,2,32),np.int64)
    for e in range(6):
        active=np.flatnonzero(~O[e]);rows[e,0,active]=41;rows[e,0,active[0]]=40
        rows[e,1,X[e]|O[e]]=128
    assert np.all(rows[:,0].sum(1)==1024) and np.all(rows[:,1].sum(1)==1536)
    return rows


def physical_image(rows,s):
    cursor=np.zeros(32,np.int64);table=np.zeros((65,2,32),np.int64)
    descriptors=[];logical=[];inventory=Counter()
    for e in range(65):
        for phase in range(2):
            C=1536 if phase==0 else (512 if e<64 else 1024)
            R=(1024 if phase==0 else 1536) if e<64 else (2048 if phase==0 else 1536)
            rowstart=0;coverage=np.zeros(R,np.uint8)
            for c in range(32):
                n=int(rows[e,phase,c])
                if not n:
                    continue
                op=service_one(n,C,s)
                assert int(cursor[c])+op["waves"]<=192
                coverage[rowstart:rowstart+n]+=1
                descriptor={"job":len(descriptors),"expert":e,"phase":phase,"chip":c,"block":0,
                            "start_wordline":int(cursor[c]),"reserved_wordlines":op["waves"],"pages":op["pages"],
                            "first_plane":0,"logical_row_start":rowstart,"rows":n,"full_input_columns":C,
                            "logical_row_IDs":list(range(rowstart,rowstart+n)),
                            "row_scale_IDs":"same expert/phase/logicalrow IDs", "copy":False,
                            "emitted_complete_rows":op["emitted_rows"],
                            "discard_complete_fringe_rows":op["emitted_rows"]-n,
                            "drain_partial_row_bytes":op["partial_row_bytes"],"owned_service_ns":op["work"],
                            "service_components_ns":{k:op[k] for k in ["command","input","read","output"]}}
                assert descriptor["discard_complete_fringe_rows"]>=0
                descriptors.append(descriptor);cursor[c]+=op["waves"];rowstart+=n
                logical.extend((e,phase,r) for r in descriptor["logical_row_IDs"])
                table[e,phase,c]=op["work"]
                for key,value in [("jobs",1),("rows",n),("useful_bytes",n*C),("pages",op["pages"]),
                                  ("waves",op["waves"]),("emitted_complete_rows",op["emitted_rows"]),
                                  ("discard_complete_fringe_rows",op["emitted_rows"]-n),
                                  ("partial_row_bytes",op["partial_row_bytes"])]:
                    inventory[key]+=value
            assert rowstart==R and np.all(coverage==1)
    assert len(set(logical))==len(logical)
    assert inventory["rows"]==167424 and inventory["useful_bytes"]==155713536
    assert (inventory["jobs"],inventory["pages"],inventory["waves"])==(2424,9764,2441)
    return descriptors,table,cursor,dict(inventory)


def run():
    p=json.loads((OUT/"protocol.frozen.json").read_text())
    for path,h in p["inputs_sha256"].items():
        assert sha(path)==h,path
    spec=importlib.util.spec_from_file_location("frozen_m32_free_backend",p["backend"])
    backend=importlib.util.module_from_spec(spec);sys.modules[spec.name]=backend;spec.loader.exec_module(backend)
    with np.load(PLAN) as z:
        shared=z["rows"][64].copy();routes=z["routes"].copy()
    with np.load(HINT) as z:
        hint=z["rows"].copy()
    assert np.array_equal(hint[64],shared) and routes.shape==(512,6)
    rootp=json.loads((BASE/"g4-M32-coupled-cp/protocol.frozen.json").read_text())
    reports=[];saved={"routes":routes,"shared_rows":shared};images={}
    for copies in [1,4]:
        s=dict(rootp["service"],input_channel_copies=copies)
        g=service_one(41,1536,s)["work"];d=service_one(128,512,s)["work"]
        assert service_one(40,1536,s)["work"]==g
        assert (g,d,d-g)==((29102,28762,-340) if copies==1 else (31406,29530,-1876))
        sw=np.zeros(32,np.int64);waves=np.zeros(32,np.int64)
        for c in range(32):
            for phase,C in [(0,1536),(1,1024)]:
                op=service_one(int(shared[phase,c]),C,s);sw[c]+=op["work"];waves[c]+=op["waves"]
        assert waves.sum()==73
        totalquerywaves=6*(25+12)+int(waves.sum())
        assert totalquerywaves==295
        highwave_states=[(6*g+int(sw[c])+d*x+(d-g)*o,c,x,o,6+x+int(waves[c]))
                         for c in range(32) for x in range(7) for o in range(7-x) if 6+x+int(waves[c])>=10]
        coarse,cc,xx,oo,ww=min(highwave_states)
        T,dp_columns,certificate=solve_dp(sw,g,d)
        assert T==coarse==(287752 if copies==1 else 298504)
        manual=manual_columns(waves)
        work=6*g+sw+d*manual[:,0]+(d-g)*manual[:,1]
        assert work.max()==T
        X,O,multigraph,colors=edge_coloring(manual)
        routed_rows=rows_for(X,O)
        fullrows=hint.copy();fullrows[routes[0]]=routed_rows
        A=fullrows[:64,0]>0;D=fullrows[:64,1]>0
        assert np.all((A&D).sum(1)==5) and np.all((~A).sum(1)==7) and np.all(A|D)
        desc,table,cursor,inventory=physical_image(fullrows,s)
        service=backend.Service(**s)
        plan=backend.lower_rows(fullrows[:,0],fullrows[:,1],1536,[512]*64+[1024],service)
        backend.validate_plan(plan,65,service)
        assert np.array_equal(backend.occupancy_table(plan,service)["chip_ns"],table)
        actual=table[routes[0]].sum((0,1))+table[64].sum(0)
        assert np.array_equal(actual,work)
        # Check all service components in the original graph builder, without
        # executing its scheduler or running the C++ native implementation.
        graph=backend.ReferenceGraph.build(plan,np.r_[routes[0],64],service)
        node_work=np.zeros(32,np.int64)
        for node in graph.nodes:
            if node["job"]>=0:
                node_work[node["chip"]]+=node["duration"]
        assert np.array_equal(node_work,work)
        for e in range(6):
            assert np.array_equal(routed_rows[e,0]>0,~O[e])
            assert np.array_equal(routed_rows[e,1]>0,X[e]|O[e])
        images[f"input{copies}"]={"selected_ordered_TRAIN_route_q0":routes[0].tolist(),
                                  "image_scope":"One fixed full64expert+shared image forq0 only; otherqueries require independent remapping in the relaxation.",
                                  "descriptors":desc}
        for name,value in [("shared_work",sw),("shared_waves",waves),("DP_columns_x_o",dp_columns),
                           ("manual_columns_x_o",manual),("X",X),("O",O),("multigraph",multigraph),
                           ("perfect_matchings",colors),("selected_rows",routed_rows),("full_rows",fullrows),
                           ("chip_work",work),("jobs",plan["jobs"]),("offsets",plan["offsets"]),
                           ("resident_waves_per_chip",cursor)]:
            saved[f"input{copies}.{name}"]=value
        owned=T;nativeLB=owned+s["host_submit_ns"]
        minheavy=totalquerywaves-32*9
        assert minheavy==7
        rank_case=s["host_submit_ns"]+coarse+(minheavy-1)*s["command_ns"]
        eleven_case=s["host_submit_ns"]+11*s["ordinary_read_ns"]
        rankLB=min(rank_case,eleven_case)
        assert rankLB==(291252 if copies==1 else 302004)
        # Every job acquires on the same command resource for250ns, and holds
        # the single chip owner until output ends. This inspects structure,
        # not the scheduler order or a numerical native trace.
        jobnodes=[n for n in graph.nodes if n["job"]>=0]
        acquires=[n for n in jobnodes if n["acquire"]]
        assert len(acquires)==6*37+56
        assert all(n["resource"]==1 and n["duration"]==s["command_ns"] for n in acquires)
        assert all(sum(n["acquire"] for n in jobnodes if n["job"]==j)==1 and
                   sum(n["release"] for n in jobnodes if n["job"]==j)==1 for j in range(len(acquires)))
        reports.append({"input_copies":copies,"routed_GU_job_ns":g,"routed_Down_job_ns":d,"O_delta_ns":d-g,
                        "shared_wave_histogram":dict(sorted(Counter(map(int,waves)).items())),
                        "shared_work_histogram_ns":dict(sorted(Counter(map(int,sw)).items())),
                        "query_readwaves":295,"at_least_one_chip_readwaves":10,
                        "enumerated_10plus_wave_work_floor_ns":coarse,
                        "minimum_10plus_state":{"chip":cc,"x":xx,"o":oo,"readwaves":ww},
                        "exact_free_position_owned_work_optimum_ns":owned,"initial_host_ns":s["host_submit_ns"],
                        "native_necessary_LB_ns":nativeLB,"DP":certificate,
                        "schedule_independent_command_rank_LB":{
                            "query_readwaves":295,"owners":32,"minimum_10wave_owners_if_max10":minheavy,
                            "global_command_duration_ns":s["command_ns"],
                            "latest_first_acquire_among7heavy_relative_to_initial_host_ns":6*s["command_ns"],
                            "max10_case_native_LB_ns":rank_case,"any11plus_case_readalone_native_LB_ns":eleven_case,
                            "unconditional_native_LB_ns":rankLB,
                            "proof":"Ifmaxwaves<=10,295waves forcesatleast7ownerswith10waves. Each owns>=enumeratedfloor service. Their7distinctfirst-acquire commands serialize, so one starts>=host+6commands. Singlechipowner serializes all itschargedjoboperations, hencefinish>=firststart+work. Ifanyownerhas>=11waves,readalone+host>=310000. Take minimumofthetwocases.",
                            "structural_Graph_checks":"Oneglobalcommand resource, eachjobacquiresoncefor250ns and releasesafteroutput. No fixedchip order, freeplanes or selector latency assumed.",
                            "scope":"Any legal scheduler of this fixedowner/resource/Service model andfixed295wave geometry, not allhardware/executors or native attainability."},
                        "manual_column_witness":[{"chip":c,"sharedwaves":int(waves[c]),"shared_ns":int(sw[c]),
                                                   "x":int(manual[c,0]),"o":int(manual[c,1]),
                                                   "GU_jobs":6-int(manual[c,1]),"Down_jobs":int(manual[c].sum()),
                                                   "readwaves":6+int(manual[c,0])+int(waves[c]),"owned_ns":int(work[c])} for c in range(32)],
                        "edge_coloring":{"left_X_vertices":5,"left_O_vertices":7,"dummy_vertices":20,
                                         "edges":192,"degree":6,"perfect_matchings":6,"exact_X5_O7_per_color":True},
                        "q0_physical_image":{"inventory":inventory,"resident_waves_per_chip":cursor.tolist(),
                                             "allocated_blocks":32,"allocated_bytes":32*192*65536,"copies":0,
                                             "all_rows_and_scales_keep_logical_IDs":True,"shared_geometry_unchanged":True,
                                             "backend_occupancy_table_exact":True,"original_Graph_node_work_exact":True,
                                             "graph_nodes_inspected_not_scheduled":len(graph.nodes)},
                        "limit":"Exact for freeperquerypositions and fixedshared/R2geometry ownedwork only; host+work is a necessary native bound, not native attainability or a fixedstaticmapping guarantee."})
    dump_new(OUT/"report.json",{"status":"PASS exact free-position work bound and one-route physical witness",
                               "protocol_sha256":sha(OUT/"protocol.frozen.json"),"results":reports,
                               "correction":"Frozen shared has12threewave/17twowave/3onewave chips, not9threewave/23twowave. The corrected witness attains the same bound.",
                               "standard_math":"Bipartite multigraph edge-colouring/perfect-matching decomposition, a classical Hall/Koenig consequence, not a new theorem or deployable algorithm."})
    path=OUT/"arrays.npz";assert not path.exists();np.savez_compressed(path,**saved)
    path=OUT/"physical_row_IDs.json.gz";assert not path.exists()
    with gzip.open(path,"wt") as f:
        json.dump(images,f,separators=(",",":"))
    print(json.dumps({"bounds":[{"copies":r["input_copies"],"owned":r["exact_free_position_owned_work_optimum_ns"],"nativeLB":r["native_necessary_LB_ns"]} for r in reports],
                      "report_sha256":sha(OUT/"report.json")},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["freeze","run"]);a=ap.parse_args()
    freeze() if a.mode=="freeze" else run()
