#!/usr/bin/env python3
"""Single-layer 3D-FPIM placement: static native trace/movement units, no cycles.

All layouts have 1280 persistent matrix cores in the same 54 compute tiles:
53 tiles contain 24 cores and the last contains 8. No replication/depth packing.
The two reserved I/O tiles and all original hardware constants are retained.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
ROUND = ROOT.parent
REPO = ROUND.parents[2]
UPSTREAM = ROUND / "3dfpim-audit/upstream"
ROUTES = ROUND / "neuron-layout-prior/granite4tiny-routes/screen/routes.npz"
EXPECTED_COMMIT = "bd17a3a3c7a6a2ef98beb7a5379e2110f94df706"
N = 1280


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def api():
    subprocess.run(["g++", "-std=c++11", "-O3", "-march=native", "-shared", "-fPIC",
                    str(ROOT/"movement_solver.cpp"), "-o", str(ROOT/"movement_solver.so")],check=True)
    lib=ctypes.CDLL(str(ROOT/"movement_solver.so"))
    ptr=np.ctypeslib.ndpointer(np.int32,flags="C_CONTIGUOUS")
    lib.score_movement.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.score_movement.restype=ctypes.c_double
    lib.optimize_movement.argtypes=[ptr,ptr]+[ctypes.c_int]*4+[ctypes.c_uint]
    lib.optimize_movement.restype=ctypes.c_double
    lib.original_shuffle.argtypes=[ptr,ctypes.c_uint]
    return lib


def valid(mapping):
    assert mapping.shape==(1280,) and mapping.dtype==np.int32
    assert mapping.min()==0 and mapping.max()==53
    np.testing.assert_array_equal(np.bincount(mapping,minlength=54),[24]*53+[8])


def movement(mapping,routes,aggregate,owner=None):
    """Original partitioner rules, expressed in 128-element vector units."""
    valid(mapping)
    tiles=mapping.reshape(64,20)
    gu=[sum(1<<int(t) for t in set(row[:8])) for row in tiles]
    down=[sum(1<<int(t) for t in set(row[8:])) for row in tiles]
    if owner is None: owner=np.broadcast_to(np.arange(4,8),(64,4))
    internal=[]
    for e in range(64):
        cross=sum(int(tiles[e,j]!=tiles[e,4+j]) for j in range(4))
        z=sum((down[e]&~(1<<int(tiles[e,owner[e,j]]))).bit_count() for j in range(4))
        internal.append((cross,z))
    components=[]
    for selected in routes:
        mask=0
        for e in selected: mask|=gu[e]
        common=1536*mask.bit_count()
        gu_cross=128*sum(internal[e][0] for e in selected)
        z_cross=128*sum(internal[e][1] for e in selected)
        if aggregate:
            output=0
            for j in range(12):
                mask=0
                for e in selected: mask|=1<<int(tiles[e,8+j])
                output+=128*mask.bit_count()
        else: output=1536*len(selected)
        components.append([common,gu_cross,z_cross,output])
    components=np.asarray(components,np.int64)
    total=components.sum(axis=1)
    k=routes.shape[1]
    return dict(mean_send_units=float(total.mean()),mean_receive_units=float(total.mean()),
                mean_network_send_plus_receive_units=float(2*total.mean()),
                component_order=["common_input","gate_up_pair","activation_to_down","output"],
                mean_components=components.mean(axis=0).tolist(),
                min_send_units=int(total.min()),max_send_units=int(total.max()),
                p50_send_units=float(np.quantile(total,.5)),p95_send_units=float(np.quantile(total,.95)),
                load_units=int(k*(8*1536+4*128+12*512)+(k-1)*1536*aggregate),
                store_units=int(k*(512+512+1536)),
                MVM_depth_instructions=int(k*(8*12+12*4)),
                ALU_output_reduction_add_vectors=int((k-1)*12*aggregate),
                ALU_routing_coefficient_mul_vectors=int(k*12))


def initial_layouts(lib,routes):
    row=np.arange(N,dtype=np.int32)//24
    freq=np.bincount(routes.ravel(),minlength=64)
    order=np.argsort(-freq,kind="stable")
    hot=np.empty(N,np.int32)
    for position,e in enumerate(order):hot[e*20:(e+1)*20]=np.arange(position*20,(position+1)*20)//24
    phase_order=[e*20+j for e in range(64) for j in range(8)]+[e*20+j for e in range(64) for j in range(8,20)]
    phase=np.empty(N,np.int32);phase[phase_order]=np.arange(N)//24
    output_order=[e*20+j for e in range(64) for j in range(8)]+[e*20+8+j for j in range(12) for e in range(64)]
    output=np.empty(N,np.int32);output[output_order]=np.arange(N)//24
    layouts={"original_row":row,"original_column":row.copy(),"expert_affinity_hotrank":hot,
             "phase_contiguous":phase,"output_row_affinity":output}
    random_ranks={}
    for seed in (1,17,101):
        rank=np.empty(N,np.int32);lib.original_shuffle(rank,seed)
        layouts[f"original_random_{seed}"]=rank//24
        layouts[f"random_fixed_native_affinity_{seed}"]=rank//24
        random_ranks[f"original_random_{seed}"]=rank
    return layouts,random_ranks


def native_crosscheck(mapping,route,aggregate,label,partition="row",seed=1,original=False,owner=None):
    out=ROOT/"crosschecks"/label;out.mkdir(parents=True,exist_ok=True)
    routepath=out/"route.txt";routepath.write_text("64\n"+" ".join(map(str,route))+"\n")
    env=os.environ.copy()
    source=UPSTREAM/"3DFPIM-Compiler/src" if original else ROOT/"compiler-adapter/src"
    env["LD_LIBRARY_PATH"]=str(source)
    env.pop("FPIM_TILE_MAP",None)
    if not original:
        mapfile=out/"mapping.txt";np.savetxt(mapfile,mapping,fmt="%d")
        env["FPIM_TILE_MAP"]=str(mapfile)
    p=subprocess.run([str(ROOT/"native_graph_probe"),"probe",str(routepath),
                      "aggregate" if aggregate else "private",partition,str(seed)],
                     cwd=out,env=env,text=True,capture_output=True)
    (out/"compile.log").write_text(p.stdout+p.stderr)
    if p.returncode and original and partition=="random":
        return dict(label=label,exit_code=p.returncode,original_unmodified_library=True,
                    status="original compiler assertion; not a successful executable trace",
                    stderr=p.stderr,analytic_not_validated=True)
    assert p.returncode==0,(label,p.stderr)
    report=(out/"probe-report.out").read_text()
    counters={k:int(v) for k,v in re.findall(r"^# ([^=\n]+?) = (\d+)$",report,re.M)}
    analytic=movement(mapping,np.asarray([route],np.int32),aggregate,owner)
    expected={"load bytes":analytic["load_units"],"store bytes":analytic["store_units"],
              "send bytes":int(analytic["mean_send_units"]),"receive bytes":int(analytic["mean_receive_units"])}
    for k,v in expected.items():assert counters[k]==v,(label,k,counters[k],v)
    assert counters["load bytes from spilling"]==0 and counters["store bytes from spilling"]==0
    active=[];hashes={};mvm_count=0;assignment={}
    for path in sorted(out.glob("*.3dfpim")):
        txt=path.read_text();hashes[path.name]=digest(path)
        match=re.search(r"tile(\d+)-core(\d+)\.3dfpim$",path.name)
        if match:
            tile,core=map(int,match.groups())
            for line in txt.splitlines():
                if not line.startswith("mvm("):continue
                mvm_count+=1
                name=re.search(r"MVM: ([GUD])(\d+)\[(\d+)\]\[(\d+)\]",line)
                assert name,line
                role,e,r,c=name.groups();e,r=int(e),int(r)
                node=e*20+{"G":0,"U":4,"D":8}[role]+r
                assert mapping[node]==tile-2,(label,node,mapping[node],tile)
                assert assignment.setdefault(str(node),[tile,core])==[tile,core]
            if "mvm(" in txt:active.append(path.name)
    assert mvm_count==analytic["MVM_depth_instructions"]
    (out/"trace_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    # Preserve full native traces while avoiding tens of thousands of tiny files.
    archive=out/"native_traces.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        for path in sorted(out.glob("*.3dfpim")):tar.add(path,arcname=path.name)
    for path in out.glob("*.3dfpim"):path.unlink()
    return dict(label=label,exit_code=0,original_unmodified_library=original,
                report=counters,analytic_expected=expected,MVM_instructions=mvm_count,
                active_core_count=len(active),selected_matrix_fixed_assignment=assignment,
                compressed_traces_sha256=digest(archive))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--steps",type=int,default=40000);args=ap.parse_args()
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=UPSTREAM,text=True).strip()==EXPECTED_COMMIT
    assert subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=UPSTREAM,text=True)==""
    with np.load(ROUTES) as f:
        allroutes=f["routes"][:32,:64,:3]
        ids=f["prompt_ids"].tolist();layers=f["layers"][:3].tolist()
    assert all(x.startswith("train_") for x in ids)
    assert allroutes.shape==(32,64,3,6)
    lib=api();result=[];saved={};crosschecks=[];start=time.monotonic()
    for li,layer in enumerate(layers):
        routes=np.ascontiguousarray(np.sort(allroutes[:,:,li].reshape(-1,6),axis=1),np.int32)
        layouts,random_ranks=initial_layouts(lib,routes)
        for aggregate in (False,True):
            policy="aggregate" if aggregate else "private"
            records=[];custom={k:v.copy() for k,v in layouts.items()}
            # Original random assignment changes the original affinity choice.
            # Optimized/custom maps retain semantic G-before-U native affinity.
            owners={}
            for label,rank in random_ranks.items():
                rr=rank.reshape(64,20)
                owners[label]=np.where(rr[:,:4]>rr[:,4:8],np.arange(4),np.arange(4,8))
            for objective,label in [(2,"corrected_weighted_graph_greedy"),(1,"marginal_multicast_affinity")]:
                best=None;search=[];native_best=None
                for si,initial_name in enumerate(["original_row","expert_affinity_hotrank","phase_contiguous","output_row_affinity"]):
                    mapping=layouts[initial_name].copy()
                    value=lib.optimize_movement(mapping,routes,len(routes),int(aggregate),objective,args.steps,20261102+li*101+si*13+objective)
                    native_value=lib.score_movement(mapping,routes,len(routes),int(aggregate),0)
                    search.append(dict(initial=initial_name,objective=value,native_send_units=native_value*128))
                    if best is None or value<best[0]:best=(value,mapping)
                    if native_best is None or native_value<native_best[0]:native_best=(native_value,mapping)
                custom[label]=best[1];saved[f"L{layer}_{policy}_{label}"]=best[1]
                custom[label+"_best_native_restart"]=native_best[1]
                records.append(dict(label=label,objective_kind=objective,search=search))
            # Joint route-aware optimizer uses exactly the same native multicast
            # and output-local reduction rules as both static affinity controls.
            bases=["original_row","marginal_multicast_affinity_best_native_restart","corrected_weighted_graph_greedy_best_native_restart"]
            joint_best=None;joint_search=[]
            for si,label in enumerate(bases):
                mapping=custom[label].copy()
                value=lib.optimize_movement(mapping,routes,len(routes),int(aggregate),0,args.steps,20262102+li*101+si*13+aggregate)
                joint_search.append(dict(initial=label,mean_send_units=value*128))
                if joint_best is None or value<joint_best[0]:joint_best=(value,mapping)
            custom["query_aware_native_multicast"]=joint_best[1]
            saved[f"L{layer}_{policy}_query_aware_native_multicast"]=joint_best[1]
            scores=[]
            for label,mapping in custom.items():
                valid(mapping)
                stats=movement(mapping,routes,aggregate,owners.get(label))
                if label not in owners:
                    np.testing.assert_allclose(stats["mean_send_units"],lib.score_movement(mapping,routes,len(routes),int(aggregate),0)*128,rtol=0,atol=1e-7)
                scores.append(dict(label=label,**stats,mapping_sha256=hashlib.sha256(mapping.tobytes()).hexdigest(),
                                   executable_status="unmodified random full FFN failed; analytical profile only" if label in owners else "native trace composition"))
                saved[f"L{layer}_{policy}_{label}"]=mapping
            baseline=min((s for s in scores if not s['label'].startswith(('query_','original_random_'))),key=lambda s:s['mean_send_units'])
            chosen=next(s for s in scores if s['label']=='query_aware_native_multicast')
            entry=dict(layer=layer,policy=policy,routes=len(routes),scores=scores,
                       strongest_static_baseline=baseline['label'],
                       query_gain_fraction=1-chosen['mean_send_units']/baseline['mean_send_units'],
                       static_search=records,joint_search=joint_search)
            result.append(entry)
            print(json.dumps({"layer":layer,"policy":policy,"baseline":baseline['label'],
                              "baseline_send_units":baseline['mean_send_units'],
                              "query_send_units":chosen['mean_send_units'],
                              "query_gain_percent":entry['query_gain_fraction']*100}),flush=True)
            # One exact selected route per fitted mapping/policy, plus original
            # row/column/random controls. Full 64E weights remain instantiated.
            selected=routes[0].tolist()
            for label in ["marginal_multicast_affinity","query_aware_native_multicast"]:
                crosschecks.append(native_crosscheck(custom[label],selected,aggregate,f"L{layer}-{policy}-{label}"))
            if li==0:
                for label,partition,seed in [("original_row","row",1),("original_column","col",1),("original_random_1","random",1)]:
                    crosschecks.append(native_crosscheck(custom[label],selected,aggregate,f"L0-{policy}-{label}",partition,seed,True,owners.get(label)))
                crosschecks.append(native_crosscheck(custom['random_fixed_native_affinity_1'],selected,aggregate,f"L0-{policy}-random-fixed-affinity"))
                crosschecks.append(native_crosscheck(custom['query_aware_native_multicast'],routes[1].tolist(),aggregate,f"L0-{policy}-query-second-route"))
            (ROOT/"partial.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(ROOT/"fixed_mappings.npz",**saved)
    document=dict(schema_version=1,scope="train-only single-layer original 3D-FPIM offline native trace movement experiment",upstream_commit=EXPECTED_COMMIT,
        routes_path=str(ROUTES),routes_sha256=digest(ROUTES),training_ids=ids,training_slice="[:32,:64,:3]",layers=layers,
        hardware=dict(MVMU_dimension=128,MVMU_depth=64,MVMUs_per_core=1,cores_per_tile=24,
                      resident_matrix_cores=1280,compute_tiles=54,reserved_IO_tiles=2,original_max_tiles=224,
                      full_40_layer_core_requirement=51200,forty_layers_fit_one_original_chip=False),
        units="128-element vector movement expanded back to element-length units; original reports call these bytes without publishing a numeric ABI. No dtype-byte conversion, timing or speedup claimed.",
        numerical_scope="Original native SIG/MUL/MULI/ADD graph; alpha immediate differs per expert. No weights or activations are numerically executed. Precision/scale/rounding/reduction-order fidelity is unverified; aggregate policy is a quality/ABI-conditional composition, measured equally for all layouts.",
        aggregation_control="private expert outputs versus identical per-tile/output-row local native ADD then global ADD; no horizontal weight fusion or extra context",
        optimizer=dict(steps_per_start=args.steps,graph="Correct consumer endpoints with marginal-weighted input-affinity star and G/U->D communication; own greedy/annealing solver, NOT KaHIP",marginal="Independent expert Bernoulli model with exact multicast-domain probability and observed marginals; no joint query tuples used by this objective",joint="Same native work and coalescing; actual training selected-set union objective",capacity="Pair swaps preserve exactly 53*24+8 persistent cores across every mapping; no replicas, empty-core tricks or extra matrices"),
        results=result,native_trace_crosschecks=crosschecks,
        private_route_relaxed_send_lower_bound=15360,
        lower_bound_scope="K6 private outputs only: >=2 input tiles*1536, plus >=3072 activation traffic at two-input-tile packing, plus 9216 outputs; additional input tiles increase the relaxed lower bound. No aggregate-policy or global fixed-placement optimality claim.",
        elapsed_seconds=time.monotonic()-start,files={p.name:digest(p) for p in [ROOT/"movement_solver.cpp",ROOT/"native_graph_probe.cpp",Path(__file__)]})
    (ROOT/"results.json").write_text(json.dumps(document,indent=2)+"\n")
    print(json.dumps({"native_crosschecks":len(crosschecks),"elapsed_seconds":document['elapsed_seconds']}),flush=True)


if __name__=="__main__":main()
