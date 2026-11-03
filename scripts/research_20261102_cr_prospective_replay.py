#!/usr/bin/env python3
"""Data-blind frozen comparisons on newly captured diagnostic routes."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02/round3-cr-ipr"
OUT = BASE / "prospective-full-ffn"
DATA = BASE / "prospective-routes"
SOURCE = ROOT / "scripts/research_20261102_cr_g2_availability_ffn.py"
spec = importlib.util.spec_from_file_location("frozen_g2_driver", SOURCE)
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
f, r, a, k = g2.f, g2.r, g2.a, g2.k
PLANS = ["g4triples", "g2_fitted", "g2_IDslot", "g2_seed"]
FAMILIES = ["candidate", "RR"]


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "protocol.frozen.json").exists()
    # The exact comparison axes were already frozen in the capture protocol.
    # Replay code was completed while capture ran. Record honestly if capture
    # finished first; no new routes/results have been inspected at this point.
    capture_plan=f.read(DATA / "protocol.frozen.json")["evaluation"]
    paths = [Path(__file__).resolve(), SOURCE, g2.DRIVER, g2.KERNEL,
             DATA / "protocol.frozen.json", g2.OUT / "support_maps.frozen.npz",
             g2.OUT / "resident_layouts.frozen.json", g2.OUT / "resident_jobs.frozen.json",
             BASE / "g2-availability-headroom/selected_adjacency.frozen.npz",
             r.OUT / "Down_job_maps.frozen.json", r.SHARED / "selection.frozen.json",
             r.SHARED / "menus.frozen.npz", r.BASE / "g8-addendum/group_maps.frozen.npz",
             f.ROOT / "scripts/research_20261102_cr_ipr_full_ffn.py", f.PIPE, f.CPP]
    f.dump(OUT / "protocol.frozen.json", dict(
        utc=f.utc(), status="frozen_before_new_route_inspection", inputs_sha256={str(p):f.sha(p) for p in paths},
        capture_archive_already_exists=(DATA / "decode.routes.npz").exists(),
        earlier_pre_capture_evaluation_plan=capture_plan,
        timing_receipt="Comparison axes frozen in capture protocol before inference; executable replay source completed after capture archive appeared, before root inspected any route or matching/performance outcome. No false pre-inference-code-freeze claim.",
        plans=PLANS, GU_families=FAMILIES, input_copies=[1,4], priority="GF", layers=[0,1,2],
        samples="All16prompts x16decode inputs, without selection, as authored and frozen in capture protocol.",
        dispatch="Same deterministic cap1 matching then cap2 fallback; no unseen-route fitting, no cost oracle.",
        reports="Per-event mean/P99; aligned3layer sum P99; paired regressions; prompt-cluster bootstrap 95% interval of mean paired reduction, seed2026110217,2000resamples. Capacity and all matching failures retained.",
        all40="Fixed canonical IDslot support only, repeated identically across all40layers. Report matching statistics, not full40 FFN speed.",
        payload_MiB=dict(g4triples=244.5,g2_fitted=289.5,g2_IDslot=289.5,g2_seed=289.5),
        FIFO_capacity_bytes=128,
        limits="Conditional W8/A8 service-model replay,16authored diagnostic prompts,not benchmark quality/nativehardware or newCBC. GU source permutation and preECCdrop remain conditional. Matching computation overhead not measured here; no original-GGUF-quality or fresh placed-logit equivalence claim."))
    print(json.dumps(dict(status="frozen",protocol_sha256=f.sha(OUT / "protocol.frozen.json"))),flush=True)


def bootstrap(costs, anchor):
    # Preserve the16 decoding steps and3layers within each prompt cluster.
    new = costs.reshape(3,16,16).sum(axis=(0,2)).astype(np.float64)
    old = anchor.reshape(3,16,16).sum(axis=(0,2)).astype(np.float64)
    rng = np.random.default_rng(2026110217)
    draw = rng.integers(0,16,size=(2000,16))
    reductions = 100*(1-new[draw].sum(axis=1)/old[draw].sum(axis=1))
    return np.percentile(reductions,[2.5,97.5]).tolist()


def run():
    protocol = f.read(OUT / "protocol.frozen.json")
    for path,expected in protocol["inputs_sha256"].items():
        assert f.sha(path)==expected,path
    capture = f.read(DATA / "report.json")
    assert capture["status"] == "complete" and capture["prompt_count"] == 16
    assert capture["decode_sha256"] == f.sha(DATA / "decode.routes.npz")
    with np.load(DATA / "decode.routes.npz") as z:
        assert z["split"].item()=="prospective_authored_diagnostic" and z["completed"].all()
        all_routes=z["routes"].transpose(2,0,1,3).reshape(40,256,6).copy()
    routes=all_routes[:3]
    with np.load(BASE / "g2-availability-headroom/selected_adjacency.frozen.npz") as z:
        maps=dict(g2_IDslot=z["IDslot"].copy(),g2_seed=z["best_single_seed"].copy())
    layouts=dict(g2_fitted=f.read(g2.OUT / "resident_layouts.frozen.json"))
    layouts.update({name:[k.build_layout(g2.supports(x),128) for x in matrix]
                    for name,matrix in maps.items()})
    resident={name:[g2.resident_jobs(layout) for layout in ls] for name,ls in layouts.items()}
    oldmaps=f.read(r.OUT / "Down_job_maps.frozen.json")
    with np.load(r.BASE / "g8-addendum/group_maps.frozen.npz") as z:
        groups=z["learned"].copy()
    assert not (OUT / "report.json").exists()
    f.dump(OUT / "run.frozen.json",dict(utc=f.utc(),protocol_sha256=f.sha(OUT / "protocol.frozen.json"),
            capture_sha256=f.sha(DATA / "report.json"),route_sha256=f.sha(DATA / "decode.routes.npz"),
            source_sha256=f.sha(__file__)))
    recipes=r.Recipes(f.load_pipeline(),a.load_shaper())
    executor=f.Executor(OUT)
    cache,arrays,records={}, {}, []
    began=time.monotonic()
    axes=[(family,plan,copies) for family in FAMILIES for copies in [1,4] for plan in PLANS]
    for ci,(family,plan,copies) in enumerate(axes):
        name=f"{family}.{plan}.input{copies}.GF"
        costs=np.full((3,256),-1,dtype=np.int64)
        fifos=costs.copy()
        matching=costs.copy()
        counts=Counter()
        saved_fallback=False
        for layer in range(3):
            for qi,route in enumerate(routes[layer]):
                if plan=="g4triples":
                    gm=oldmaps
                    graphplan="triple_private_shared4"
                    cap=0
                else:
                    choice=g2.dispatch(route,layouts[plan][layer])
                    selected=[job for expert,group in choice["assignment"].items()
                              for job in resident[plan][layer][(int(expert),int(group))]]
                    selected+=resident[plan][layer][(64,0)]
                    gm={family+".g2availability":[selected]*3}
                    graphplan="g2availability"
                    cap=choice["capacity"]
                graph=r.graph_for(route,layer,family,graphplan,f.Service(copies,0,0),recipes,groups,gm)
                assert graph is not None,(name,layer,qi)
                signature,data=graph.signature()
                signature=(signature,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                need_trace=(qi==0) or (cap==2 and not saved_fallback)
                if signature in cache and not need_trace:
                    value,audit=cache[signature]
                else:
                    result=executor.run(graph,data)
                    audit=f.audit(graph,result,full=need_trace)
                    value=result["cost_ns"]
                    cache[signature]=value,audit
                    if need_trace:
                        suffix="first" if qi==0 else "fallback"
                        f.dump_trace(OUT / f"trace.L{layer}.{name}.{suffix}.q{qi}.json",graph,result,audit)
                if cap==2:
                    saved_fallback=True
                costs[layer,qi]=value
                fifos[layer,qi]=audit["peak_output_fifo_bytes"]
                matching[layer,qi]=cap
                for key,value in audit["counts"].items():
                    counts[key]+=value
        assert np.all(costs>=0) and np.all(fifos<=128)
        arrays[name+".cost"]=costs
        arrays[name+".fifo"]=fifos
        arrays[name+".matching_capacity"]=matching
        rec=dict(name=name,completed_events=int(costs.size),cost_ns=f.stats(costs),
                 three_layer_sum_ns=f.stats(costs.sum(axis=0)),FIFO_peak_bytes=f.stats(fifos),
                 fallback_events=int(np.count_nonzero(matching==2)),
                 payload_MiB=protocol["payload_MiB"][plan],counts_per_event={key:value/costs.size for key,value in counts.items()})
        if plan!="g4triples":
            anchor=arrays[f"{family}.g4triples.input{copies}.GF.cost"]
            rec.update(anchor_name=f"{family}.g4triples.input{copies}.GF",mean_reduction_pct=100*(1-float(costs.mean()/anchor.mean())),
                       event_P99_reduction_pct=100*(1-float(np.percentile(costs,99)/np.percentile(anchor,99))),
                       three_layer_sum_P99_reduction_pct=100*(1-float(np.percentile(costs.sum(axis=0),99)/np.percentile(anchor.sum(axis=0),99))),
                       slower_events=int(np.count_nonzero(costs>anchor)),equal_events=int(np.count_nonzero(costs==anchor)),
                       mean_reduction_prompt_bootstrap_95pct=bootstrap(costs,anchor))
        records.append(rec)
        f.dump(OUT / "progress.json",dict(finished=ci+1,total=len(axes),last=name,elapsed_seconds=time.monotonic()-began))
        print(json.dumps(dict(finished=ci+1,total=len(axes),last=name,elapsed_seconds=time.monotonic()-began)),flush=True)
    canonical=maps["g2_IDslot"][0]
    canonical_layout=k.build_layout(g2.supports(canonical),128)
    failure_by_layer=[]
    for layer in range(40):
        failure_by_layer.append(sum(not k.assign_route(route,canonical_layout["supports"],1)["feasible"] for route in all_routes[layer]))
    np.savez_compressed(OUT / "service_arrays.npz",**arrays)
    f.dump(OUT / "report.json",dict(status="complete_conditional_prospective_diagnostic",records=records,
           source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT / "protocol.frozen.json"),
           route_sha256=f.sha(DATA / "decode.routes.npz"),elapsed_seconds=time.monotonic()-began,
           fixed_IDslot_all40=dict(events=10240,cap1_failures_by_layer=failure_by_layer,
                                   cap1_failures=sum(failure_by_layer)),
           scope=protocol["limits"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["freeze","run"])
    action=parser.parse_args().mode
    freeze() if action=="freeze" else run()
