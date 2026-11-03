#!/usr/bin/env python3
"""Frozen sparse g2-replica controls using the existing causal FFN backend.

No new hardware, fitting, or latency oracle in this driver. The supplied support
map is trained/frozen separately. Every dispatch is standard cap1/cap2 matching.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02/round3-cr-ipr"
OUT = BASE / "g2-availability-full-ffn"
DRIVER = BASE / "down-replica-full-ffn/replica_full_ffn.py"
KERNEL = BASE / "g2-availability-headroom/capacity-kernel/kernel_builder.py"
sys.path.insert(0, str(DRIVER.parent))
import replica_full_ffn as r
f, a = r.f, r.a
spec = importlib.util.spec_from_file_location("g2_capacity_kernel", KERNEL)
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
FAMILIES = ["candidate", "RR"]


def maps_from(path, key, index=None):
    with np.load(path) as archive:
        values = archive[key].copy()
    if index is not None:
        values = values[index]
    assert values.shape == (3, 64, 6), values.shape
    assert np.all((values == 0) | (values == 1))
    assert np.all(values.sum(axis=1) == 42)
    assert all(sorted(x.sum(axis=1).tolist()) == [3] * 4 + [4] * 60 for x in values)
    return values.astype(bool)


def supports(matrix):
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def resident_jobs(layout):
    jobs = {}
    for group in range(6):
        for expert in layout["groups"][group]["resident_experts"]:
            entries = []
            for local_chip in range(2):
                meta = k.build_private_job(expert, group, local_chip, layout)["metadata"]
                entries.append(dict(expert=expert, copy=0, resident_copy=group,
                    resident_job_id=f"e{expert}.g{group}.c{2*group+local_chip}",
                    chip=meta["physical_chip"], rows=768, columns=512,
                    block=meta["block"], wl_start=meta["WL_start"], pages=24,
                    row_start=meta["source_row_start"], reserved_WL=6))
            jobs[(expert, group)] = entries
    jobs[(64, 0)] = []
    for local_chip in range(4):
        meta = k.build_shared_job(local_chip)["metadata"]
        jobs[(64, 0)].append(dict(expert=64, copy=0, resident_copy=0,
            resident_job_id=f"shared.c{12+local_chip}", chip=12+local_chip,
            rows=384, columns=1024, block=0, wl_start=128, pages=24,
            row_start=meta["source_row_start"], reserved_WL=6))
    return jobs


def dispatch(route, layout):
    result = k.assign_route(route, layout["supports"], 1)
    if not result["feasible"]:
        result = k.assign_route(route, layout["supports"], 2)
    assert result["feasible"]
    assert sum(result["group_load"]) == 6
    return result


def freeze(path, key, index=None):
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "protocol.frozen.json").exists()
    matrix = maps_from(path, key, index)
    np.savez_compressed(OUT / "support_maps.frozen.npz", selected=matrix)
    layouts = [k.build_layout(supports(x), 128) for x in matrix]
    f.dump(OUT / "resident_layouts.frozen.json", layouts)
    jobs = [{"entries": [entry for entries in resident_jobs(x).values() for entry in entries]}
            for x in layouts]
    f.dump(OUT / "resident_jobs.frozen.json", jobs)
    paths = [Path(__file__).resolve(), DRIVER, KERNEL, Path(path).resolve(),
             OUT / "support_maps.frozen.npz", OUT / "resident_layouts.frozen.json",
             OUT / "resident_jobs.frozen.json", f.PIPE, f.CPP, f.TRAIN,
             ROOT / "scripts/research_20261102_cr_ipr_full_ffn.py",
             r.OUT / "TRAIN512/report.json", r.OUT / "TRAIN512/service_arrays.npz"]
    f.dump(OUT / "protocol.frozen.json", dict(
        status="frozen_before_causal_replay", utc=f.utc(), inputs_sha256={str(p): f.sha(p) for p in paths},
        selected_input_map_key=key, selected_input_map_index=index, GU_families=FAMILIES,
        native="G2 privateDown R768 C512 /24pages/6waves perchip, complete input, one independent cold GEMV. Shared R384 C1024 on chips12..15 also24pages/6waves. Same existing GU and finite-PB policy.",
        dispatch="Standard deterministic augmenting matching capacity1; if impossible use capacity2. No query latency/FFN oracle and no route migration or replica fitting in this driver.",
        storage="252 complete privateDown instances across six2chip groups,42experts/group; original onecopy64instances. Extra141MiB/layer, total289.5MiB. Compared g4triples244.5MiB, +45MiB. Same32chip-block budget; every private job resides wholly in one block.",
        capacity="GU<=128WL required. Ten g2bands in block0 WL128..187,32 in block1 WL0..191, candidate highplane128+252=380. NOT composable with186WL replicatedGU under this local2block allocation.",
        comparators="Both GU families receive identical frozen availability. Existing sealed samefamily g4triples reused as a named lower-storage reference, not an equal-payload claim.",
        data="Previously observed W8 TRAIN first3layers,32prompts/512alignedinputs. Support fitting uses this TRAIN separately. No heldout/new routes.",
        screen_indices=np.linspace(0, 511, 64, dtype=int).tolist(),
        gate="All192events PB/FIFO128valid for bothGU; freeze full execution if at least one family's mean improves>5% over sealed samefamilyg4triples. Report every regression/P99/storage cost.",
        full_axes=dict(input_copies=[1,4], priority=["GF","DF"]), activation_ns=0,
        limitations="Conditional PB/CR/source/drop/ECC-PE/FIFO contract inherited. This is standard replica geometry plus matching, no newCBC or measured device claim. Matching/control runtime not measured by this offline replay; commands and host submissions retain prior costs.",
    ))
    print(json.dumps(dict(status="frozen", source_sha256=f.sha(__file__), protocol_sha256=f.sha(OUT / "protocol.frozen.json"))))


def checked():
    protocol = f.read(OUT / "protocol.frozen.json")
    for path, expected in protocol["inputs_sha256"].items():
        assert f.sha(path) == expected, path
    routes, _, _, groups = r.inputs()
    layouts = f.read(OUT / "resident_layouts.frozen.json")
    jobs = [resident_jobs(x) for x in layouts]
    return protocol, routes, groups, layouts, jobs


def execute(mode):
    protocol, routes, groups, layouts, resident = checked()
    screen = mode == "screen"
    if not screen:
        selection = f.read(OUT / "selection.frozen.json")
        assert selection["gate_passed"]
        assert selection["screen_sha256"] == f.sha(OUT / "screen/report.json")
    indices = np.asarray(protocol["screen_indices"]) if screen else np.arange(512)
    axes = [(family, copies, df) for family in FAMILIES
            for copies in ([1] if screen else [1,4]) for df in ([0] if screen else [0,1])]
    dest = OUT / ("screen" if screen else "TRAIN512")
    dest.mkdir(exist_ok=True)
    assert not (dest / "report.json").exists()
    f.dump(dest / "run.frozen.json", dict(utc=f.utc(), source_sha256=f.sha(__file__), protocol_sha256=f.sha(OUT / "protocol.frozen.json"), indices=indices.tolist(), axes=axes))
    recipes = r.Recipes(f.load_pipeline(), a.load_shaper())
    executor = f.Executor(OUT)
    arrays, records, cache = {}, [], {}
    unique = 0
    began = time.monotonic()
    with np.load(r.OUT / "TRAIN512/service_arrays.npz") as anchor_file:
        anchors = {name: anchor_file[name].copy() for name in anchor_file.files}
    for ci, (family, copies, df) in enumerate(axes):
        name = f"{family}.g2availability.input{copies}.{'DF' if df else 'GF'}"
        costs = np.full((3, len(indices)), -1, dtype=np.int64)
        fifos = costs.copy()
        capacities = costs.copy()
        aggregate = Counter()
        for layer in range(3):
            for qi, index in enumerate(indices):
                route = routes[layer, index]
                decision = dispatch(route, layouts[layer])
                selected = [job for expert, group in decision["assignment"].items()
                            for job in resident[layer][(int(expert), int(group))]]
                selected += resident[layer][(64,0)]
                maps = {family + ".g2availability": [selected] * 3}
                graph = r.graph_for(route, layer, family, "g2availability", f.Service(copies, df, 0), recipes, groups, maps)
                assert graph is not None, "strict PB failed"
                signature, data = graph.signature()
                signature = (signature, df, json.dumps(dict(graph.pipeline_totals), sort_keys=True))
                if signature in cache:
                    value, audit = cache[signature]
                else:
                    result = executor.run(graph, data)
                    audit = f.audit(graph, result, full=qi in (0, len(indices)-1))
                    value = result["cost_ns"]
                    cache[signature] = value, audit
                    unique += 1
                    if qi == 0:
                        f.dump_trace(dest / f"trace.L{layer}.{name}.json", graph, result, audit)
                costs[layer, qi] = value
                fifos[layer, qi] = audit["peak_output_fifo_bytes"]
                capacities[layer, qi] = decision["capacity"]
                for kind, count in audit["counts"].items():
                    aggregate[kind] += count
        anchor_name = f"{family}.triple_private_shared4.input{copies}.{'DF' if df else 'GF'}.cost"
        anchor = anchors[anchor_name][:, indices]
        assert np.all(costs >= 0) and np.all(fifos <= 128)
        arrays[name + ".cost"] = costs
        arrays[name + ".fifo"] = fifos
        arrays[name + ".matching_capacity"] = capacities
        records.append(dict(name=name, completed_events=int(costs.size), cost_ns=f.stats(costs),
            three_layer_sum_ns=f.stats(costs.sum(axis=0)), FIFO_peak_bytes=f.stats(fifos),
            perfect_matching_events=int(np.count_nonzero(capacities == 1)), fallback_events=int(np.count_nonzero(capacities == 2)),
            anchor_name=anchor_name, anchor_mean_ns=float(anchor.mean()),
            mean_reduction_pct=100*(1-float(costs.mean()/anchor.mean())),
            event_P99_reduction_pct=100*(1-float(np.percentile(costs,99)/np.percentile(anchor,99))),
            three_layer_sum_P99_reduction_pct=100*(1-float(np.percentile(costs.sum(axis=0),99)/np.percentile(anchor.sum(axis=0),99))),
            slower_events=int(np.count_nonzero(costs > anchor)), equal_events=int(np.count_nonzero(costs == anchor)),
            counts_per_event={key: value/costs.size for key,value in aggregate.items()}))
        progress = dict(finished=ci+1,total=len(axes),unique_graphs=unique,elapsed_seconds=time.monotonic()-began,last=name)
        f.dump(dest / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    np.savez_compressed(dest / "service_arrays.npz", **arrays)
    f.dump(dest / "report.json", dict(status="complete_conditional_TRAIN_g2_availability", mode=mode,
        records=records, source_sha256=f.sha(__file__), protocol_sha256=f.sha(OUT / "protocol.frozen.json"),
        unique_graphs=unique, elapsed_seconds=time.monotonic()-began,
        scope="Standard sparse-replica matching, two GU controls, conditional model, reused TRAIN; not measured device or novelCBC."))


def select():
    path = OUT / "screen/report.json"
    report = f.read(path)
    assert not (OUT / "selection.frozen.json").exists()
    passed = any(x["mean_reduction_pct"] > 5 for x in report["records"])
    f.dump(OUT / "selection.frozen.json", dict(utc=f.utc(), gate_passed=passed,
        screen_sha256=f.sha(path), source_sha256=f.sha(__file__), selection="No refit; same map and dispatch for both families/all axes."))
    print(json.dumps(dict(gate_passed=passed, comparisons=[{key: r[key] for key in ["name","mean_reduction_pct","event_P99_reduction_pct","slower_events","fallback_events"]} for r in report["records"]])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze","screen","select","run"])
    parser.add_argument("--maps")
    parser.add_argument("--map-key", default="selected")
    parser.add_argument("--map-index", type=int)
    args = parser.parse_args()
    if args.mode == "freeze":
        assert args.maps
        freeze(args.maps, args.map_key, args.map_index)
    elif args.mode == "select":
        select()
    else:
        execute(args.mode)
