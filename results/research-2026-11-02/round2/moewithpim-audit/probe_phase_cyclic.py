#!/usr/bin/env python3
"""Two-layer, fixed-capacity phase placement null against native DAG floor.

This is a small occupancy regime, not a claim about a full-size G4 deployment.
The critical path uses the original backend latency functions without changing
the scheduler or interpreting a relaxed resource model as runnable hardware.
"""
import collections
import contextlib
import copy
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
backend = sys.argv[1] if len(sys.argv) > 1 else "moe"
assert backend in ["moe", "ibm"]
sys.argv = ["probe_native_backend.py", "--backend", backend]
spec = importlib.util.spec_from_file_location("native_probe", HERE / "probe_native_backend.py")
n = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = n
spec.loader.exec_module(n)
torch = n.torch
from threedsim.graph.processing import add_comm_nodes, prune_and_merge_ops, remove_unimportant_nodes_and_reconnect
from threedsim.inference.scheduler import _assign_latency


class Cascade(torch.nn.Module):
    def __init__(self, modules, routes):
        super().__init__()
        self.layers = torch.nn.ModuleList([n.RouteReplay(m, r) for m, r in zip(modules, routes)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def critical_path(model, cfg):
    graph = copy.deepcopy(n.trace_model(model, bsz=1).graph)
    prune_and_merge_ops(graph)
    remove_unimportant_nodes_and_reconnect(graph)
    add_comm_nodes(graph)
    nodes = [v for v in graph.nodes if v.op != "output"]
    node_set = set(nodes)
    degree = {v: sum(u in node_set for u in v._input_nodes) for v in nodes}
    queue = collections.deque(v for v in nodes if degree[v] == 0)
    finish = {}
    while queue:
        v = queue.popleft()
        _assign_latency(v, cfg)
        finish[v] = v.latency + max((finish[u] for u in v._input_nodes if u in node_set), default=0)
        for w in v.users:
            if w not in node_set:
                continue
            degree[w] -= 1
            if degree[w] == 0:
                queue.append(w)
    assert len(finish) == len(nodes), "Native cleaned graph must be acyclic"
    return max(finish.values()), len(nodes)


def make_model(strategy=None, down_permutation=(0, 1, 2, 3)):
    acc = n.Accelerator(n.config(tiles=4, tiers=6), device="cpu")
    modules = [n.MoELayer(8, 12, 4, 2, activation="relu", device="cpu") for _ in range(2)]
    model = Cascade(modules, [(0, 1), (2, 3)])
    n.assign_acc(model, acc)
    if strategy is not None:
        density, _, _ = n.pair_density()
        mapper = n.Mapper(accelerator=acc, model=model, map_strategy=n.MapStrategy(strategy=strategy, split_ffn=True, stack_embedding=True), density=density)
        with contextlib.redirect_stdout(io.StringIO()):
            mapper.map_network()
    else:
        used = [0] * 4
        def allocate(linear, tile):
            nr, nc = linear.weight.shape
            linear.set_mapping([[(tile, used[tile], nr*nc/(512*512), nr, nc)]])
            used[tile] += 1
        # Each phase of every DNN layer has exactly one expert atom per bank;
        # DNN layers/projections share original tier capacity, not new contexts.
        for module in modules:
            allocate(module.router, 0)
            for expert in range(4):
                allocate(module.experts[expert].ffn1, expert)
                allocate(module.experts[expert].ffn2, down_permutation[expert])
        assert max(used) <= 6 and sum(used) == 18
    n.fill_name_fields(model)
    for layer, module in enumerate(modules):
        w = torch.zeros(8, 4)
        w[0] = torch.tensor([1., .25, -.25, .5])
        n.set_weight(module.router, w)
        for e, expert in enumerate(module.experts):
            n.set_weight(expert.ffn1, ((torch.arange(96).reshape(8, 12)+e+layer)%7-3).float()/8)
            n.set_weight(expert.ffn2, ((torch.arange(96).reshape(12, 8)+3*e+layer)%9-4).float()/16)
    return model, acc


def check(model, acc, routes, tag):
    x = torch.tensor([2., 1., -1., .5, 0., 0., 0., 0.])
    target = x
    for layer, route in zip(model.layers, routes):
        layer.indices = tuple(route)
        target = n.reference(layer.moe, target, route)
    n.make_traceable(model, False)
    actual = model(x)
    assert torch.allclose(actual, target, rtol=1e-6, atol=1e-6)
    n.make_traceable(model, True)
    floor, nodes = critical_path(model, acc.config)
    scheduled, traced = n.run_schedule(model, acc, tag)
    assert torch.allclose(traced(x), target, rtol=1e-6, atol=1e-6)
    assert len(scheduled["tier_operations"]) == 10
    return {"routes": routes, "native_ns": scheduled["execution_time_native_ns"], "native_DAG_floor_ns": floor, "gap_ns": scheduled["execution_time_native_ns"]-floor, "nodes": nodes, "numeric_max_abs_error": (actual-target).abs().max().item()}


def main():
    routes = list(itertools.combinations(range(4), 2))
    reference_model, acc = make_model()
    fixed_mapping = [n.map_digest(l.moe) for l in reference_model.layers]
    checks = [check(reference_model, acc, (a,b), f"cyclic-{i}-{j}") for i,a in enumerate(routes) for j,b in enumerate(routes)]
    assert all(c["gap_ns"] == 0 for c in checks)
    assert [n.map_digest(l.moe) for l in reference_model.layers] == fixed_mapping
    permutations = []
    for idx, permutation in enumerate(itertools.permutations(range(4))):
        model, acc = make_model(down_permutation=permutation)
        result = check(model, acc, ((0,2),(1,3)), f"phase-permutation-{idx}")
        permutations.append({"down_permutation": permutation, **result})
    assert all(c["gap_ns"] == 0 for c in permutations)
    baselines = []
    for strategy in [n.Strategy.GREEDY_IN_ORDER, n.Strategy.GREEDY_UTIL, n.Strategy.GREEDY_TOP2]:
        model, acc = make_model(strategy=strategy)
        checks_base = [check(model, acc, (a,b), f"cascade-{strategy.name}-{i}-{j}") for i,a in enumerate(routes) for j,b in enumerate(routes)]
        baselines.append({"strategy": strategy.name, "uniform_pair_mean_native_ns": sum(c["native_ns"] for c in checks_base)/len(checks_base), "min_ns": min(c["native_ns"] for c in checks_base), "max_ns": max(c["native_ns"] for c in checks_base), "checks": checks_base})
    report = {
        "backend": backend, "upstream_commit": n.subprocess.check_output(["git","rev-parse","HEAD"],cwd=n.UPSTREAM,text=True).strip(),
        "status": "PASS native numerical/schedule null: phase-cyclic reaches same-native-DAG critical path",
        "config": {"tiles": 4, "tiers_per_tile": 6, "tier_shape": [512,512], "DNN_layers": 2, "experts_per_layer": 4, "topk": 2, "resident_tier_weights": 18, "capacity_tiers": 24, "model_ott": True, "native_timing_overrides": False},
        "cross_layer_phase_cyclic": checks,
        "all24_down_phase_permutations": permutations,
        "original_mapper_controls": baselines,
        "maximum_phase_cyclic_gap_ns": max(c["gap_ns"] for c in checks),
        "conclusion": "In this low-occupancy regime, route-oblivious per-phase distinct-bank placement plus cross-DNN-layer tier packing removes all schedulable resource delay. Changing projection bank relationships has zero remaining native timing headroom. Original whole-expert order limitations are not a research novelty claim.",
        "scope_limit": "This does not cover full-size G4 matrices, input/output ranges, many routes or large concurrency. Each complete tiny projection occupies one native tier; when a phase has more atoms than peripherals the zero-collision argument no longer applies. Route replay uses known toy weights, not downloaded model weights. This abstract AIMC model is not a calibrated NAND chip.",
        "upstream_git_diff": n.subprocess.check_output(["git","diff","--stat"],cwd=n.UPSTREAM,text=True),
    }
    assert report["upstream_git_diff"] == ""
    out = HERE / f"{backend}_phase_cyclic.json"
    out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(out), "phase_cyclic_ns": sorted(set(c["native_ns"] for c in checks)), "all36_routes_hit_floor": True, "all24_phase_permutations_hit_floor": True, "baseline_mean_ns": {r["strategy"]:r["uniform_pair_mean_native_ns"] for r in baselines}}))


if __name__ == "__main__":
    main()
