#!/usr/bin/env python3
"""Known-weight numeric + native schedule probes; upstream is read-only.

No timing, MAC, quantization or cache implementation is added. RouteReplay uses
existing Linear/activation/digital_add and existing cross_dependency bookkeeping.
The cleaned scheduling graph is deliberately never treated as a numeric graph.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["moe", "ibm"], default="moe")
parser.add_argument("--exhaustive", action="store_true")
args = parser.parse_args()
UPSTREAM = HERE / ("upstream" if args.backend == "moe" else "ibm-upstream")
sys.path.insert(0, str(UPSTREAM / "src"))

import numpy as np
import torch
from torch import fx
from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.inference import schedule_execution, trace_model
from threedsim.mapping import Mapper, MapStrategy, Strategy
from threedsim.modules import Linear, MoELayer
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable, set_weight
from threedsim.utils import cross_dependency, digital_add

fx.wrap(cross_dependency)
fx.wrap(digital_add)
torch.set_num_threads(1)
torch.manual_seed(7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(tiles=4, tiers=4, tier_shape=(512, 512)):
    # Default native MVM/ALU/communication timings; cache extensions disabled.
    return AcceleratorConfig(tiles=tiles, tiers=tiers, tier_shape=tier_shape)


def default_info():
    return {"token_id": 0, "seq_len": 1, "decoding_id": 0}


def graph_inventory(graph):
    records = []
    for n in graph.nodes:
        if "tier_linear" in n.name:
            op = dict(n.kwargs["op_info"])
            records.append({"name": n.name, **op})
    return records


def run_schedule(model, acc, tag):
    make_traceable(model, True)
    traced = trace_model(model, bsz=1)
    inventory = graph_inventory(traced.graph)
    for tile in acc.tiles:
        tile.active_time = 0
        tile.num_conflicts = 0
        tile.current_op = None
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = schedule_execution(
            traced.graph, acc, plot=False, communication=True,
            save_memory=True, save_peak_memory=True,
        )
    out = HERE / "native-probe"
    out.mkdir(exist_ok=True)
    (out / f"{args.backend}-{tag}.schedule.log").write_text(log.getvalue())
    (out / f"{args.backend}-{tag}.fx.txt").write_text(str(traced.graph))
    return {
        "execution_time_native_ns": result[0], "peak_memory_reported": result[2],
        "energy_native_nJ": result[3], "flops_reported": result[4],
        "energy_breakdown": result[5], "latency_breakdown": result[6],
        "tier_operations": inventory, "tile_active_times": [t.active_time for t in acc.tiles],
        "tile_conflicts": [t.num_conflicts for t in acc.tiles],
    }, traced


def map_entries(lin):
    return [[list(t) for t in row] for row in lin.mapping]


def map_digest(moe):
    return hashlib.sha256(json.dumps({n: map_entries(m) for n, m in moe.named_modules() if isinstance(m, Linear)}, sort_keys=True).encode()).hexdigest()


def weight_inventory_digest(moe):
    h = hashlib.sha256()
    for n, m in moe.named_modules():
        if isinstance(m, Linear):
            h.update(n.encode())
            for row in m.mapping:
                for tile, tier, _, _, _ in row:
                    h.update(m.accelerator.tiles[tile].tiers[tier].weight.numpy().tobytes())
    return h.hexdigest()


class Pair(torch.nn.Module):
    def __init__(self, lin_a, lin_b):
        super().__init__()
        self.lin_a, self.lin_b = lin_a, lin_b

    def forward(self, x):
        a = self.lin_a(x, op_info=default_info())
        b = self.lin_b(x, op_info=default_info())
        return digital_add(a, b, op_info={"size": 8, **default_info()})


def probe_tier_isolation(shared_tile):
    acc = Accelerator(config(), device="cpu")
    model = Pair(Linear(8, 8, device="cpu"), Linear(8, 8, device="cpu"))
    assign_acc(model, acc)
    a_addr = (0, 0)
    b_addr = (0, 1) if shared_tile else (1, 0)
    for lin, (tile, tier) in zip([model.lin_a, model.lin_b], [a_addr, b_addr]):
        lin.set_mapping([[(tile, tier, 64 / (512 * 512), 8, 8)]])
    fill_name_fields(model)
    a = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 8
    b = torch.eye(8) * -3
    set_weight(model.lin_a, a)
    set_weight(model.lin_b, b)
    x = torch.tensor([1, -2, 3, -4, 5, -6, 7, -8.], dtype=torch.float32)
    y = model(x)
    target = x @ a + x @ b
    assert torch.equal(y, target)
    # Repeated A->B->A verifies stateless numeric tier selection, not device CAP reset.
    assert torch.equal(model.lin_a(x), x @ a)
    assert torch.equal(model.lin_b(x), x @ b)
    assert torch.equal(model.lin_a(x), x @ a)
    result, traced = run_schedule(model, acc, "pair-shared" if shared_tile else "pair-spread")
    assert torch.equal(traced(x), target)
    result.update({"numeric_exact": True, "output": y.tolist(), "addresses": [a_addr, b_addr], "native_mvm_count": len(result["tier_operations"])})
    assert result["native_mvm_count"] == 2
    return result


class SingleLinear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x, op_info=default_info())


def probe_rectangular_linear():
    acc = Accelerator(config(), device="cpu")
    lin = Linear(600, 700, device="cpu")
    model = SingleLinear(lin)
    assign_acc(model, acc)
    mapping = []
    for i in range(2):
        row = []
        for j in range(2):
            nr, nc = min(512, 600 - i * 512), min(512, 700 - j * 512)
            row.append((i * 2 + j, 0, nr * nc / (512 * 512), nr, nc))
        mapping.append(row)
    lin.set_mapping(mapping)
    fill_name_fields(model)
    w = (torch.arange(600 * 700).reshape(600, 700) % 11 - 5).float()
    x = (torch.arange(600) % 7 - 3).float()
    # The original programmer shadows n_cols by the last mapping entry's
    # fringe width, then splits the next input-row block by that smaller width.
    # Keep the failure visible; the outside adapter only corrects programming
    # indices using already-supported public tier weights and fixed addresses.
    original_programmer_error = None
    try:
        set_weight(lin, w)
    except AssertionError as e:
        original_programmer_error = str(e)
        lin.weight.data = w
        for i, row in enumerate(lin.mapping):
            for j, (tile, tier, _, nr, nc) in enumerate(row):
                block = w[i * 512:i * 512 + nr, j * 512:j * 512 + nc]
                assert tuple(block.shape) == (nr, nc)
                acc.tiles[tile].tiers[tier].weight[:nr, :nc] = block
    actual, target = model(x), x @ w
    assert torch.equal(actual, target)
    result, traced = run_schedule(model, acc, "rectangular-600x700")
    assert torch.equal(traced(x), target)
    assert len(result["tier_operations"]) == 4
    assert sum("digital_add" in n.name for n in traced.graph.nodes) == 2
    result.update({"shape": [600, 700], "tier_shape": [512, 512], "mapping": map_entries(lin), "numeric_exact": True, "partial_sums": 2, "output_concat": True, "original_programmer_error": original_programmer_error, "outside_programmer_fix": original_programmer_error is not None, "output_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
    return result


class RouteReplay(torch.nn.Module):
    """Host-known explicit route, native per-expert input/activation/output graph.

    Coefficients are gathered for these exact indices, not topk values paired
    with unrelated random indices. This models neither routing hardware nor a
    new operation. cross_dependency is existing simulator bookkeeping.
    """
    def __init__(self, moe, indices):
        super().__init__()
        self.moe = moe
        self.indices = tuple(int(i) for i in indices)

    def forward(self, x):
        logits = self.moe.router(x, op_info=default_info())
        p = torch.softmax(logits, dim=0)
        p, token = cross_dependency(p, x)
        output = p[self.indices[0]] * self.moe.experts[self.indices[0]](token, op_info=default_info())
        for e in self.indices[1:]:
            term = p[e] * self.moe.experts[e](token, op_info=default_info())
            output = digital_add(output, term, op_info={"size": self.moe.d_model, **default_info()})
        return output


class OrderedMapper(Mapper):
    def __init__(self, *args, order, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = list(order)

    def get_expert_indices_map_order(self, *args):
        return self.order


def make_moe(strategy, *, E=4, k=2, order=None, density=None, tier_shape=(512, 512), tiles=4, tiers=4):
    acc = Accelerator(config(tiles, tiers, tier_shape), device="cpu")
    moe = MoELayer(8, 12, E, k, activation="relu", device="cpu")
    assign_acc(moe, acc)
    # Original mapper expects a model container, not a bare MoELayer root.
    kwargs = dict(accelerator=acc, model=torch.nn.Sequential(moe), map_strategy=MapStrategy(strategy=strategy, split_ffn=True, stack_embedding=True), density=density)
    mapper = Mapper(**kwargs) if order is None else OrderedMapper(**kwargs, order=order)
    with contextlib.redirect_stdout(io.StringIO()):
        mapper.map_network()
    fill_name_fields(moe)
    router = torch.zeros(8, E)
    router[0] = torch.tensor([1., .25, -.25, .5]) if E == 4 else torch.arange(E).float() / 128
    set_weight(moe.router, router)
    for e, expert in enumerate(moe.experts):
        w1 = ((torch.arange(96).reshape(8, 12) + e) % 7 - 3).float() / 8
        w2 = ((torch.arange(96).reshape(12, 8) + 3 * e) % 9 - 4).float() / 16
        set_weight(expert.ffn1, w1)
        set_weight(expert.ffn2, w2)
    return moe, acc, mapper


def reference(moe, x, indices):
    p = torch.softmax(x @ moe.router.weight, dim=0)
    return sum(p[e] * (torch.relu(x @ moe.experts[e].ffn1.weight) @ moe.experts[e].ffn2.weight) for e in indices)


def pair_density():
    q = [(0, 1), (2, 3), (0, 2), (1, 3)]
    probabilities = [.4, .4, .1, .1]
    density = torch.zeros(4, 4)
    for (a, b), p in zip(q, probabilities):
        density[a, b] = density[b, a] = p / 2
    return density, q, probabilities


def probe_routes(strategy, label, order=None):
    density, routes, probabilities = pair_density()
    moe, acc, mapper = make_moe(strategy, order=order, density=density)
    x = torch.tensor([2., 1., -1., .5, 0., 0., 0., 0.])
    actual_gate_route = torch.topk(torch.softmax(x @ moe.router.weight, dim=0), k=2).indices.tolist()
    wrapper = RouteReplay(moe, actual_gate_route)
    target = reference(moe, x, actual_gate_route)
    assert torch.allclose(wrapper(x), target, rtol=1e-6, atol=1e-6)
    frozen_mapping, frozen_weights = map_digest(moe), weight_inventory_digest(moe)
    schedules = []
    max_error = 0.0
    for route in [tuple(actual_gate_route)] + routes:
        wrapper.indices = tuple(route)
        make_traceable(wrapper, False)
        expected = reference(moe, x, route)
        got = wrapper(x)
        err = (got - expected).abs().max().item()
        max_error = max(max_error, err)
        assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6)
        result, traced = run_schedule(wrapper, acc, f"{label}-route-{'-'.join(map(str, route))}")
        assert torch.allclose(traced(x), expected, rtol=1e-6, atol=1e-6)
        assert len(result["tier_operations"]) == 5
        schedules.append({"route": route, **result})
        assert map_digest(moe) == frozen_mapping
        assert weight_inventory_digest(moe) == frozen_weights
    return {
        "label": label, "strategy": strategy.name, "order_override": order,
        "native_actual_gate_topk": actual_gate_route, "max_numeric_abs_error": max_error,
        "all_eager_and_unpruned_fx_numeric_pass": True,
        "mapping_invariant": True, "weight_inventory_invariant": True,
        "mapping_sha256": frozen_mapping, "weight_inventory_sha256": frozen_weights,
        "addresses": {n: map_entries(m) for n, m in moe.named_modules() if isinstance(m, Linear)},
        "query_distribution": {str(q): p for q, p in zip(routes, probabilities)},
        "weighted_native_time_ns": sum(p * s["execution_time_native_ns"] for p, s in zip(probabilities, schedules[1:])),
        "schedules": schedules,
    }


def real_route_smoke():
    path = HERE.parent / "neuron-layout-prior/granite4tiny-routes/screen/routes.npz"
    data = np.load(path, allow_pickle=True)
    # The already-frozen train file only, no heldout routes or model download.
    route_key = "routes" if "routes" in data else list(data.keys())[0]
    routes = data[route_key]
    assert routes.shape == (32, 64, 40, 6), (route_key, routes.shape)
    chosen = routes[0, :2, 0, :].astype(int).tolist()
    moe, acc, _ = make_moe(Strategy.GREEDY_IN_ORDER, E=64, k=6, tier_shape=(256, 256), tiles=16, tiers=9)
    digest, wdigest = map_digest(moe), weight_inventory_digest(moe)
    wrapper = RouteReplay(moe, chosen[0])
    checks = []
    x = torch.tensor([2., 1., -1., .5, 0., 0., 0., 0.])
    for i, route in enumerate(chosen):
        wrapper.indices = tuple(route)
        make_traceable(wrapper, False)
        y, target = wrapper(x), reference(moe, x, route)
        assert torch.allclose(y, target, rtol=1e-6, atol=1e-6)
        result, traced = run_schedule(wrapper, acc, f"real-selected-set-{i}")
        assert torch.allclose(traced(x), target, rtol=1e-6, atol=1e-6)
        assert len(result["tier_operations"]) == 13
        assert map_digest(moe) == digest and weight_inventory_digest(moe) == wdigest
        checks.append({"route": route, "max_numeric_abs_error": (y-target).abs().max().item(), **result})
    return {
        "route_source": str(path), "route_source_sha256": sha(path), "selection": "train[0,0:2,layer0]",
        "semantic_scope": "Real selected expert IDs replayed on known toy weights; not the actual G4 weights/activations or its router probabilities. Golden reference uses exactly the supplied route and toy router coefficient for each selected expert.",
        "fixed_mapping_sha256": digest, "fixed_weights_sha256": wdigest,
        "hardware_config": {"tiles": 16, "tiers": 9, "tier_shape": [256, 256]},
        "checks": checks,
    }


def main():
    result = {
        "backend": args.backend, "upstream": str(UPSTREAM),
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip(),
        "torch": torch.__version__, "device": "CPU",
        "scheduler_import": "PASS unmodified native import; plotting tracked files exist",
        "upstream_source_changes": "none",
        "numerical_scope": "Native ideal torch.float32 F.linear/digital_add. No ADC quantization, scale, noise, retention or fabricated-device precision proof.",
        "timing_scope": "Unmodified backend default event-scheduler model only; no physical NAND/AiF calibration. Router softmax/topk/gate multiplies are not separately timed by the original IMPORTANT_NODES list.",
        "pair_shared": probe_tier_isolation(True),
        "pair_spread": probe_tier_isolation(False),
        "rectangular": probe_rectangular_linear(),
        "moe_layouts": [probe_routes(Strategy.GREEDY_IN_ORDER, "in-order"), probe_routes(Strategy.GREEDY_UTIL, "util"), probe_routes(Strategy.GREEDY_TOP2, "top2")],
    }
    assert result["pair_shared"]["execution_time_native_ns"] > result["pair_spread"]["execution_time_native_ns"]
    if args.exhaustive:
        exhaustive = [probe_routes(Strategy.GREEDY_IN_ORDER, "order-"+"".join(map(str, p)), order=list(p)) for p in itertools.permutations(range(4))]
        best = min(exhaustive, key=lambda r: r["weighted_native_time_ns"])
        result["exhaustive_same_geometry_control"] = {"best_order": best["order_override"], "best_weighted_native_time_ns": best["weighted_native_time_ns"], "trials": [{"order": r["order_override"], "time_ns": r["weighted_native_time_ns"]} for r in exhaustive], "scope": "All24 expert-order permutations through original shape allocator, not all possible atom mappings; same query weights and scheduler for every layout."}
    if args.backend == "moe":
        result["real_route_smoke"] = real_route_smoke()
    result["upstream_git_diff"] = subprocess.check_output(["git", "diff", "--stat"], cwd=UPSTREAM, text=True)
    assert result["upstream_git_diff"] == ""
    output = HERE / f"{args.backend}_native_probe.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": "PASS", "pair_ns": [result["pair_shared"]["execution_time_native_ns"], result["pair_spread"]["execution_time_native_ns"]], "moe_weighted_ns": {r["label"]: r["weighted_native_time_ns"] for r in result["moe_layouts"]}}))


if __name__ == "__main__":
    main()
