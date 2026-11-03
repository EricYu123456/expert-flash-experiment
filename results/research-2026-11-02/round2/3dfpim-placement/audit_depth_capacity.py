#!/usr/bin/env python3
"""Conditional geometry audit, not a 3D-FPIM ISA or numerical emulator.

The public backend does not establish absolute-depth start/reset/programming
semantics. This checks the placement combinatorics assuming those existing
capabilities are later confirmed. It neither changes hardware nor claims they
are present.
"""
import collections
import hashlib
import json
import math
from pathlib import Path


def cyclic_layers(layers, experts, rowtiles, depth, first_core):
    """Pack distinct DNN layers per core, retaining every matrix tile once."""
    assert 64 % depth == 0
    per_core = 64 // depth
    assert per_core <= layers
    mapping = []
    for item in range(experts * rowtiles):
        for layer in range(layers):
            n = item * layers + layer
            mapping.append({
                "layer": layer, "expert": item // rowtiles,
                "rowtile": item % rowtiles,
                "core": first_core + n // per_core,
                "depth_base": (n % per_core) * depth,
                "depths": depth,
            })
    return mapping


def verify(mapping, layers, experts, rowtiles, expected_depth):
    assert len(mapping) == layers * experts * rowtiles
    identities = {(m["layer"], m["expert"], m["rowtile"]) for m in mapping}
    assert len(identities) == len(mapping)
    occupied = set()
    by_layer_core = collections.Counter()
    occupancy = collections.Counter()
    for m in mapping:
        assert m["depths"] == expected_depth
        assert 0 <= m["depth_base"] < 64
        assert m["depth_base"] + m["depths"] <= 64
        by_layer_core[m["layer"], m["core"]] += 1
        for d in range(m["depth_base"], m["depth_base"] + m["depths"]):
            address = m["core"], d
            assert address not in occupied
            occupied.add(address)
            occupancy[m["core"]] += 1
    # This proves the stronger all-experts-selected property, for every route.
    assert max(by_layer_core.values()) == 1
    return {
        "matrix_tiles": len(mapping), "cores": len(occupancy),
        "used_depth_slots": len(occupied),
        "min_depth_slots_per_core": min(occupancy.values()),
        "max_depth_slots_per_core": max(occupancy.values()),
        "max_same_layer_tiles_per_core": max(by_layer_core.values()),
        "any_route_same_phase_core_collision": False,
    }


def main():
    gu = cyclic_layers(24, 32, 8, 8, 0)
    down = cyclic_layers(24, 32, 8, 4, 768)
    gu_report = verify(gu, 24, 32, 8, 8)
    down_report = verify(down, 24, 32, 8, 4)
    assert {m["core"] for m in gu}.isdisjoint({m["core"] for m in down})
    slots = gu_report["used_depth_slots"] + down_report["used_depth_slots"]
    cores = gu_report["cores"] + down_report["cores"]
    assert slots == 73728 and cores == 1152 and slots == cores * 64
    layout_json = json.dumps({"GU": gu, "down": down}, sort_keys=True).encode()
    report = {
        "status": "PASS conditional placement combinatorics only",
        "not_proven": [
            "3D-FPIM can start an independent MVM at nonzero absolute depth",
            "isLast or another existing control resets the physical accumulator",
            "public weight programming ABI accepts multiple matrices per core",
            "numerical QA-ADC precision/scale for nonzero-depth groups",
            "GGUF or pretrained model quantization compatibility",
            "full model fits including non-expert parameters, buffers and code",
            "latency, throughput, energy or physical area",
        ],
        "assumption": "Single-token DNN layers execute sequentially; GU/down core pools are disjoint, so intra-layer overlap causes no new alias.",
        "native_geometry": {"rows": 128, "columns_per_depth": 128, "depths_per_core": 64, "cores_per_tile": 24, "max_tiles": 224},
        "granite3_expert_geometry": {
            "layers": 24, "experts": 32,
            "GU": {"rows": 1024, "columns": 1024, "report": gu_report},
            "down": {"rows": 1024, "columns": 512, "report": down_report},
            "one_matrix_tile_per_core": len(gu) + len(down),
            "depth_capacity_lower_bound_cores": math.ceil(slots / 64),
            "construction_cores": cores,
            "construction_reaches_depth_capacity_lower_bound": True,
            "compute_tiles": math.ceil(cores / 24),
            "max_core_capacity": 224 * 24,
            "fits_expert_weight_geometry": cores <= 224 * 24,
            "mapping_sha256": hashlib.sha256(layout_json).hexdigest(),
            "construction": "For each projection order all matrix tiles by item=(expert,rowtile), then DNN layer. Put each consecutive 64/depth items in one core at disjoint depth offsets. Since 64/depth <= 24, no core contains two tiles from the same layer. GU and down use disjoint core ranges.",
        },
        "granite4tiny_expert_geometry": {
            "layers": 40, "experts": 64,
            "GU_core_tiles_per_expert": 8, "GU_depth_per_tile": 12,
            "down_core_tiles_per_expert": 12, "down_depth_per_tile": 4,
            "one_matrix_tile_per_core": 40 * 64 * 20,
            "used_depth_slots": 40 * 64 * (8 * 12 + 12 * 4),
            "depth_capacity_lower_bound_cores": math.ceil(40 * 64 * (8 * 12 + 12 * 4) / 64),
            "max_core_capacity": 224 * 24,
            "does_not_fit_even_relaxed_expert_geometry": math.ceil(40 * 64 * (8 * 12 + 12 * 4) / 64) > 224 * 24,
        },
        "novelty_implication": "Granite3 expert-weight capacity and same-phase collision elimination can already be optimal through route-oblivious cross-DNN-layer packing. An anti-coactivation placement must improve a separate measured resource or workload regime beyond this strong baseline; capacity fit alone is insufficient.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target = Path(__file__).with_name("depth_capacity_audit.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(target.resolve()), "status": report["status"], "granite3_cores": cores, "granite4tiny_core_lower_bound": report["granite4tiny_expert_geometry"]["depth_capacity_lower_bound_cores"]}))


if __name__ == "__main__":
    main()
