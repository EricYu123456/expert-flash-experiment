#!/usr/bin/env python3
"""Exact four-owner shared-native-job work oracle, training first3 layers.

No schedule, no fitting, no row splitting, no new physical input/output ABI.
Generic indivisible-item RDA over the same copies is constructively identical.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/research-2026-11-02/round2"
SRC = BASE/"ordinary-read-compiler/full-campaign-v3"
OUT = BASE/"native-mechanism-next"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cost(j):
    r, c, p, w = (j[k] for k in ("rows", "columns", "pages", "waves"))
    assert p == (r*c+16383)//16384 and w == (p+3)//4
    emit = p*16384//c
    return w*28000+250+(c+1)//2+2*emit


def vectors(costs):
    n = len(costs)
    assignments = ((np.arange(4**n)[:, None] >> (2*np.arange(n))) & 3).astype(np.int8)
    loads = np.zeros((len(assignments), 4), dtype=np.int64)
    for k, value in enumerate(costs):
        loads[np.arange(len(assignments)), assignments[:, k]] += value
    return np.unique(loads, axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    total = {k:np.zeros(64, dtype=np.int64) for k in
             ("observed", "onecopy", "whole_expert_view", "phase_independent_views", "native_rda_quad")}
    receipts = {}
    for layer in range(3):
        path = SRC/f"qwen38.L{layer}.cmd250/whole_rows.plan.json"
        data = path.with_name("whole_rows.npz")
        receipts[str(path)] = sha(path); receipts[str(data)] = sha(data)
        d = json.loads(path.read_text()); z = np.load(data)
        assert d["service"]["output_fringe"] == 1
        assert d["service"]["command_ns"] == 250
        costs = np.zeros((513, 16), dtype=np.int64)
        shared = np.zeros((2, 16), dtype=np.int64)
        for j in d["logical"]:
            costs[j["expert"], j["chip"]] += cost(j)
            if j["expert"] == 512: shared[j["phase"], j["chip"]] += cost(j)
        routes = z["selection_routes"]
        assert routes.shape == (64, 10)
        routed = costs[routes].sum(axis=1)
        results = dict(observed=z["cost"].copy(), onecopy=2000+(routed+shared.sum(axis=0)).max(axis=1))
        view1 = []; view2 = []
        for gu in range(4):
            for down in range(4):
                load = np.roll(shared[0], gu*4)+np.roll(shared[1], down*4)
                v = 2000+(routed+load).max(axis=1)
                view2.append(v)
                if gu == down: view1.append(v)
        results["whole_expert_view"] = np.min(view1, axis=0)
        results["phase_independent_views"] = np.min(view2, axis=0)
        quad = np.zeros(64, dtype=np.int64); sizes = []
        for group in range(4):
            chips = np.arange(group, 16, 4)
            jobs = shared[:, chips].reshape(-1)
            assert len(jobs) == 8
            options = vectors(jobs.tolist()); sizes.append(len(options))
            best = np.empty(64, dtype=np.int64)
            for q in range(64):
                best[q] = (options+routed[q, chips]).max(axis=1).min()
            quad = np.maximum(quad, best)
            # Separate scalar brute-force control on first route: no array table.
            brute = min(max(int(routed[0, chips[c]])+sum(int(jobs[j]) for j in range(8) if a[j] == c)
                            for c in range(4)) for a in itertools.product(range(4), repeat=8))
            assert brute == best[0]
        results["native_rda_quad"] = 2000+quad
        assert np.all(results["native_rda_quad"] <= results["phase_independent_views"])
        assert np.all(results["phase_independent_views"] <= results["whole_expert_view"])
        assert np.all(results["whole_expert_view"] <= results["onecopy"])
        for k, v in results.items(): total[k] += v
        records.append(dict(layer=layer, distinct_quad_load_options=sizes,
                            mean_ns={k:float(v.mean()) for k,v in results.items()},
                            p99_ns={k:float(np.percentile(v,99)) for k,v in results.items()}))
    reference = total["observed"]
    stats = {k:dict(mean_ns=float(v.mean()), p99_ns=float(np.percentile(v,99)),
                    optimistic_mean_headroom_pct=float(100*(1-v.mean()/reference.mean())))
             for k,v in total.items()}
    result = dict(status="exact fixed-copy-work oracle, not achieved runtime", created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), protocol="First3layers current whole_rows plans; their64fixed training-only selection queries; no refit or external data.",
        inventory="Each existing shared native job stored at four owners{j,j+4,j+8,j+12}mod16. Exactly three full-shared-expert-equivalent extra copies. No routed copies or moved row endpoints.",
        copy_fraction_full_ffn=3/513, copy_useful_bytes_first3=3*3*(1280*2560+2560*640),
        totals=stats, layers=records, input_sha256=receipts,
        generic_equivalence="native_rda_quad is the generic RDA baseline itself; no new selector mechanism or algorithm is claimed.",
        limits=["All candidate values except observed are necessary native context-work bounds; GU readiness, A8 global scale, resource contention and online selection cost remain.",
                "Whole-expert-view control picks one complete preplaced replica per query; phase-independent control can choose GU/down copies independently; RDA chooses each indivisible native job.",
                "No runtime weights are moved; immutable copies use existing full-row native shapes and one chip context.",
                "Neither the incumbent onecopy shared profile nor four-owner inventory is globally optimal. Samebudget onecopy row/owner refit, flexible shared geometry and generic coalescing are mandatory before any contribution claim.",
                "Population mean and p99 are64training selection ordinals summed over3layers, not heldout or fullmodel latency."])
    np.savez_compressed(OUT/"qwen38_shared_quad_oracle.npz", **total)
    p = OUT/"qwen38_shared_quad_oracle.json"; p.write_text(json.dumps(result,indent=2)+"\n")
    assert all(sha(p)==v for p,v in receipts.items())
    print(json.dumps(dict(path=str(p), totals=stats),indent=2))


if __name__ == "__main__":
    main()
