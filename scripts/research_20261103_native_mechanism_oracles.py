#!/usr/bin/env python3
"""Training-only headroom bounds; no placement fit, schedule, or new native ABI.

Qwen3.8 copies below duplicate complete existing shared-expert native matrices.
The exact pair selector is also the generic replicated-declustering baseline.
This file does not import, modify, or execute any sealed generator/replayer.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02/round2"
SRC = BASE / "ordinary-read-compiler/full-campaign-v3"
OUT = BASE / "native-mechanism-next"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_cost(job, service):
    r, c, p, w = (int(job[k]) for k in ("rows", "columns", "pages", "waves"))
    assert p == (r*c + 16383)//16384 and w == (p+3)//4
    assert w <= 192
    mode = service["output_fringe"]
    emit = r if mode == 0 else (p*16384//c if mode == 1 else w*65536//c)
    # The chip context is held through channel drain, not PCIe drain.
    return w*service["ordinary_read_ns"] + service["command_ns"] + (c+1)//2 + 2*emit


def pair_bound(routed, shared, phases, phase_set):
    """Exact indivisible-job minmax over two owners, ignoring future releases.

    This is a necessary work bound, not an executable schedule. The same finite
    list is the generic native-matrix RDA oracle, so their equality is exact.
    """
    q = len(routed)
    result = np.zeros(q, dtype=np.int64)
    for a in range(8):
        b = a+8
        movable = [j for j, (chip, _) in enumerate(shared)
                   if chip in (a, b) and phases[j] in phase_set]
        fixed_a = sum(cost for j, (chip, cost) in enumerate(shared)
                      if chip == a and j not in movable)
        fixed_b = sum(cost for j, (chip, cost) in enumerate(shared)
                      if chip == b and j not in movable)
        sums = sorted({sum(shared[j][1] for j, bit in zip(movable, bits) if bit)
                       for bits in itertools.product((0, 1), repeat=len(movable))})
        total = sum(shared[j][1] for j in movable)
        best = np.full(q, np.iinfo(np.int64).max, dtype=np.int64)
        for x in sums:
            best = np.minimum(best, np.maximum(routed[:, a]+fixed_a+x,
                                               routed[:, b]+fixed_b+total-x))
        result = np.maximum(result, best)
    return result


def stat(values):
    return dict(mean_ns=float(np.mean(values)), p99_ns=float(np.percentile(values, 99)),
                min_ns=int(np.min(values)), max_ns=int(np.max(values)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = {}
    arrays = {}
    records = []
    totals = {}
    capacity = dict(useful_bytes=0, programmed_page_bytes=0, reserved_wave_bytes=0,
                    per_chip_useful_bytes=[0]*16, per_chip_programmed_page_bytes=[0]*16,
                    per_chip_reserved_wave_bytes=[0]*16)
    base_payload = 0
    gu_payload = 0
    for layer in range(48):
        plan = SRC/f"qwen38.L{layer}.cmd250/whole_rows.plan.json"
        npz = plan.with_name("whole_rows.npz")
        ledger[str(plan)] = sha(plan); ledger[str(npz)] = sha(npz)
        d = json.loads(plan.read_text()); z = np.load(npz)
        routes = np.asarray(z["selection_routes"], dtype=np.int64)
        assert routes.shape == (64, 10)
        assert np.all(np.diff(np.sort(routes, axis=1), axis=1) > 0)
        assert (routes >= 0).all() and (routes < 512).all()
        service = d["service"]
        assert service["command_ns"] == 250 and service["ordinary_read_ns"] == 28000
        assert service["input_bytes"] == 1 and service["output_bytes"] == 4
        rowsum = np.zeros((513, 2), dtype=np.int64)
        costs = np.zeros((513, 16), dtype=np.int64)
        shared = []; phases = []
        for job in d["logical"]:
            e, phase, c = (job[k] for k in ("expert", "phase", "chip"))
            rowsum[e, phase] += job["rows"]
            cost = native_cost(job, service)
            costs[e, c] += cost
            if e == 512:
                shared.append((c, cost)); phases.append(phase)
                dest = c ^ 8
                for k, n in [("useful_bytes", job["useful_bytes"]),
                             ("programmed_page_bytes", job["pages"]*16384),
                             ("reserved_wave_bytes", job["waves"]*65536)]:
                    capacity[k] += n
                    capacity["per_chip_"+k][dest] += n
                if phase == 0: gu_payload += job["useful_bytes"]
        assert np.all(rowsum == np.array([1280, 2560]))
        assert len(shared) == 32
        base_payload += d["useful_bytes"]
        routed = costs[routes].sum(axis=1)
        base_work = (routed + costs[512]).max(axis=1)
        bounds = dict(onecopy_work=base_work)
        for name, ps in [("paired_shared_gu", {0}), ("paired_shared_down", {1}),
                         ("paired_shared_both", {0, 1})]:
            bounds[name] = pair_bound(routed, shared, phases, ps)
        # Free, fractional any-owner relocation: more optimistic than duplicating
        # the entire shared expert at every chip, and not a finite-budget plan.
        pooled = np.ceil((routed.sum(axis=1)+costs[512].sum())/16).astype(np.int64)
        bounds["shared_unlimited_fractional"] = np.maximum(routed.max(axis=1), pooled)
        # Fixed complete-native-matrix volume, all routed/shared owners free.
        bounds["all_jobs_unlimited_fractional"] = pooled
        observed = np.asarray(z["cost"], dtype=np.int64)
        assert np.all(observed >= service["host_submit_ns"]+base_work)
        arrays[f"L{layer}.routes"] = routes
        arrays[f"L{layer}.observed"] = observed
        summary = {}
        for name, work in bounds.items():
            b = work + service["host_submit_ns"]
            assert np.all(b <= observed)
            arrays[f"L{layer}.{name}"] = b
            totals[name] = totals.get(name, np.zeros(64, dtype=np.int64)) + b
            summary[name] = stat(b)
        totals["observed"] = totals.get("observed", np.zeros(64, dtype=np.int64)) + observed
        records.append(dict(layer=layer, observed=stat(observed), bounds=summary,
                            plan_path=str(plan), plan_sha256=ledger[str(plan)],
                            array_path=str(npz), array_sha256=ledger[str(npz)]))
    assert capacity["useful_bytes"]*513 == base_payload
    assert gu_payload*3 == capacity["useful_bytes"]*2
    capacity["base_payload_bytes"] = base_payload
    capacity["extra_useful_fraction_full_ffn"] = capacity["useful_bytes"]/base_payload
    capacity["extra_gu_only_useful_fraction_full_ffn"] = gu_payload/base_payload
    capacity["extra_down_only_useful_fraction_full_ffn"] = (capacity["useful_bytes"]-gu_payload)/base_payload
    capacity["fresh_block_upper_bound_bytes"] = sum((x+192*65536-1)//(192*65536)*(192*65536)
                                                   for x in capacity["per_chip_reserved_wave_bytes"])
    capacity["notes"] = "Useful bytes, programmed pages, wave reservations and fresh-block upper bound differ. Existing free holes not repacked. No physical programming performed."
    totals_summary = {k:stat(v) for k, v in totals.items()}
    ref_mean = totals_summary["observed"]["mean_ns"]
    for k, s in totals_summary.items():
        s["optimistic_mean_headroom_pct"] = 100*(1-s["mean_ns"]/ref_mean)
        s["optimistic_p99_headroom_pct"] = 100*(1-s["p99_ns"]/totals_summary["observed"]["p99_ns"])
    result = dict(status="training-only bounds; not achieved timing or new replication algorithm",
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        protocol="48 layers; current ordinary-read compiler whole_rows incumbent; its64fixed training selection routes/layer; no fit, no external payload read;250ns commands; page-fringe output mode preserved.",
        native_mechanism="Immutable complete shared-GU/down native jobs copied once to chip xor8; each selected job executes once at one of two fixed owners. Single context per chip unchanged.",
        generic_rda_equivalence="Exact by construction: every native job is one item; copied owner set is{chip,chip xor8}; weighted indivisible-job maxload selector enumerates the same2^4 choices/pair. No algorithmic speed advantage over the same-inventory native RDA control.",
        capacity=capacity, totals=totals_summary, layers=records,
        scopes=["All values are necessary chip-context-work lower bounds; future GU readiness, per-expert activation/quantization, host submissions, channels and PCIe can increase service.",
                "Positive observed-minus-bound gap is an optimistic upper limit on candidate speedup, not an achieved or guaranteed speedup.",
                "Paired bounds exact only for the fixed native job inventory and unconstrained availability; no shared geometry or owner search was fitted.",
                "Existing onecopy shared geometry is a strong incumbent, not certified global onecopy optimum. Static row/owner refit and whole-shared copies at the same budget remain required controls.",
                "Full48 paired aggregation reuses64selection ordinals; source route IDs are retained. No independent workload claim.",
                "Nominal I8/A8/4B service for routes collected from NVFP4 checkpoint; AiF output/scale ABI and timing calibration remain unresolved."],
        input_sha256=ledger)
    np.savez_compressed(OUT/"qwen38_shared_copy_oracles.npz", **arrays, **{"total."+k:v for k,v in totals.items()})
    (OUT/"qwen38_shared_copy_oracles.json").write_text(json.dumps(result, indent=2)+"\n")
    assert all(sha(p) == h for p, h in ledger.items())
    print(json.dumps(dict(path=str(OUT/"qwen38_shared_copy_oracles.json"),
                          capacity=capacity, totals=totals_summary), indent=2))


if __name__ == "__main__":
    main()
