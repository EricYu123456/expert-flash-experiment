#!/usr/bin/env python3
"""TRAIN-only native-batch work arithmetic; no SLIM/AiF native execution."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02/round2"
OUT = BASE / "existing-capability-mechanism-screen/slim-train-union-bound"
CASES = {"granite4tiny": (1536, 512, 1024), "deepseek": (2048, 1408, 2816)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    p = OUT / name
    assert not p.exists(), p
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def protocol():
    paths = [Path(__file__), BASE / "multi-request-onecopy-screen/protocol.frozen.json",
             BASE / "existing-capability-mechanism-screen/sources.json",
             BASE / "existing-capability-mechanism-screen/primary/slim.v1.pdf",
             BASE / "existing-capability-mechanism-screen/primary/slim.v1.txt"]
    paths += [BASE/f"w8-format-training/{model}/decode.routes.npz" for model in CASES]
    return {
        "inputs_sha256": {str(p): sha(p) for p in paths},
        "cases_H_I_sharedI": CASES,
        "scope": "Layer0 only, TRAIN32prompts*16decode steps, fixed cohortsB4/B8. No native simulator, inference, placement/selector fit, arrivals or SLO.",
        "cohort_rule": "Same as frozen multi-request-onecopy-screen: adjacent distinct promptIDs in groupsB, same decode index; allready synthetic schedule.",
        "counting": "Each unique selected expert whole GU/Down fetched once per cohort; each request retains all6experts, complete GU, private activation/A8 and private Down. Shared weights fetched once but multipliedBtimes. This is an optimistic batched-projection work relaxation, not proof the fused SLIM command supports the numerical ABI.",
        "strong_control": "Exactly16die equal g16 whole rows for routed/shared projections; native4KiB SLC or16KiB TLC page accounting, not AiF64KiB/fourplane quanta. Every local rowband is exact page multiple, no duplicates/fringe. It attains both global perprojection page and MAC work/D minima for allcohorts.",
        "SRAM": "Table2 64KiB is totalPE SRAM. Input/output split and scratch are unknown. Report explicit A8input+32bitoutput lower footprint; no claim all64KiB is input or fullABI fits.",
        "extra_DSE": "32/64die legal page-exact chunk-balanced row multisets only, no optimized placement or timing; no64die SLIM result inferred from16die construction.",
        "forbidden": "No AiF batch reuse capability, sparsity/thresholding, fused earlypartialA8, newcontexts, newMAC, modelnumeric/ECC changes, freePB reuse, heldout, native latency or totalbatch gain presented as placementgain."
    }


def stats(a):
    a = np.asarray(a)
    return {"min": float(a.min()), "mean": float(a.mean()), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def pack_pages(jobs, block_pages=512):
    bins = []
    placements = []
    for ident, n in sorted(enumerate(jobs), key=lambda x: (-x[1], x[0])):
        assert 0 < n <= block_pages
        fits = [i for i, used in enumerate(bins) if used+n <= block_pages]
        if fits:
            bid = max(fits, key=lambda i: bins[i])
            off = bins[bid]
            bins[bid] += n
        else:
            bid, off = len(bins), 0
            bins.append(n)
        placements.append({"job_index": ident, "block": bid, "page_offset": off, "pages": n})
    assert sum(bins) == sum(jobs)
    return bins, placements


def row_quantum(R, C, page, D):
    q = page // math.gcd(C, page)
    assert R % q == 0
    chunks = R // q
    rows = np.full(D, chunks // D, np.int64)*q
    rows[:chunks % D] += q
    assert rows.sum() == R and np.all(rows*C % page == 0)
    return {"row_quantum": q, "rows": rows.tolist(), "pages": (rows*C//page).tolist(),
            "positive_jobs": int((rows > 0).sum()), "logical_rows": R,
            "stored_rows": int(rows.sum()), "fringe_or_copies": 0}


def run():
    all_arrays, reports = {}, []
    for model, (H, I, SI) in CASES.items():
        with np.load(BASE/f"w8-format-training/{model}/decode.routes.npz") as f:
            assert f["split"].item() == "train" and f["completed"].all()
            assert all(str(x).startswith("train_") for x in f["prompt_ids"])
            routes = f["routes"][:, :, 0].copy()
            prompt_ids = f["prompt_ids"].copy()
        assert routes.shape == (32, 16, 6)
        assert np.all(np.sort(routes, axis=-1)[:, :, 1:] != np.sort(routes, axis=-1)[:, :, :-1])
        shapes = [(2*I, H), (H, I), (2*SI, H), (H, SI)]
        row16 = [R//16 for R, C in shapes]
        assert all(R % 16 == 0 for R, C in shapes)
        coefficient_per_expert = 3*H*I
        coefficient_shared = 3*H*SI
        model_report = {"model": model, "H": H, "I": I, "sharedI": SI,
                        "projection_order": ["routed_GU", "routed_Down", "shared_GU", "shared_Down"],
                        "logical_rows_columns": shapes, "g16_rows_per_die": row16,
                        "zero_fault_row_contract": "Each view is a one-copy, exact full-row band of all input columns, no aliases/copies/fringe. Global GU/A8 barriers remain; no fused partial activation is executed.",
                        "page_configurations": [], "cohorts": []}
        for page in [4096, 16384]:
            per_die_pages = []
            for (R, C), r in zip(shapes, row16):
                assert r*C % page == 0
                per_die_pages.append(r*C//page)
            resident_jobs = [p for p in per_die_pages[:2] for _ in range(64)] + per_die_pages[2:]
            bins, address = pack_pages(resident_jobs)
            # In these finite cases BFD reaches the simple total-page lower bound.
            assert len(bins) == math.ceil(sum(resident_jobs)/512)
            model_report["page_configurations"].append({
                "page_bytes": page, "g16_per_die_pages_in_projection_order": per_die_pages,
                "all_projection_bands_exact_page_multiple": True,
                "resident_useful_bytes": 64*coefficient_per_expert+coefficient_shared,
                "resident_programmed_page_bytes": sum(resident_jobs)*16*page,
                "g16_per_die_blocks512pages": len(bins), "g16_total_allocated_bytes": len(bins)*16*512*page,
                "per_die_used_pages_per_block": bins, "per_die_address_witness": address,
                "legal_32_64die_page_exact_chunk_profiles": {str(D): [row_quantum(R,C,page,D) for R,C in shapes] for D in [32,64]},
                "DSE_caveat": "Profiles are legal whole-row page-exact arithmetic, not fixedplacement/native timing or proof of uniformly balanced cohort work on32/64dies."
            })
        for B in [4, 8]:
            rr = routes.reshape(32//B, B, 16, 6).transpose(0, 2, 1, 3).reshape(-1, B, 6)
            mult = np.zeros((len(rr), 64), np.int64)
            for q, r in enumerate(rr):
                np.add.at(mult[q], r.reshape(-1), 1)
            assert np.all(mult.sum(1) == 6*B) and mult.max() <= B
            selected = mult > 0
            union = selected.sum(1)
            useful_read = union*coefficient_per_expert+coefficient_shared
            separate_read = B*(6*coefficient_per_expert+coefficient_shared)
            total_mac = np.full(len(rr), separate_read, np.int64)
            assert np.all(total_mac % 16 == 0)
            # Staticg16: each die owns identical full-row coefficient counts.
            mac16 = total_mac//16
            phase_mac16 = np.stack([np.full(len(rr), B*(6*2*I*H+2*SI*H)//16, np.int64),
                                   np.full(len(rr), B*(6*H*I+H*SI)//16, np.int64)], axis=1)
            assert np.array_equal(phase_mac16.sum(1), mac16)
            pstats = []
            for page in [4096, 16384]:
                assert np.all(useful_read % (16*page) == 0)
                per_die_pages = useful_read//(16*page)
                explicit = selected @ np.full((64,16), coefficient_per_expert//(16*page), np.int64)
                explicit += coefficient_shared//(16*page)
                assert np.all(explicit == per_die_pages[:,None])
                pstats.append({"page_bytes": page, "global_fetched_pages": stats(useful_read//page),
                               "g16_per_die_fetched_pages": stats(per_die_pages),
                               "g16_placement_only_page_work_improvement_ceiling_pct": 0.0,
                               "global_page_floor_attained_each_cohort": True})
            # Footprint is explicit in the original W8/A8 stage contract, not a
            # verified standalone projection ABI on SLIM's published fusedFFN.
            footprint = []
            for name, (R,C), r in zip(model_report["projection_order"],shapes,row16):
                input_bytes = B*C
                output_bytes = B*r*4
                assert C*127*127 < 2**31
                footprint.append({"projection": name, "max_full_batch_A8input_bytes": input_bytes,
                                  "max_full_batch_32bitoutput_bytes_per_die": output_bytes,
                                  "sum_without_scales_scratch_metadata": input_bytes+output_bytes})
            histogram = np.bincount(mult.reshape(-1), minlength=B+1)
            cohort = {"B": B, "cohort_count": len(rr), "requests": len(rr)*B,
                      "same_decode_step_distinct_prompts": True, "synthetic_allready": True,
                      "selected_expert_union": stats(union),
                      "multiplicity_histogram_including_unselected0": histogram.tolist(),
                      "total_logical_expert_evaluations_each_cohort": 6*B,
                      "routed_logical_output_rows_per_request_GU_Down": [6*2*I,6*H],
                      "shared_output_rows_per_request_GU_Down": [2*SI,H],
                      "global_fetched_coefficient_bytes": stats(useful_read),
                      "ideal_read_saving_vs_separate_requests_pct": stats(100*(1-useful_read/separate_read)),
                      "read_saving_is_batching_not_placement": True,
                      "global_exact_MACs_per_cohort": int(total_mac[0]),
                      "g16_exact_MACs_per_die": int(mac16[0]),
                      "g16_MAC_work_floor_attained_each_cohort": True,
                      "g16_peak16MAC_1GHz_compute_time_LB_ns": int(mac16[0]//16),
                      "page_counts": pstats, "required_buffer_lower_footprints": footprint,
                      "largest_input_plus_output_footprint_bytes": max(f["sum_without_scales_scratch_metadata"] for f in footprint),
                      "SRAM_status": "Below64KiB total for these ideal stage buffers, but input/output partition, scratch, scales and actual fullvectorABI remain unverified. No all64KiB input assumption or native fit claim.",
                      "unresolved_headroom": "Commands,input/reload/output,channel/PCIe,GU/privateDown readiness and compute/read overlap are outside these necessary resource counts. They do not justify a new placement contribution; samebatch allwidth/unequalrow controls are mandatory."}
            model_report["cohorts"].append(cohort)
            all_arrays[f"{model}.B{B}.routes"] = rr
            all_arrays[f"{model}.B{B}.expert_multiplicity"] = mult
            all_arrays[f"{model}.B{B}.union"] = union
            all_arrays[f"{model}.B{B}.g16_MAC_per_phase_per_die"] = phase_mac16
        all_arrays[f"{model}.prompt_ids"] = prompt_ids
        reports.append(model_report)
    save("report.json", {"status": "Bounded work null for16die g16; no native/serving claim",
                          "protocol_sha256": sha(OUT/"protocol.frozen.json"), "models": reports,
                          "conclusion": "Actual batching saves coefficient reads, while fixed onecopy g16 attains the usefulpage and MAC balance floors on16dies. Therefore these resources establish no placement-only opportunity beyond a same-batching strongcontrol. Other native costs/interfaces remain unknown;32/64die rowprofileDSE does not establish a gain."})
    p = OUT / "arrays.npz"
    assert not p.exists()
    np.savez_compressed(p, **all_arrays)
    print(json.dumps({m["model"]: [{"B": c["B"], "mean_union": c["selected_expert_union"]["mean"],
                                   "batching_read_saving_pct": c["ideal_read_saving_vs_separate_requests_pct"]["mean"],
                                   "buffer_bytes": c["largest_input_plus_output_footprint_bytes"]} for c in m["cohorts"]] for m in reports}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    p = protocol()
    if args.freeze:
        save("protocol.frozen.json", p)
        print(sha(OUT/"protocol.frozen.json"))
    else:
        # JSON stores tuples as lists; compare through canonical serialization.
        assert json.loads((OUT/"protocol.frozen.json").read_text()) == json.loads(json.dumps(p))
        run()
