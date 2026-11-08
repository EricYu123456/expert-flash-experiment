#!/usr/bin/env python3
"""Small calibration-VALID summaries of completed native-prototype side fits.

This summarizes existing selections without fitting, opening target arrays or
claiming fresh holdout/model quality. The 1108_2 native-shared residual study
has priority; these expanded prototype fits are a separately reported branch.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formal-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "calibration VALID native-hidden-state diagnostics, no fresh TEST or live LLM quality",
              "branch": "native hot prototype pool; side branch after 1108_2_spec prioritization",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "layers": {}, "TEST_arrays_opened": False}
    rows = []
    for layer in (1, 13, 26):
        path = args.formal_dir / f"layer{layer}/selection.json"
        d = json.loads(path.read_text())
        if d["status"] != "formal_TRAIN_VALID_no_TEST":
            raise ValueError("incomplete/unrecognized formal selection")
        owners = set(d["prototype_ids"])
        raw = {m["expert"]: m for m in d["families"]["raw_x_eq"]}
        nonlinear = {m["expert"]: m for m in d["families"]["best_cross_prototype"]}
        common = sorted(set(raw) & set(nonlinear))
        legal = [e for e in common if e not in owners]
        summary = {"selection": str(path.resolve()), "selection_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "prototype_ids": sorted(owners), "paired_experts": len(common), "legal_nonowners": len(legal),
                   "lambdas": d["lambdas"], "betas": d["betas"], "fit_wall_seconds": d["wall_seconds"],
                   "aggregate_VALID": d["aggregate_VALID"], "cost_notes": d["cost_notes"]}
        for name, experts in (("all64_diagnostic", common), ("legal_nonowners", legal)):
            raw_sse = sum(raw[e]["weighted_squared_error"] for e in experts)
            proto_sse = sum(nonlinear[e]["weighted_squared_error"] for e in experts)
            energy = sum(raw[e]["selected_VALID_weighted_target_energy"] for e in experts)
            summary[name] = {"experts": len(experts), "prototype_beats_raw_count": sum(nonlinear[e]["weighted_squared_error"] < raw[e]["weighted_squared_error"] for e in experts),
                             "prototype_at_least_10percent_better_count": sum(nonlinear[e]["weighted_squared_error"] <= .9 * raw[e]["weighted_squared_error"] for e in experts),
                             "raw_alpha2_NMSE_pooled": raw_sse / energy,
                             "prototype_alpha2_NMSE_pooled": proto_sse / energy,
                             "prototype_to_raw_pooled_SSE_ratio": proto_sse / raw_sse}
        result["layers"][str(layer)] = summary
        for e in common:
            a, b = raw[e], nonlinear[e]
            rows.append({"layer": layer, "expert": e, "owner_kept_native": e in owners,
                         "raw_width": a["width"], "prototype_width": b["width"], "prototype": b["prototype"],
                         "raw_alpha2_NMSE": a["selected_alpha2_nmse"], "prototype_alpha2_NMSE": b["selected_alpha2_nmse"],
                         "prototype_to_raw_SSE_ratio": b["weighted_squared_error"] / a["weighted_squared_error"],
                         "raw_beta": a["beta"], "raw_lambda": a["lambda"],
                         "prototype_beta": b["beta"], "prototype_lambda": b["lambda"],
                         "TRAIN_selected_events": a["TRAIN_selected_event_count"], "VALID_selected_events": a["VALID_selected_event_count"],
                         "VALID_weighted_target_energy": a["selected_VALID_weighted_target_energy"]})
    (args.output / "prototype_local_VALID_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "prototype_local_VALID_per_expert.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({l: {"legal": v["legal_nonowners"], "fit_seconds": v["fit_wall_seconds"]} for l, v in result["layers"].items()}), flush=True)


if __name__ == "__main__":
    main()
