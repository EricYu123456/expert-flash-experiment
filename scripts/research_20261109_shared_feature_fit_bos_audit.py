#!/usr/bin/env python3
"""Split prototype-vs-raw calibration VALID gains into BOS and non-BOS.

Predictions use canonical deployed W8/A8 payloads on native captured states.
Article bootstrap intervals are descriptive after VALID model selection; this
is not a fresh holdout. No TEST tensors or inputs are accepted by this script.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).with_name("research_20261109_shared_feature_fit_proto.py")
spec = importlib.util.spec_from_file_location("bos_prototype_fit", SOURCE)
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)
base, old = proto.base, proto.old


def audit(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("BOS audit output exists")
    out.mkdir(parents=True)
    selected_path = Path(args.selection).resolve()
    parent = json.loads(selected_path.read_text())
    layer = parent["layer"]
    valid = proto.Dataset(args.valid, args.valid_target_manifest, layer)
    if valid.identity() != parent["valid_identity"]:
        raise ValueError("BOS audit data differ from parent VALID")
    metadata_path = Path(args.prompt_manifest).resolve()
    records = json.loads(metadata_path.read_text())["records"]
    article_map = {r["id"]: r["article_id"] for r in records if r["split"] == "validation"}
    prompt_ids = valid.capture.prompts()
    articles = np.asarray([article_map[p] for p in prompt_ids])
    article_ids = np.unique(articles)
    article_index = np.searchsorted(article_ids, articles)
    positions = valid.capture.get("positions")
    raw = {m["expert"]: m for m in parent["families"]["raw_x_eq"]}
    nonlinear = {m["expert"]: m for m in parent["families"]["best_cross_prototype"]}
    owners = set(parent["prototype_ids"])
    cache, comparisons = {}, []
    rng = np.random.default_rng(args.seed + layer)
    counts = rng.multinomial(len(article_ids), np.ones(len(article_ids)) / len(article_ids), size=args.bootstrap)
    for expert in sorted(set(raw) & set(nonlinear) - owners):
        a2 = base.alpha_squared(valid.capture, expert)
        rows = np.flatnonzero(a2 > 0)
        y = np.asarray(valid.targets.get(expert)[rows], np.float64)
        energy = np.sum(y ** 2, axis=1)
        predictions = {}
        for name, entry in (("raw", raw[expert]), ("prototype", nonlinear[expert])):
            enc = proto.cached_encoded(valid, entry, cache)
            xx = {k: v[rows] for k, v in enc.items()}
            predictions[name] = base.predict_encoded(xx, proto.load_deployed_model(entry)).astype(np.float64)
        errors = {name: np.sum((pred - y) ** 2, axis=1) for name, pred in predictions.items()}
        row = {"layer": layer, "expert": expert, "prototype": nonlinear[expert]["prototype"],
               "partitions": {}, "raw_beta": raw[expert]["beta"], "prototype_beta": nonlinear[expert]["beta"]}
        for name, mask in (("all", np.ones(len(rows), bool)), ("BOS", positions[rows] == 0), ("nonBOS", positions[rows] > 0)):
            selected_rows, weights = rows[mask], a2[rows[mask]]
            sums = {k: float(weights @ v[mask]) for k, v in errors.items()}
            total_energy = float(weights @ energy[mask])
            partition = {"selected_rows": int(np.sum(mask)), "prompts": len(np.unique(prompt_ids[selected_rows])),
                         "articles": len(np.unique(articles[selected_rows])), "alpha2_sum": float(weights.sum()),
                         "weighted_target_energy": total_energy,
                         "raw_weighted_squared_error": sums["raw"], "prototype_weighted_squared_error": sums["prototype"],
                         "raw_alpha2_NMSE": sums["raw"] / total_energy if total_energy else None,
                         "prototype_alpha2_NMSE": sums["prototype"] / total_energy if total_energy else None,
                         "prototype_to_raw_SSE_ratio": sums["prototype"] / sums["raw"] if sums["raw"] else None}
            if name == "nonBOS" and sums["raw"]:
                article_sse = {k: np.bincount(article_index[selected_rows], weights=weights * v[mask], minlength=len(article_ids)) for k, v in errors.items()}
                numerator, denominator = counts @ article_sse["prototype"], counts @ article_sse["raw"]
                bootstrap_ratio = numerator[denominator > 0] / denominator[denominator > 0]
                partition["article_bootstrap_SSE_ratio_CI95"] = np.quantile(bootstrap_ratio, [.025, .975]).tolist()
            row["partitions"][name] = partition
        row["BOS_alpha2_fraction"] = row["partitions"]["BOS"]["alpha2_sum"] / row["partitions"]["all"]["alpha2_sum"]
        row["BOS_target_energy_fraction"] = row["partitions"]["BOS"]["weighted_target_energy"] / row["partitions"]["all"]["weighted_target_energy"]
        comparisons.append(row)
    candidates = [r for r in comparisons if r["partitions"]["nonBOS"]["prototype_to_raw_SSE_ratio"] is not None
                  and r["partitions"]["nonBOS"]["prototype_to_raw_SSE_ratio"] <= .9]
    candidates.sort(key=lambda r: r["partitions"]["nonBOS"]["prototype_to_raw_SSE_ratio"])
    groups = {}
    for row in candidates:
        groups.setdefault(str(row["prototype"]), []).append(row["expert"])
    receipt = {"status": "calibration_VALID_BOS_nonBOS_diagnostic", "layer": layer,
               "selection": str(selected_path), "selection_sha256": old.sha(selected_path),
               "source_sha256": old.sha(__file__), "proto_source_sha256": old.sha(SOURCE),
               "prompt_manifest_metadata_only_sha256": old.sha(metadata_path), "valid_identity": valid.identity(),
               "nonowner_experts": len(comparisons), "bootstrap": args.bootstrap,
               "bootstrap_unit": "original official VALID article; all article prompts sampled together",
               "CI_limit": "descriptive post-selection calibration intervals, not an independent holdout",
               "comparisons": comparisons, "nonBOS_10percent_candidates": candidates,
               "nonBOS_10percent_candidate_groups": groups, "no_TEST_tensors_or_inputs_opened": True}
    old.dump(out / "BOS_nonBOS_VALID.json", receipt)
    csv_rows = []
    for row in comparisons:
        for partition, metrics in row["partitions"].items():
            csv_rows.append({"layer": layer, "expert": row["expert"], "prototype": row["prototype"],
                             "partition": partition, **{k: v for k, v in metrics.items() if not isinstance(v, list)}})
    with (out / "BOS_nonBOS_VALID.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({"layer": layer, "nonBOS_10percent_groups": groups,
                      "candidates": [{"expert": r["expert"], "prototype": r["prototype"],
                                      "ratio": r["partitions"]["nonBOS"]["prototype_to_raw_SSE_ratio"],
                                      "CI": r["partitions"]["nonBOS"]["article_bootstrap_SSE_ratio_CI95"]} for r in candidates]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "valid", "valid-target-manifest", "prompt-manifest", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261109)
    args = p.parse_args()
    with threadpool_limits(limits=args.threads, user_api="blas"):
        audit(args)
