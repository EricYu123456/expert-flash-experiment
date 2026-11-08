#!/usr/bin/env python3
"""TRAIN aggregate-residual correction folded into existing W8 readouts.

Each group shares one actual runtime feature vector. For that group fit
delta_y ~= g*(R*QDQ(v)+b_R), then VALID chooses eta from the fixed grid.
eta=0 preserves original payload bytes. No correction head exists at runtime.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

PROTO_PATH = Path(__file__).with_name("research_20261109_shared_feature_fit_proto.py")
spec = importlib.util.spec_from_file_location("joint_proto_tools", PROTO_PATH)
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)
base, old = proto.base, proto.old


def fit_correction(encoded, residual, g, ridge):
    x = np.multiply(encoded["q"].astype(np.float32), encoded["scale"][:, None], dtype=np.float32).astype(np.float64)
    g = np.asarray(g, np.float64)
    g2 = g ** 2
    sw = g2.sum()
    if sw == 0:
        return None
    mean_x = g2 @ x / sw
    mean_y = g @ residual / sw
    covariance = (x.T @ (x * g2[:, None])) / sw - np.outer(mean_x, mean_x)
    cross = (x.T @ (residual * g[:, None])) / sw - np.outer(mean_x, mean_y)
    vals, vecs, rank = proto.symmetric_eigen(covariance)
    c = vecs @ ((vecs.T @ cross) / (vals[:, None] + ridge))
    return {"R": c.T.astype(np.float32), "bias_R": (mean_y - mean_x @ c).astype(np.float32),
            "Gram_rank": rank, "effective_df": float(np.sum(vals / (vals + ridge))),
            "effective_g_rows": float(sw ** 2 / np.sum(g2 ** 2)), "nonzero_g_rows": int(np.sum(g != 0))}


def fold(entry, correction, eta):
    if eta == 0:
        return dict(entry)
    original = proto.load_deployed_model(entry)
    deployed_float = np.multiply(original["qweight"].astype(np.float32), original["wscale"][:, None], dtype=np.float32)
    delta = np.multiply(correction["R"], np.float32(eta), dtype=np.float32)
    coefficient = np.add(deployed_float, delta, dtype=np.float32)
    qweight, wscale = old.quantize_rows(coefficient)
    bias = np.add(original["bias"], np.multiply(correction["bias_R"], np.float32(eta), dtype=np.float32), dtype=np.float32)
    result = dict(entry)
    result["_payload"] = {"qweight": qweight, "wscale": wscale, "bias": bias}
    result["joint_eta"] = eta
    return result


def joint(args):
    start = time.monotonic()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("joint output exists")
    out.mkdir(parents=True)
    parent_path = Path(args.parent_selection).resolve()
    parent = json.loads(parent_path.read_text())
    layer = parent["layer"]
    train = proto.Dataset(args.train, args.train_target_manifest, layer)
    valid = proto.Dataset(args.valid, args.valid_target_manifest, layer)
    if train.identity() != parent["train_identity"] or valid.identity() != parent["valid_identity"]:
        raise ValueError("joint data differ from parent fitting data")
    models = [dict(m) for m in parent["families"][args.family]]
    if args.exclude_owners:
        models = [m for m in models if m["expert"] not in train.prototype_ids]
    train_cache, valid_cache = {}, {}
    initial = proto.aggregate_family(valid, models, valid_cache)
    report = {"status": "TRAIN_joint_fit_VALID_eta_selection_no_TEST", "layer": layer,
              "family": args.family, "exclude_native_owners": args.exclude_owners,
              "parent_selection": str(parent_path), "parent_selection_sha256": old.sha(parent_path),
              "train_identity": train.identity(), "valid_identity": valid.identity(),
              "ridge": args.ridge, "eta_grid": args.etas, "groups": [], "initial_VALID": initial,
              "source_sha256": old.sha(__file__), "proto_source_sha256": old.sha(PROTO_PATH),
              "runtime_extra_coefficients": 0, "runtime_extra_heads": 0,
              "limits": "Local native-hidden-state aggregate calibration; fresh live LLM quality remains separate."}
    group_keys = sorted(set(m["transform_prefix"] for m in models))
    ids, alpha = train.capture.get("selected_experts"), train.capture.get("routing_weights")
    for gi, group in enumerate(group_keys):
        group_entries = [m for m in models if m["transform_prefix"] == group]
        group_experts = [m["expert"] for m in group_entries]
        _, native, current = proto.aggregate_family(train, models, train_cache, return_arrays=True)
        residual = native.astype(np.float64) - current.astype(np.float64)
        g = np.sum(np.where(np.isin(ids, group_experts), alpha, 0), axis=1, dtype=np.float64)
        x = proto.cached_encoded(train, group_entries[0], train_cache)
        correction = fit_correction(x, residual, g, args.ridge)
        if correction is None:
            report["groups"].append({"transform_prefix": group, "experts": group_experts, "skipped": "g is zero"})
            continue
        np.savez(out / f"offline_correction.group{gi}.npz", R=correction["R"], bias_R=correction["bias_R"])
        group_report = {"transform_prefix": group, "experts": group_experts,
                        "TRAIN": {k: v for k, v in correction.items() if k not in ("R", "bias_R")},
                        "offline_R_not_resident_or_executed": True, "candidates": []}
        best_score, chosen_eta, chosen_models = float("inf"), None, None
        for eta in args.etas:
            candidates = [fold(m, correction, eta) if m["transform_prefix"] == group else dict(m) for m in models]
            score = proto.aggregate_family(valid, candidates, valid_cache)
            group_report["candidates"].append({"eta": eta, "VALID": score})
            objective = score["routed"]["squared_error"]
            if objective < best_score:
                best_score, chosen_eta, chosen_models = objective, eta, candidates
        for entry in chosen_models:
            if "_payload" not in entry:
                continue
            prefix = out / f"joint.group{gi}.{entry['basis']}.expert{entry['expert']}"
            model = entry.pop("_payload")
            entry["pre_joint_readout_prefix"] = entry["readout_prefix"]
            entry["readout_prefix"] = str(prefix)
            entry["payload_sha256"] = proto.write_model(prefix, model)
            if "selected_alpha2_nmse" in entry:
                entry["pre_joint_selected_alpha2_nmse"] = entry["selected_alpha2_nmse"]
        models = chosen_models
        group_report["selected_eta"] = chosen_eta
        group_report["eta0_keeps_original_payload_bytes"] = True
        report["groups"].append(group_report)
        print(json.dumps({"group": gi, "basis": group_entries[0]["basis"], "eta": chosen_eta,
                          "VALID_aggregate_squared_error": best_score}), flush=True)
        old.dump(out / "progress.json", {"groups": report["groups"]})
        del native, current, residual
    report["final_VALID"] = proto.aggregate_family(valid, models, valid_cache)
    for entry in models:
        e = entry["expert"]
        a2 = base.alpha_squared(valid.capture, e)
        rows = np.flatnonzero(a2 > 0)
        xx = proto.cached_encoded(valid, entry, valid_cache)
        sub = {key: value[rows] for key, value in xx.items()}
        pred = base.predict_encoded(sub, proto.load_deployed_model(entry))
        selected_y = valid.targets.get(e)[rows]
        entry["selected_alpha2_nmse"] = base.weighted_nmse(selected_y, pred, a2[rows])
        energy = float(np.sum(a2[rows, None] * np.asarray(selected_y, np.float64) ** 2))
        entry["selected_VALID_weighted_target_energy"] = energy
        entry["weighted_squared_error"] = entry["selected_alpha2_nmse"] * energy
        if "all_VALID_row_nmse" in entry:
            entry["pre_joint_all_VALID_row_nmse"] = entry.pop("all_VALID_row_nmse")
    report["models"] = models
    report["wall_seconds"] = time.monotonic() - start
    (out / "registry.tsv").write_text("\n".join(proto.registry_line(layer, m) for m in models) + "\n")
    old.dump(out / "joint_selection.json", report)
    print(json.dumps({"complete": str(out / "joint_selection.json"), "seconds": report["wall_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-selection", required=True)
    parser.add_argument("--family", choices=("raw_x_eq", "best_cross_prototype", "proto_or_raw"), required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--train-target-manifest", required=True)
    parser.add_argument("--valid-target-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge", type=float, default=1)
    parser.add_argument("--etas", nargs="+", type=float, default=[0, .25, .5, 1])
    parser.add_argument("--exclude-owners", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.ridge <= 0 or not args.etas or args.etas[0] != 0:
        raise ValueError("positive TRAIN ridge and eta grid beginning with zero required")
    with threadpool_limits(limits=args.threads, user_api="blas"):
        joint(args)


if __name__ == "__main__":
    main()
