#!/usr/bin/env python3
"""Same-width linear control: calibration-selected 1408 raw-x coordinates.

Parent full-width W8 readouts were selected with VALID, so the pooled C-column
ranking is explicitly calibration-selected, not TRAIN-only feature selection.
No TEST argument exists. The selected gather/affine executes before a NEW A8
boundary; the row-W8 readouts are fitted afresh to those QDQ features.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

PROTO_PATH = Path(__file__).with_name("research_20261109_shared_feature_fit_proto.py")
spec = importlib.util.spec_from_file_location("raw_subset_prototype_tools", PROTO_PATH)
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)
base, old = proto.base, proto.old


def column_ranking(entries):
    """Pooled deployed C^2; deterministic lower coordinate breaks exact ties."""
    width = entries[0]["width"]
    importance = np.zeros(width, np.float64)
    for entry in entries:
        if entry["width"] != width or entry["prototype"] is not None:
            raise ValueError("expected uniform full raw-x parent readouts")
        model = proto.load_deployed_model(entry)
        coefficient = model["qweight"].astype(np.float64) * model["wscale"][:, None]
        importance += np.sum(coefficient ** 2, axis=0)
    ranking = np.lexsort((np.arange(width), -importance))
    return ranking, importance


def gathered_encoded(dataset, transform):
    if np.any(transform["sources"] != 0):
        raise ValueError("raw-x subset control requires source0 only")
    raw = dataset.capture.get("x")[:, transform["indices"]]
    return base.encode(base.transform(raw, transform["center"], transform["inv_scale"]))


def fit(args):
    start = time.monotonic()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("raw-subset output exists")
    out.mkdir(parents=True)
    parent_path = Path(args.parent_selection).resolve()
    parent = json.loads(parent_path.read_text())
    layer = parent["layer"]
    train = proto.Dataset(args.train, args.train_target_manifest, layer)
    valid = proto.Dataset(args.valid, args.valid_target_manifest, layer)
    if train.identity() != parent["train_identity"] or valid.identity() != parent["valid_identity"]:
        raise ValueError("subset control data differ from parent fitting data")
    audit = base.capture_audit(train.capture, valid.capture)
    parent_models = parent["families"]["raw_x_eq"]
    expected = set(train.ids)
    if set(m["expert"] for m in parent_models) != expected:
        raise ValueError("require all parent full-width raw-x experts")
    ranking, importance = column_ranking(parent_models)
    if not 0 < args.width < len(ranking):
        raise ValueError("subset width must be positive and smaller than parent")
    # Sort the chosen coordinates into native x order. This does not change the
    # importance-based set, and fixes a stable gather/row-W8 coefficient order.
    indices = np.sort(ranking[:args.width]).astype(np.int32)
    full_transform = proto.load_transform(parent_models[0]["transform_prefix"])
    if not (np.all(full_transform["sources"] == 0)
            and np.array_equal(full_transform["indices"], np.arange(len(ranking)))):
        raise ValueError("parent full raw-x transform contract changed")
    if any(m["transform_prefix"] != parent_models[0]["transform_prefix"] for m in parent_models):
        raise ValueError("parent raw-x transforms differ")
    transform = {"sources": np.zeros(args.width, np.uint8), "indices": indices,
                 "center": full_transform["center"][indices],
                 "inv_scale": full_transform["inv_scale"][indices],
                 "prototypes": np.full(args.width, -1, np.int32)}
    basis = f"raw_x_select{args.width}_eq"
    tp = out / f"transform.{basis}"
    proto.save_transform(tp, transform)
    importance.tofile(out / "parent_pooled_column_squared_norm.f64")
    ranking.astype(np.int32).tofile(out / "parent_coordinate_ranking.i32")
    experts = list(train.ids if args.experts is None else args.experts)
    if not experts or not set(experts) <= set(train.ids):
        raise ValueError("requested experts absent from targets")
    report = {"status": "calibration_selected_raw_subset_TRAIN_VALID_no_TEST", "layer": layer,
              "width": args.width, "basis": basis, "feature_selection_uses_VALID_indirectly": True,
              "feature_selection": "pooled squared dequantized parent W8 C columns; parent C selected by VALID; ties by coordinate",
              "parent_selection": str(parent_path), "parent_selection_sha256": old.sha(parent_path),
              "feature_ranking_experts": sorted(m["expert"] for m in parent_models),
              "coordinate_basis_shared_across_experts": True,
              "train_identity": train.identity(), "valid_identity": valid.identity(), "capture_audit": audit,
              "prototype_ids": train.prototype_ids, "experts": experts,
              "selected_indices": indices.tolist(), "lambdas": args.lambdas, "betas": args.betas,
              "transform_prefix": str(tp), "contract": base.CONTRACT,
              "source_sha256": old.sha(__file__), "proto_source_sha256": old.sha(PROTO_PATH),
              "host_extra_GU_coefficients": 0, "runtime_extra_GU_MAC": 0,
              "deployment": "gather raw x coordinates; TRAIN affine; whole-vector A8; row-W8 C plus bias",
              "models": [], "grid": []}
    old.dump(out / "protocol.json", report)
    tx, vx = gathered_encoded(train, transform), gathered_encoded(valid, transform)
    moments = proto.all_moments(tx["qdq"])
    for expert in experts:
        a2, va2 = base.alpha_squared(train.capture, expert), base.alpha_squared(valid.capture, expert)
        selected = np.flatnonzero(va2 > 0)
        if not len(selected):
            raise ValueError(f"no VALID routing events for expert{expert}")
        yy, vy = train.targets.get(expert), valid.targets.get(expert)
        em = proto.expert_moments(moments, yy, a2)
        sx = {key: value[selected] for key, value in vx.items()}
        sy, sw = np.asarray(vy[selected]), va2[selected]
        energy = float(np.sum(sw[:, None] * sy.astype(np.float64) ** 2))
        best_score, best = float("inf"), None
        for beta in args.betas:
            combination = proto.combined(moments, em, beta)
            if combination is None:
                continue
            for ridge in args.lambdas:
                model = proto.solve_model(combination, ridge)
                prediction = base.predict_encoded(sx, model)
                score = base.weighted_nmse(sy, prediction, sw)
                record = {"expert": expert, "basis": basis, "prototype": None,
                          "beta": beta, "lambda": ridge, "selected_alpha2_nmse": score,
                          "selected_VALID_rows": len(selected),
                          "selected_VALID_weighted_target_energy": energy,
                          "weighted_squared_error": score * energy,
                          "VALID_selected_alpha2_sum": float(sw.sum()),
                          "TRAIN_selected_event_count": int(np.sum(train.capture.get("selected_experts") == expert)),
                          "VALID_selected_event_count": int(np.sum(valid.capture.get("selected_experts") == expert)),
                          "Gram_rank": combination["rank"],
                          "effective_df": float(np.sum(combination["values"] / (combination["values"] + ridge)))}
                report["grid"].append(record)
                if score < best_score:
                    best_score, best = score, (record, {k: v.copy() for k, v in model.items()})
        record, model = best
        prediction = base.predict_encoded(vx, model)
        record["all_VALID_row_nmse"] = base.weighted_nmse(vy, prediction, np.ones(len(prediction)))
        cp = out / f"{basis}.expert{expert}"
        entry = {**record, "width": args.width, "transform_prefix": str(tp), "readout_prefix": str(cp),
                 "payload_sha256": proto.write_model(cp, model), "TRAIN_rounding_audit": proto.moment_rounding_audit(tx, model),
                 "cost": base.model_cost(args.width, hidden=train.meta["hidden"]),
                 "cross_expert_reuse": False, "owner_self": False}
        report["models"].append(entry)
        print(json.dumps({"fit": basis, "expert": expert, "beta": entry["beta"],
                          "lambda": entry["lambda"], "VALID_alpha2_NMSE": best_score}), flush=True)
    # Pre-populate actual gathered encodings. The immutable v1 helper's generic
    # native raw-x path deliberately has no selected-coordinate assumptions.
    aggregate_cache = {str(tp): {"q": vx["q"], "scale": vx["scale"]}}
    report["aggregate_VALID"] = {}
    for suffix, entries in (("", report["models"]),
                            (".excluding_owners", [m for m in report["models"] if m["expert"] not in train.prototype_ids])):
        (out / f"registry{suffix}.tsv").write_text("\n".join(proto.registry_line(layer, m) for m in entries) + "\n")
        report["aggregate_VALID"][basis + suffix] = proto.aggregate_family(valid, entries, aggregate_cache)
    report["families"] = {basis: report["models"]}
    report["wall_seconds"] = time.monotonic() - start
    old.dump(out / "selection.json", report)
    print(json.dumps({"complete": str(out / "selection.json"), "seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-selection", "train", "valid", "train-target-manifest", "valid-target-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--width", type=int, default=1408)
    parser.add_argument("--experts", nargs="+", type=int)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1, 3, 10])
    parser.add_argument("--betas", nargs="+", type=float, default=[0, .99])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if any(v <= 0 for v in args.lambdas) or any(not 0 <= v < 1 for v in args.betas):
        raise ValueError("positive ridge and beta in [0,1) required")
    with threadpool_limits(limits=args.threads, user_api="blas"):
        fit(args)
