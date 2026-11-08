#!/usr/bin/env python3
"""TRAIN-only prototype ranking, then full-output TRAIN/VALID readout fitting.

No TEST arguments exist. Prototype owners are excluded from cross-expert
ranking. Runtime affine transforms remain before A8. Cached hot GU/native D
is a separate exact control, never counted as cross-expert approximation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

BASE_PATH = Path(__file__).with_name("research_20261109_shared_feature_fit.py")
spec = importlib.util.spec_from_file_location("shared_fit_1109_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
old = base.old


class Dataset:
    def __init__(self, capture, targets, layer):
        self.capture = old.Capture(capture, layer)
        self.targets = base.Targets(self.capture, targets, layer)
        self.path = Path(targets).resolve()
        self.meta = self.targets.meta
        if self.meta["layer"] != layer or self.meta["rows"] != len(self.capture.get("x")):
            raise ValueError("target/capture layer or row mismatch")
        if self.meta["capture_manifest_sha256"] != self.capture.identity():
            raise ValueError("target/capture manifest hash mismatch")
        for path, expected in self.meta["input_sha256"].items():
            if old.sha(path) != expected:
                raise ValueError(f"target/input payload hash mismatch: {path}")
        self.ids = self.targets.ids
        self.prototype_ids = sorted(map(int, self.meta["prototype_hidden"]))

    def prototype(self, owner):
        rec = self.meta["prototype_hidden"][str(owner)]
        path = Path(rec["path"])
        if not path.is_absolute():
            path = self.path.parent / path
        shape, dtype = tuple(rec["shape"]), np.dtype(rec.get("dtype", "float32"))
        if path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError("prototype file size mismatch")
        return np.memmap(path, mode="r", dtype=dtype, shape=shape)

    def identity(self):
        return {"target_manifest": str(self.path), "target_manifest_sha256": old.sha(self.path),
                "capture_manifests": self.capture.identity(), "input_sha256": self.meta["input_sha256"]}


def definition(dataset, owner=None):
    raw = dataset.capture.get("x") if owner is None else dataset.prototype(owner)
    center, inv_scale = base.mean_std(raw)
    width = raw.shape[1]
    return {"sources": np.full(width, 0 if owner is None else 2, np.uint8),
            "indices": np.arange(width, dtype=np.int32), "center": center, "inv_scale": inv_scale,
            "prototypes": np.full(width, -1 if owner is None else owner, np.int32)}


def encoded(dataset, transform, owner=None, rows=None):
    raw = dataset.capture.get("x") if owner is None else dataset.prototype(owner)
    if rows is not None:
        raw = raw[rows]
    return base.encode(base.transform(raw, transform["center"], transform["inv_scale"]))


def save_transform(prefix, transform):
    base.save_transform(prefix, transform)
    transform["prototypes"].tofile(str(prefix) + ".prototypes.i32")


def load_transform(prefix):
    return {key: np.fromfile(str(prefix) + suffix, dtype=dtype) for key, suffix, dtype in
            [("sources", ".sources.u8", np.uint8), ("indices", ".indices.i32", np.int32),
             ("center", ".center.f32", np.float32), ("inv_scale", ".inv_scale.f32", np.float32),
             ("prototypes", ".prototypes.i32", np.int32)]}


def native_self_controls(dataset):
    """Small all-output-channel bit checks; own native D uses RAW phi A8."""
    reports = []
    for owner in dataset.prototype_ids:
        rec = dataset.meta["prototype_hidden"][str(owner)]
        prefix = rec["native_D_prefix"]
        phi = dataset.prototype(owner)
        h, width = dataset.meta["hidden"], phi.shape[1]
        weight = np.fromfile(prefix + ".i8", np.int8).reshape(h, width)
        scale = np.fromfile(prefix + ".scales.f32", np.float32)
        rows = np.asarray(sorted({0, len(phi) // 3, len(phi) // 2, len(phi) - 1}))
        q, sx = old.quantize_rows(phi[rows])
        pred = old.rescale(old.exact_dots(q, weight), scale, sx)
        target = np.ascontiguousarray(dataset.targets.get(owner)[rows])
        if not np.array_equal(pred.view(np.uint32), target.view(np.uint32)):
            raise ValueError(f"prototype {owner} native-D exact self control mismatch")
        reports.append({"owner": owner, "rows": rows.tolist(), "all_output_floats_bitchecked": int(pred.size),
                        "own_native_D_exact": True, "cross_expert_reuse": False})
    return reports


def rank(args):
    started = time.monotonic()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("ranking output exists")
    out.mkdir(parents=True)
    train = Dataset(args.train, args.train_target_manifest, args.layer)
    ids = train.capture.prompts()
    prompts = np.unique(ids)
    rng = np.random.default_rng(args.seed + args.layer)
    rng.shuffle(prompts)
    count = max(1, int(len(prompts) * .8))
    fit_prompts, score_prompts = prompts[:count], prompts[count:]
    fit_rows = np.flatnonzero(np.isin(ids, fit_prompts))
    score_rows = np.flatnonzero(np.isin(ids, score_prompts))
    if len(fit_rows) > args.rank_fit_rows:
        fit_rows = np.sort(rng.choice(fit_rows, args.rank_fit_rows, replace=False))
    if len(score_rows) > args.rank_score_rows:
        score_rows = np.sort(rng.choice(score_rows, args.rank_score_rows, replace=False))
    if set(ids[fit_rows]) & set(ids[score_rows]):
        raise AssertionError("internal ranking split leaks prompts")
    if not len(score_rows):
        raise ValueError("no prompt-disjoint internal rank-score rows")
    h = train.meta["hidden"]
    sketch = (rng.choice([-1.0, 1.0], size=(h, args.sketch_width)) / np.sqrt(args.sketch_width)).astype(np.float32)
    sketch.tofile(out / "TRAIN_output_sketch.f32")
    fit_rows.astype(np.int64).tofile(out / "rank_fit_rows.i64")
    score_rows.astype(np.int64).tofile(out / "rank_score_rows.i64")
    experts = list(args.experts if args.experts is not None else train.ids)
    ys_fit, ys_score = [], []
    score_a2 = {}
    for expert in experts:
        y = train.targets.get(expert)
        ys_fit.append(np.asarray(y[fit_rows], np.float64) @ sketch.astype(np.float64))
        ys_score.append(np.asarray(y[score_rows], np.float64) @ sketch.astype(np.float64))
        score_a2[expert] = base.alpha_squared(train.capture, expert)[score_rows]
    yfit = np.concatenate(ys_fit, axis=1)
    yscore = np.concatenate(ys_score, axis=1)
    rankings = {e: [] for e in experts}
    basis_records = []
    for owner in train.prototype_ids:
        transform = definition(train, owner)
        prefix = out / f"transform.prototype_{owner}_eq"
        save_transform(prefix, transform)
        xf = encoded(train, transform, owner, fit_rows)["qdq"]
        xs = encoded(train, transform, owner, score_rows)["qdq"]
        gram = base.prepare_gram(xf, np.ones(len(xf)))
        target_mean, rhs = base.solve_components(gram, yfit)
        model = base.solve_one(gram, target_mean, rhs, args.rank_lambda)
        pred = xs.astype(np.float64) @ model["coefficient"].astype(np.float64).T + model["bias"]
        for ei, expert in enumerate(experts):
            a, b = ei * args.sketch_width, (ei + 1) * args.sketch_width
            metric = base.weighted_nmse(yscore[:, a:b], pred[:, a:b], score_a2[expert])
            if metric is None:
                metric = base.weighted_nmse(yscore[:, a:b], pred[:, a:b], np.ones(len(score_rows)))
            rankings[expert].append({"prototype": owner, "internal_TRAIN_score": metric,
                                      "owner_self": owner == expert,
                                      "rank_score_selected_rows": int(np.sum(score_a2[expert] > 0))})
        basis_records.append({"prototype": owner, "transform_prefix": str(prefix),
                              "definition": "native prototype pre-Down phi, TRAIN equalized then A8; full width1408"})
        print(json.dumps({"ranked_prototype": owner, "experts": len(experts)}), flush=True)
    choices = {}
    for expert, records in rankings.items():
        cross = sorted([r for r in records if not r["owner_self"]], key=lambda r: (r["internal_TRAIN_score"], r["prototype"]))
        choices[str(expert)] = [r["prototype"] for r in cross[:args.top_prototypes]]
    report = {"status": "TRAIN_only_prototype_ranking", "layer": args.layer, "seed": args.seed,
              "train_identity": train.identity(), "fit_prompt_ids": fit_prompts.tolist(),
              "score_prompt_ids": score_prompts.tolist(), "rank_fit_rows": len(fit_rows),
              "rank_score_rows": len(score_rows), "sketch_width": args.sketch_width,
              "sketch_projection": "fixed seeded Rademacher; offline ranking only, absent from runtime",
              "rank_lambda": args.rank_lambda, "ranking_fit_weight": "all-row; ranking score uses routed alpha^2",
              "prototype_ids": train.prototype_ids, "experts": experts,
              "rankings": {str(k): v for k, v in rankings.items()}, "choices": choices,
              "bases": basis_records, "self_controls_TRAIN": native_self_controls(train),
              "no_VALID_or_TEST_targets_opened": True,
              "source_sha256": old.sha(__file__), "wall_seconds": time.monotonic() - started}
    old.dump(out / "ranking.json", report)
    print(json.dumps({"ranking": str(out / "ranking.json"), "seconds": report["wall_seconds"]}), flush=True)


def symmetric_eigen(gram):
    vals, vecs = eigh((gram + gram.T) * .5, driver="evd", check_finite=False)
    tol = max(1, float(vals[-1])) * 1e-10
    if vals[0] < -tol:
        raise ValueError("weighted covariance negative beyond roundoff")
    vals = np.maximum(vals, 0)
    return vals, vecs, int(np.sum(vals > tol))


def all_moments(x):
    xx = np.asarray(x, np.float64)
    mean = xx.mean(axis=0)
    second = (xx.T @ xx) / len(xx)
    eig = symmetric_eigen(second - np.outer(mean, mean))
    return {"x": xx, "mean": mean, "second": second, "eig": eig}


def expert_moments(moments, y, a2):
    x = moments["x"]
    yy = np.asarray(y, np.float64)
    ymean = yy.mean(axis=0)
    xy = (x.T @ yy) / len(x)
    rows = np.flatnonzero(a2 > 0)
    if not len(rows):
        return {"y0": ymean, "xy0": xy, "selected": None}
    weights = a2[rows]
    sx, sy = x[rows], yy[rows]
    sw = weights.sum()
    selected = {"x": weights @ sx / sw, "y": weights @ sy / sw,
                "xx": (sx.T @ (sx * weights[:, None])) / sw,
                "xy": (sx.T @ (sy * weights[:, None])) / sw}
    return {"y0": ymean, "xy0": xy, "selected": selected}


def combined(moments, expert, beta):
    if beta == 0:
        mean_x, mean_y, cross, eig = moments["mean"], expert["y0"], expert["xy0"], moments["eig"]
    else:
        selected = expert["selected"]
        if selected is None:
            return None
        mean_x = (1 - beta) * moments["mean"] + beta * selected["x"]
        mean_y = (1 - beta) * expert["y0"] + beta * selected["y"]
        second = (1 - beta) * moments["second"] + beta * selected["xx"]
        cross = (1 - beta) * expert["xy0"] + beta * selected["xy"]
        eig = symmetric_eigen(second - np.outer(mean_x, mean_x))
    values, vectors, rank = eig
    rhs = vectors.T @ (cross - np.outer(mean_x, mean_y))
    return {"xmean": mean_x, "ymean": mean_y, "values": values, "vectors": vectors,
            "rank": rank, "rhs": rhs}


def solve_model(combination, ridge):
    c = combination["vectors"] @ (combination["rhs"] / (combination["values"][:, None] + ridge))
    coefficient = c.T.astype(np.float32)
    qw, sw = old.quantize_rows(coefficient)
    # TRAIN weighted moments of the QDQ-feature / dequantized-W8 linear map.
    # This avoids N*H*width predictor evaluation for every candidate. Canonical
    # predictor roundoff is audited on TRAIN probes for each selected payload.
    dequantized = qw.astype(np.float64) * sw[:, None].astype(np.float64)
    bias = (combination["ymean"] - combination["xmean"] @ dequantized.T).astype(np.float32)
    return {"qweight": qw, "wscale": sw, "bias": bias}


def moment_rounding_audit(encoded_train, model):
    q, sx = encoded_train["q"], encoded_train["scale"]
    rows = np.asarray(sorted({0, len(q) // 3, len(q) // 2, len(q) - 1}))
    sub = {k: v[rows] for k, v in encoded_train.items()}
    actual = base.predict_encoded(sub, model)
    wd = model["qweight"].astype(np.float64) * model["wscale"][:, None].astype(np.float64)
    ideal = sub["qdq"].astype(np.float64) @ wd.T + model["bias"].astype(np.float64)
    diff = actual.astype(np.float64) - ideal
    return {"TRAIN_probe_rows": rows.tolist(), "all_output_elements": int(diff.size),
            "max_abs_canonical_vs_dequantized_linear_roundoff": float(np.max(np.abs(diff))),
            "relative_L2_roundoff": float(np.linalg.norm(diff) / max(np.linalg.norm(ideal), 1e-300)),
            "calibration": "TRAIN weighted moments of QDQ-feature/dequantized-W8 map; canonical rounding not exactly linear"}


def write_model(prefix, model):
    base.save_model(prefix, model)
    return {str(Path(str(prefix) + suffix)): old.sha(str(prefix) + suffix)
            for suffix in (".i8", ".scales.f32", ".bias.f32", ".npz")}


def registry_line(layer, model):
    return f"{layer}\t{model['expert']}\t{model['basis']}\t{model['width']}\t{model['transform_prefix']}\t{model['readout_prefix']}"


def load_deployed_model(entry):
    if "_payload" in entry:
        return entry["_payload"]
    prefix = entry["readout_prefix"]
    return {"qweight": np.fromfile(prefix + ".i8", np.int8).reshape(-1, entry["width"]),
            "wscale": np.fromfile(prefix + ".scales.f32", np.float32),
            "bias": np.fromfile(prefix + ".bias.f32", np.float32)}


def cached_encoded(dataset, entry, cache):
    key = entry["transform_prefix"]
    if key not in cache:
        transform = load_transform(key)
        cache[key] = encoded(dataset, transform, entry["prototype"])
        # Aggregation/prediction needs only q and scale. QDQ can be reconstructed
        # explicitly later for a TRAIN joint correction, without persisting it.
        del cache[key]["qdq"]
    return cache[key]


def aggregate_family(dataset, models, cache=None, return_arrays=False):
    """True F32 slot-order sum, preserving every native fallback slot."""
    cache = {} if cache is None else cache
    cap = dataset.capture
    ids, alpha = cap.get("selected_experts"), cap.get("routing_weights")
    native_slots = cap.get("routed_outputs")
    n, k = ids.shape
    h = native_slots.shape[2]
    native, approx = np.zeros((n, h), np.float32), np.zeros((n, h), np.float32)
    predictions = {}
    for entry in models:
        expert = entry["expert"]
        rows = np.flatnonzero(np.any(ids == expert, axis=1))
        if not len(rows):
            continue
        all_encoded = cached_encoded(dataset, entry, cache)
        xx = {key: value[rows] for key, value in all_encoded.items()}
        predictions[expert] = (rows, base.predict_encoded(xx, load_deployed_model(entry)))
    for slot in range(k):
        actual = np.asarray(native_slots[:, slot], np.float32)
        value = actual.copy()
        for expert, (rows, pred) in predictions.items():
            selected = np.flatnonzero(ids[:, slot] == expert)
            if len(selected):
                value[selected] = pred[np.searchsorted(rows, selected)]
        native += np.multiply(actual, alpha[:, slot, None], dtype=np.float32)
        approx += np.multiply(value, alpha[:, slot, None], dtype=np.float32)
    shared = cap.get("shared_output") if cap.has("shared_output") else np.zeros_like(native)
    eligible = np.isin(ids, [m["expert"] for m in models])
    need = np.zeros((n, len(dataset.prototype_ids)), np.bool_)
    for entry in models:
        if entry["prototype"] is not None:
            j = dataset.prototype_ids.index(entry["prototype"])
            need[:, j] |= np.any(ids == entry["expert"], axis=1)
    native_owner_selected = np.stack([np.any(ids == owner, axis=1) for owner in dataset.prototype_ids], axis=1)
    report = {"routed": old.metrics(native, approx),
              "shared_plus_routed": old.metrics(native + shared, approx + shared),
              "eligible_slots": int(eligible.sum()), "total_slots": int(ids.size),
              "slot_coverage": float(eligible.mean()),
              "eligible_alpha_mass_fraction": float(np.sum(np.abs(alpha) * eligible) / max(np.sum(np.abs(alpha)), 1e-300)),
              "mean_required_prototypes_per_token": float(need.sum(axis=1).mean()),
              "mean_extra_prototypes_beyond_native_hot_GU_hits": float((need & ~native_owner_selected).sum(axis=1).mean()),
              "cross_expert_slots": int(sum(np.sum(ids == m["expert"]) for m in models if m["prototype"] is not None)),
              "contract": "slot-order F32(alpha*expert-output) sums; every unregistered expert remains native"}
    return (report, native, approx) if return_arrays else report


def fit(args):
    started = time.monotonic()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("formal fitting output exists")
    out.mkdir(parents=True)
    train = Dataset(args.train, args.train_target_manifest, args.layer)
    valid = Dataset(args.valid, args.valid_target_manifest, args.layer)
    audit = base.capture_audit(train.capture, valid.capture)
    ranking_path = Path(args.ranking).resolve()
    ranking = json.loads(ranking_path.read_text())
    if ranking["train_identity"] != train.identity() or ranking["layer"] != args.layer:
        raise ValueError("ranking/fit TRAIN identity mismatch")
    if train.prototype_ids != valid.prototype_ids:
        raise ValueError("prototype pools differ across TRAIN/VALID")
    experts = list(args.experts if args.experts is not None else train.ids)
    basis_jobs = [("raw_x_eq", None, experts, definition(train))]
    by_owner = {r["prototype"]: r for r in ranking["bases"]}
    for owner in train.prototype_ids:
        assigned = [e for e in experts if owner in ranking["choices"][str(e)]]
        if any(owner == e for e in assigned):
            raise ValueError("owner self entered cross-expert assignments")
        if assigned:
            d = load_transform(by_owner[owner]["transform_prefix"])
            basis_jobs.append((f"prototype_{owner}_eq", owner, assigned, d))
    report = {"status": "formal_TRAIN_VALID_no_TEST", "layer": args.layer,
              "capture_audit": audit, "train_identity": train.identity(), "valid_identity": valid.identity(),
              "ranking_path": str(ranking_path), "ranking_sha256": old.sha(ranking_path),
              "prototype_ids": train.prototype_ids, "experts": experts,
              "lambdas": args.lambdas, "betas": args.betas,
              "source_sha256": old.sha(__file__), "base_source_sha256": old.sha(BASE_PATH),
              "contract": base.CONTRACT,
              "bias_calibration": "weighted TRAIN moments of QDQ/dequantized-W8 linear map; canonical roundoff probes retained",
              "candidate_VALID_scope": "only naturally selected rows; alpha²=0 elsewhere, exactly same selection objective",
              "models": [], "grid": [], "missing_VALID_experts": []}
    old.dump(out / "protocol.json", report)
    for basis_id, owner, assigned, transform in basis_jobs:
        tp = out / f"transform.{basis_id}"
        save_transform(tp, transform)
        tx, vx = encoded(train, transform, owner), encoded(valid, transform, owner)
        moments = all_moments(tx["qdq"])
        width = tx["qdq"].shape[1]
        for expert in assigned:
            a2 = base.alpha_squared(train.capture, expert)
            va2 = base.alpha_squared(valid.capture, expert)
            selected = np.flatnonzero(va2 > 0)
            if not len(selected):
                report["missing_VALID_experts"].append({"expert": expert, "basis": basis_id})
                continue
            yy = train.targets.get(expert)
            vy = valid.targets.get(expert)
            em = expert_moments(moments, yy, a2)
            selected_vx = {key: val[selected] for key, val in vx.items()}
            selected_y = np.asarray(vy[selected])
            selected_weights = va2[selected]
            target_energy = float(np.sum(selected_weights[:, None] * selected_y.astype(np.float64) ** 2))
            train_events = int(np.sum(train.capture.get("selected_experts") == expert))
            valid_events = int(np.sum(valid.capture.get("selected_experts") == expert))
            best_score, best = float("inf"), None
            for beta in args.betas:
                combination = combined(moments, em, beta)
                if combination is None:
                    continue
                for ridge in args.lambdas:
                    model = solve_model(combination, ridge)
                    pred = base.predict_encoded(selected_vx, model)
                    score = base.weighted_nmse(selected_y, pred, va2[selected])
                    record = {"expert": expert, "basis": basis_id, "prototype": owner,
                              "beta": beta, "lambda": ridge, "selected_VALID_rows": len(selected),
                              "selected_alpha2_nmse": score,
                              "selected_VALID_weighted_target_energy": target_energy,
                              "weighted_squared_error": score * target_energy,
                              "VALID_selected_alpha2_sum": float(selected_weights.sum()),
                              "TRAIN_selected_event_count": train_events,
                              "VALID_selected_event_count": valid_events,
                              "selected_row_nmse": base.weighted_nmse(selected_y, pred, np.ones(len(pred))),
                              "Gram_rank": combination["rank"],
                              "effective_df": float(np.sum(combination["values"] / (combination["values"] + ridge)))}
                    report["grid"].append(record)
                    if score < best_score:
                        best_score, best = score, (record, {k: v.copy() for k, v in model.items()})
            if best is None:
                continue
            record, model = best
            pred_all = base.predict_encoded(vx, model)
            record["all_VALID_row_nmse"] = base.weighted_nmse(vy, pred_all, np.ones(len(pred_all)))
            prefix = out / f"{basis_id}.expert{expert}"
            files = write_model(prefix, model)
            entry = {**record, "width": width, "transform_prefix": str(tp), "readout_prefix": str(prefix),
                     "payload_sha256": files, "TRAIN_rounding_audit": moment_rounding_audit(tx, model),
                     "cost": base.model_cost(width, hidden=train.meta["hidden"]),
                     "cross_expert_reuse": owner is not None, "owner_self": False}
            report["models"].append(entry)
            print(json.dumps({"fit": basis_id, "expert": expert, "beta": entry["beta"],
                              "lambda": entry["lambda"], "VALID_alpha2_NMSE": best_score}), flush=True)
        del moments, tx, vx
        # Durable family progress is an audit, never a TEST selection.
        old.dump(out / "progress.json", {"completed_basis": basis_id, "models": report["models"]})
    families = {"raw_x_eq": [], "best_cross_prototype": [], "proto_or_raw": []}
    for expert in experts:
        raw = next((m for m in report["models"] if m["expert"] == expert and m["prototype"] is None), None)
        candidates = [m for m in report["models"] if m["expert"] == expert and m["prototype"] is not None]
        proto = min(candidates, key=lambda m: (m["selected_alpha2_nmse"], m["prototype"])) if candidates else None
        if raw:
            families["raw_x_eq"].append(raw)
        if proto:
            families["best_cross_prototype"].append(proto)
        choices = [m for m in (raw, proto) if m is not None]
        if choices:
            families["proto_or_raw"].append(min(choices, key=lambda m: (m["selected_alpha2_nmse"], m["width"])))
    report["families"] = families
    report["self_controls_VALID"] = native_self_controls(valid)
    report["aggregate_VALID"] = {}
    aggregate_cache = {}
    for name, records in families.items():
        for suffix, subset in (("", records), (".excluding_owners", [m for m in records if m["expert"] not in train.prototype_ids])):
            (out / f"registry.{name}{suffix}.tsv").write_text("\n".join(registry_line(args.layer, m) for m in subset) + "\n")
            if valid.capture.has("routed_outputs"):
                report["aggregate_VALID"][name + suffix] = aggregate_family(valid, subset, aggregate_cache)
    report["cost_notes"] = {
        "host_GU_per_prototype_W8_scales_bytes": 2 * train.meta["hidden"] * 1408 + 8 * 1408,
        "number_resident_prototypes": len(train.prototype_ids),
        "GU_MAC_per_computed_prototype_row": 2 * train.meta["hidden"] * 1408,
        "prototype_execution": "only required prototype/input rows; de-duplicate within token, cost never free",
        "required_strong_control": "same hot GU cache + exact native owners + enhanced raw_x for remaining experts",
        "native_owner_hits_are_cross_reuse": False,
        "latency_or_speedup_claim": False}
    report["wall_seconds"] = time.monotonic() - started
    old.dump(out / "selection.json", report)
    print(json.dumps({"completed": str(out / "selection.json"), "seconds": report["wall_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ranking = sub.add_parser("rank")
    fitting = sub.add_parser("fit")
    for p in (ranking, fitting):
        p.add_argument("--train", required=True)
        p.add_argument("--train-target-manifest", required=True)
        p.add_argument("--layer", type=int, required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--experts", nargs="+", type=int)
        p.add_argument("--threads", type=int, default=8)
    ranking.add_argument("--rank-fit-rows", type=int, default=4096)
    ranking.add_argument("--rank-score-rows", type=int, default=1024)
    ranking.add_argument("--sketch-width", type=int, default=128)
    ranking.add_argument("--top-prototypes", type=int, default=2)
    ranking.add_argument("--rank-lambda", type=float, default=3)
    ranking.add_argument("--seed", type=int, default=20261109)
    fitting.add_argument("--valid", required=True)
    fitting.add_argument("--valid-target-manifest", required=True)
    fitting.add_argument("--ranking", required=True)
    fitting.add_argument("--lambdas", nargs="+", type=float, default=[1, 3, 10])
    fitting.add_argument("--betas", nargs="+", type=float, default=[0, .99])
    args = parser.parse_args()
    if args.command == "fit" and (any(x <= 0 for x in args.lambdas) or any(not 0 <= x < 1 for x in args.betas)):
        raise ValueError("positive ridges and beta in [0,1) required")
    with threadpool_limits(limits=args.threads, user_api="blas"):
        (rank if args.command == "rank" else fit)(args)


if __name__ == "__main__":
    main()
