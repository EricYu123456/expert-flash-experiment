#!/usr/bin/env python3
"""CPU TRAIN/VALID screening of equalized, QDQ-aware, routed-weighted readouts.

No TEST argument exists. The legacy captures/targets are read-only. Affine
equalization remains at runtime before whole-vector A8 and is NOT folded into
C. All C models fit reconstructed A8 inputs and recalibrate bias on TRAIN after
row-W8. Deployment is six-column TSV plus explicitly typed binary payloads.
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


LEGACY_PATH = Path(__file__).with_name("research_20261108_shared_feature_fit.py")
spec = importlib.util.spec_from_file_location("shared_fit_1108_readonly", LEGACY_PATH)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

BASES = ("raw_x_eq", "shared_z_eq", "shared_z_select1408", "hybrid_x_z512", "hybrid_x_z768")
CONTRACT = {
    "feature": "gather x/shared_z; F32(F32(raw-center)*inv_scale); whole-vector A8; source shared Down unchanged",
    "fit": "weighted ridge on actual F32 QDQ(A8(transformed features)); TRAIN-only transformations/subsets/bias",
    "readout": old.NUMERIC_CONTRACT,
    "weight": "w=(1-beta)+beta*selected_alpha_squared/mean_TRAIN(selected_alpha_squared)",
    "selection": "minimum VALID selected-alpha-squared-weighted NMSE; all-row scores also retained",
    "feature_source_codes": {"0": "raw_x", "1": "native_shared_z"},
    "TEST": "not accepted or opened by this program",
}


class Targets:
    """Per-expert contiguous target files, or the original small capture array."""
    def __init__(self, cap, manifest=None, layer=None):
        self.cap = cap
        self.path = Path(manifest).resolve() if manifest else None
        if self.path:
            meta = json.loads(self.path.read_text())
            self.meta = meta["layers"][str(layer)] if "layers" in meta else meta
            self.ids = list(map(int, self.meta["expert_ids"]))
        else:
            self.meta = None
            self.ids = cap.expert_ids

    def get(self, expert):
        if not self.path:
            return self.cap.get("expert_outputs")[:, self.ids.index(expert)]
        rec = self.meta["targets"][str(expert)]
        path = Path(rec["path"])
        if not path.is_absolute():
            path = self.path.parent / path
        if path.suffix == ".npy":
            return np.load(path, mmap_mode="r", allow_pickle=False)
        shape, dtype = tuple(rec["shape"]), np.dtype(rec.get("dtype", "float32"))
        if path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"target size mismatch: {path}")
        return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def alpha_squared(cap, expert):
    ids = np.asarray(cap.get("selected_experts"))
    alpha = np.asarray(cap.get("routing_weights"), np.float64)
    if ids.shape != alpha.shape:
        raise ValueError("router IDs/alpha shape mismatch")
    return np.sum(np.where(ids == expert, alpha ** 2, 0), axis=1)


def blend_weights(a2, beta):
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0,1), preserving positive all-row weight")
    mean = np.mean(a2)
    if mean == 0:
        if beta:
            raise ValueError("cannot route-weight an unobserved TRAIN expert")
        return np.ones(len(a2), np.float64)
    return (1 - beta) + beta * a2 / mean


def weighted_nmse(y, pred, weights):
    weights = np.asarray(weights, np.float64)
    if not np.any(weights > 0):
        return None
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    err = np.sum((y - pred) ** 2, axis=1)
    energy = np.sum(y ** 2, axis=1)
    return float(weights @ err / max(weights @ energy, 1e-300))


def validation_metrics(y, pred, a2):
    selected = a2 > 0
    return {"all_row_nmse": weighted_nmse(y, pred, np.ones(len(y))),
            "selected_alpha2_nmse": weighted_nmse(y, pred, a2),
            "selected_row_nmse": weighted_nmse(y, pred, selected.astype(np.float64)),
            "selected_rows": int(selected.sum()), "rows": len(y)}


def mean_std(a):
    a = np.asarray(a, np.float64)
    mean, std = a.mean(axis=0), a.std(axis=0)
    floor = max(float(np.median(std)) * 1e-6, 1e-12)
    return mean.astype(np.float32), (1 / np.maximum(std, floor)).astype(np.float32)


def transform(raw, center, inv_scale):
    raw = np.asarray(raw, np.float32)
    return old.finite(np.multiply(np.subtract(raw, center, dtype=np.float32), inv_scale, dtype=np.float32))


def encode(raw):
    q, scale = old.quantize_rows(raw)
    reconstructed = np.multiply(q.astype(np.float32), scale[:, None], dtype=np.float32)
    return {"q": q, "scale": scale, "qdq": reconstructed}


def prepare_gram(x, weights):
    x = np.asarray(x, np.float64)
    weights = np.asarray(weights, np.float64)
    sw = weights.sum()
    mean = (weights @ x) / sw
    xc = x - mean
    gram = (xc.T @ (xc * weights[:, None])) / sw
    vals, vecs = eigh(gram, check_finite=False, driver="evd")
    tolerance = max(float(vals[-1]), 1) * 1e-10
    if vals[0] < -tolerance:
        raise ValueError("negative Gram eigenvalue beyond roundoff")
    vals = np.maximum(vals, 0)
    return {"xc": xc, "mean": mean, "weights": weights, "weight_sum": sw,
            "values": vals, "vectors": vecs,
            "numerical_rank": int(np.sum(vals > tolerance)),
            "effective_rows": float(sw ** 2 / np.sum(weights ** 2))}


def solve_components(gram, y):
    y = np.asarray(y, np.float64)
    w, sw = gram["weights"], gram["weight_sum"]
    mean = (w @ y) / sw
    rhs = gram["vectors"].T @ ((gram["xc"].T @ ((y - mean) * w[:, None])) / sw)
    return mean, rhs


def solve_one(gram, target_mean, rhs, ridge):
    c = gram["vectors"] @ (rhs / (gram["values"][:, None] + ridge))
    coefficient = c.T.astype(np.float32)
    bias = (target_mean - gram["mean"] @ c).astype(np.float32)
    qw, sw = old.quantize_rows(coefficient)
    return {"coefficient": coefficient, "bias": bias, "qweight": qw, "wscale": sw}


def predict_encoded(encoded, model, use_bias=True, batch=512):
    q, sx = encoded["q"], encoded["scale"]
    out = np.empty((len(q), len(model["wscale"])), np.float32)
    for start in range(0, len(q), batch):
        d = old.exact_dots(q[start:start + batch], model["qweight"])
        p = old.rescale(d, model["wscale"], sx[start:start + batch])
        if use_bias:
            p = np.add(p, model["bias"], dtype=np.float32)
        out[start:start + len(p)] = p
    return old.finite(out)


def recalibrate_bias(encoded_train, model, target, weights):
    before_bias = predict_encoded(encoded_train, model, use_bias=False)
    residual = np.asarray(target, np.float64) - before_bias.astype(np.float64)
    model["bias"] = ((weights @ residual) / weights.sum()).astype(np.float32)


class FeatureFactory:
    def __init__(self, train, valid):
        self.train, self.valid = train, valid
        self.xmean, self.xgain = mean_std(train.get("x"))
        self.zmean, self.zgain = mean_std(train.get("shared_z"))
        self.raw_x_train = transform(train.get("x"), self.xmean, self.xgain)
        self.raw_z_train = transform(train.get("shared_z"), self.zmean, self.zgain)
        self.x_encoded_train = encode(self.raw_x_train)
        self.target_order = np.arange(len(self.zmean))
        self.residual_order = np.arange(len(self.zmean))
        self.scores = {}

    def select_features_train_only(self, targets, experts):
        """Common TRAIN-only feature indices, no VALID model choice in selector."""
        z = self.raw_z_train.astype(np.float64)
        target_score, residual_score = np.zeros(z.shape[1]), np.zeros(z.shape[1])
        x = self.x_encoded_train["qdq"]
        for expert in experts:
            y = np.asarray(targets.get(expert), np.float64)
            a2 = alpha_squared(self.train, expert)
            weights = blend_weights(a2, .99 if np.any(a2 > 0) else 0)
            gram = prepare_gram(x, weights)
            ym, rhs = solve_components(gram, y)
            # Fixed lambda=1 selector, selected on TRAIN alone. No W8 model is
            # installed: this merely scores candidate resident z dimensions.
            m = solve_one(gram, ym, rhs, 1.0)
            residual = y - (x.astype(np.float64) @ m["coefficient"].astype(np.float64).T + m["bias"])
            zmean = weights @ z / weights.sum()
            variance = weights @ ((z - zmean) ** 2) / weights.sum()
            for response, score in ((y - ym, target_score), (residual, residual_score)):
                response = response - (weights @ response) / weights.sum()
                cross = (z.T @ (response * weights[:, None])) / weights.sum()
                energy = float(np.sum(weights[:, None] * response ** 2) / weights.sum())
                score += np.sum(cross ** 2, axis=1) / np.maximum(variance * energy, 1e-30)
            del gram
        self.target_order = np.argsort(-target_score, kind="stable")
        self.residual_order = np.argsort(-residual_score, kind="stable")
        self.scores = {"target": target_score, "residual": residual_score}

    def definition(self, basis):
        h, r = len(self.xmean), len(self.zmean)
        if basis == "raw_x_eq":
            sources, indices = np.zeros(h, np.uint8), np.arange(h, dtype=np.int32)
        elif basis == "shared_z_eq":
            sources, indices = np.ones(r, np.uint8), np.arange(r, dtype=np.int32)
        elif basis.startswith("shared_z_select"):
            width = int(basis.split("select")[1])
            sources = np.ones(width, np.uint8)
            indices = np.sort(self.target_order[:width]).astype(np.int32)
        elif basis.startswith("hybrid_x_z"):
            count = int(basis.split("_z")[1])
            sources = np.concatenate((np.zeros(h, np.uint8), np.ones(count, np.uint8)))
            indices = np.concatenate((np.arange(h, dtype=np.int32), np.sort(self.residual_order[:count]).astype(np.int32)))
        else:
            raise ValueError(basis)
        center, inv_scale = np.empty(len(sources), np.float32), np.empty(len(sources), np.float32)
        for source, means, gains in ((0, self.xmean, self.xgain), (1, self.zmean, self.zgain)):
            mask = sources == source
            center[mask], inv_scale[mask] = means[indices[mask]], gains[indices[mask]]
        if len(sources) > 2816:
            raise ValueError("readout input-width budget exceeded")
        return {"sources": sources, "indices": indices, "center": center, "inv_scale": inv_scale}

    @staticmethod
    def make(cap, definition):
        sources, indices = definition["sources"], definition["indices"]
        n = len(cap.get("x"))
        raw = np.empty((n, len(sources)), np.float32)
        for source, name in ((0, "x"), (1, "shared_z")):
            mask = sources == source
            if np.any(mask):
                raw[:, mask] = cap.get(name)[:, indices[mask]]
        return encode(transform(raw, definition["center"], definition["inv_scale"]))


def save_transform(prefix, definition):
    for key, suffix in (("sources", ".sources.u8"), ("indices", ".indices.i32"),
                        ("center", ".center.f32"), ("inv_scale", ".inv_scale.f32")):
        definition[key].tofile(str(prefix) + suffix)


def save_model(prefix, model):
    model["qweight"].tofile(str(prefix) + ".i8")
    model["wscale"].tofile(str(prefix) + ".scales.f32")
    model["bias"].tofile(str(prefix) + ".bias.f32")
    np.savez(str(prefix) + ".npz", **model)


def model_cost(width, hidden=2048, private_width=1408):
    return {"input_width": width, "C_W8_bytes": hidden * width,
            "C_row_scales_bias_bytes": 8 * hidden, "transform_bytes": 13 * width,
            "readout_MAC_per_token_expert": hidden * width,
            "transform_subtract_multiply_per_token_basis": 2 * width,
            "additional_GU_coefficients": 0, "additional_dense_projection_coefficients": 0,
            "native_GUD_coefficients": 3 * hidden * private_width,
            "raw_coefficient_reduction": 1 - width / (3 * private_width),
            "native_shared_GU_still_required": True,
            "latency_or_system_speedup_claim": False}


def capture_audit(train, valid):
    a, b = set(map(str, train.prompts())), set(map(str, valid.prompts()))
    if a & b:
        raise ValueError("TRAIN/VALID prompt overlap")
    return {"train_prompts": len(a), "valid_prompts": len(b),
            "train_rows": len(train.get("x")), "valid_rows": len(valid.get("x")),
            "train_capture_manifests": train.identity(), "valid_capture_manifests": valid.identity()}


def fit_screen(args):
    started = time.monotonic()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("refuse output overwrite")
    out.mkdir(parents=True)
    train, valid = old.Capture(args.train, args.layer), old.Capture(args.valid, args.layer)
    tt = Targets(train, args.train_target_manifest, args.layer)
    vt = Targets(valid, args.valid_target_manifest, args.layer)
    experts = list(args.experts if args.experts is not None else tt.ids)
    if set(experts) - set(tt.ids) or set(experts) - set(vt.ids):
        raise ValueError("target expert identities unavailable")
    audit = capture_audit(train, valid)
    manifest = {"status": "VALID_screening_no_TEST_access", "layer": args.layer,
                "capture_audit": audit, "experts": experts, "bases": args.bases,
                "betas": args.betas, "lambdas": args.lambdas, "contract": CONTRACT,
                "source_sha256": old.sha(__file__), "readonly_legacy_source_sha256": old.sha(LEGACY_PATH),
                "transform_selection": "pooled TRAIN target correlation for pure-z subsets; fixed-lambda TRAIN raw-x residual correlation for hybrid subsets",
                "models": [], "grid": []}
    old.dump(out / "protocol.json", manifest)
    factory = FeatureFactory(train, valid)
    factory.select_features_train_only(tt, experts)
    np.savez(out / "TRAIN_feature_selection.npz", target_order=factory.target_order,
             residual_order=factory.residual_order, **factory.scores)
    train_a2 = {e: alpha_squared(train, e) for e in experts}
    valid_a2 = {e: alpha_squared(valid, e) for e in experts}
    for basis in args.bases:
        definition = factory.definition(basis)
        width = len(definition["sources"])
        tprefix = out / f"transform.{basis}"
        save_transform(tprefix, definition)
        tx, vx = factory.make(train, definition), factory.make(valid, definition)
        best = {e: (float("inf"), None) for e in experts}
        shared_gram = None
        for beta in args.betas:
            if beta == 0:
                shared_gram = prepare_gram(tx["qdq"], np.ones(len(tx["q"])))
            for expert in experts:
                if beta and not np.any(train_a2[expert] > 0):
                    continue
                weights = blend_weights(train_a2[expert], beta)
                gram = shared_gram if beta == 0 else prepare_gram(tx["qdq"], weights)
                train_y, valid_y = tt.get(expert), vt.get(expert)
                target_mean, rhs = solve_components(gram, train_y)
                for ridge in args.lambdas:
                    model = solve_one(gram, target_mean, rhs, ridge)
                    recalibrate_bias(tx, model, train_y, weights)
                    pred = predict_encoded(vx, model)
                    scores = validation_metrics(valid_y, pred, valid_a2[expert])
                    record = {"basis": basis, "expert": expert, "beta": beta, "lambda": ridge,
                              "effective_rows": gram["effective_rows"], "Gram_rank": gram["numerical_rank"],
                              "effective_df": float(np.sum(gram["values"] / (gram["values"] + ridge))),
                              "valid": scores}
                    manifest["grid"].append(record)
                    objective = scores["selected_alpha2_nmse"]
                    if objective is None:
                        objective = scores["all_row_nmse"]
                    if objective < best[expert][0]:
                        best[expert] = (objective, (record, {k: v.copy() for k, v in model.items()}))
                if beta:
                    del gram
            if shared_gram is not None:
                del shared_gram
                shared_gram = None
        registry = []
        for expert in experts:
            _, (record, model) = best[expert]
            prefix = out / f"{basis}.expert{expert}"
            save_model(prefix, model)
            ent = {**record, "transform_prefix": str(tprefix), "readout_prefix": str(prefix),
                   "cost": model_cost(width, hidden=train.get("x").shape[1]), "payload_sha256": {str(Path(str(prefix) + suffix)): old.sha(str(prefix) + suffix)
                                                                for suffix in (".i8", ".scales.f32", ".bias.f32", ".npz")}}
            manifest["models"].append(ent)
            registry.append(f"{args.layer}\t{expert}\t{basis}\t{width}\t{tprefix}\t{prefix}")
            print(json.dumps({"basis": basis, "expert": expert, "chosen_beta": record["beta"],
                              "chosen_lambda": record["lambda"], "VALID": record["valid"]}), flush=True)
        (out / f"registry.{basis}.tsv").write_text("\n".join(registry) + "\n")
        del tx, vx
    winners, lines = [], []
    for expert in experts:
        candidates = [m for m in manifest["models"] if m["expert"] == expert]
        best = min(candidates, key=lambda m: (m["valid"]["selected_alpha2_nmse"] if m["valid"]["selected_alpha2_nmse"] is not None else float("inf"), m["cost"]["input_width"]))
        winners.append(best)
        lines.append(f"{args.layer}\t{expert}\t{best['basis']}\t{best['cost']['input_width']}\t{best['transform_prefix']}\t{best['readout_prefix']}")
    manifest["best_per_expert"] = winners
    (out / "registry.best_per_expert.tsv").write_text("\n".join(lines) + "\n")
    manifest["wall_seconds"] = time.monotonic() - started
    manifest["all_payload_sha256"] = {str(p): old.sha(p) for p in sorted(out.iterdir()) if p.suffix not in (".json",)}
    old.dump(out / "selection.json", manifest)
    print(json.dumps({"complete": str(out / "selection.json"), "wall_seconds": manifest["wall_seconds"]}), flush=True)


def numeric_self_test(path):
    rng = np.random.default_rng(20261109)
    raw = rng.normal(size=(800, 14)).astype(np.float32)
    raw[:, 0] *= 1000
    raw[:, 1] += 200
    center, gain = mean_std(raw[:600])
    full = encode(transform(raw, center, gain))
    target = (full["qdq"] @ rng.normal(size=(14, 5))).astype(np.float32)
    weights = np.ones(600)
    gram = prepare_gram(full["qdq"][:600], weights)
    mean, rhs = solve_components(gram, target[:600])
    model = solve_one(gram, mean, rhs, 1e-8)
    train = {k: v[:600] for k, v in full.items()}
    valid = {k: v[600:] for k, v in full.items()}
    recalibrate_bias(train, model, target[:600], weights)
    score = old.metrics(target[600:], predict_encoded(valid, model))
    if score["nmse"] > .001:
        raise AssertionError("QDQ linear teacher recovery")
    # Independent scalar contract for feature subtraction then multiplication.
    scalar = np.empty_like(raw)
    for i in range(4):
        for j in range(raw.shape[1]):
            scalar[i, j] = np.float32(np.float32(raw[i, j] - center[j]) * gain[j])
    if not np.array_equal(scalar[:4].view(np.uint32), transform(raw[:4], center, gain).view(np.uint32)):
        raise AssertionError("transform operation order")
    a2 = np.asarray([0, .25, 0, 1.0])
    blend = blend_weights(a2, .99)
    assert np.all(blend > 0) and abs(blend.mean() - 1) < 1e-12
    old.dump(path, {"status": "PASS", "CPU_only": True, "transform_F32_order": True,
                    "QDQ_teacher_heldout": score, "positive_weight_floor_mean_one": True,
                    "no_TEST_captures": True, "source_sha256": old.sha(__file__)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("screen")
    for name in ("train", "valid", "output"):
        fit.add_argument("--" + name, required=True)
    fit.add_argument("--layer", type=int, required=True)
    fit.add_argument("--train-target-manifest")
    fit.add_argument("--valid-target-manifest")
    fit.add_argument("--experts", type=int, nargs="+")
    fit.add_argument("--bases", nargs="+", default=list(BASES))
    fit.add_argument("--betas", type=float, nargs="+", default=[0, .99])
    fit.add_argument("--lambdas", type=float, nargs="+", default=[.03, .3, 3])
    fit.add_argument("--threads", type=int, default=8)
    numeric = sub.add_parser("self-test")
    numeric.add_argument("--output", required=True)
    numeric.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads, user_api="blas"):
        if args.command == "screen":
            if any(lam <= 0 for lam in args.lambdas):
                raise ValueError("positive ridge required")
            fit_screen(args)
        else:
            numeric_self_test(args.output)


if __name__ == "__main__":
    main()
