#!/usr/bin/env python3
"""CPU-only, prompt-disjoint native shared-feature readout diagnosis.

TRAIN/VALID/TEST are separate capture manifests. Each manifest has `arrays`
whose entries contain path/dtype/shape for raw row-major arrays (or .npy), and
`expert_ids`. Required: x, shared_z, shared_output, expert_outputs, prompt_ids.
Optional: selected_experts, routing_weights, routed_sum. shared_z is the native
shared Down *input*, never its output. TEST is evaluated only after all VALID
choices and model payloads have been written and hashed to selection.json.

The nonlinear control is a seeded random SwiGLU of identical width, with two
H x r matrices. Its additional resident weights and MACs are charged explicitly.
This tool reports approximation quality and coefficient costs, never speedup.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from threadpoolctl import threadpool_limits


BASES = ("x", "shared_output", "x_plus_shared_output", "shared_z", "random_swiglu")
LAMBDAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
NUMERIC_CONTRACT = (
    "row-W8 and whole-vector A8: F32 amax/127, zero scale=1; F32 quotient "
    "then round ties away from zero to [-127,127]; exact integer dot; "
    "F32(dot) * F32(weight_scale * input_scale); separate F32 bias addition. "
    "FP diagnostic: stored F32 coefficient, FP64 dot, cast F32 then F32 bias; "
    "not the C++ FMA reduction contract."
)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(a):
    if not np.isfinite(a).all():
        raise ValueError("nonfinite input or coefficient")
    return a


def quantize_rows(a):
    """Match std::round(F32(x / F32(amax/127))), including near halves."""
    a = finite(np.asarray(a, dtype=np.float32))
    if a.ndim != 2 or not a.shape[1]:
        raise ValueError("quantization requires nonempty row vectors")
    amax = np.max(np.abs(a), axis=1)
    scale = np.where(amax == 0, np.float32(1), amax / np.float32(127)).astype(np.float32)
    if not np.all((scale > 0) & np.isfinite(scale)):
        raise ValueError("invalid/subnormal quantization scale")
    ratio = np.divide(a, scale[:, None], dtype=np.float32).astype(np.float64)
    # The +0.5 must be FP64: F32 addition rounds just-below-half up incorrectly.
    q = np.clip(np.copysign(np.floor(np.abs(ratio) + 0.5), ratio), -127, 127).astype(np.int8)
    return q, scale


def exact_dots(q, w):
    if q.dtype != np.int8 or w.dtype != np.int8 or q.shape[1] != w.shape[1]:
        raise ValueError("integer dot shapes/dtypes")
    if q.shape[1] * 128 ** 2 >= 2 ** 53:
        raise ValueError("FP64 integer-exact accumulation bound exceeded")
    # All products and any signed partial sum are exact integers in FP64.
    d = q.astype(np.float64) @ w.astype(np.float64).T
    if len(q) and len(w):
        qi = sorted({0, len(q) // 2, len(q) - 1})
        wi = sorted({0, len(w) // 2, len(w) - 1})
        ref = q[qi].astype(np.int64) @ w[wi].astype(np.int64).T
        if not np.array_equal(d[np.ix_(qi, wi)], ref):
            raise AssertionError("integer dot audit")
    return d


def rescale(d, sw, sx):
    scale = np.multiply(sx[:, None], sw[None, :], dtype=np.float32)
    return np.multiply(d.astype(np.float32), scale, dtype=np.float32)


def predict(x, model, precision, batch=512):
    out = np.empty((len(x), len(model["bias"])), np.float32)
    for a in range(0, len(x), batch):
        xx = np.asarray(x[a:a + batch], np.float32)
        if precision == "fp":
            p = (xx.astype(np.float64) @ model["coefficient"].astype(np.float64).T).astype(np.float32)
        elif precision == "w8a8":
            q, sx = quantize_rows(xx)
            p = rescale(exact_dots(q, model["qweight"]), model["wscale"], sx)
        else:
            raise ValueError(precision)
        out[a:a + len(xx)] = np.add(p, model["bias"], dtype=np.float32)
    return finite(out)


def metrics(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    err = np.sum((y - p) ** 2, axis=1)
    energy = np.sum(y ** 2, axis=1)
    penergy = np.sum(p ** 2, axis=1)
    denom = np.sqrt(energy * penergy)
    ok = denom > 0
    cosine = np.zeros(len(y))
    cosine[ok] = np.sum(y[ok] * p[ok], axis=1) / denom[ok]
    cosine[(energy == 0) & (penergy == 0)] = 1.0
    return {
        "rows": len(y), "nmse": float(err.sum() / max(energy.sum(), 1e-300)),
        "mean_cosine": float(cosine.mean()) if len(y) else None,
        "row_nmse_median": float(np.median(err / np.maximum(energy, 1e-300))) if len(y) else None,
        "row_nmse_p95": float(np.percentile(err / np.maximum(energy, 1e-300), 95)) if len(y) else None,
        "target_energy": float(energy.sum()), "squared_error": float(err.sum()),
    }


class Capture:
    def __init__(self, path, layer=None):
        self.path = Path(path).resolve()
        self.cache = {}
        self.children = []
        self.layer = layer
        if self.path.is_dir():
            manifests = sorted(self.path.glob("shard*/capture_manifest.json"))
            if len(manifests) != 4:
                raise ValueError(f"expected four completed shard capture manifests in {self.path}; got {len(manifests)}")
            self.children = [Capture(p, layer) for p in manifests]
            self.expert_ids = self.children[0].expert_ids
            if any(c.expert_ids != self.expert_ids for c in self.children):
                raise ValueError("expert IDs differ across shards")
            self.meta = {"expert_ids": self.expert_ids}
            self.records = self.children[0].records
            return
        self.meta = json.loads(self.path.read_text())
        if "layers" in self.meta:
            if layer is None:
                raise ValueError("multilayer manifest requires --layer")
            self.meta = self.meta["layers"][str(layer)]
        self.records = self.meta.get("arrays", self.meta.get("files", {}))
        self.expert_ids = [int(e) for e in self.meta["expert_ids"]]

    def identity(self):
        if self.children:
            return {str(c.path): sha(c.path) for c in self.children}
        return {str(self.path): sha(self.path)}

    def payload_identity(self):
        if self.children:
            return {p: h for c in self.children for p, h in c.payload_identity().items()}
        paths = set()
        for rec in self.records.values():
            if isinstance(rec, list):
                continue
            path = Path(rec if isinstance(rec, str) else rec["path"])
            if not path.is_absolute():
                path = self.path.parent / path
            paths.add(path.resolve())
            metadata = path.parent / "rows.jsonl"
            if metadata.exists():
                paths.add(metadata.resolve())
        return {str(path): sha(path) for path in sorted(paths)}

    def has(self, name):
        if name == "routed_sum" and "routed_outputs" in self.records:
            return True
        return name in self.records or name in self.meta

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        if self.children:
            out = np.concatenate([c.get(name) for c in self.children], axis=0)
            self.cache[name] = out
            return out
        if name == "routed_sum" and name not in self.records and "routed_outputs" in self.records:
            y, alpha = self.get("routed_outputs"), self.get("routing_weights")
            out = np.zeros((len(y), y.shape[2]), np.float32)
            for slot in range(y.shape[1]):
                out += np.multiply(y[:, slot], alpha[:, slot, None], dtype=np.float32)
            self.cache[name] = out
            return out
        rec = self.records.get(name, self.meta.get(name))
        if rec is None:
            raise ValueError(f"{self.path}: missing {name}")
        if isinstance(rec, list):
            out = np.asarray(rec)
        else:
            path = Path(rec if isinstance(rec, str) else rec["path"])
            if not path.is_absolute():
                path = self.path.parent / path
            if path.suffix == ".npy":
                out = np.load(path, mmap_mode="r", allow_pickle=False)
            else:
                dtype = np.dtype(rec.get("dtype", "float32"))
                shape = tuple(rec["shape"])
                if path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
                    raise ValueError(f"capture size mismatch: {path}")
                out = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        self.cache[name] = out
        return out

    def prompts(self):
        if self.children:
            return np.concatenate([c.prompts() for c in self.children])
        # The canonical capture uses worker-local integer indices in its array,
        # but globally unique corpus prompt IDs in this companion JSONL.
        if "x" in self.records:
            rec = self.records["x"]
            xp = Path(rec if isinstance(rec, str) else rec["path"])
            if not xp.is_absolute():
                xp = self.path.parent / xp
            metadata = xp.parent / "rows.jsonl"
            if metadata.exists():
                return np.asarray([json.loads(line)["prompt_id"] for line in metadata.read_text().splitlines()])
        if self.has("prompt_ids"):
            return self.get("prompt_ids")
        if "rows" in self.meta:
            return np.asarray([row["prompt_id"] for row in self.meta["rows"]])
        raise ValueError("capture needs per-row prompt_ids")


def split_audit(captures, minimum_test_prompts):
    sets, report = {}, {}
    for name, cap in captures.items():
        ids = cap.prompts()
        sets[name] = set(map(str, ids))
        x = cap.get("x")
        if len(ids) != len(x):
            raise ValueError("prompt ID / row count mismatch")
        if cap.get("shared_z").shape[0] != len(x):
            raise ValueError("shared_z / x row mismatch")
        report[name] = {"prompts": len(sets[name]), "rows": len(x),
                        "manifest_sha256": cap.identity()}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if a not in sets or b not in sets:
            continue
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"prompt overlap {a}/{b}: {list(overlap)[:3]}")
    if "test" in sets and len(sets["test"]) < minimum_test_prompts:
        raise ValueError(f"TEST has {len(sets['test'])} prompts, requires {minimum_test_prompts}")
    if any(c.expert_ids != captures["train"].expert_ids for c in captures.values()):
        raise ValueError("expert identities differ across splits")
    return report


class Features:
    def __init__(self, train, width, seed):
        self.width = width
        x = np.asarray(train.get("x"), np.float64)
        self.xmean = x.mean(axis=0)
        self.xstd = np.maximum(x.std(axis=0), 1e-8)
        rng = np.random.default_rng(seed)
        gate = rng.standard_normal((x.shape[1], width)) / np.sqrt(x.shape[1])
        up = rng.standard_normal((x.shape[1], width)) / np.sqrt(x.shape[1])
        # Fold TRAIN-only normalization into row projections and biases, then
        # apply the same row-W8/whole-input-A8 contract as native GU.
        gate, up = gate / self.xstd[:, None], up / self.xstd[:, None]
        self.gate, self.up = gate.T.astype(np.float32), up.T.astype(np.float32)
        self.gate_bias = (-self.xmean @ gate).astype(np.float32)
        self.up_bias = (-self.xmean @ up).astype(np.float32)
        self.qgate, self.sgate = quantize_rows(self.gate)
        self.qup, self.sup = quantize_rows(self.up)

    def make(self, cap, basis):
        if basis == "x_plus_shared_output":
            return finite(np.concatenate((cap.get("x"), cap.get("shared_output")), axis=1).astype(np.float32))
        if basis != "random_swiglu":
            return finite(np.asarray(cap.get(basis), np.float32))
        qx, sx = quantize_rows(cap.get("x"))
        gate = np.add(rescale(exact_dots(qx, self.qgate), self.sgate, sx), self.gate_bias, dtype=np.float32)
        up = np.add(rescale(exact_dots(qx, self.qup), self.sup, sx), self.up_bias, dtype=np.float32)
        return finite((gate * expit(gate) * up).astype(np.float32))

    def save(self, path):
        np.savez(path, xmean=self.xmean, xstd=self.xstd, gate=self.gate, up=self.up,
                 qgate=self.qgate, qup=self.qup, sgate=self.sgate, sup=self.sup,
                 gate_bias=self.gate_bias, up_bias=self.up_bias)

    @classmethod
    def load(cls, path):
        out = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                setattr(out, key, data[key])
        out.width = len(out.qgate)
        return out


def decompose(train_x, allow_underdetermined=False):
    t = time.monotonic()
    x = np.asarray(train_x, np.float64)
    n, p = x.shape
    if n <= p and not allow_underdetermined:
        raise ValueError(f"TRAIN {n} rows <= {p} features; add TRAIN data or explicitly label diagnostic")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    constants = std < 1e-12
    std[constants] = 1.0
    x = (x - mean) / std
    gram = (x.T @ x) / n
    vals, vecs = eigh(gram, check_finite=False, driver="evd")
    tol = max(float(vals[-1]), 1.0) * 1e-10
    if vals[0] < -tol:
        raise ValueError("Gram has negative eigenvalues beyond roundoff")
    vals = np.maximum(vals, 0)
    report = {"train_rows": n, "features": p, "rows_per_feature": n / p,
              "underdetermined_by_row_count": n <= p,
              "numerical_rank_rel1e10": int(np.sum(vals > tol)),
              "constant_features": int(constants.sum()),
              "largest_eigenvalue": float(vals[-1]), "smallest_eigenvalue": float(vals[0]),
              "gram_f64_bytes": gram.nbytes, "seconds": time.monotonic() - t}
    return x, mean, std, vals, vecs, report


def fit_candidates(decomp, train_y, lambdas):
    x, mean, std, vals, vecs, _ = decomp
    y = np.asarray(train_y, np.float64)
    ym = y.mean(axis=0)
    rhs = vecs.T @ ((x.T @ (y - ym)) / len(x))
    for ridge in lambdas:
        c = (vecs @ (rhs / (vals[:, None] + ridge))) / std[:, None]
        b = ym - mean @ c
        coefficient, bias = c.T.astype(np.float32), b.astype(np.float32)
        qweight, wscale = quantize_rows(coefficient)
        yield ridge, {"coefficient": coefficient, "bias": bias, "qweight": qweight, "wscale": wscale}, float(np.sum(vals / (vals + ridge)))


def save_model(path, model, basis, expert, ridge):
    np.savez(path, **model, basis=np.asarray(basis), expert=np.int32(expert), ridge=np.float64(ridge),
             numeric_contract=np.asarray(NUMERIC_CONTRACT))


def fit_all(args):
    start = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise FileExistsError("selection exists; use evaluate to reuse frozen models")
    split_names = ("train", "valid") if getattr(args, "defer_test", False) else ("train", "valid", "test")
    caps = {name: Capture(getattr(args, name), args.layer) for name in split_names}
    audit = split_audit(caps, args.minimum_test_prompts)
    train, valid = caps["train"], caps["valid"]
    shared_width = train.get("shared_z").shape[1]
    feat = Features(train, shared_width, args.seed)
    feat.save(output / "random_swiglu_basis.npz")
    selection = {"status": "all_choices_frozen_before_TEST_target_read", "cpu_only": True,
                 "numeric_contract": NUMERIC_CONTRACT, "split_audit": audit,
                 "seed": args.seed, "layer": args.layer, "lambdas": args.lambdas, "basis_fits": {},
                 "capture_manifests": {name: str(Path(getattr(args, name)).resolve()) for name in ("train", "valid", "test")},
                 "train_valid_payload_sha256": {name: caps[name].payload_identity() for name in ("train", "valid")},
                 "test_targets_opened_during_fit": False,
                 "test_capture_deferred": getattr(args, "defer_test", False),
                 "source_sha256": sha(__file__), "models": [],
                 "limits": ["Local teacher-forced approximation diagnosis; not autoregressive quality or speedup.",
                            "x+shared_output has more readout coefficients than shared_z and is not a matched-capacity control.",
                            "Random input-only control is a fixed random basis, not an optimized nonlinear predictor.",
                            "Rows within a prompt are correlated; row count above width alone does not prove generalization."]}
    train_y, valid_y = train.get("expert_outputs"), valid.get("expert_outputs")
    for basis in args.bases:
        bx = feat.make(train, basis)
        vx = feat.make(valid, basis)
        decomp = decompose(bx, args.allow_underdetermined)
        selection["basis_fits"][basis] = decomp[-1]
        for ei, expert in enumerate(train.expert_ids):
            yy, vy = train_y[:, ei], valid_y[:, ei]
            best = {precision: (float("inf"), None) for precision in ("fp", "w8a8")}
            validations = []
            for ridge, model, edf in fit_candidates(decomp, yy, args.lambdas):
                record = {"lambda": ridge, "effective_df": edf, "valid": {}}
                for precision in best:
                    score = metrics(vy, predict(vx, model, precision))
                    record["valid"][precision] = score
                    if score["nmse"] < best[precision][0]:
                        best[precision] = (score["nmse"], (ridge, {k: v.copy() for k, v in model.items()}, edf))
                validations.append(record)
            for precision, (_, (ridge, model, edf)) in best.items():
                path = output / f"{basis}.expert{expert}.{precision}.npz"
                save_model(path, model, basis, expert, ridge)
                exports = {}
                if precision == "w8a8":
                    prefix = output / f"{basis}.expert{expert}.deploy"
                    model["qweight"].tofile(str(prefix) + ".i8")
                    model["wscale"].tofile(str(prefix) + ".scales.f32")
                    model["bias"].tofile(str(prefix) + ".bias.f32")
                    exports = {str(Path(str(prefix) + suffix).resolve()): sha(str(prefix) + suffix)
                               for suffix in (".i8", ".scales.f32", ".bias.f32")}
                selection["models"].append({"basis": basis, "expert": expert, "precision": precision,
                                             "lambda": ridge, "effective_df": edf, "path": str(path.resolve()),
                                             "sha256": sha(path), "deploy_files_sha256": exports, "validation_grid": validations})
            print(json.dumps({"fit": basis, "expert": expert, "valid_nmse": {p: v[0] for p, v in best.items()}}), flush=True)
        del decomp, bx, vx
    selection["fit_wall_seconds"] = time.monotonic() - start
    selection["random_basis_sha256"] = sha(output / "random_swiglu_basis.npz")
    registry = []
    aliases = {"x": "raw_x", "x_plus_shared_output": "x_shared_output", "random_swiglu": "nonlinear_control"}
    for ent in selection["models"]:
        if ent["precision"] != "w8a8":
            continue
        prefix = (output / f"{ent['basis']}.expert{ent['expert']}.deploy").resolve()
        width = selection["basis_fits"][ent["basis"]]["features"]
        registry.append(f"{args.layer}\t{ent['expert']}\t{aliases.get(ent['basis'], ent['basis'])}\t{width}\t{prefix}")
    (output / "registry.tsv").write_text("\n".join(registry) + "\n")
    selection["registry_sha256"] = sha(output / "registry.tsv")
    dump(output / "selection.json", selection)
    if not getattr(args, "defer_test", False):
        evaluate(output, args.minimum_test_prompts)


def aggregate_predictions(cap, preds, expert_ids):
    """Replace only the fixed experts; preserve all remaining native slots."""
    ids = np.asarray(cap.get("selected_experts"))
    alpha = np.asarray(cap.get("routing_weights"), np.float32)
    n, k = ids.shape
    h = cap.get("x").shape[1]
    delta = np.zeros((n, h), np.float32)
    selected_target = np.zeros_like(delta)
    selected_pred = np.zeros_like(delta)
    target = cap.get("expert_outputs")
    coverage = 0
    for slot in range(k):
        for ei, e in enumerate(expert_ids):
            rows = np.flatnonzero(ids[:, slot] == e)
            coverage += len(rows)
            if not len(rows):
                continue
            wt = alpha[rows, slot, None]
            native = np.multiply(target[rows, ei], wt, dtype=np.float32)
            approx = np.multiply(preds[ei][rows], wt, dtype=np.float32)
            selected_target[rows] += native
            selected_pred[rows] += approx
            delta[rows] += approx - native
    result = {"selected_slots": coverage, "total_slots": n * k,
              "selected_slot_fraction": coverage / (n * k),
              "selected_contribution": metrics(selected_target, selected_pred)}
    if cap.has("routed_sum"):
        native = np.asarray(cap.get("routed_sum"), np.float32)
        approx = native + delta
        if cap.has("routed_outputs"):
            slots = cap.get("routed_outputs")
            approx = np.zeros_like(native)
            for slot in range(k):
                value = np.asarray(slots[:, slot], np.float32).copy()
                for ei, e in enumerate(expert_ids):
                    rows = np.flatnonzero(ids[:, slot] == e)
                    value[rows] = preds[ei][rows]
                approx += np.multiply(value, alpha[:, slot, None], dtype=np.float32)
        result["routed_with_native_fallback"] = metrics(native, approx)
        # The native shared output is computed once, remains unchanged.
        whole = native + cap.get("shared_output")
        result["shared_plus_routed_with_native_fallback"] = metrics(whole, approx + cap.get("shared_output"))
        result["aggregation_contract"] = "slot-order F32(alpha * expert_output) additions; unchanged native slots, shared F32 addition"
    else:
        result["routed_with_native_fallback"] = None
        result["limitation"] = "routed_sum absent; only replaced expert contribution is scored"
    return result


def evaluate(output, minimum_test_prompts=1000):
    output = Path(output)
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text())
    if (output / "test_report.json").exists():
        raise FileExistsError("TEST report exists; refuses duplicate test evaluation")
    caps = {name: Capture(selection["capture_manifests"][name], selection.get("layer")) for name in ("train", "valid", "test")}
    audit = split_audit(caps, minimum_test_prompts)
    cap = caps["test"]
    for name in ("train", "valid"):
        if audit[name] != selection["split_audit"][name]:
            raise ValueError(f"{name} capture metadata changed after fit")
        if caps[name].payload_identity() != selection["train_valid_payload_sha256"][name]:
            raise ValueError(f"{name} payload changed after fit")
    feat = Features.load(output / "random_swiglu_basis.npz")
    if sha(output / "random_swiglu_basis.npz") != selection["random_basis_sha256"]:
        raise ValueError("random basis payload changed")
    report = {"status": "completed_heldout_local_test", "selection_sha256": sha(selection_path),
              "split_audit": audit,
              "test_payload_sha256": cap.payload_identity(),
              "cpu_only": True, "test_prompts": len(set(map(str, cap.prompts()))), "per_expert": [],
              "aggregates": [], "limits": selection["limits"]}
    truth = cap.get("expert_outputs")
    for basis in dict.fromkeys(m["basis"] for m in selection["models"]):
        bx = feat.make(cap, basis)
        for precision in ("fp", "w8a8"):
            predictions = []
            for ei, expert in enumerate(cap.expert_ids):
                ent = next(m for m in selection["models"] if m["basis"] == basis and m["precision"] == precision and m["expert"] == expert)
                if sha(ent["path"]) != ent["sha256"]:
                    raise ValueError("model payload changed after VALID selection")
                with np.load(ent["path"], allow_pickle=False) as model:
                    p = predict(bx, model, precision)
                predictions.append(p)
                score = metrics(truth[:, ei], p)
                if cap.has("selected_experts"):
                    selected = np.any(cap.get("selected_experts") == expert, axis=1)
                    score["native_selected_rows"] = int(selected.sum())
                    score["native_selected_only"] = metrics(truth[selected, ei], p[selected]) if selected.any() else None
                report["per_expert"].append({"basis": basis, "expert": expert, "precision": precision,
                                               "lambda": ent["lambda"], **score})
                # Compact per-prompt sufficient statistics allow paired uncertainty
                # checks with prompts, not correlated token rows, as resampling unit.
                ids, inverse = np.unique(cap.prompts(), return_inverse=True)
                yd, pd = np.asarray(truth[:, ei], np.float64), p.astype(np.float64)
                sq = np.sum((yd - pd) ** 2, axis=1)
                en = np.sum(yd ** 2, axis=1)
                np.savez(output / f"test.{basis}.expert{expert}.{precision}.prompt_stats.npz", prompt_ids=ids,
                         squared_error=np.bincount(inverse, weights=sq), target_energy=np.bincount(inverse, weights=en),
                         token_rows=np.bincount(inverse))
            if cap.has("selected_experts") and cap.has("routing_weights"):
                report["aggregates"].append({"basis": basis, "precision": precision,
                                              **aggregate_predictions(cap, predictions, cap.expert_ids)})
            del predictions
        del bx
    dump(output / "test_report.json", report)
    dump(output / "coefficient_costs.json", coefficient_costs(
        h=cap.get("x").shape[1], shared_width=cap.get("shared_z").shape[1]))
    print(json.dumps({"test_report": str(output / "test_report.json"), "test_prompts": report["test_prompts"]}), flush=True)
    return report


def coefficient_costs(h=2048, private_width=1408, shared_width=2816, experts=64, layers=26, k=6):
    def w8(rows, columns, bias=False):
        return rows * columns + 4 * rows + (4 * rows if bias else 0)
    original = 3 * h * private_width
    readout = h * shared_width
    gu_bytes = 2 * w8(shared_width, h)
    shared_down_bytes = w8(h, shared_width)
    cbytes = w8(h, shared_width, True)
    native_bytes = 2 * w8(private_width, h) + w8(h, private_width)
    total_readouts = layers * experts * cbytes
    # Available budget is after attention, embeddings, KV, common shared branch,
    # scratch, and metadata. The shared GU residency is already common here.
    budgets = []
    for gib in (0, .25, .5, 1, 2, 4, 8):
        budget = int(gib * 2 ** 30)
        resident = min(layers * experts, budget // cbytes)
        budgets.append({"incremental_host_GiB": gib, "resident_readouts": resident,
                        "total_readouts": layers * experts,
                        "all_readouts_fit": resident == layers * experts,
                        "resident_readout_fraction": resident / (layers * experts),
                        "remaining_flash_readout_W8_bytes": (layers * experts - resident) * cbytes})
    return {
        "scope": "Raw coefficients/MACs and W8+F32 row scales/biases; no hardware speedup claim.",
        "dimensions": {"H": h, "I": private_width, "Is_r": shared_width, "E": experts, "L": layers, "K": k},
        "per_expert_coefficients": {"native_G": h * private_width, "native_U": h * private_width,
                                     "native_D": h * private_width, "total_native": original,
                                     "refitted_C": readout, "raw_reduction": 1 - readout / original},
        "per_layer_host_shared_GU_W8_bytes": gu_bytes,
        "per_layer_common_shared_Down_W8_bytes": shared_down_bytes,
        "all_layers_host_shared_GU_W8_bytes": layers * gu_bytes,
        "per_expert_native_W8_bytes": native_bytes,
        "per_expert_C_W8_scales_bias_bytes": cbytes,
        "all_layers_all_experts_C_W8_scales_bias_bytes": total_readouts,
        "all_layers_all_experts_C_W8_scales_bias_GiB": total_readouts / 2 ** 30,
        "MAC_per_token_layer": {"native_routed": k * original, "refitted_routed_C": k * readout,
                                 "shared_GU": 2 * h * shared_width,
                                 "shared_Down": h * shared_width,
                                 "random_input_control_additional_GU": 2 * h * shared_width},
        "input_only_control_additional_W8_GU_scales_bias_bytes": 2 * w8(shared_width, h, True),
        "input_only_control_train_normalization_folded_into_GU_and_bias": True,
        "input_only_control_same_nominal_GU_readout_coefficient_count": True,
        "input_only_control_capacity_match_scope": "Isolated generator width and GU+readout coefficient count match; preserving the native shared branch adds a second GU generator for the random control, charged above. Total host footprint is therefore larger for that control.",
        "x_plus_shared_output_readout_coefficients": 2 * h * h,
        "host_readout_resident_strong_baseline": budgets,
        "budget_contract": "Budget rows are incremental host capacity after common native shared branch, non-FFN weights, KV, scales already counted, scratch and metadata. Full system residency still requires those common costs.",
        "unmodeled_for_latency": ["shared feature readiness serial dependency", "host effective bandwidth and MAC throughput",
                                  "commands, NAND page packing and starts", "input/output transfer", "native fallbacks and all contexts"],
    }


def numeric_self_test(out):
    rng = np.random.default_rng(20261108)
    a = rng.normal(size=(19, 2816)).astype(np.float32)
    a[0] = 0
    q, sx = quantize_rows(a)
    w, sw = quantize_rows(rng.normal(size=(13, 2816)).astype(np.float32))
    d = exact_dots(q, w)
    assert np.array_equal(d, q.astype(np.int64) @ w.astype(np.int64).T)
    # Make max=127, hence scale=1, so these are actual adversarial A8 ties.
    edge = np.asarray([[127, .5, -.5, np.nextafter(np.float32(.5), np.float32(0)),
                        np.nextafter(np.float32(-.5), np.float32(0)), 1.5, -1.5]], np.float32)
    qe, se = quantize_rows(edge)
    assert qe.tolist() == [[127, 1, -1, 0, 0, 2, -2]]
    assert sx[0] == 1 and np.all(q[0] == 0)
    report = {"status": "PASS", "CPU_only": True, "near_half_ties_away": True,
              "full_2816_exact_integer_dots": True, "zero_scale_contract": True,
              "numeric_contract": NUMERIC_CONTRACT}
    # Use immutable project C++ canonical primitive if available. It exports
    # exactly the historic contract, providing an independent language check.
    libpath = Path(__file__).resolve().parents[1] / "results/research-2026-11-03-no-cr/round02-local-affine/stage0/freezes/rev003/numeric.so"
    if libpath.exists():
        os.environ.setdefault("OMP_NUM_THREADS", "8")
        lib = ctypes.CDLL(str(libpath))
        ptr = lambda x: ctypes.c_void_p(x.ctypes.data)
        qc, sc = np.empty_like(q), np.empty_like(sx)
        assert lib.la_quantize(ptr(a), len(a), a.shape[1], ptr(qc), ptr(sc)) == 0
        assert np.array_equal(qc, q) and np.array_equal(sc.view(np.uint32), sx.view(np.uint32))
        qec, sec = np.empty_like(qe), np.empty_like(se)
        assert lib.la_quantize(ptr(edge), len(edge), edge.shape[1], ptr(qec), ptr(sec)) == 0
        assert np.array_equal(qec, qe)
        outc = np.empty(d.shape, np.float32)
        assert lib.la_rescale(ptr(d), ptr(sw), ptr(sx), len(d), d.shape[1], ptr(outc)) == 0
        assert np.array_equal(outc.view(np.uint32), rescale(d, sw, sx).view(np.uint32))
        report["canonical_Cpp_quantize_rescale_bitwise_match"] = True
        report["canonical_library_sha256"] = sha(libpath)
    # An identifiable synthetic shared teacher is recovered; independent GU
    # features form a negative control, with more rows than features.
    n, h, width, outputs = 1200, 9, 24, 7
    x = rng.normal(size=(n, h))
    g, u = rng.normal(size=(h, width)), rng.normal(size=(h, width))
    z = ((x @ g) * expit(x @ g) * (x @ u)).astype(np.float32)
    c = rng.normal(size=(width, outputs))
    y = (z @ c).astype(np.float32)
    train, test = np.arange(800), np.arange(800, n)
    dec = decompose(z[train])
    _, model, _ = next(fit_candidates(dec, y[train], [1e-8]))
    score = metrics(y[test], predict(z[test], model, "fp"))
    assert score["nmse"] < 1e-8
    report["synthetic_shared_teacher_test"] = score
    bad = rng.normal(size=z.shape).astype(np.float32)
    dec = decompose(bad[train])
    _, model, _ = next(fit_candidates(dec, y[train], [1.0]))
    negative = metrics(y[test], predict(bad[test], model, "fp"))
    assert negative["nmse"] > .5
    report["synthetic_independent_basis_negative_control"] = negative
    dump(out, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    for name in ("train", "valid", "test", "output"):
        fit.add_argument("--" + name, required=True)
    fit.add_argument("--minimum-test-prompts", type=int, default=1000)
    fit.add_argument("--layer", type=int)
    fit.add_argument("--seed", type=int, default=20261108)
    fit.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    fit.add_argument("--bases", nargs="+", choices=BASES, default=list(BASES))
    fit.add_argument("--allow-underdetermined", action="store_true")
    fit.add_argument("--defer-test", action="store_true", help="fit and freeze using TRAIN/VALID only; evaluate later")
    fit.add_argument("--threads", type=int, default=8)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--minimum-test-prompts", type=int, default=1000)
    evaluate_parser.add_argument("--threads", type=int, default=8)
    numeric = sub.add_parser("self-test")
    numeric.add_argument("--output", required=True)
    numeric.add_argument("--threads", type=int, default=8)
    costs = sub.add_parser("costs")
    costs.add_argument("--output", required=True)
    costs.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads, user_api="blas"):
        if args.command == "fit":
            if min(args.lambdas) <= 0:
                raise ValueError("strictly positive ridge grid required")
            fit_all(args)
        elif args.command == "evaluate":
            evaluate(args.output, args.minimum_test_prompts)
        elif args.command == "self-test":
            numeric_self_test(args.output)
        else:
            dump(args.output, coefficient_costs())


if __name__ == "__main__":
    main()
