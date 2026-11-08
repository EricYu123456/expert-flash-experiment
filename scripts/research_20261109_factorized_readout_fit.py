#!/usr/bin/env python3
"""TRAIN activation-aware low-rank readouts; all VALID is CPU integer-exact.

Each fixed parent C is projected to top TRAIN output-covariance directions:
P=top_eigenvectors(C G C.T), U=P, V=P.T C. The actual deployment candidate is
input affine/A8 -> rowW8 V -> F32 latent affine -> new A8 -> rowW8 U + bias.
Latent affine and bias use TRAIN only. Full-C fallback files remain immutable.
No TEST interface or runtime adapter is implemented here.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("factorized_prototype_tools", HERE / "research_20261109_shared_feature_fit_proto.py")
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)
base, old = proto.base, proto.old
spec = importlib.util.spec_from_file_location("factorized_exact_numeric", HERE / "research_20261109_shared_feature_targets.py")
targets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(targets)


def source_features(dataset, transform, rows=None):
    sources, indices = transform["sources"], transform["indices"]
    n = len(dataset.capture.get("x")) if rows is None else len(rows)
    raw = np.empty((n, len(sources)), np.float32)
    for source, name in ((0, "x"), (1, "shared_z")):
        mask = sources == source
        if np.any(mask):
            values = dataset.capture.get(name)
            raw[:, mask] = values[:, indices[mask]] if rows is None else values[np.ix_(rows, indices[mask])]
    for owner in set(map(int, transform["prototypes"][sources == 2])):
        mask = (sources == 2) & (transform["prototypes"] == owner)
        values = dataset.prototype(owner)
        raw[:, mask] = values[:, indices[mask]] if rows is None else values[np.ix_(rows, indices[mask])]
    return base.transform(raw, transform["center"], transform["inv_scale"])


def weighted_latent_affine(latent, weights):
    value = latent.astype(np.float64)
    total = weights.sum()
    mean = weights @ value / total
    std = np.sqrt(np.maximum(weights @ (value - mean) ** 2 / total, 0))
    floor = max(float(np.median(std)) * 1e-6, 1e-12)
    return mean.astype(np.float32), (1 / np.maximum(std, floor)).astype(np.float32), floor


def train_bias(encoded, model, target, weights):
    total = np.zeros(len(model["wscale"]), np.float64)
    for start in range(0, len(weights), 512):
        sl = slice(start, start + 512)
        xx = {k: v[sl] for k, v in encoded.items() if k != "qdq"}
        pred = base.predict_encoded(xx, model, use_bias=False)
        total += weights[sl] @ (np.asarray(target[sl], np.float64) - pred.astype(np.float64))
    return (total / weights.sum()).astype(np.float32)


def validate_pipeline(numeric, input_features, V, center, gain, U, expected):
    """Independent frozen C++ rescale/transform checks on selected VALID probes."""
    x = np.ascontiguousarray(input_features)
    latent = numeric.matvecs(x, V["qweight"], V["wscale"])
    python_latent = base.predict_encoded(base.encode(x), V, use_bias=False)
    assert np.array_equal(latent.view(np.uint32), python_latent.view(np.uint32)), "V CPU/C++ bits"
    actual = np.empty_like(latent)
    rank = latent.shape[1]
    sources = np.zeros(rank, np.uint8)
    indices = np.arange(rank, dtype=np.int32)
    dummy = np.zeros((len(x), 1), np.float32)
    assert numeric.lib.sf_transform(len(x), rank, 1, rank, latent, dummy, sources, indices, center, gain, actual) == 0
    transformed = base.transform(latent, center, gain)
    assert np.array_equal(actual.view(np.uint32), transformed.view(np.uint32)), "latent affine C++ bits"
    final = np.add(numeric.matvecs(actual, U["qweight"], U["wscale"]), U["bias"], dtype=np.float32)
    assert np.array_equal(final.view(np.uint32), expected.view(np.uint32)), "factor predictor C++ bits"
    return dict(status="PASS", rows=len(x), latent_floats_bitchecked=int(latent.size),
                output_floats_bitchecked=int(final.size))


def partition_metrics(y, prediction, a2, positions):
    yy, pred = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    squared, energy = np.sum((yy - pred) ** 2, axis=1), np.sum(yy ** 2, axis=1)
    result = {}
    for name, mask in (("all", np.ones(len(y), bool)), ("BOS", positions == 0), ("nonBOS", positions > 0)):
        error, total = float(a2[mask] @ squared[mask]), float(a2[mask] @ energy[mask])
        result[name] = dict(rows=int(mask.sum()), weighted_squared_error=error, weighted_target_energy=total,
                            alpha2_NMSE=error / total if total else None)
    return result


def aggregate(valid, predictions):
    ids, alpha = valid.capture.get("selected_experts"), valid.capture.get("routing_weights")
    slots = valid.capture.get("routed_outputs")
    n, k, hidden = slots.shape
    native, approx = np.zeros((n, hidden), np.float32), np.zeros((n, hidden), np.float32)
    for slot in range(k):
        actual = np.asarray(slots[:, slot], np.float32)
        value = actual.copy()
        for expert, (rows, pred) in predictions.items():
            selected = np.flatnonzero(ids[:, slot] == expert)
            value[selected] = pred[np.searchsorted(rows, selected)]
        native += np.multiply(actual, alpha[:, slot, None], dtype=np.float32)
        approx += np.multiply(value, alpha[:, slot, None], dtype=np.float32)
    positions, shared = valid.capture.get("positions"), valid.capture.get("shared_output")
    return {"routed": old.metrics(native, approx), "shared_plus_routed": old.metrics(native + shared, approx + shared),
            "routed_nonBOS": old.metrics(native[positions > 0], approx[positions > 0]),
            "routed_BOS": old.metrics(native[positions == 0], approx[positions == 0]),
            "eligible_slots": int(np.isin(ids, list(predictions)).sum()), "total_slots": int(ids.size)}


def save_model(prefix, V, center, gain, U):
    payload = {}
    arrays = {".V.i8": V["qweight"], ".V.scales.f32": V["wscale"],
              ".latent.center.f32": center, ".latent.inv_scale.f32": gain,
              ".U.i8": U["qweight"], ".U.scales.f32": U["wscale"], ".bias.f32": U["bias"]}
    for suffix, value in arrays.items():
        path = str(prefix) + suffix
        value.tofile(path)
        payload[path] = old.sha(path)
    return payload


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or set(os.sched_getaffinity(0)) != {64, 65}:
        raise ValueError("use CUDA0 and CPU affinity64-65")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(8 * 1024 ** 3 / properties.total_memory, device)
    torch.cuda.reset_peak_memory_stats(device)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    handoff_path = Path(args.handoff).resolve()
    handoff = json.loads(handoff_path.read_text())
    experts = list(handoff["cohort_order"] if args.experts is None else args.experts)
    if not set(experts) <= set(handoff["cohort_order"]):
        raise ValueError("unknown cohort expert")
    train = proto.Dataset(args.train, args.train_target_manifest, 1)
    valid = proto.Dataset(args.valid, args.valid_target_manifest, 1)
    reference = json.loads(Path(next(iter(handoff["input_selection_sha256"]))).read_text())
    if reference["train_identity"] != train.identity() or reference["valid_identity"] != valid.identity():
        raise ValueError("factor data differ from full-C baseline")
    audit = base.capture_audit(train.capture, valid.capture)
    numeric = targets.Numeric(2)
    if numeric.binary_sha256 != "dedb64bee2d6ee55a252ffa318ece561eb3c55dcb6901496972584f3aa31b8f5":
        raise ValueError("canonical numeric library changed")
    report = dict(status="TRAIN_activation_aware_factorization_CPU_exact_VALID_no_TEST", ranks=args.ranks,
        experts=experts, families=args.families, train_identity=train.identity(), valid_identity=valid.identity(),
        capture_audit=audit, source_sha256=old.sha(__file__), handoff_sha256=old.sha(handoff_path),
        numeric_sha256=numeric.binary_sha256, models=[], aggregate_VALID={}, pooled_VALID={},
        projection="top TRAIN weighted C G C^T output eigenvectors P; U=P,V=P^TC; parent beta",
        latent_affine="TRAIN weighted mean/std of actual canonical W8 V output; not folded out of runtime",
        bias_calibration="full TRAIN actual CPU canonical residual, parent beta weights",
        runtime_implemented=False, hardware=dict(GPU=properties.name, GPU_device="cuda:0", CPU_affinity=[64, 65],
            VALID_CPU_threads=2, torch=torch.__version__, CUDA=torch.version.cuda,
            nvidia_smi=subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                                     capture_output=True, text=True, check=True).stdout.strip()))
    started = time.monotonic()
    for family in args.families:
        entries = {v["expert"]: v for v in handoff["models"][family]}
        tp = handoff["transforms"][family]["unified_prefix"]
        transform = proto.load_transform(tp)
        raw_train = source_features(train, transform)
        tx = base.encode(raw_train)
        del raw_train
        raw_valid = source_features(valid, transform)
        vx = base.encode(raw_valid)
        x = torch.from_numpy(tx["qdq"]).to(device=device, dtype=torch.float64)
        mean_all, second_all = x.mean(dim=0), (x.T @ x) / len(x)
        predictions_by_rank = {rank: {} for rank in args.ranks}
        parent_predictions = {}
        for expert in experts:
            entry = entries[expert]
            for path, expected in entry["payload_sha256"].items():
                if old.sha(path) != expected:
                    raise ValueError("immutable full-C payload changed")
            parent = proto.load_deployed_model(entry)
            coefficient = np.multiply(parent["qweight"].astype(np.float32), parent["wscale"][:, None], dtype=np.float32)
            c = torch.from_numpy(coefficient).to(device=device, dtype=torch.float64)
            a2, va2 = base.alpha_squared(train.capture, expert), base.alpha_squared(valid.capture, expert)
            weights = base.blend_weights(a2, entry["beta"])
            if entry["beta"] == 0:
                mean, second = mean_all, second_all
            else:
                selected = np.flatnonzero(a2 > 0)
                xx = x[torch.from_numpy(selected).to(device)]
                aw = torch.from_numpy(a2[selected]).to(device)
                mean_sel, second_sel = (aw @ xx) / aw.sum(), (xx.T @ (xx * aw[:, None])) / aw.sum()
                mean = (1 - entry["beta"]) * mean_all + entry["beta"] * mean_sel
                second = (1 - entry["beta"]) * second_all + entry["beta"] * second_sel
                del xx, aw
            covariance = second - mean[:, None] * mean[None, :]
            output_covariance = c @ covariance @ c.T
            eigenvalues, eigenvectors = torch.linalg.eigh((output_covariance + output_covariance.T) * .5)
            eigenvalues = eigenvalues.clamp(min=0).flip(0)
            pmax = eigenvectors[:, -max(args.ranks):].flip(1)
            vbig = pmax.T @ c
            rows = np.flatnonzero(va2 > 0)
            selected_vx = {k: val[rows] for k, val in vx.items() if k != "qdq"}
            target_valid = np.asarray(valid.targets.get(expert)[rows])
            target_train = train.targets.get(expert)
            parent_pred = base.predict_encoded(selected_vx, parent)
            parent_predictions[expert] = (rows, parent_pred)
            parent_metrics = partition_metrics(target_valid, parent_pred, va2[rows], valid.capture.get("positions")[rows])
            assert np.isclose(parent_metrics["all"]["weighted_squared_error"], entry["weighted_squared_error"], rtol=1e-11, atol=1e-9)
            for rank in args.ranks:
                rank_started = time.monotonic()
                ufloat, vfloat = pmax[:, :rank].cpu().numpy(), vbig[:rank].cpu().numpy().astype(np.float32)
                vq, vs = old.quantize_rows(vfloat)
                V = dict(qweight=vq, wscale=vs, bias=np.zeros(rank, np.float32))
                latent_train = base.predict_encoded(tx, V, use_bias=False)
                center, gain, floor = weighted_latent_affine(latent_train, weights)
                encoded_latent = base.encode(base.transform(latent_train, center, gain))
                del latent_train
                # Retain the actual F32 gain in runtime and compensate U for
                # that coordinate system before row-W8. No affine is omitted.
                transformed_u = (ufloat / gain.astype(np.float64)[None, :]).astype(np.float32)
                uq, us = old.quantize_rows(transformed_u)
                U = dict(qweight=uq, wscale=us, bias=np.zeros(2048, np.float32))
                U["bias"] = train_bias(encoded_latent, U, target_train, weights)
                valid_latent = base.predict_encoded(selected_vx, V, use_bias=False)
                valid_encoded_latent = base.encode(base.transform(valid_latent, center, gain))
                prediction = base.predict_encoded(valid_encoded_latent, U)
                metrics = partition_metrics(target_valid, prediction, va2[rows], valid.capture.get("positions")[rows])
                probes = np.asarray(sorted({0, len(rows) // 3, len(rows) // 2, len(rows) - 1}))
                check = validate_pipeline(numeric, np.ascontiguousarray(raw_valid[rows[probes]]), V, center, gain, U, prediction[probes])
                prefix = out / f"{family}.expert{expert}.rank{rank}"
                payload = save_model(prefix, V, center, gain, U)
                parent_error = parent_metrics["all"]["weighted_squared_error"]
                model = dict(family=family, expert=expert, rank=rank, input_width=1408, hidden=2048,
                    input_transform_prefix=tp, factor_prefix=str(prefix), parent_readout_prefix=entry["readout_prefix"],
                    parent_beta=entry["beta"], TRAIN_latent_variance_floor=floor,
                    TRAIN_output_covariance_energy_retained=float(eigenvalues[:rank].sum().cpu() / eigenvalues.sum().cpu()),
                    VALID=metrics, parent_VALID=parent_metrics,
                    VALID_SSE_ratio_vs_fullC=metrics["all"]["weighted_squared_error"] / parent_error,
                    nonBOS_VALID_SSE_ratio_vs_fullC=metrics["nonBOS"]["weighted_squared_error"] / parent_metrics["nonBOS"]["weighted_squared_error"],
                    payload_sha256=payload, CPP_numeric_audit=check,
                    factor_coefficients=rank * (2048 + 1408),
                    factor_payload_bytes=rank * (2048 + 1408) + 12 * rank + 8 * 2048,
                    original_fullC_scales_bias_bytes=2048 * 1408 + 8 * 2048,
                    latent_extra_affine_and_A8_per_expert=True, native_GU_owner=25 if family == "prototype25" else None,
                    seconds=time.monotonic() - rank_started)
                actual_bytes = sum(Path(p).stat().st_size for p in payload)
                assert actual_bytes == model["factor_payload_bytes"]
                report["models"].append(model)
                predictions_by_rank[rank][expert] = (rows, prediction)
                print(json.dumps({"family": family, "expert": expert, "rank": rank,
                                  "VALID_NMSE": metrics["all"]["alpha2_NMSE"],
                                  "SSE_ratio_fullC": model["VALID_SSE_ratio_vs_fullC"], "seconds": model["seconds"]}), flush=True)
                del encoded_latent, valid_latent, valid_encoded_latent
            del c, output_covariance, eigenvalues, eigenvectors, pmax, vbig
            old.dump(out / "progress.json", {"models": report["models"]})
        report["aggregate_VALID"][family] = {"fullC": aggregate(valid, parent_predictions)}
        report["pooled_VALID"][family] = {}
        for rank in args.ranks:
            report["aggregate_VALID"][family][str(rank)] = aggregate(valid, predictions_by_rank[rank])
            models = [m for m in report["models"] if m["family"] == family and m["rank"] == rank]
            pooled = {}
            for partition in ("all", "BOS", "nonBOS"):
                error = sum(m["VALID"][partition]["weighted_squared_error"] for m in models)
                original_error = sum(m["parent_VALID"][partition]["weighted_squared_error"] for m in models)
                energy = sum(m["VALID"][partition]["weighted_target_energy"] for m in models)
                pooled[partition] = dict(alpha2_NMSE=error / energy if energy else None,
                                         SSE_ratio_fullC=error / original_error if original_error else None,
                                         weighted_squared_error=error, weighted_target_energy=energy)
            pooled["heads_with_SSE_increase_le5percent"] = sum(m["VALID_SSE_ratio_vs_fullC"] <= 1.05 for m in models)
            pooled["heads"] = len(models)
            report["pooled_VALID"][family][str(rank)] = pooled
        del x, mean_all, second_all, tx, vx, raw_valid, predictions_by_rank, parent_predictions
    report["GPU_peak_memory"] = dict(allocated_bytes=torch.cuda.max_memory_allocated(device), reserved_bytes=torch.cuda.max_memory_reserved(device))
    assert report["GPU_peak_memory"]["reserved_bytes"] < 8 * 1024 ** 3
    report["seconds"] = time.monotonic() - started
    old.dump(out / "selection.json", report)
    print(json.dumps({"complete": str(out / "selection.json"), "seconds": report["seconds"], "pooled": report["pooled_VALID"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("handoff", "train", "valid", "train-target-manifest", "valid-target-manifest", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--experts", nargs="+", type=int)
    parser.add_argument("--ranks", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--families", nargs="+", choices=("prototype25", "raw_x_select1408_eq"), default=["prototype25", "raw_x_select1408_eq"])
    args = parser.parse_args()
    if not args.ranks or any(not 0 < rank <= 1408 for rank in args.ranks):
        raise ValueError("factor ranks must be in 1..1408")
    with threadpool_limits(limits=2, user_api="blas"):
        run(args)
