#!/usr/bin/env python3
"""CUDA TRAIN-only GU STE with fixed C/affine and CPU-exact VALID.

This is an independent extension of the immutable CPU learned-GU experiment.
Only the surrogate/optimizer run on CUDA. Every VALID score uses canonical
CPU integer-exact W8/A8 and the frozen native SwiGLU implementation. Updated
GU weights form a new bank and cannot reuse native owner25's exact GU work.
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
from threadpoolctl import threadpool_limits, threadpool_info
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cpu_learned_gu_reference", HERE / "research_20261109_learned_gu.py")
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
fit, targets, base, old = cpu.fit, cpu.targets, cpu.base, cpu.old


def exact_features(numeric, x, gate, up, transform, native_weights=None, reference_phi=None):
    """CPU only. Native step0 bypasses dequantization/requantization of W8."""
    if native_weights is None:
        gq, gs = old.quantize_rows(gate)
        uq, us = old.quantize_rows(up)
    else:
        gq, gs = native_weights["gate"]
        uq, us = native_weights["up"]
    result, bitchecked = [], 0
    for start in range(0, len(x), 1024):
        xx = np.ascontiguousarray(x[start:start + 1024], dtype=np.float32)
        g, u = numeric.matvecs(xx, gq, gs), numeric.matvecs(xx, uq, us)
        phi = np.empty(g.shape, np.float32)
        assert numeric.lib.sf_swiglu(*phi.shape, g, u, phi) == 0
        if reference_phi is not None:
            ref = np.ascontiguousarray(reference_phi[start:start + len(phi)])
            if not np.array_equal(phi.view(np.uint32), ref.view(np.uint32)):
                raise ValueError("step0 native GU phi differs from frozen capture bits")
            bitchecked += phi.size
        result.append(base.transform(phi, transform["center"], transform["inv_scale"]))
    return base.encode(np.concatenate(result)), bitchecked


def evaluate(numeric, dataset, pair_data, models, gate, up, transform, native_weights=None):
    """No CUDA tensors or Torch arithmetic occur in this evaluator."""
    pair_rows, pair_experts, pair_targets, pair_a2 = pair_data
    unique, inverse = np.unique(pair_rows, return_inverse=True)
    reference = dataset.prototype(25)[unique] if native_weights is not None else None
    features, bitchecked = exact_features(numeric, dataset.capture.get("x")[unique], gate, up,
                                         transform, native_weights, reference)
    positions = dataset.capture.get("positions")
    error, energy, records = 0., 0., []
    partition_totals = {name: [0., 0.] for name in ("BOS", "nonBOS")}
    for expert, model in models.items():
        selected = np.flatnonzero(pair_experts == expert)
        enc = {k: v[inverse[selected]] for k, v in features.items()}
        pred = base.predict_encoded(enc, model)
        yy = pair_targets[selected].astype(np.float64)
        ee = np.sum((yy - pred.astype(np.float64)) ** 2, axis=1)
        nn = np.sum(yy ** 2, axis=1)
        a2 = pair_a2[selected]
        sse, target_energy = float(a2 @ ee), float(a2 @ nn)
        record = dict(expert=expert, selected_rows=len(selected), weighted_sse=sse,
                      weighted_target_energy=target_energy, alpha2_nmse=sse / target_energy,
                      partitions={})
        for name, mask in (("BOS", positions[pair_rows[selected]] == 0),
                           ("nonBOS", positions[pair_rows[selected]] > 0)):
            se, en = float(a2[mask] @ ee[mask]), float(a2[mask] @ nn[mask])
            record["partitions"][name] = dict(selected_rows=int(mask.sum()), weighted_sse=se,
                weighted_target_energy=en, alpha2_nmse=se / en if en else None)
            partition_totals[name][0] += se
            partition_totals[name][1] += en
        records.append(record)
        error += sse
        energy += target_energy
    return dict(alpha2_nmse=error / energy, weighted_sse=error, weighted_target_energy=energy,
                experts=records, canonical_W8A8=True, evaluation_device="CPU",
                GU_unique_input_rows=len(unique), native_phi_floats_bitchecked=bitchecked,
                partitions={k: dict(weighted_sse=v[0], weighted_target_energy=v[1],
                                     alpha2_nmse=v[0] / v[1] if v[1] else None)
                            for k, v in partition_totals.items()})


def save_gu(folder, gate, up, native=None):
    folder.mkdir(parents=True, exist_ok=False)
    payload = {}
    for name, value in (("gate", gate), ("up", up)):
        codes, scales = native[name] if native is not None else old.quantize_rows(value)
        for suffix, array in ((".i8", codes), (".scales.f32", scales)):
            path = folder / (name + suffix)
            array.tofile(path)
            payload[str(path)] = old.sha(path)
    return payload


def run(args):
    if args.prototype != 25 or args.device not in ("cuda", "cuda:0"):
        raise ValueError("this frozen screen uses only physical CUDA0 and initial native prototype25")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("launch with CUDA_VISIBLE_DEVICES=0")
    if set(os.sched_getaffinity(0)) != {64, 65}:
        raise ValueError("CPU exact evaluation must stay pinned to cores64-65")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("compatible CUDA environment required")
    prop = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(min(1., args.gpu_memory_limit_gib * 1024 ** 3 / prop.total_memory), device)
    torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text())
    layer = selection["layer"]
    records = selection["families"]["best_cross_prototype"]
    experts = sorted(m["expert"] for m in records)
    if len(records) != 12 or len(set(experts)) != 12 or 25 in experts:
        raise ValueError("exactly twelve nonowner expert readouts required")
    if any(m["prototype"] != 25 or m["width"] != 1408 for m in records):
        raise ValueError("all twelve readouts must use native prototype25 full width1408")
    train = fit.Dataset(args.train, args.train_target_manifest, layer)
    valid = fit.Dataset(args.valid, args.valid_target_manifest, layer)
    if selection["train_identity"] != train.identity() or selection["valid_identity"] != valid.identity():
        raise ValueError("selection/data identity mismatch")
    capture_audit = base.capture_audit(train.capture, valid.capture)
    transform = fit.load_transform(records[0]["transform_prefix"])
    assert all(m["transform_prefix"] == records[0]["transform_prefix"] for m in records)
    assert len(transform["indices"]) == 1408 and np.all(transform["sources"] == 2)
    assert np.all(transform["prototypes"] == 25)
    assert np.array_equal(transform["indices"], np.arange(1408, dtype=np.int32))
    fixed_payload_hashes, models = {}, {}
    for record in records:
        prefix = record["readout_prefix"]
        models[record["expert"]] = dict(qweight=np.fromfile(prefix + ".i8", np.int8).reshape(2048, 1408),
            wscale=np.fromfile(prefix + ".scales.f32", np.float32), bias=np.fromfile(prefix + ".bias.f32", np.float32))
        for suffix in (".i8", ".scales.f32", ".bias.f32"):
            path = prefix + suffix
            actual = old.sha(path)
            if actual != record["payload_sha256"][path]:
                raise ValueError("fixed baseline C payload changed")
            fixed_payload_hashes[path] = actual
    tp = records[0]["transform_prefix"]
    for suffix in (".sources.u8", ".indices.i32", ".center.f32", ".inv_scale.f32", ".prototypes.i32"):
        fixed_payload_hashes[tp + suffix] = old.sha(tp + suffix)
    weights_path = Path(args.weights)
    weights_meta = json.loads((weights_path / "weights.json").read_text())
    assert weights_meta["layer"] == layer and "25" in weights_meta["experts"]
    native = targets.load_weights(weights_path, 25)
    gate0 = (native["gate"][0].astype(np.float32) * native["gate"][1][:, None]).astype(np.float32)
    up0 = (native["up"][0].astype(np.float32) * native["up"][1][:, None]).astype(np.float32)
    gate_reference = torch.from_numpy(gate0.copy()).to(device)
    up_reference = torch.from_numpy(up0.copy()).to(device)
    gate = torch.nn.Parameter(gate_reference.clone())
    up = torch.nn.Parameter(up_reference.clone())
    optimizer = torch.optim.Adam([gate, up], lr=args.learning_rate)
    tr, va = cpu.pairs(train, experts), cpu.pairs(valid, experts)
    q, sx = old.quantize_rows(train.capture.get("x"))
    xqdq = torch.from_numpy(np.multiply(q.astype(np.float32), sx[:, None], dtype=np.float32)).to(device)
    y = torch.from_numpy(tr[2]).to(device)
    a2 = torch.from_numpy(tr[3].astype(np.float32)).to(device)
    torch_heads = {e: (torch.from_numpy(np.multiply(m["qweight"].astype(np.float32), m["wscale"][:, None], dtype=np.float32)).to(device),
                       torch.from_numpy(m["bias"]).to(device)) for e, m in models.items()}
    center, inv = torch.from_numpy(transform["center"]).to(device), torch.from_numpy(transform["inv_scale"]).to(device)
    assert all(not t.requires_grad for pair in torch_heads.values() for t in pair)
    assert not center.requires_grad and not inv.requires_grad
    normalization = float(np.sum(tr[3] * np.sum(tr[2].astype(np.float64) ** 2, axis=1)) / len(tr[0]))
    gate_denominator, up_denominator = max(float(np.mean(gate0 ** 2)), 1e-20), max(float(np.mean(up0 ** 2)), 1e-20)
    numeric = targets.Numeric(args.threads)
    if numeric.binary_sha256 != "dedb64bee2d6ee55a252ffa318ece561eb3c55dcb6901496972584f3aa31b8f5":
        raise ValueError("frozen canonical CPU numeric library changed")
    gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True).stdout.strip()
    hardware = dict(device=str(device), CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"], gpu_name=prop.name,
        gpu_uuid=str(getattr(prop, "uuid", "unknown")), total_device_memory_bytes=prop.total_memory,
        torch_version=torch.__version__, torch_CUDA_runtime=torch.version.cuda, nvidia_smi_at_start=gpu_query,
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        mixed_precision=False, torch_compile=False, deterministic_algorithms=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)), exact_VALID_CPU_threads=args.threads,
        memory_limit_GiB=args.gpu_memory_limit_gib, BLAS=threadpool_info())
    old.dump(out / "hardware.json", hardware)
    checkpoints, training = [], []
    started, best_score, best_step = time.monotonic(), float("inf"), None
    for step in range(args.steps + 1):
        if step in args.checkpoints or step == args.steps:
            g, u = gate.detach().cpu().numpy().copy(), up.detach().cpu().numpy().copy()
            valid_start = time.monotonic()
            metric = evaluate(numeric, valid, va, models, g, u, transform, native if step == 0 else None)
            if step == 0:
                expected = {m["expert"]: m for m in records}
                for m in metric["experts"]:
                    ref = expected[m["expert"]]
                    assert np.isclose(m["weighted_sse"], ref["weighted_squared_error"], rtol=1e-11, atol=1e-9)
                    assert np.isclose(m["weighted_target_energy"], ref["selected_VALID_weighted_target_energy"], rtol=1e-11, atol=1e-9)
                expected_error = sum(m["weighted_squared_error"] for m in records)
                expected_energy = sum(m["selected_VALID_weighted_target_energy"] for m in records)
                assert abs(metric["alpha2_nmse"] - expected_error / expected_energy) < 1e-9
            payload = save_gu(out / f"checkpoint{step}", g, u, native if step == 0 else None)
            rec = dict(step=step, elapsed_seconds=time.monotonic() - started,
                       CPU_VALID_seconds=time.monotonic() - valid_start, VALID=metric, GU_payload_sha256=payload,
                       step0_original_W8_codes_preserved=step == 0)
            checkpoints.append(rec)
            if metric["alpha2_nmse"] < best_score:
                best_score, best_step = metric["alpha2_nmse"], step
            old.dump(out / "progress.json", dict(checkpoints=checkpoints, training=training))
            print(json.dumps(rec), flush=True)
        if step == args.steps:
            break
        ix = rng.integers(len(tr[0]), size=args.batch)
        optimizer.zero_grad(set_to_none=True)
        xx = xqdq[torch.from_numpy(tr[0][ix]).to(device)]
        phi = torch.nn.functional.silu(xx @ cpu.ste_qdq(gate).T) * (xx @ cpu.ste_qdq(up).T)
        feature = cpu.ste_qdq((phi - center) * inv)
        loss = torch.zeros((), dtype=torch.float32, device=device)
        for expert in np.unique(tr[1][ix]):
            loc = np.flatnonzero(tr[1][ix] == expert)
            li = torch.from_numpy(loc).to(device)
            yi = torch.from_numpy(ix[loc]).to(device)
            cw, cb = torch_heads[int(expert)]
            prediction = feature[li] @ cw.T + cb
            loss = loss + ((prediction - y[yi]).square().sum(dim=1) * a2[yi]).sum()
        loss = loss / (args.batch * normalization)
        penalty = ((gate - gate_reference).square().mean() / gate_denominator
                   + (up - up_reference).square().mean() / up_denominator)
        objective = loss + args.regularization * penalty
        if not bool(torch.isfinite(objective)):
            raise ValueError("non-finite TRAIN GPU objective")
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([gate, up], 10.)
        optimizer.step()
        training.append(dict(step=step + 1, TRAIN_surrogate_loss=float(loss.detach().cpu()),
                             relative_displacement_penalty=float(penalty.detach().cpu()),
                             objective=float(objective.detach().cpu()), gradient_norm=float(gradient_norm.detach().cpu())))
    torch.cuda.synchronize(device)
    peak = dict(allocated_bytes=torch.cuda.max_memory_allocated(device), reserved_bytes=torch.cuda.max_memory_reserved(device))
    assert peak["reserved_bytes"] < args.gpu_memory_limit_gib * 1024 ** 3
    for path, expected in fixed_payload_hashes.items():
        if old.sha(path) != expected:
            raise ValueError("fixed C/affine payload was mutated")
    result = dict(status="TRAIN_CUDA_STE_VALID_CPU_exact_no_TEST", layer=layer, native_initial_prototype=25,
        updated_bank_distinct_from_native25=True, experts=experts, best_step=best_step,
        best_VALID_alpha2_NMSE=best_score, checkpoints=checkpoints, training=training,
        learning_rate=args.learning_rate, regularization=args.regularization, seed=args.seed, batch=args.batch,
        train_pair_count=len(tr[0]), valid_pair_count=len(va[0]), capture_audit=capture_audit,
        selection_path=str(selection_path), selection_sha256=old.sha(selection_path),
        source_sha256=old.sha(__file__), CPU_reference_source_sha256=old.sha(HERE / "research_20261109_learned_gu.py"),
        numeric_sha256=numeric.binary_sha256, fixed_C_and_affine_payload_sha256=fixed_payload_hashes,
        hardware=hardware, GPU_peak_memory=peak, training_arithmetic="CUDA FP32 STE surrogate; no TF32/AMP/compile",
        reported_VALID_arithmetic="CPU2threads integer-exact W8A8, F32 rescale, pinned native SwiGLU",
        native_owner_GU_reuse_allowed=False, updated_bank_GU_W8_scales_bytes=5778432,
        retaining_native_hot_GU25_extra_bytes=5778432, both_GU_banks_W8_scales_bytes=11556864,
        updated_GU_MAC_per_unique_input_row=5767168,
        native_and_updated_GU_work_can_deduplicate=False, total_seconds=time.monotonic() - started)
    old.dump(out / "selection.json", result)
    print(json.dumps(dict(output=str(out), best_step=best_step, best_VALID_nmse=best_score, GPU_peak_memory=peak)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "train", "valid", "train-target-manifest", "valid-target-manifest", "weights", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cuda:0"))
    p.add_argument("--prototype", type=int, default=25)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[0, 16, 32, 64])
    p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--regularization", type=float, default=.001)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--gpu-memory-limit-gib", type=float, default=8.)
    args = p.parse_args()
    if args.threads != 2 or args.learning_rate <= 0 or args.regularization < 0 or args.steps <= 0 or args.batch <= 0:
        raise ValueError("positive LR/steps/batch and exactly2 CPU VALID threads required")
    if 0 not in args.checkpoints:
        raise ValueError("step0 original-code baseline check is mandatory")
    with threadpool_limits(limits=args.threads, user_api="blas"):
        run(args)
