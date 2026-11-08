#!/usr/bin/env python3
"""TRAIN-only drop/constant controls for the original six expert replacements.

All arms deploy an actual width-one source0 transform and a 2048x1 zero W8 C,
unit row scales and F32 bias. No empty matrix or special live-runtime path is
needed. An extra alpha-squared-weighted mean is the strongest constant for the
same selected-alpha-squared local least-squares objective used by readouts.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("research_20261109_shared_feature_fit.py")
spec = importlib.util.spec_from_file_location("constant_shared_fit", SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
old = base.old

EXPERTS = {1: [14, 40], 13: [10, 56], 26: [0, 1]}


def selected_constants(target, alpha_squared):
    selected = np.flatnonzero(alpha_squared > 0)
    if not len(selected):
        raise ValueError("TRAIN contains no naturally selected rows for expert")
    y = np.asarray(target[selected], np.float64)
    a2 = np.asarray(alpha_squared[selected], np.float64)
    return {"drop_zero": np.zeros(y.shape[1], np.float32),
            "selected_mean": y.mean(axis=0).astype(np.float32),
            "selected_alpha2_mean": ((a2 @ y) / a2.sum()).astype(np.float32)}


def run(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("constant-control output exists")
    out.mkdir(parents=True)
    arms = {name: [] for name in ("drop_zero", "selected_mean", "selected_alpha2_mean")}
    expert_spec = dict(EXPERTS)
    if args.expert_spec:
        expert_spec = {}
        for spec in args.expert_spec:
            layer, values = spec.split(":")
            layer, experts = int(layer), list(map(int, values.split(",")))
            if layer in expert_spec or not experts or len(experts) != len(set(experts)):
                raise ValueError("duplicate/empty expert specification")
            expert_spec[layer] = experts
    receipt = {"status": "TRAIN_only_constant_controls", "CPU_only": True,
               "source_sha256": old.sha(__file__), "base_source_sha256": old.sha(SOURCE),
               "train": str(Path(args.train).resolve()), "train_layers": {}, "models": [],
               "arithmetic": "source0 raw x[0], center0/gain1 -> whole-vector A8; zero rowW8 C, unit W8 scales, F32 bias",
               "zero": "All output channels are exact FP32 zero; actual nonempty width-one readout.",
               "mean": "FP64 mean across TRAIN rows where native router selected expert; cast once to F32; no VALID",
               "alpha2_mean": "FP64 TRAIN selected alpha^2 weighted mean; cast once to F32; minimum constant selected-alpha^2 SSE before bias rounding",
               "train_target_contract": "canonical captured native rowW8/vectorA8 expert output",
               "runtime_extra_GU_coefficients": 0, "runtime_C_width": 1,
               "expert_spec": expert_spec,
               "No_VALID_or_TEST_inputs_or_targets_opened": True}
    for layer, experts in expert_spec.items():
        cap = old.Capture(args.train, layer)
        target_manifest = Path(args.target_root) / f"layer{layer}/targets.json" if args.target_root else None
        targets = base.Targets(cap, target_manifest, layer)
        if target_manifest:
            meta = targets.meta
            if (meta["status"] != "complete" or meta["layer"] != layer
                    or meta["capture_manifest_sha256"] != cap.identity()
                    or meta["rows"] != len(cap.get("selected_experts"))):
                raise ValueError("target/capture manifest identity mismatch")
        if not set(experts) <= set(targets.ids):
            raise ValueError("capture lacks original fixed-expert targets")
        ids = cap.get("selected_experts")
        receipt["train_layers"][str(layer)] = {"capture_manifests": cap.identity(),
                                                "input_payload_sha256": cap.payload_identity(),
                                                "rows": len(ids), "prompts": len(np.unique(cap.prompts()))}
        if target_manifest:
            receipt["train_layers"][str(layer)]["target_manifest"] = str(target_manifest.resolve())
            receipt["train_layers"][str(layer)]["target_manifest_sha256"] = old.sha(target_manifest)
        tp = out / f"layer{layer}.width1"
        base.save_transform(tp, {"sources": np.zeros(1, np.uint8), "indices": np.zeros(1, np.int32),
                                 "center": np.zeros(1, np.float32), "inv_scale": np.ones(1, np.float32)})
        np.full(1, -1, np.int32).tofile(str(tp) + ".prototypes.i32")
        for expert in experts:
            a2 = base.alpha_squared(cap, expert)
            selected = np.flatnonzero(a2 > 0)
            target = targets.get(expert)
            target_identity = {}
            if target_manifest:
                rec = targets.meta["targets"][str(expert)]
                path = Path(rec["path"])
                if not path.is_absolute():
                    path = target_manifest.parent / path
                actual_sha = old.sha(path)
                if rec.get("target_sha256") is not None and actual_sha != rec["target_sha256"]:
                    raise ValueError("native target payload hash mismatch")
                target_identity = {"native_target_path": str(path.resolve()), "native_target_sha256": actual_sha}
            constants = selected_constants(target, a2)
            yy, weights = np.asarray(target[selected], np.float64), a2[selected]
            energy = float(np.sum(weights[:, None] * yy ** 2))
            for name, bias in constants.items():
                cp = out / f"layer{layer}.{name}.expert{expert}"
                model = {"qweight": np.zeros((len(bias), 1), np.int8),
                         "wscale": np.ones(len(bias), np.float32), "bias": bias}
                base.save_model(cp, model)
                error = float(np.sum(weights[:, None] * (yy - bias.astype(np.float64)) ** 2))
                payload = {str(cp) + suffix: old.sha(str(cp) + suffix)
                           for suffix in (".i8", ".scales.f32", ".bias.f32", ".npz")}
                receipt["models"].append({"layer": layer, "expert": expert, "basis": name,
                    **target_identity,
                    "width": 1, "transform_prefix": str(tp), "readout_prefix": str(cp),
                    "payload_sha256": payload, "TRAIN_selected_event_count": int(np.sum(ids == expert)),
                    "TRAIN_selected_rows": len(selected), "TRAIN_selected_alpha2_sum": float(weights.sum()),
                    "TRAIN_selected_alpha2_effective_rows": float(weights.sum() ** 2 / np.sum(weights ** 2)),
                    "TRAIN_weighted_target_energy": energy, "TRAIN_weighted_squared_error": error,
                    "TRAIN_alpha2_NMSE": error / energy})
                arms[name].append(f"{layer}\t{expert}\t{name}\t1\t{tp}\t{cp}")
    for name, entries in arms.items():
        path = out / f"registry.{name}.tsv"
        path.write_text("\n".join(entries) + "\n")
    receipt["registry_sha256"] = {str(out / f"registry.{name}.tsv"): old.sha(out / f"registry.{name}.tsv") for name in arms}
    old.dump(out / "controls.json", receipt)
    print(str(out / "controls.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-root", help="TRAIN-only contiguous per-expert targets root; layerL/targets.json")
    p.add_argument("--expert-spec", nargs="+", help="Explicit layer:expert,expert list; default original six")
    run(p.parse_args())
