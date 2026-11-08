#!/usr/bin/env python3
"""Reproduce the local TEST plot or VALID-only quantization diagnosis.

This reads frozen models/results from --fit-dir. It does not refit a model,
choose a new candidate, or run the LLM. --kind diagnosis reads VALID only.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def load_fit_module(folder):
    source = folder / "research_20261108_shared_feature_fit.frozen.py"
    spec = importlib.util.spec_from_file_location("frozen_shared_fit", source)
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    return fit


def plot(folder, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((folder / "local_TEST_summary.json").read_text())
    experts = [(1, 14), (1, 40), (13, 10), (13, 56), (26, 0), (26, 1)]
    bases = [("x", "Raw x (2048)"), ("shared_output", "Shared output (2048)"),
             ("x_plus_shared_output", "x + shared output (4096)"),
             ("shared_z", "Native pre-Down z (2816)"), ("random_swiglu", "Random SwiGLU (2816)")]
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    width, positions = .155, np.arange(len(experts))
    colors = ["#2f5b99", "#d79645", "#93999e", "#347d68", "#8464a6"]
    for j, (basis, label) in enumerate(bases):
        values = [next(row["nmse"] for row in data["rows"] if row["layer"] == layer and row["expert"] == expert
                       and row["basis"] == basis and row["precision"] == "w8a8") for layer, expert in experts]
        ax.bar(positions + (j - 2) * width, values, width, label=label, color=colors[j])
    ax.set_xticks(positions, [f"Layer {layer}, expert {expert}" for layer, expert in experts])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out expert output NMSE (lower is better)")
    ax.set_title("Native shared features lose to the smaller raw-input readout in all six tests")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper center", fontsize=9)
    fig.text(.5, -.02, "1000 held-out prompts; 16000 native-state token rows per expert; VALID-selected ridge; canonical readout W8/A8. Local diagnosis, not live LLM quality.", ha="center", fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"local_TEST_nmse.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fp_predict(x, coefficient, bias):
    out = np.empty((len(x), len(bias)), np.float32)
    for a in range(0, len(x), 512):
        pred = (x[a:a + 512].astype(np.float64) @ coefficient.astype(np.float64).T).astype(np.float32)
        out[a:a + len(pred)] = np.add(pred, bias, dtype=np.float32)
    return out


def quantiles(x):
    return {f"p{q}": float(np.percentile(x, q)) for q in (0, 5, 50, 95, 99, 100)}


def diagnosis(folder, output):
    fit = load_fit_module(folder)
    if not all((folder / f"layer{layer}/selection.json").exists() for layer in (1, 13, 26)):
        raise ValueError("all three selections must already be frozen")
    selected = json.loads((folder / "layer26/selection.json").read_text())
    cap = fit.Capture(selected["capture_manifests"]["valid"], 26)
    z = np.asarray(cap.get("shared_z"), np.float32)
    qz, sz = fit.quantize_rows(z)
    reconstructed = np.multiply(qz.astype(np.float32), sz[:, None], dtype=np.float32)
    zrms = np.sqrt(np.mean(z.astype(np.float64) ** 2, axis=0))
    row_max = np.max(np.abs(z), axis=1)
    row_rms = np.sqrt(np.mean(z.astype(np.float64) ** 2, axis=1))
    report = {
        "status": "completed_VALID_only_diagnosis_no_new_candidates", "rows": len(z),
        "prompts": len(set(cap.prompts())), "selection_sha256": fit.sha(folder / "layer26/selection.json"),
        "test_payload_opened": False,
        "activation_stats": {
            "A8_reconstruction": fit.metrics(z, reconstructed),
            "A8_zero_fraction_per_row": quantiles(np.mean(qz == 0, axis=1)),
            "A8_scale_per_row": quantiles(sz),
            "amax_to_rms_per_row": quantiles(row_max / np.maximum(row_rms, 1e-30)),
            "feature_RMS": quantiles(zrms),
        }, "experts": [],
        "contract": "Same W8A8-VALID-selected float C/bias in every variant. Single-tensor quantization variants reconstruct into F32 and use the declared FP64-dot diagnostic. Attribution controls only, no new deployment candidates. Canonical W8A8 uses exact integer-dot and F32 rescale.",
    }
    truth = cap.get("expert_outputs")
    for ei, expert in enumerate(cap.expert_ids):
        ent = next(model for model in selected["models"] if model["basis"] == "shared_z"
                   and model["expert"] == expert and model["precision"] == "w8a8")
        if fit.sha(ent["path"]) != ent["sha256"]:
            raise ValueError("frozen model hash mismatch")
        with np.load(ent["path"], allow_pickle=False) as arrays:
            model = {key: arrays[key] for key in ("coefficient", "bias", "qweight", "wscale")}
        wrec = np.multiply(model["qweight"].astype(np.float32), model["wscale"][:, None], dtype=np.float32)
        preds = {
            "same_C_full_precision": fp_predict(z, model["coefficient"], model["bias"]),
            "C_weight_W8_only": fp_predict(z, wrec, model["bias"]),
            "z_activation_A8_only": fp_predict(reconstructed, model["coefficient"], model["bias"]),
            "both_quantized_dequantized_FP_diagnostic": fp_predict(reconstructed, wrec, model["bias"]),
            "canonical_W8A8": fit.predict(z, model, "w8a8"),
        }
        w = model["coefficient"].astype(np.float64)
        wrms = np.sqrt(np.mean(w ** 2, axis=1))
        count = np.bincount(np.argmax(np.abs(w), axis=1), minlength=w.shape[1])
        most = np.argsort(-count)[:10]
        sensitivity = np.mean((z.astype(np.float64) - reconstructed) ** 2, axis=0) * np.sum(w ** 2, axis=0)
        sensitive = np.argsort(-sensitivity)[:10]
        report["experts"].append({
            "expert": expert, "W8A8_selected_lambda": ent["lambda"],
            "versus_native_target": {name: fit.metrics(truth[:, ei], pred) for name, pred in preds.items()},
            "quantization_error_versus_same_C_FP": {name: fit.metrics(preds["same_C_full_precision"], pred)
                                                   for name, pred in preds.items() if name != "same_C_full_precision"},
            "weight_stats": {
                "W8_reconstruction": fit.metrics(model["coefficient"], wrec),
                "W8_zero_fraction_per_output_row": quantiles(np.mean(model["qweight"] == 0, axis=1)),
                "W8_scale_per_output_row": quantiles(model["wscale"]),
                "amax_to_rms_per_output_row": quantiles(np.max(np.abs(w), axis=1) / np.maximum(wrms, 1e-300)),
                "most_often_max_weight_feature": [{"feature": int(j), "output_rows": int(count[j]), "VALID_feature_RMS": float(zrms[j])} for j in most],
                "largest_diagonal_activation_error_sensitivity": [{"feature": int(j), "score": float(sensitivity[j]), "VALID_feature_RMS": float(zrms[j])} for j in sensitive],
            },
        })
    fit.dump(output / "layer26.quantization_VALID_diagnosis.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("plot", "diagnosis"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads, user_api="blas"):
        (plot if args.kind == "plot" else diagnosis)(args.fit_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
