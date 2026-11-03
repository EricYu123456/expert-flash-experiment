#!/usr/bin/env python3
"""Summarize sealed TRAIN results; never select layouts or open new routes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02/round3-cr-ipr"
INPUT = BASE / "down-replica-full-ffn/TRAIN512/report.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    document = json.loads(INPUT.read_text())
    by_name = {r["name"]: r for r in document["records"]}
    names = [
        "candidate.original_g4_shared16.input1.GF",
        "candidate.onecopy_shared4.input1.GF",
        "candidate.triple_private_shared4.input1.GF",
        "RR.triple_private_shared4.input1.GF",
        "g8_fixed.triple_private_shared4.input1.GF",
    ]
    rows = []
    for index, name in enumerate(names):
        record = by_name[name]
        rows.append({
            "name": name, "mean_us": record["cost_ns"]["mean"] / 1000,
            "event_p99_us": record["cost_ns"]["p99"] / 1000,
            "three_layer_sum_p99_us": record["three_layer_sum_ns"]["p99"] / 1000,
            "useful_payload_MiB_per_layer": 148.5 if index < 2 else 244.5,
            "scope": "existing TRAIN, 32 prompts, 512 aligned inputs, three MoE layers; conditional service model",
        })
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    labels = [["g4 / shared g16", "Shared partition", "Down triples"],
              ["Compensated GU", "RR GU", "Fixed-source g8 GU"]]
    for panel, (axis, subset) in enumerate(zip(axes, [rows[:3], rows[2:]])):
        x = np.arange(3)
        means = [r["mean_us"] for r in subset]
        tails = [r["event_p99_us"] for r in subset]
        first = axis.bar(x - .18, means, width=.34, color="#3977a8", label="Mean")
        second = axis.bar(x + .18, tails, width=.34, color="#d79740", label="Event P99")
        axis.bar_label(first, fmt="%.1f", fontsize=9, padding=3)
        axis.bar_label(second, fmt="%.1f", fontsize=9, padding=3)
        axis.set_xticks(x, labels[panel], rotation=12, ha="right")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.set_ylim(0, 455)
    axes[0].set_ylabel("Conditional FFN service time (microseconds)")
    axes[0].set_title("Standard controls account for the main gain")
    axes[1].set_title("Same Down triples granted to every GU control")
    axes[0].legend(frameon=False, loc="upper right")
    figure.suptitle("G4 W8/A8: existing TRAIN, first 3 MoE layers, input1 / GF", fontsize=12)
    figure.text(.04, .02,
        "Down triples add 96 MiB/layer (148.5 to 244.5 MiB). Nominal copy500 / CR9.7us model; not measured device or full inference.\n"
        "512 aligned inputs from 32 prompts; 1,536 layer-events. P99 is pooled layer-event latency, not a queueing SLO.",
        fontsize=9)
    figure.tight_layout(rect=(0, .15, 1, .93))
    outputs = [BASE / "overview.png", BASE / "overview.pdf"]
    for output in outputs:
        figure.savefig(output)
    plt.close(figure)
    summary = {
        "status": "descriptive_summary_of_sealed_results",
        "inputs_sha256": {str(INPUT): sha(INPUT), str(Path(__file__).resolve()): sha(__file__)},
        "rows": rows,
        "candidate_vs_same_triples_RR_mean_reduction_pct": 100 * (1 - rows[2]["mean_us"] / rows[3]["mean_us"]),
        "candidate_triples_vs_onecopy_partition_mean_reduction_pct": 100 * (1 - rows[2]["mean_us"] / rows[1]["mean_us"]),
        "candidate_triples_vs_onecopy_partition_event_p99_reduction_pct": 100 * (1 - rows[2]["event_p99_us"] / rows[1]["event_p99_us"]),
        "outputs_sha256": {str(p): sha(p) for p in outputs},
        "analysis_scope": "Posthoc descriptive summary only; no new layouts, test data, hyperparameters or statistical independence claims.",
    }
    (BASE / "overview.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "rows": len(rows), "outputs": [str(p) for p in outputs]}))


if __name__ == "__main__":
    main()
