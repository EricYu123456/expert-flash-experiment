#!/usr/bin/env python3
"""Apply the independent scalar/DAG audit to frozen asymmetric Down traces.

This reuses the independent implementation, not the experiment executor.
Only saved representative traces are audited; all TRAIN queries are not replayed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02/round3-cr-ipr"
INPUT = BASE / "full-ffn/asymmetric-down"
OUT = BASE / "asymmetric-down-independent"
AUDITOR = ROOT / "scripts/research_20261102_cr_ipr_full_ffn_audit.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    began = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "report.json").exists(), "preserve sealed output"
    spec = importlib.util.spec_from_file_location("independent_cr_ffn_audit", AUDITOR)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    files = sorted((INPUT / "TRAIN512").glob("trace.*.json"))
    assert len(files) == 12
    paths = [Path(__file__).resolve(), AUDITOR, INPUT / "protocol.frozen.json",
             INPUT / "selection.frozen.json", INPUT / "TRAIN512/report.json",
             INPUT / "selected_oneblock_capacity_audit.json", *files]
    hashes = {str(p): sha(p) for p in paths}
    dump(OUT / "execution.frozen.json", {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_and_input_sha256": hashes,
        "scope": "All twelve saved asymmetric winner traces; independent scalar scheduling and semantic checks. No new routes, fits, or timing parameters.",
    })
    records = []
    for path in files:
        record = json.loads(path.read_text())
        times, cost = auditor.scalar_execute(record["nodes"], record["service"]["down_first"])
        expected = np.asarray([(n["start_ns"], n["end_ns"]) for n in record["nodes"]], dtype=np.int64)
        assert np.array_equal(times, expected), path
        assert cost == record["cost_ns"], path
        summary = auditor.semantic_audit(record, times)
        records.append({"path": str(path), "sha256": hashes[str(path)], "cost_ns": cost, **summary})
    assert hashes == {str(p): sha(p) for p in paths}, "sealed inputs changed"
    result = {
        "status": "pass", "traces": len(records),
        "total_nodes": sum(r["nodes"] for r in records),
        "total_edges": sum(r["edges"] for r in records),
        "total_jobs": sum(r["jobs"] for r in records),
        "all_node_start_end_times_exact": True,
        "FIFO128_valid_traces": sum(r["FIFO128_valid"] for r in records),
        "peak_FIFO_bytes": max(r["peak_FIFO_bytes"] for r in records),
        "records": records, "inputs_sha256": hashes,
        "elapsed_seconds": time.monotonic() - began,
        "limitations": [
            "Representative saved traces only, not every TRAIN query.",
            "Conditional service-model verification, not physical device measurements or model quality.",
            "Uses the previously validated independent scalar implementation and semantics checker; it is not a new independent implementation.",
            "Capacity certificate is recorded as an input; this script does not reconstruct weight bytes or physical addresses.",
            "Programmable source order, pre-ECC inactive discard, PB overwrite/copy timing and actual FIFO size remain unverified capabilities.",
        ],
    }
    dump(OUT / "report.json", result)
    (OUT / "report.md").write_text(
        f"All {len(records)} saved asymmetric winner traces pass the independent scalar/DAG audit. "
        f"Every start/end and makespan matches exactly across {result['total_nodes']:,} nodes, "
        f"{result['total_edges']:,} edges and {result['total_jobs']:,} jobs. "
        f"All traces fit 128 B in the declared continuous-drain output model; peak is {result['peak_FIFO_bytes']} B.\n\n"
        "Checks cover full GU arrival before private whole-vector activation and upload, chip ownership, "
        "resource exclusion, page-output causal timing, all seven Down outputs before combine, and byte conservation.\n\n"
        + "\n".join(result["limitations"]) + "\n"
    )
    print(json.dumps({k: result[k] for k in ["status", "traces", "total_nodes", "total_edges", "total_jobs", "peak_FIFO_bytes", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
