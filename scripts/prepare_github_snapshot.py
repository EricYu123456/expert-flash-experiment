#!/usr/bin/env python3
"""Prepare a bounded Git publication snapshot; never run research workloads.

Existing tracked paths are preserved. New project source/docs are included;
generated results use the documented size/type policy. Omitted files are only
stat'ed, so this command never opens sealed evaluation bodies or raw tensors.
"""

import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/github_snapshot_manifest.json"
OMITTED = ROOT / "docs/github_snapshot_omitted.jsonl.gz"
TEXT_TYPES = {
    ".md", ".json", ".jsonl", ".csv", ".tsv", ".png", ".svg", ".pdf",
    ".txt", ".log", ".diff", ".patch", ".py", ".cpp", ".hpp", ".h",
    ".sh", ".stdout", ".stderr",
}
LOCAL_DIRECTORIES = {
    "data", "metadata", "texts", "books", "source", "sources", "upstream",
    "ibm-upstream", "artifact", "traces", "row_metrics", "predictions",
    "inputs", "targets", "prior-sources", "primary", "nasbench", "QCFlash",
    "3DFPIM-NVSim", "3DFPIM-Compiler", "3DFPIM-Quant", "screenshots",
    "prompts", "corpus", "public-data", "datasets",
}


def git_paths(*args):
    raw = subprocess.check_output(["git", "ls-files", *args, "-z"], cwd=ROOT)
    return [os.fsdecode(p) for p in raw.split(b"\0") if p]


def omission_reason(path, info):
    p = Path(path)
    if not stat.S_ISREG(info.st_mode):
        return "nonregular_file"
    if any(x.startswith(".env") or x in {".git", ".ssh", "credentials.json"}
           for x in p.parts):
        return "local_configuration"
    if p.parts[0] != "results":
        return None
    if any(x in LOCAL_DIRECTORIES or x.endswith("-primary") for x in p.parts):
        return "external_source_data_or_raw_trace"
    if p.name in {"prompts.jsonl", "prompts.json", "corpus.json",
                  "actual_token_windows.json", "selected_train_prompts.json"}:
        return "evaluation_or_training_body_cache"
    if p.suffix not in TEXT_TYPES:
        return "generated_binary_or_archive"
    cap = 2 * 1024**2 if "research-2026-11-03-no-cr" in p.parts else 256 * 1024
    if p.suffix in {".png", ".svg", ".pdf", ".md"}:
        cap = 2 * 1024**2
    if info.st_size > cap:
        return "generated_result_exceeds_size_policy"
    if p.name.endswith(".plan.json") and "research-2026-11-03-no-cr" not in p.parts:
        return "repeated_generated_placement_plan"
    return None


def main():
    tracked = set(git_paths())
    candidates = sorted(set(git_paths("--others", "--exclude-standard")))
    selected = sorted(p for p in tracked if (ROOT / p).exists())
    omitted = []
    reasons = collections.defaultdict(lambda: {"files": 0, "bytes": 0})
    for p in candidates:
        if p in {str(REPORT.relative_to(ROOT)), str(OMITTED.relative_to(ROOT))}:
            continue
        info = (ROOT / p).lstat()
        reason = omission_reason(p, info)
        if reason:
            omitted.append({"path": p, "bytes": info.st_size, "reason": reason})
            reasons[reason]["files"] += 1
            reasons[reason]["bytes"] += info.st_size
        else:
            selected.append(p)
    entries = []
    for p in sorted(selected):
        if p in {str(REPORT.relative_to(ROOT)), str(OMITTED.relative_to(ROOT))}:
            continue
        info = (ROOT / p).stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Unexpected tracked nonregular path: {p}")
        entries.append({"path": p, "bytes": info.st_size,
                        "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()})
    # mtime=0 gives a deterministic inventory without reading omitted contents.
    with OMITTED.open("wb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
            for item in omitted:
                gz.write((json.dumps(item, ensure_ascii=False) + "\n").encode())
    inventory = {"path": str(OMITTED.relative_to(ROOT)),
                 "bytes": OMITTED.stat().st_size,
                 "sha256": hashlib.sha256(OMITTED.read_bytes()).hexdigest()}
    report = {
        "status": "publication_snapshot_not_complete_raw_artifact_backup",
        "repository": "EricYu123456/expert-flash-experiment",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                text=True).strip(),
        "policy": {
            "existing_tracked_files": "preserved",
            "new_project_sources_and_docs": "included",
            "new_result_types": sorted(TEXT_TYPES),
            "new_no_cr_result_max_bytes": 2 * 1024**2,
            "other_new_result_max_bytes": 256 * 1024,
            "new_figure_and_markdown_max_bytes": 2 * 1024**2,
            "omitted_file_contents_opened": False,
            "baseline_gitignored_files_in_inventory": False,
            "local_only_directories": sorted(LOCAL_DIRECTORIES),
        },
        "included_files_excluding_manifest_and_inventory": len(entries),
        "included_bytes_excluding_manifest_and_inventory": sum(x["bytes"] for x in entries),
        "omitted_files": len(omitted),
        "omitted_bytes": sum(x["bytes"] for x in omitted),
        "omitted_by_reason": dict(reasons),
        "omitted_inventory": inventory,
        "manifest_self_hash_omitted_to_avoid_circularity": True,
        "included": entries,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # This file is local Git administration, not part of the scientific record.
    stage_paths = sorted(set(selected) | {str(REPORT.relative_to(ROOT)),
                                         str(OMITTED.relative_to(ROOT))})
    git_dir = Path(subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                          cwd=ROOT, text=True).strip())
    (git_dir / "publication_snapshot_paths.zlist").write_bytes(
        b"\0".join(os.fsencode(p) for p in stage_paths) + b"\0")
    print(json.dumps({k: report[k] for k in [
        "included_files_excluding_manifest_and_inventory",
        "included_bytes_excluding_manifest_and_inventory", "omitted_files", "omitted_bytes",
    ]}, indent=2))


if __name__ == "__main__":
    main()
