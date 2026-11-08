#!/usr/bin/env python3
"""Freeze new sources and watch each layer's complete TRAIN/VALID targets.

Runs rank -> full fit -> C++ numeric audit independently per ready layer.
No TEST files or quality runs are part of this launcher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def ready(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except json.JSONDecodeError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data, repo = args.data_dir.resolve(), args.repo.resolve()
    formal = data / "formal"
    formal.mkdir(parents=True, exist_ok=True)
    execution = formal / "execution"
    if execution.exists():
        raise FileExistsError("formal execution already exists; no implicit rerun")
    execution.mkdir()
    source_dir = formal / "source-v1"
    source_dir.mkdir()
    source_files = ["research_20261108_shared_feature_fit.py", "research_20261109_shared_feature_fit.py",
                    "research_20261109_shared_feature_fit_proto.py", "research_20261109_shared_feature_fit_joint.py",
                    "research_20261109_shared_feature_fit_numeric_check.py", Path(__file__).name]
    identities = {}
    for name in source_files:
        original = repo / "scripts" / name
        dest = source_dir / name
        shutil.copy2(original, dest)
        identities[name] = {"source": str(original), "sha256": sha(dest)}
    dump(execution / "source_freeze.json", {"sources": identities, "source_dir": str(source_dir),
                                           "TRAIN_VALID_only": True,
                                           "rank_fit_rows": 4096, "rank_score_rows": 1024,
                                           "sketch_width": 128, "top_cross_prototypes": 2,
                                           "formal_lambdas": [1, 3, 10], "formal_betas": [0, .99]})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
    states = {layer: {"state": "waiting_targets"} for layer in (1, 13, 26)}
    slots = {"24-31": None, "56-63": None}
    third_enabled = False
    last_print = 0

    def launch(layer, cpus, phase, argv):
        log_path = execution / f"layer{layer}.{phase}.log"
        log = log_path.open("w")
        command = ["taskset", "-c", cpus, "python3", *map(str, argv)]
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {"layer": layer, "phase": phase, "cpus": cpus, "pid": proc.pid,
                  "argv": command, "started_unix": time.time(), "CUDA_VISIBLE_DEVICES": "",
                  "OMP_NUM_THREADS": 8, "OPENBLAS_NUM_THREADS": 8}
        dump(execution / f"layer{layer}.{phase}.launch.json", record)
        states[layer] = {"state": phase, "cpus": cpus, "process": proc, "log": log,
                         "record": record, "started": time.monotonic()}
        slots[cpus] = layer
        print(json.dumps({"launched_layer": layer, "phase": phase, "cpus": cpus, "pid": proc.pid}), flush=True)

    while any(s["state"] not in ("complete", "failed") for s in states.values()):
        if not third_enabled and all(ready(data / f"targets64/validation/layer{layer}/targets.json") for layer in states):
            slots["32-39"] = None
            third_enabled = True
            print(json.dumps({"third_CPU_group_enabled": "32-39", "all_VALID_target_workers_complete": True}), flush=True)
        for layer, state in list(states.items()):
            if "process" not in state:
                continue
            code = state["process"].poll()
            if code is None:
                continue
            state["log"].close()
            phase, cpus = state["state"], state["cpus"]
            dump(execution / f"layer{layer}.{phase}.exit.json", {"layer": layer, "phase": phase, "exit_code": code,
                                                               "seconds": time.monotonic() - state["started"]})
            print(json.dumps({"layer": layer, "completed_phase": phase, "exit_code": code}), flush=True)
            if code:
                states[layer] = {"state": "failed", "phase": phase, "exit_code": code}
                slots[cpus] = None
                continue
            common = ["--train", data / "captures64/train", "--train-target-manifest", data / f"targets64/train/layer{layer}/targets.json", "--layer", layer]
            if phase == "rank":
                argv = [source_dir / "research_20261109_shared_feature_fit_proto.py", "fit", *common,
                        "--valid", data / "captures64/validation", "--valid-target-manifest", data / f"targets64/validation/layer{layer}/targets.json",
                        "--ranking", formal / f"rank-layer{layer}/ranking.json", "--output", formal / f"layer{layer}", "--threads", 8]
                launch(layer, cpus, "fit", argv)
            elif phase == "fit":
                output = formal / f"layer{layer}"
                registry = output / "registry.numeric_audit.tsv"
                registry.write_text((output / "registry.raw_x_eq.tsv").read_text() + (output / "registry.best_cross_prototype.tsv").read_text())
                argv = [source_dir / "research_20261109_shared_feature_fit_numeric_check.py", "--registry", registry,
                        "--valid-capture", data / "captures64/validation", "--valid-target-manifest", data / f"targets64/validation/layer{layer}/targets.json",
                        "--numeric-so", data / "build/shared-feature-numeric.so", "--output", output / "VALID_cpp_numeric_audit.json", "--threads", 8]
                launch(layer, cpus, "numeric", argv)
            else:
                output = formal / f"layer{layer}"
                dump(execution / f"layer{layer}.complete.json", {"status": "complete", "layer": layer,
                                                               "selection_sha256": sha(output / "selection.json"),
                                                               "numeric_audit_sha256": sha(output / "VALID_cpp_numeric_audit.json")})
                states[layer] = {"state": "complete"}
                slots[cpus] = None
        for layer in states:
            if states[layer]["state"] != "waiting_targets":
                continue
            targets = [data / f"targets64/{split}/layer{layer}/targets.json" for split in ("train", "validation")]
            free = next((cpus for cpus, occupant in slots.items() if occupant is None), None)
            if free is not None and all(ready(p) for p in targets):
                argv = [source_dir / "research_20261109_shared_feature_fit_proto.py", "rank",
                        "--train", data / "captures64/train", "--train-target-manifest", targets[0], "--layer", layer,
                        "--output", formal / f"rank-layer{layer}", "--threads", 8]
                launch(layer, free, "rank", argv)
        snapshot = {str(layer): {key: value for key, value in state.items() if key not in ("process", "log", "started")}
                    for layer, state in states.items()}
        dump(execution / "progress.json", snapshot)
        if time.monotonic() - last_print > 60:
            print(json.dumps({"formal_progress": {str(l): s["state"] for l, s in states.items()}}), flush=True)
            last_print = time.monotonic()
        if any(s["state"] not in ("complete", "failed") for s in states.values()):
            time.sleep(5)
    dump(execution / "completion.json", {"status": "complete" if all(s["state"] == "complete" for s in states.values()) else "failed", "layers": states})


if __name__ == "__main__":
    main()
