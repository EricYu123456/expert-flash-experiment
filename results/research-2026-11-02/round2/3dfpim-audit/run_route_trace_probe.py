#!/usr/bin/env python3
"""Reproduce a trace-expression probe, not numerical or cycle simulation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
COMPILER = UPSTREAM / "3DFPIM-Compiler"
EXPECTED_COMMIT = "bd17a3a3c7a6a2ef98beb7a5379e2110f94df706"


def checked(args: list[str], cwd: Path, env: dict | None = None) -> str:
    p = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{args}: exit {p.returncode}\n{p.stdout}{p.stderr}")
    return p.stdout + p.stderr


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    commit = checked(["git", "rev-parse", "HEAD"], UPSTREAM).strip()
    assert commit == EXPECTED_COMMIT, commit
    assert checked(["git", "status", "--porcelain", "--untracked-files=no"], UPSTREAM) == ""
    build_log = checked(["make", "-j2"], COMPILER / "src")
    (ROOT / "compiler_build.log").write_text(build_log)
    checked([
        "g++", "-std=c++11", "-O0", "-g",
        "-I" + str(COMPILER / "include"),
        str(ROOT / "route_trace_probe.cpp"),
        "-L" + str(COMPILER / "src"), "-l3dfpim",
        "-o", str(ROOT / "route_trace_probe"),
    ], ROOT)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(COMPILER / "src")
    runs = []
    invariant = {}
    for part in ("row", "col", "random"):
        for input_mode in ("shared", "private"):
            for route in (1, 2, 3):
                case = f"{part}-{input_mode}-r{route}"
                out = ROOT / "route-traces" / case
                out.mkdir(parents=True, exist_ok=True)
                log = checked([
                    str(ROOT / "route_trace_probe"), "toy", str(route), input_mode, part
                ], out, env)
                (out / "compile.log").write_text(log)
                mapping = {}
                active = []
                instructions = []
                trace_hashes = {}
                for path in sorted(out.glob("*.3dfpim")):
                    trace = path.read_text()
                    trace_hashes[path.name] = sha256(path)
                    m = re.search(r"-tile(\d+)-core(\d+)\.3dfpim$", path.name)
                    if m is None:
                        continue
                    tile, core = map(int, m.groups())
                    lines = [x for x in trace.splitlines() if x.startswith("mvm(")]
                    if lines:
                        active.append([tile, core])
                    for line in lines:
                        match = re.search(r"expert(\d+)\[(\d+)\]\[(\d+)\]", line)
                        assert match, line
                        expert, rowtile, coltile = map(int, match.groups())
                        depth = int(re.search(r"depth = (\d+)", line).group(1))
                        key = f"expert{expert}[{rowtile}][{coltile}]"
                        assignment = [tile, core, coltile]
                        assert mapping.setdefault(key, assignment) == assignment
                        fullkey = (part, key)
                        assert invariant.setdefault(fullkey, assignment) == assignment
                        assert route & (1 << expert), line
                        instructions.append({"expert": expert, "row_tile": rowtile,
                                             "column_tile": coltile, "depth": depth,
                                             "tile": tile, "core": core})
                for expert in range(2):
                    observed = [x for x in instructions if x["expert"] == expert]
                    if route & (1 << expert):
                        assert len(observed) == 16
                        for rowtile in range(4):
                            depths = sorted(x["depth"] for x in observed if x["row_tile"] == rowtile)
                            assert depths == [0, 1, 2, 3]
                    else:
                        assert not observed
                expected_cores = 8 if route == 3 else 4
                assert len(active) == expected_cores
                report = (out / "toy-report.out").read_text()
                reported = {k: int(v) for k, v in re.findall(r"^# ([^=\n]+?) = (\d+)$", report, re.M)}
                runs.append({"case": case, "partition": part, "input_mode": input_mode,
                             "route_mask": route, "exit_code": 0, "mapping": mapping,
                             "active_cores": active, "mvm_instruction_count": len(instructions),
                             "report_labels_and_values": reported, "trace_sha256": trace_hashes})
    both = {(r["partition"], r["input_mode"]): r for r in runs if r["route_mask"] == 3}
    for part in ("row", "col", "random"):
        assert both[(part, "shared")]["report_labels_and_values"]["send bytes"] == 1536
        assert both[(part, "private")]["report_labels_and_values"]["send bytes"] == 2048
        assert both[(part, "shared")]["report_labels_and_values"]["load bytes"] == 4096
    result = {
        "schema_version": 1, "upstream_commit": commit,
        "probe": "fixed two-matrix inventory, route-conditioned static operations",
        "weight_shape_each": [512, 512], "experts": 2,
        "original_core_parameters_unchanged": True, "passes": len(runs),
        "interpretation": {
            "proved": "Unmodified row/column/random compiler emits selected-only MVMs with invariant matrix tile/core/depth assignment across route masks and shared/private inputs.",
            "movement_units": "Report labels say bytes; implementation increments by vector length. Record original labels without asserting an undisclosed element-to-byte ABI.",
            "not_proved": ["numerical MVM correctness", "weight programming binary",
                           "native host runtime and dispatch overhead", "cycle or energy timing",
                           "KaHIP fixed-layout trace generation", "multiple expert matrices in one MVMU depth range"],
        },
        "source_sha256": {p.name: sha256(p) for p in [ROOT / "route_trace_probe.cpp", Path(__file__)]},
        "runs": runs,
    }
    (ROOT / "route_trace_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passes": len(runs), "fixed_mapping_verified": True,
                      "output": str(ROOT / "route_trace_probe.json")}))


if __name__ == "__main__":
    main()
