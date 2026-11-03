#!/usr/bin/env python3
"""Prospective, authored diagnostic routes using the sealed canonical W8 helper.

No fitting and no reopening an earlier holdout. These prompts are small routing
diagnostics, not a representative language benchmark or a quality experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/research-2026-11-02"
OUT = BASE / "round3-cr-ipr/second-prospective-routes"
TRAIN = BASE / "round2/w8-format-training"
BINARY = TRAIN / "build/granite4tiny-canonical-capture"
MODEL = Path("/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-q4_k_m.gguf")
PLAN = BASE / "round2/full-ffn-integer-reference/placement.tsv"
SELF = Path(__file__).resolve()

# Authored once, before running the model. No response/route based selection.
PROMPTS = [
    [
        "reasoning",
        "A theater sells adult tickets for 12 dollars and child tickets for 7 dollars. Five tickets cost 50 dollars. Determine how many of each were sold and check the result."
    ],
    [
        "reasoning",
        "Every copper key is kept in drawer A. No item in drawer A is wooden. Which conclusion follows about copper keys, and does the converse follow? Explain briefly."
    ],
    [
        "reasoning",
        "A rectangular sign has perimeter 34 centimeters. Its length is 3 centimeters greater than its width. Find its dimensions and explain your equations."
    ],
    [
        "reasoning",
        "Four parcels weigh 2, 3, 5, and 8 kilograms. Divide them between two shelves so the difference in total weight is smallest. Explain why your division works."
    ],
    [
        "code",
        "Write a JavaScript function that merges two already sorted arrays of numbers. Explain how two indices avoid sorting the result again."
    ],
    [
        "code",
        "Show a small Rust function that accepts Option<i32> and returns twice the value or zero when it is None. Explain the match arms."
    ],
    [
        "code",
        "Write a compact C function that reverses a null-terminated string in place. State how the empty string is handled."
    ],
    [
        "code",
        "Explain how an HTTP client can retry a failed GET request with exponential backoff. Give language-independent pseudocode with a fixed retry limit."
    ],
    [
        "zh",
        "請向國小學生解釋月亮為什麼會有不同的盈虧形狀，並指出月球本身是否發光。"
    ],
    [
        "zh",
        "請為一批家庭旅行照片設計容易排序的檔名規則，給出三個範例並說明如何避免同名。"
    ],
    [
        "zh",
        "請比較獨奏與合奏練習時需要注意的事情，重點放在速度、聆聽與彼此配合。"
    ],
    [
        "zh",
        "請寫一張簡短的社區公告，邀請居民在週日上午交換閒置文具，並提醒自備袋子與標示物品。"
    ],
    [
        "explanation",
        "Explain how a barcode check digit can help detect a mistyped number. Use a tiny invented arithmetic example and state what it cannot guarantee."
    ],
    [
        "explanation",
        "Describe how a lever can reduce the force needed to lift an object. Explain the roles of the fulcrum and the distances on each side."
    ],
    [
        "explanation",
        "Explain why a museum might limit the light falling on an old watercolor painting. Describe the tradeoff between visibility and preservation."
    ],
    [
        "explanation",
        "Write a short description of a clockmaker opening a workshop before sunrise. Include two concrete sounds and one ordinary tool."
    ]
]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "protocol.frozen.json").exists()
    prompts = OUT / "prompts"
    prompts.mkdir(exist_ok=True)
    jobs = []
    for index, (task, text) in enumerate(PROMPTS):
        identifier = f"prospective_{index:02d}_{task}"
        path = prompts / f"{identifier}.txt"
        assert not path.exists()
        path.write_text(text + "\n")
        jobs.append(dict(index=index, id=identifier, task=task, path=str(path),
                         prompt_sha256=sha(path), prompt_bytes=path.stat().st_size))
    for batch in range(2):
        (OUT / f"batch{batch}.jobs.tsv").write_text("".join(
            f"{job['path']}\t{job['id']}\n" for job in jobs[batch::2]))
    old = json.loads((TRAIN / "protocol.frozen.json").read_text())
    assert sha(BINARY) == old["inputs_sha256"][str(BINARY)]
    paths = [SELF, BINARY, PLAN, TRAIN / "protocol.frozen.json",
             ROOT / "scripts/research_20261103_w8_training_capture_granite4tiny.cpp",
             BASE / "round3-cr-ipr/g2-availability-full-ffn/protocol.frozen.json",
             BASE / "round3-cr-ipr/g2-availability-full-ffn/support_maps.frozen.npz",
             BASE / "round3-cr-ipr/g2-availability-full-ffn/resident_jobs.frozen.json",
             BASE / "round3-cr-ipr/g2-availability-headroom/selected_adjacency.frozen.npz",
             BASE / "round3-cr-ipr/down-replica-full-ffn/Down_job_maps.frozen.json",
             BASE / "round3-cr-ipr/down-headroom/final-run/selection.frozen.json",
             BASE / "round3-cr-ipr/down-headroom/final-run/menus.frozen.npz",
             BASE / "round3-cr-ipr/full-ffn/g8-addendum/group_maps.frozen.npz",
             *[OUT / f"batch{batch}.jobs.tsv" for batch in range(2)]]
    write(OUT / "protocol.frozen.json", dict(
        status="frozen_before_new_model_execution", utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256={str(path): sha(path) for path in paths}, jobs=jobs,
        model=str(MODEL), model_sha256_inherited=old["inputs_sha256"][str(MODEL)],
        model_stat=dict(size=MODEL.stat().st_size, mtime_ns=MODEL.stat().st_mtime_ns),
        operator="Sealed canonical fixed full-FFN W8/A8 helper; no placed execution or quality claim. Attention/router remain original GGUF.",
        sample="16 newly authored prompts, four per listed task; no old heldout opened; no content/route/response filtering or replacement. Prospective diagnostic only, not representative benchmark.",
        decode_steps=16, layers=list(range(40)), active_experts=6, experts=64,
        batch_rule="Two processes, 8 prompts each at indices batch::2, 8 CPU threads/process.",
        inference_wall_budget_seconds=1200,
        timeout_policy="Preserve partial files; fail completion and forbid full-sample claims if any job incomplete. No replacement.",
        evaluation="No refitting. First3layers paired fullFFN comparisons: both candidate/RR GU, frozen g4triples and trained/IDslot/best_single_seed g2 supports, input1/4 and GF priority. Independent matching statistics on all40layers for the fixed IDslot support. Report every fallback, regression, payload, eventP99 and aligned3layerP99. Any extension is separately frozen, never selected from these outcomes.",
        limitations="Existing conditional hardware contract still unresolved; no native AiF measurement, unchanged original-model-quality claim or new CBC theorem. These new route rows have not been paired with a placed model operator."))
    print(json.dumps(dict(status="frozen", prompts=len(jobs), source_sha256=sha(SELF),
                          protocol_sha256=sha(OUT / "protocol.frozen.json"))), flush=True)


def checked():
    protocol = json.loads((OUT / "protocol.frozen.json").read_text())
    for path, expected in protocol["inputs_sha256"].items():
        assert sha(path) == expected, path
    for job in protocol["jobs"]:
        assert sha(job["path"]) == job["prompt_sha256"]
    assert MODEL.stat().st_size == protocol["model_stat"]["size"]
    assert MODEL.stat().st_mtime_ns == protocol["model_stat"]["mtime_ns"]
    return protocol


def extract(protocol):
    routes = np.full((16, 16, 40, 6), -1, dtype=np.int32)
    prefill, summaries, completed = [], [], np.zeros(16, dtype=bool)
    expected = {f"blk.{layer}.ffn_{projection}.weight"
                for layer in range(40) for projection in
                ["gate_exps", "up_exps", "down_exps", "gate_shexp", "up_shexp", "down_shexp"]}
    ledgers = defaultdict(list)
    reports = []
    for batch in range(2):
        base = OUT / f"batch{batch}"
        report = json.loads((base / "capture_report.json").read_text())
        assert report["status"] == "complete" and report["jobs"] == 8
        assert report["full_FFN_outputs_all_tokens_finite"]
        reports.append(report)
        for line in (base / "coverage_ledger.jsonl").read_text().splitlines():
            row = json.loads(line)
            ledgers[row["job"]].append(row)
    for job in protocol["jobs"]:
        index = job["index"]
        base = OUT / f"batch{index % 2}/traces/{job['id']}.unsharded"
        summary = json.loads(Path(str(base) + ".summary.json").read_text())
        assert summary["decoded_tokens"] == 16 and len(summary["generated_token_ids"]) == 16
        n = summary["prompt_tokens"]
        pref = np.full((n, 40, 6), -1, dtype=np.int32)
        seen = set()
        routepath = Path(str(base) + ".routing.jsonl")
        for line in routepath.read_text().splitlines():
            row = json.loads(line)
            layer, phase = row["layer"], row["phase"]
            selected = row["expert_ids"]
            assert 0 <= layer < 40 and len(set(selected)) == 6 and all(0 <= e < 64 for e in selected)
            if phase == "decode":
                step = row["step"]
                assert 0 <= step < 16 and row["position"] == n + step
                assert row["token_id"] == summary["generated_token_ids"][step]
                key = (phase, step, layer)
                routes[index, step, layer] = selected
            else:
                assert phase == "prefill"
                position = row["position"]
                assert 0 <= position < n and row["token_id"] == summary["prompt_token_ids"][position]
                key = (phase, position, layer)
                pref[position, layer] = selected
            assert key not in seen
            seen.add(key)
        assert np.all(pref >= 0) and np.all(routes[index] >= 0)
        assert len(seen) == (n + 16) * 40
        totals, callbacks = Counter(), Counter()
        for row in ledgers[job["id"]]:
            assert row["weight"] in expected and row["mode"] == "unsharded"
            assert row["integer_mismatches"] == row["scaled_bit_mismatches"] == row["cross_run_output_bit_mismatches"] == 0
            tokens = row["tokens"]
            columns, rows, experts = row["weight_shape"][:3]
            active = 6 if experts == 64 else 1
            assert row["selected_experts"] == active and row["output_floats"] == tokens * active * rows
            assert row["input_shape"][0] == columns
            if row["phase"] == "decode":
                assert tokens == 1
            totals[row["phase"], row["weight"]] += tokens
            callbacks[row["phase"], row["weight"]] += 1
        for weight in expected:
            assert totals["prefill", weight] == n and totals["decode", weight] == 16
            assert callbacks["prefill", weight] == (n + 511) // 512 and callbacks["decode", weight] == 16
        prefill.append(pref)
        completed[index] = True
        summaries.append(dict(index=index, id=job["id"], task=job["task"], prompt_tokens=n,
                              generated_token_ids=summary["generated_token_ids"], route_sha256=sha(routepath)))
    assert completed.all()
    common = dict(prompt_ids=np.asarray([j["id"] for j in protocol["jobs"]]),
                  tasks=np.asarray([j["task"] for j in protocol["jobs"]]),
                  layers=np.arange(40, dtype=np.int32), completed=completed,
                  split=np.asarray("prospective_authored_diagnostic"),
                  representation=np.asarray("fixed_full_FFN_W8A8_canonical_only"))
    np.savez_compressed(OUT / "decode.routes.npz", routes=routes, **common)
    np.savez_compressed(OUT / "prefill.routes.npz", routes=np.concatenate(prefill),
                        offsets=np.concatenate([[0], np.cumsum([len(x) for x in prefill])]), **common)
    write(OUT / "report.json", dict(status="complete", prompt_count=16, decode_inputs=256,
          first3_layer_events=768, all40_layer_events=10240, jobs=summaries, capture_reports=reports,
          protocol_sha256=sha(OUT / "protocol.frozen.json"), source_sha256=sha(SELF),
          decode_sha256=sha(OUT / "decode.routes.npz"), prefill_sha256=sha(OUT / "prefill.routes.npz"),
          scope="Prospective authored diagnostics. Finite canonical fullW8 FFN only; new placed row/logit comparison and task quality not measured."))


def run():
    protocol = checked()
    assert not (OUT / "report.json").exists()
    running = []
    began = time.monotonic()
    try:
        for batch in range(2):
            base = OUT / f"batch{batch}"
            assert not base.exists()
            log = open(OUT / f"batch{batch}.log", "w")
            argv = [str(BINARY), str(MODEL), str(OUT / f"batch{batch}.jobs.tsv"),
                    "16", "8", str(PLAN), str(base)]
            proc = subprocess.Popen(argv, stdout=log, stderr=log,
                                    env={**os.environ, "OMP_NUM_THREADS": "8"}, start_new_session=True)
            running.append((proc, log))
            write(OUT / f"batch{batch}.command.json", dict(argv=argv, pid=proc.pid))
        while any(proc.poll() is None for proc, _ in running):
            if time.monotonic() - began > protocol["inference_wall_budget_seconds"]:
                for proc, _ in running:
                    if proc.poll() is None:
                        os.killpg(proc.pid, 15)
                raise RuntimeError("Prospective capture budget exceeded; partial files preserved")
            time.sleep(1)
        assert all(proc.returncode == 0 for proc, _ in running), [proc.returncode for proc, _ in running]
    finally:
        for _, log in running:
            log.close()
    extract(protocol)
    print(json.dumps(dict(status="complete", elapsed_seconds=time.monotonic() - began,
                          decode_sha256=sha(OUT / "decode.routes.npz"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run"])
    action = parser.parse_args().mode
    freeze() if action == "freeze" else run()
