#!/usr/bin/env python3
"""Run deterministic CPU MoE decode and confirm tracing preserves output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PROMPTS = {
    "explanation": "Flash memory stores data in pages and blocks. A mixture of experts language model selects several experts for each token. Explain how distributing expert weights across memory chips can affect parallel execution.\n",
    "programming": "Write a Python function that counts the occurrences of each word in a string, then explain its time complexity.\n",
    "mathematics": "Solve this problem step by step: a train travels 120 kilometers in 90 minutes. What is its average speed in kilometers per hour?\n",
}


def validate_trace(path, metadata, steps):
    architecture = metadata["metadata"]["general.architecture"]
    expert_count = metadata["metadata"][architecture + ".expert_count"]
    active = metadata["metadata"][architecture + ".expert_used_count"]
    layers = metadata["metadata"][architecture + ".block_count"]
    routed_layer_ids = {tensor["layer"] for tensor in metadata["tensors"]
                        if "projection" in tensor and "expert_count" in tensor}
    assert routed_layer_ids, "No routed expert layers found in GGUF metadata"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == steps * len(routed_layer_ids), (len(records), steps * len(routed_layer_ids))
    seen = set()
    for event in records:
        key = (event["token_index"], event["layer"])
        assert key not in seen
        seen.add(key)
        assert 0 <= event["token_index"] < steps
        assert event["layer"] in routed_layer_ids
        assert len(event["expert_ids"]) == active
        assert len(set(event["expert_ids"])) == active
        assert all(0 <= index < expert_count for index in event["expert_ids"])
    return {"routing_records": len(records), "decode_tokens": steps,
            "layers": layers, "moe_layers": len(routed_layer_ids),
            "routed_layer_ids": sorted(routed_layer_ids),
            "experts": expert_count, "active_experts": active}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path(
        "/home/nfs_client/ericyu/expert_flash_experiment/build/llama-route-trace/llama-route-trace"))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    results = []
    for name, prompt in PROMPTS.items():
        prompt_path = args.output_dir / (name + ".prompt.txt")
        prompt_path.write_text(prompt)
        for traced in (True, False):
            label = name + (".traced" if traced else ".baseline")
            command = [str(args.binary), str(args.model), str(prompt_path),
                       str(args.output_dir / (name + ".routing.jsonl")) if traced else "-",
                       str(args.output_dir / (label + ".output.txt")), str(args.steps),
                       str(args.threads), str(args.output_dir / (label + ".summary.json"))]
            (args.output_dir / (label + ".command.json")).write_text(json.dumps(command, indent=2) + "\n")
            print("Running", label, flush=True)
            with (args.output_dir / (label + ".stderr.log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=log, check=True)
        traced_output = (args.output_dir / (name + ".traced.output.txt")).read_bytes()
        baseline_output = (args.output_dir / (name + ".baseline.output.txt")).read_bytes()
        assert traced_output == baseline_output, name + ": tracing changed greedy output"
        result = validate_trace(args.output_dir / (name + ".routing.jsonl"), metadata, args.steps)
        result.update(prompt=name, baseline_output_identical=True,
                      output_sha256=hashlib.sha256(traced_output).hexdigest())
        results.append(result)
        (args.output_dir / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
