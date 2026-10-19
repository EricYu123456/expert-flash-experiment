#!/usr/bin/env python3
"""Download pinned public GGUF files to NFS and verify SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MODELS = {
    "olmoe": {
        "repo": "allenai/OLMoE-1B-7B-0924-GGUF",
        "revision": "70df85ed7132bf21b5acbcb33817f58e7d3cb949",
        "filename": "olmoe-1b-7b-0924-q4_k_m.gguf",
        "bytes": 4213511776,
        "sha256": "f3ae25f6c7e83efb0e3632ef068b321f4d4a644bcfb6c0e01dd8c00e94e53f14",
    },
    "granite": {
        "repo": "bartowski/granite-3.1-1b-a400m-instruct-GGUF",
        "revision": "940d2e1f9f65330615c7c8e980e6c5ac73d3360c",
        "filename": "granite-3.1-1b-a400m-instruct-Q4_K_M.gguf",
        "bytes": 821847360,
        "sha256": "3a2ec1c2a78cb29d901e29bbf5162dcd03381e13803d2cbdcff838d4d08142eb",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--directory", type=Path, default=Path(
        "/home/nfs_client/ericyu/expert_flash_experiment/models"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        model = MODELS[name]
        path = args.directory / model["filename"]
        if not path.exists() or path.stat().st_size != model["bytes"]:
            url = "https://huggingface.co/{repo}/resolve/{revision}/{filename}".format(**model)
            subprocess.run(["curl", "-L", "--fail", "--retry", "4", "--continue-at", "-",
                            "--output", str(path), url], check=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if path.stat().st_size != model["bytes"] or digest.hexdigest() != model["sha256"]:
            raise RuntimeError("Model size or SHA256 mismatch: " + str(path))
        manifest = dict(model, local_path=str(path.resolve()), verified=True)
        (args.directory / (name + "-verified-download.json")).write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
