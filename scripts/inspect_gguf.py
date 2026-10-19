#!/usr/bin/env python3
"""Extract GGUF metadata and exact contiguous expert tensor slices."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-hash", action="store_true", help="Extract geometry only; SHA256 is explicitly null")
    parser.add_argument("--llama-source", type=Path, default=Path(
        "/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_source / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(str(args.model))
    metadata = {key: field.contents() for key, field in reader.fields.items()
                if not key.startswith("tokenizer.") and not key.startswith("GGUF.")}
    tensors = []
    for tensor in reader.tensors:
        record = {"name": tensor.name, "ggml_type": tensor.tensor_type.name,
                  "shape_ggml": [int(value) for value in tensor.shape],
                  "n_bytes": int(tensor.n_bytes), "data_offset": int(tensor.data_offset)}
        expert = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if expert:
            record.update(layer=int(expert[1]), projection=expert[2],
                          ffn_kind="routed_expert",
                          expert_count=int(tensor.shape[2]),
                          bytes_per_expert=int(tensor.n_bytes) // int(tensor.shape[2]))
            assert len(tensor.shape) == 3
            assert tensor.n_bytes % tensor.shape[2] == 0
        else:
            shared = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight", tensor.name)
            dense = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight", tensor.name)
            if shared:
                record.update(layer=int(shared[1]), shared_projection=shared[2],
                              ffn_kind="always_active_shared_experts")
            elif dense:
                record.update(layer=int(dense[1]), dense_projection=dense[2],
                              ffn_kind="dense_ffn")
        tensors.append(record)
    digest = None
    if not args.skip_hash:
        digest = hashlib.sha256()
        with args.model.open("rb") as model_file:
            for block in iter(lambda: model_file.read(8 << 20), b""):
                digest.update(block)
    result = {"model_path": str(args.model.resolve()), "model_bytes": args.model.stat().st_size,
              "sha256": digest.hexdigest() if digest else None,
              "sha256_status": "computed" if digest else "not_computed_geometry_only",
              "metadata": metadata, "tensors": tensors,
              "tensor_shape_convention": "GGML dimension order; dimension 0 is contiguous",
              "expert_slice_formula": "data_offset + expert_id * bytes_per_expert"}
    result["routed_layer_ids"] = sorted({tensor["layer"] for tensor in tensors
                                         if tensor.get("ffn_kind") == "routed_expert"})
    result["shared_ffn_layer_ids"] = sorted({tensor["layer"] for tensor in tensors
                                             if tensor.get("ffn_kind") == "always_active_shared_experts"})
    result["dense_ffn_layer_ids"] = sorted({tensor["layer"] for tensor in tensors
                                            if tensor.get("ffn_kind") == "dense_ffn"})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    experts = [tensor for tensor in tensors if "bytes_per_expert" in tensor]
    print(json.dumps({"output": str(args.output), "sha256": result["sha256"],
                      "expert_tensors": len(experts), "first_layer_experts": experts[:3]}, indent=2))


if __name__ == "__main__":
    main()
