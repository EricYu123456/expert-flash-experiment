#!/usr/bin/env python3
"""Compare saved llama.cpp prompt IDs with the local Hugging Face tokenizer."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--deepseek-compat", action="store_true")
    args = parser.parse_args()
    import transformers
    if args.deepseek_compat:
        from convert_deepseek_compat import configured_tokenizer
        tokenizer = configured_tokenizer(str(args.source_model), local_files_only=True,
                                          trust_remote_code=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.source_model),
            local_files_only=True, trust_remote_code=False)
    checks = []
    for path in sorted(args.runs.glob("*.prompt.txt")):
        name = path.name.removesuffix(".prompt.txt")
        expected = tokenizer.encode(path.read_text(), add_special_tokens=True)
        for variant in ("traced", "baseline"):
            summary = json.loads((args.runs / f"{name}.{variant}.summary.json").read_text())
            actual = summary["prompt_token_ids"]
            assert actual == expected, (name, variant, actual, expected)
        checks.append({"prompt": name, "prompt_tokens": len(expected),
                       "hf_llama_token_ids_identical": True})
    assert checks, "No saved prompts found"
    result = {"source_model": str(args.source_model), "transformers": transformers.__version__,
              "deepseek_configured_bos_compatibility": args.deepseek_compat, "checks": checks}
    output = args.runs / "hf-tokenization-validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
