#!/usr/bin/env python3
"""Honor DeepSeek's configured BOS policy with Transformers' generic backend.

Transformers 5.8 loads this legacy LlamaTokenizerFast config as TokenizersBackend
and omits its add_bos_token policy. Restore that declared policy in memory,
without changing tokenizer files or the pinned converter's recognized hashes.
"""
import json
from pathlib import Path
import runpy
import sys

from tokenizers import processors
from transformers import AutoTokenizer

original_loader = AutoTokenizer.from_pretrained


def configured_tokenizer(*args, **kwargs):
    tokenizer = original_loader(*args, **kwargs)
    directory = Path(args[0])
    model_config = json.loads((directory / "config.json").read_text())
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
    if model_config.get("model_type") != "deepseek_v2":
        raise ValueError("This compatibility adapter only supports deepseek_v2")
    if tokenizer_config.get("add_bos_token"):
        probe = "Hello, flash memory."
        raw = tokenizer.encode(probe, add_special_tokens=False)
        expected = [tokenizer.bos_token_id] + raw
        if tokenizer.encode(probe, add_special_tokens=True) != expected:
            template = processors.TemplateProcessing(
                single=f"{tokenizer.bos_token} $A",
                pair=f"{tokenizer.bos_token} $A {tokenizer.bos_token} $B:1",
                special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id)],
            )
            tokenizer.backend_tokenizer.post_processor = processors.Sequence([
                tokenizer.backend_tokenizer.post_processor, template])
            assert tokenizer.encode(probe, add_special_tokens=True) == expected
            assert tokenizer.encode(probe, add_special_tokens=False) == raw
            print("Restored configured add_bos_token=True in memory for legacy DeepSeek tokenizer", file=sys.stderr)
    return tokenizer


AutoTokenizer.from_pretrained = configured_tokenizer

if __name__ == "__main__":
    converter = Path("/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c/convert_hf_to_gguf.py")
    sys.argv[0] = str(converter)
    runpy.run_path(str(converter), run_name="__main__")
