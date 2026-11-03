#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_dir=${EXPERIMENT_STORAGE:-/home/nfs_client/ericyu/expert_flash_experiment}
model_name=${1:-deepseek}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$storage_dir/tmp"
mkdir -p "$storage_dir/models" "$TMPDIR"
case "$model_name" in
    deepseek)
        input_dir=/home/nfs_client/ymtuan2/models/DeepSeek-V2-Lite-Chat
        output_file="$storage_dir/models/deepseek-v2-lite-chat-q8_0.gguf"
        metadata_file="$storage_dir/models/deepseek-metadata.json"
        converter="$experiment_dir/scripts/convert_deepseek_compat.py"
        ;;
    mixtral)
        input_dir=/home/nfs_client/ymtuan2/models/Mixtral-8x7B-Instruct-v0.1
        output_file="$storage_dir/models/mixtral-8x7b-instruct-v0.1-q8_0.gguf"
        metadata_file="$storage_dir/models/mixtral-metadata.json"
        converter="$storage_dir/sources/llama.cpp-927dada6c/convert_hf_to_gguf.py"
        ;;
    qwen3)
        input_dir=/home/nfs_client/ymtuan2/models/Qwen3-30B-A3B-Instruct-2507-FP8
        output_file="$storage_dir/models/qwen3-30b-a3b-instruct-2507-q8_0.gguf"
        metadata_file="$storage_dir/models/qwen3-metadata.json"
        converter="$storage_dir/sources/llama.cpp-927dada6c/convert_hf_to_gguf.py"
        ;;
    *) printf 'Usage: %s deepseek|mixtral|qwen3\n' "$0" >&2; exit 2 ;;
esac
if [[ "$model_name" != deepseek && ! -s "$storage_dir/models/$model_name-source-manifest.json" ]]; then
    python3 "$experiment_dir/scripts/record_model_source.py" "$input_dir" \
        "$storage_dir/models/$model_name-source-manifest.json"
fi
if [[ -s "$output_file" && -s "$metadata_file" ]]; then
    if [[ "$model_name" == mixtral ]]; then
        python3 "$experiment_dir/scripts/fix_mixtral_tokenizer.py" \
            --source-model "$input_dir" --model "$output_file" --check-only
    fi
    printf 'Existing completed model and metadata: %s\n' "$output_file"
    exit 0
fi
python3 - "$input_dir" <<'PY'
import json
from pathlib import Path
import sys
directory = Path(sys.argv[1])
index = json.loads((directory / 'model.safetensors.index.json').read_text())
missing = [filename for filename in set(index['weight_map'].values())
           if not (directory / filename).is_file()]
if missing:
    raise RuntimeError(f'Missing source safetensors files: {missing}')
print('Source tensor index complete:', directory, flush=True)
PY
python3 "$converter" "$input_dir" --outfile "$output_file" --outtype q8_0
if [[ "$model_name" == mixtral ]]; then
    python3 "$experiment_dir/scripts/fix_mixtral_tokenizer.py" \
        --source-model "$input_dir" --model "$output_file" \
        --output "$storage_dir/models/mixtral-tokenizer-compat-validation.json"
fi
python3 "$experiment_dir/scripts/inspect_gguf.py" "$output_file" "$metadata_file"
