#!/usr/bin/env bash
# Inside the experiment guest: validate/resume full held-out v3 queue replays.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 /mnt/experiment/runs/corpus-300-v1/MODEL/replays" >&2; exit 2; }
run_dir=$1
test -r /proc/nvmev/aif_v2_stats
exec 9>/run/expert-flash-corpus-replay.lock
flock -n 9 || { echo 'Another corpus replay owns the shared counters' >&2; exit 1; }
gcc -std=gnu11 -Wall -Wextra -Werror -O2 /mnt/workspace/scripts/aif_replay.c -o /root/aif_corpus_replay
mkdir -p "$run_dir/nvmevirt"
shopt -s nullglob
workloads=("$run_dir"/*.bin)
(( ${#workloads[@]} == 4 )) || { echo 'Expected baseline and learned prefix32/tail96 workloads' >&2; exit 1; }
for workload in "${workloads[@]}"; do
    tag=$(basename "$workload" .bin)
    validation="$run_dir/nvmevirt/$tag.validation.json"
    source_tag=$tag
    alias="$run_dir/nvmevirt/$tag.alias.json"
    if [[ -s "$alias" ]]; then
        source_tag=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_tag"])' "$alias")
        cmp -s "$workload" "$run_dir/$source_tag.bin"
    elif [[ ! -s "$validation" ]]; then
        for previous in "${workloads[@]}"; do
            previous_tag=$(basename "$previous" .bin)
            [[ "$previous_tag" != "$tag" ]] || continue
            [[ -s "$run_dir/nvmevirt/$previous_tag.validation.json" ]] || continue
            [[ ! -e "$run_dir/nvmevirt/$previous_tag.alias.json" ]] || continue
            if cmp -s "$workload" "$previous"; then
                source_tag=$previous_tag
                python3 -c 'import json,sys; json.dump({"source_tag":sys.argv[1],"reason":"Byte-identical complete descriptor stream; validate both expected counters against the same real queue replay."},open(sys.argv[2],"w"),indent=2)' "$source_tag" "$alias"
                break
            fi
        done
    fi
    if [[ "$source_tag" != "$tag" ]]; then
        echo "Validating byte-identical $tag against real replay $source_tag"
        python3 /mnt/workspace/scripts/validate_replay.py \
            "$run_dir/$tag.expected.json" "$run_dir/nvmevirt/$source_tag.stats.txt" \
            "$run_dir/nvmevirt/$source_tag.wall.json" --v2-stats "$run_dir/nvmevirt/$source_tag.v2_stats.txt" \
            --output "$validation"
        continue
    fi
    if [[ -s "$validation" ]]; then
        python3 /mnt/workspace/scripts/validate_replay.py \
            "$run_dir/$tag.expected.json" "$run_dir/nvmevirt/$tag.stats.txt" \
            "$run_dir/nvmevirt/$tag.wall.json" --v2-stats "$run_dir/nvmevirt/$tag.v2_stats.txt" \
            --output "$validation"
        continue
    fi
    echo "Replaying $tag"
    echo reset > /proc/nvmev/aif_stats
    /root/aif_corpus_replay /dev/nvme0n1 "$workload" > "$run_dir/nvmevirt/$tag.wall.json"
    cat /proc/nvmev/aif_stats > "$run_dir/nvmevirt/$tag.stats.txt"
    cat /proc/nvmev/aif_v2_stats > "$run_dir/nvmevirt/$tag.v2_stats.txt"
    python3 /mnt/workspace/scripts/validate_replay.py \
        "$run_dir/$tag.expected.json" "$run_dir/nvmevirt/$tag.stats.txt" \
        "$run_dir/nvmevirt/$tag.wall.json" --v2-stats "$run_dir/nvmevirt/$tag.v2_stats.txt" \
        --output "$validation"
done
