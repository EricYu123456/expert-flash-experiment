#!/usr/bin/env bash
# Execute inside the guest against metadata-only AiF v2, one workload at a time.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 /mnt/experiment/runs/MODEL-aif/overhead" >&2; exit 2; }
run_dir=$1
test -r /proc/nvmev/aif_v2_stats
gcc -std=gnu11 -Wall -Wextra -O2 /mnt/workspace/scripts/aif_replay.c -o /root/aif_replay
mkdir -p "$run_dir/nvmevirt"
shopt -s nullglob
workloads=("$run_dir"/*.bin)
(( ${#workloads[@]} > 0 )) || { echo "No workloads in $run_dir" >&2; exit 1; }
for workload in "${workloads[@]}"; do
    tag=$(basename "$workload" .bin)
    echo reset > /proc/nvmev/aif_stats
    /root/aif_replay /dev/nvme0n1 "$workload" > "$run_dir/nvmevirt/$tag.wall.json"
    cat /proc/nvmev/aif_stats > "$run_dir/nvmevirt/$tag.stats.txt"
    cat /proc/nvmev/aif_v2_stats > "$run_dir/nvmevirt/$tag.v2_stats.txt"
    python3 /mnt/workspace/scripts/validate_replay.py \
        "$run_dir/$tag.expected.json" "$run_dir/nvmevirt/$tag.stats.txt" \
        "$run_dir/nvmevirt/$tag.wall.json" --v2-stats "$run_dir/nvmevirt/$tag.v2_stats.txt" \
        --output "$run_dir/nvmevirt/$tag.validation.json"
done
