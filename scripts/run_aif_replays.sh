#!/usr/bin/env bash
# Run inside the QEMU guest. All output files are on the NFS-backed shared tree.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: $0 /mnt/experiment/runs/{granite,olmoe}-aif" >&2
    exit 2
fi
run_dir=$1
test -r /proc/nvmev/aif_stats
gcc -std=gnu11 -Wall -Wextra -O2 /mnt/workspace/scripts/aif_replay.c -o /root/aif_replay
mkdir -p "$run_dir/nvmevirt"
for schedule in ready serial; do
    for g in 1 2 4 8 16; do
        tag=g${g}_${schedule}
        test -s "$run_dir/$tag.bin"
        echo reset > /proc/nvmev/aif_stats
        /root/aif_replay /dev/nvme0n1 "$run_dir/$tag.bin" > "$run_dir/nvmevirt/$tag.wall.json"
        cat /proc/nvmev/aif_stats > "$run_dir/nvmevirt/$tag.stats.txt"
        python3 /mnt/workspace/scripts/validate_replay.py \
            "$run_dir/$tag.expected.json" "$run_dir/nvmevirt/$tag.stats.txt" \
            "$run_dir/nvmevirt/$tag.wall.json" --output "$run_dir/nvmevirt/$tag.validation.json"
    done
done
