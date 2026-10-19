#!/usr/bin/env bash
# Fetch the pinned upstream module and apply the complete experimental extension.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${EXPERT_FLASH_DATA:-/home/nfs_client/ericyu/expert_flash_experiment}
source_dir="$data_dir/sources/nvmevirt"
commit=61c90f7758cbd9545b4a4727e89377bf88eab060
patch="$repo_dir/patches/nvmevirt-aif.patch"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone https://github.com/snu-csl/nvmevirt "$source_dir"
    git -C "$source_dir" checkout --detach "$commit"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$commit" ]] || {
    echo "Different NVMeVirt commit at $source_dir; preserving it" >&2; exit 1;
}
if git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Experimental patch is already applied: $source_dir"
else
    git -C "$source_dir" apply --check "$patch"
    git -C "$source_dir" apply "$patch"
    echo "Applied experimental patch: $source_dir"
fi
