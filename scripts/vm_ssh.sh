#!/usr/bin/env bash
set -euo pipefail
data_dir=${EXPERT_FLASH_DATA:-/home/nfs_client/ericyu/expert_flash_experiment}
exec ssh -i "$data_dir/vm/id_ed25519" -p "${EXPERT_FLASH_SSH_PORT:-22240}" \
    -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
    -o "UserKnownHostsFile=$data_dir/vm/known_hosts" root@127.0.0.1 "$@"
