#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$script_dir/vm_ssh.sh" poweroff || true
echo 'Guest shutdown requested; QEMU exits after the guest powers off.'
