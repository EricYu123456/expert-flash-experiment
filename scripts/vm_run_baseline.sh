#!/usr/bin/env bash
# Writes only 64 MiB to the disposable NVMeVirt namespace inside the guest.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${EXPERT_FLASH_DATA:-/home/nfs_client/ericyu/expert_flash_experiment}
out="$data_dir/runs/vm_baseline"
mkdir -p "$out"
export EXPERT_FLASH_DATA="$data_dir"
python3 - "$out/raw_requests.csv" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).write_text('offset,length\n' + ''.join(
    f'{offset},131072\n' for offset in range(0, 64 * 1024 * 1024, 131072)))
PY
ssh_guest=("$repo_dir/scripts/vm_ssh.sh")
"${ssh_guest[@]}" 'mount -a; test "$(cat /sys/class/nvme/nvme0/model | xargs)" = CSL_Virt_MN_01; gcc -O2 -Wall -Wextra -Werror -pthread /mnt/workspace/scripts/vm_raw_replay.c -o /root/vm_raw_replay'
"${ssh_guest[@]}" 'uname -a; cat /proc/cmdline; cat /proc/iomem; lspci -nn; nvme list; cat /proc/nvmev/aif_stats; cat /proc/nvmevirt_flash_stats' > "$out/preflight.log"
"${ssh_guest[@]}" 'fio --name=precondition --filename=/dev/nvme0n1 --rw=write --bs=128k --size=64m --ioengine=libaio --iodepth=16 --direct=1 --numjobs=1 --output-format=json --output=/root/fio-precondition.json'
"${ssh_guest[@]}" 'cat /root/fio-precondition.json' > "$out/fio-precondition.json"
"${ssh_guest[@]}" 'cat /proc/nvmevirt_flash_stats' > "$out/flash_before.csv"
"${ssh_guest[@]}" '/root/vm_raw_replay /dev/nvme0n1 /mnt/experiment/runs/vm_baseline/raw_requests.csv 1 4' > "$out/raw_workers1.json"
"${ssh_guest[@]}" 'cat /proc/nvmevirt_flash_stats' > "$out/flash_workers1.csv"
"${ssh_guest[@]}" '/root/vm_raw_replay /dev/nvme0n1 /mnt/experiment/runs/vm_baseline/raw_requests.csv 4 4' > "$out/raw_workers4.json"
"${ssh_guest[@]}" 'cat /proc/nvmevirt_flash_stats' > "$out/flash_workers4.csv"
python3 - "$out" <<'PY'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1])
def load(name):
    with (out / name).open() as stream:
        return list(csv.DictReader(stream))
before, one, four = map(load, ('flash_before.csv', 'flash_workers1.csv', 'flash_workers4.csv'))
results = {}
for tag, first, last in (('workers1', before, one), ('workers4', one, four)):
    deltas = [{key: (int(b[key]) - int(a[key]) if key in
               ('read_commands', 'read_bytes', 'nand_read_ns') else int(b[key]))
               for key in b} for a, b in zip(first, last)]
    results[tag] = {'per_lun': deltas,
        'total_read_bytes': sum(row['read_bytes'] for row in deltas),
        'total_read_commands': sum(row['read_commands'] for row in deltas),
        'active_luns': sum(row['read_commands'] > 0 for row in deltas)}
    client = json.loads((out / f'raw_{tag}.json').read_text())
    if client['failed'] or results[tag]['total_read_bytes'] != client['bytes']:
        raise SystemExit(f'Conventional NAND/client byte mismatch: {tag}')
(out / 'flash_delta.json').write_text(json.dumps(results, indent=2) + '\n')
PY
echo "Baseline results: $out"
