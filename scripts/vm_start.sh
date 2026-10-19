#!/usr/bin/env bash
# All VM state, including the guest root disk, remains on NFS.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${EXPERT_FLASH_DATA:-/home/nfs_client/ericyu/expert_flash_experiment}
vm_dir="$data_dir/vm"
port=${EXPERT_FLASH_SSH_PORT:-22240}
if [[ -s "$vm_dir/qemu.pid" ]] && kill -0 "$(cat "$vm_dir/qemu.pid")" 2>/dev/null; then
    echo "VM is already running (PID $(cat "$vm_dir/qemu.pid")), SSH port $port"
    exit 0
fi
[[ -r /dev/kvm && -w /dev/kvm ]] || { echo '/dev/kvm is not accessible' >&2; exit 1; }
qemu-system-x86_64 \
    -name expert-flash-nvmevirt -enable-kvm -machine q35 -cpu host \
    -smp "${EXPERT_FLASH_VCPUS:-4}" -m "${EXPERT_FLASH_RAM_MB:-8192}" \
    -drive "file=$vm_dir/root.qcow2,format=qcow2,if=virtio,cache=none" \
    -drive "file=$vm_dir/seed.iso,format=raw,if=virtio,readonly=on" \
    -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:$port-:22" \
    -device virtio-net-pci,netdev=net0 \
    -virtfs "local,path=$data_dir,mount_tag=experiment,security_model=none,id=experiment" \
    -virtfs "local,path=$repo_dir,mount_tag=workspace,security_model=none,readonly=on,id=workspace" \
    -display none -serial "file:$vm_dir/serial.log" -monitor "unix:$vm_dir/monitor.sock,server=on,wait=off" \
    -pidfile "$vm_dir/qemu.pid" -daemonize
echo "Started QEMU/KVM; SSH port $port; log $vm_dir/serial.log"
