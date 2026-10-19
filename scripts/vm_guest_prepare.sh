#!/usr/bin/env bash
# Run inside guest after cloud-init completes and after reboot enables memmap.
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Run as root inside the guest' >&2; exit 1; }
if ! grep -Fq 'memmap=2G$4G' /proc/cmdline; then
    echo 'Memory reservation is not active. Wait for cloud-init and reboot guest.' >&2
    exit 1
fi
mountpoint -q /mnt/experiment || mount /mnt/experiment
mountpoint -q /mnt/workspace || mount /mnt/workspace
test -d "/lib/modules/$(uname -r)/build" || apt-get install -y "linux-headers-$(uname -r)"
src=/mnt/experiment/sources/nvmevirt
build=/mnt/experiment/build/nvmevirt
mkdir -p "$build"
# A local build directory on the guest disk avoids slow 9p metadata operations;
# that disk's backing qcow2 and the copied outputs are both on the NFS share.
mkdir -p /root/nvmevirt-build
tar -C "$src" --exclude=.git -cf - . | tar -C /root/nvmevirt-build -xf -
make -C /root/nvmevirt-build -j2
# 9p over NFS reports a large preferred block size to coreutils cp on this host.
dd if=/root/nvmevirt-build/nvmev.ko of="$build/nvmev.ko" bs=1M status=none
modprobe nvme
if ! lsmod | awk '{print $1}' | grep -qx nvmev; then
    insmod "$build/nvmev.ko" memmap_start=4G memmap_size=2G cpus=2,3
fi
udevadm settle
nvme list
cat /proc/cmdline
grep -i reserved /proc/iomem
