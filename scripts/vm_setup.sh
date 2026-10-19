#!/usr/bin/env bash
# Prepare a normal unprivileged QEMU/KVM VM; the NVMeVirt module runs in guest root.
set -euo pipefail
data_dir=${EXPERT_FLASH_DATA:-/home/nfs_client/ericyu/expert_flash_experiment}
export EXPERT_FLASH_DATA="$data_dir"
vm_dir="$data_dir/vm"
mkdir -p "$vm_dir/seed"
if [[ -s "$vm_dir/qemu.pid" ]] && kill -0 "$(cat "$vm_dir/qemu.pid")" 2>/dev/null; then
    echo 'The VM is running. Shut it down before regenerating its seed disk.' >&2
    exit 1
fi
for tool in curl qemu-img genisoimage ssh-keygen python3; do
    command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 1; }
done
base_url=${EXPERT_FLASH_IMAGE_URL:-https://cloud-images.ubuntu.com/jammy/current}
image_name=jammy-server-cloudimg-amd64.img
if [[ ! -s "$vm_dir/$image_name" ]]; then
    curl --fail --location --retry 3 "$base_url/$image_name" -o "$vm_dir/$image_name.part"
    mv "$vm_dir/$image_name.part" "$vm_dir/$image_name"
fi
if [[ ! -s "$vm_dir/SHA256SUMS" ]]; then
    curl --fail --location --retry 3 "$base_url/SHA256SUMS" -o "$vm_dir/SHA256SUMS"
fi
if [[ ! -f "$vm_dir/id_ed25519" ]]; then
    ssh-keygen -q -t ed25519 -N '' -C expert-flash-vm -f "$vm_dir/id_ed25519"
fi
python3 - <<'PY'
import hashlib, os
from pathlib import Path
vm = Path(os.environ['EXPERT_FLASH_DATA']) / 'vm'
image_name = 'jammy-server-cloudimg-amd64.img'
expected = [line.split()[0] for line in (vm / 'SHA256SUMS').read_text().splitlines()
            if line.split()[-1].lstrip('*') == image_name]
if len(expected) != 1:
    raise SystemExit('Image missing from official SHA256SUMS')
with (vm / image_name).open('rb') as image:
    actual = hashlib.file_digest(image, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if actual is None:
        h = hashlib.sha256()
        while chunk := image.read(8 * 1024 * 1024):
            h.update(chunk)
        actual = h.hexdigest()
if actual != expected[0]:
    raise SystemExit(f'Image checksum mismatch: {actual}')
(vm / 'image.sha256').write_text(f'{actual}  {image_name}\n')
key = (vm / 'id_ed25519.pub').read_text().strip()
(vm / 'seed/user-data').write_text(fr'''#cloud-config
hostname: expert-flash-vm
manage_etc_hosts: true
disable_root: false
ssh_pwauth: false
users:
  - name: root
    lock_passwd: true
    ssh_authorized_keys:
      - {key}
package_update: true
packages:
  - build-essential
  - linux-headers-generic
  - fio
  - nvme-cli
  - python3
  - cmake
write_files:
  - path: /etc/modules-load.d/9p.conf
    permissions: '0644'
    content: |
      9p
      9pnet
      9pnet_virtio
  - path: /etc/default/grub.d/99-nvmevirt.cfg
    permissions: '0644'
    content: |
      GRUB_CMDLINE_LINUX_DEFAULT='console=ttyS0 memmap=2G\$4G isolcpus=2,3 intremap=off'
  - path: /etc/fstab
    append: true
    content: |
      experiment /mnt/experiment 9p trans=virtio,version=9p2000.L,msize=65536,nofail 0 0
      workspace /mnt/workspace 9p trans=virtio,version=9p2000.L,msize=65536,ro,nofail 0 0
runcmd:
  - mkdir -p /mnt/experiment /mnt/workspace
  - mount -a
  - [apt-get, install, -y, 'linux-headers-generic']
  - update-grub
  - touch /root/nvmevirt-bootstrap-complete
''')
(vm / 'seed/meta-data').write_text('instance-id: expert-flash-20261019\nlocal-hostname: expert-flash-vm\n')
PY
genisoimage -quiet -output "$vm_dir/seed.iso" -volid cidata -joliet -rock "$vm_dir/seed"
if [[ ! -f "$vm_dir/root.qcow2" ]]; then
    qemu-img create -f qcow2 -F qcow2 -b "$vm_dir/$image_name" "$vm_dir/root.qcow2" 24G
fi
echo "Prepared $vm_dir. Start with scripts/vm_start.sh; wait for cloud-init, then reboot guest."
