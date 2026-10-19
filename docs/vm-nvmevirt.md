# 無 host sudo 的 NVMeVirt 虛擬機

這台主機的使用者 `ericyu` 已在 `kvm` 群組，可以直接使用 `/dev/kvm`。QEMU 6.2 以一般使用者執行，Ubuntu 22.04 guest 內以 root 安裝、編譯及載入真正的 NVMeVirt Linux module。QEMU 的系統碟使用 virtio-blk；沒有使用 QEMU 的一般 `-device nvme` 來冒充 NVMeVirt。

所有 image、sources、build、工作負載及完整 logs 放在 `/home/nfs_client/ericyu/expert_flash_experiment`。Guest 的 `/root` 位於 NFS 上的 `vm/root.qcow2`，因此 guest 的 kernel build 也使用 NFS 容量。只有小型腳本、patch 與文件放在本機專案。

## 已驗證的環境與結果

2026-10-19 使用 4 vCPU、8 GiB RAM。Guest kernel 為 `5.15.0-198-generic`，有匹配的 headers。Guest kernel cmdline 含 `memmap=2G$4G isolcpus=2,3 intremap=off`，`/proc/iomem` 確認實體位址 `0x100000000` 至 `0x17fffffff` 為 reserved，與 guest kernel 記憶體不重疊。

NVMeVirt 使用 upstream commit `61c90f7758cbd9545b4a4727e89377bf88eab060` 加上專案 patch。Module 以 `memmap_start=4G memmap_size=2G cpus=2,3` 載入。`dmesg` 記錄 NVMeVirt 建立的 PCI function `0001:10:00.0`，vendor/device `0c51:0110`，dispatcher 在 CPU 2、worker 在 CPU 3。`nvme list` 顯示 `/dev/nvme0n1`、model `CSL_Virt_MN_01`、serial `CSL_Virt_SN_01`，namespace 約 2.01 GB。

Conventional SSD 路徑保留 upstream Samsung 970 Pro 組態：8 channels × 2 LUNs/channel × 1 plane/LUN、32 KiB flash page、4 個 FTL partitions。它與另外的 AiF descriptor 模型（16 chips × 4 planes、16 KiB pages）必須分別解讀。

先以 direct write 預寫 64 MiB，再讀取同區域 4 次；每次測試總計 2048 個 128 KiB O_DIRECT requests，即 256 MiB。結果如下，wall time 包含 guest kernel、QEMU/KVM 排程與 CPU overhead：

| 同時讀取 workers | Wall time | Mean request latency | NAND 活躍 LUNs | 每個 LUN 讀取量 |
|---:|---:|---:|---:|---:|
| 1 | 269.108 ms | 131.270 µs | 16/16 | 16 MiB |
| 4 | 77.621 ms | 150.940 µs | 16/16 | 16 MiB |

兩次 client 均回報 0 errors，NAND counters 各為 8192 次 NAND reads、256 MiB，與 client byte count 完全吻合。這證明 conventional NVMeVirt FTL 路徑能將這組 sequential workload 平均分散到 16 LUNs；PE 執行與 MoE expert 小矩陣的利用率，需另外看 AiF extension 的結果。

證據位於 NFS `runs/vm_baseline/`：`preflight.log`、`fio-precondition.json`、`raw_workers1.json`、`raw_workers4.json`、三份 `flash_*.csv`、`flash_delta.json`。Counters 為 `/proc/nvmevirt_flash_stats` 的差值，`nand_read_ns` 是累積 NAND sensing service time，不包含 queue wait 或 channel transfer，也不是 wall-clock utilization。

## 重跑

已建立的 VM 可以直接：

```bash
bash scripts/vm_start.sh
bash scripts/vm_ssh.sh 'mount -a; bash /mnt/workspace/scripts/vm_guest_prepare.sh'
bash scripts/vm_run_baseline.sh
```

全新環境先執行 `bash scripts/vm_setup.sh`，會下載官方 Ubuntu cloud image、核對 SHA256、建立 SSH key、cloud-init seed 與 24 GiB sparse qcow2 overlay。接著 `vm_start.sh`，等待 `vm_ssh.sh 'cloud-init status --wait'` 完成，再用 `vm_ssh.sh reboot` 讓 reserved-memory 設定生效。第一次開機約需數分鐘安裝套件。SSH 僅綁定 `127.0.0.1:22240`，key 位於 NFS `vm/id_ed25519`，權限為 0600。

`vm_guest_prepare.sh` 先檢查 cmdline，將 sources 複製到 guest root disk 後編譯，將 module 留在 NFS `build/nvmevirt/nvmev.ko`。它不會自動替換已載入的 module。更新 kernel code 後，先確認沒有 replay 或開啟 namespace 的程序，於 guest 中 `rmmod nvmev`，再執行 prepare。重新載入會清空 volatile 模擬裝置內容，baseline 應重新 precondition。

如需跑另外的 AiF descriptor replay，編譯 guest client `gcc -O2 /mnt/workspace/scripts/aif_replay.c -o /root/aif_replay`，再執行對應的 `scripts/run_aif_replays.sh`。普通 raw replay 與 AiF replay 不應同時執行，以避免影響 timing 與 counter snapshots。

新增的 v3 ABI 會模型化 chip command control、host command，以及 host 傳一次後由 controller fanout 的 input/output vector transfer，並保留 v1/v2 支援；詳細 wire layout、counter 意義與模型界限見 [vm-aif-abi.md](vm-aif-abi.md)。Vector 內容與 GEMV 沒有真的執行。

可用 `bash scripts/vm_stop.sh` 關閉 guest。改變資料路徑、SSH port、RAM 或 vCPU 數量可設定 `EXPERT_FLASH_DATA`、`EXPERT_FLASH_SSH_PORT`、`EXPERT_FLASH_RAM_MB`、`EXPERT_FLASH_VCPUS`；guest 的 reservation 與 isolated CPU 設定仍需配套調整。

## 操作細節

GRUB 設定檔中必須保存 `memmap=2G\$4G` 的反斜線，讓 GRUB 傳給 kernel 的參數是 `memmap=2G$4G`。若少了反斜線，GRUB 會展開 `$4` 成為錯誤的 `memmap=2GG`；prepare 會拒絕載入 module。

9p share 使用 `msize=65536`，並先載入 `9pnet_virtio`。第一次以較大 `msize=1048576` 操作時，guest 出現 `cp: memory exhausted`、malloc corruption 與 PID 1 segfault。重設 guest、降低 msize 後，copy、編譯、module 載入與上述 direct I/O 測試均完成。這是環境相關的穩定性處理，尚未證明最初 crash 的根本原因；原始序列 log 保存在 `runs/vm_baseline/serial-with-first-boot-crash.log`。

Image SHA256 為 `012d81fade7e8ff4428fc35dfeee711d66d1f47d5ea9c5c8ced96a85b959835c`，已比對官方下載的 `vm/SHA256SUMS`。參考：[NVMeVirt upstream README](https://github.com/snu-csl/nvmevirt)、[Ubuntu Jammy cloud images](https://cloud-images.ubuntu.com/jammy/current/)。
