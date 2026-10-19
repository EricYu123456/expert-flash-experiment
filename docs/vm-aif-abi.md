# NVMeVirt AiF metadata ABI v1 / v2 / v3

本專案的 opcode `0xc1` 是 NVMeVirt 的實驗性 timing extension。它接收工作負載描述、計算完成時間並延遲 NVMe completion；不讀取模型權重、不執行 GEMV，也不傳輸真正的 input/output vectors。Vector byte counters 與 transfer latency 是模型值，不能當成硬體流量或 MAC lane utilization 實測。

v1 只包含 flash demand。v2 在相同 flash 模型上加入 serialized chip command control、host command 與 input/output distribution barriers。各成本可透過 descriptor 指定，預設的 250 ns/chip matrix command 與 2 µs/host command 是公開的假設，沒有以實體 AiF 硬體校準。

v3 將 host PCIe 流量與 controller 對各 chip 的 ONFI 流量分開。相同 input vector 可以由 host 傳一次，再由 controller 複製到多個 chips；v2 原有的 per-chip host unicast 假設仍保留為相容的參考模型。

## Transport

使用 guest `/dev/nvme0n1` 的 `NVME_IOCTL_IO_CMD`，opcode `0xc1`、namespace 1、payload 為 4096-byte aligned buffer 的前 400 或 800 bytes；PRP1 必須 page aligned，整個 descriptor 在單一實體 page 中。新 client 在 command dword 10 傳入 descriptor 長度：v1 為 400、v2/v3 為 800。舊 v1 client 的 cdw10=0 仍支援。v2/v3 必須為 800；不支援的 version、長度或欄位回覆 `NVME_SC_INVALID_FIELD`，且不更新 counters。

所有欄位為 little endian，無可變長度 array。Workload `.bin` 直接串接完整 descriptors，沒有額外 file header；新版 `scripts/aif_replay.c` 可在同一檔案中自動辨識 v1/v2/v3，也會拒絕被截斷的 header/body。

## Wire layout

v1 保留 400 bytes：16-byte header `<IIII`，依序為 magic `0x31464941`、version `1`、nchips、reserved `0`；其後固定 16 個 24-byte records `<QQII>`：

| 欄位 | 型別 | 意義 |
|---|---|---|
| useful_bytes | u64 | 有效權重 bytes |
| physical_pages | u64 | 16 KiB physical pages |
| read_rounds | u32 | 4 planes/chip 的 sensing rounds |
| cold_reads | u32 | 需要完整 28 µs 延遲的 rounds |

v2 為 800 bytes：32-byte header `<IIIIIIII>`：

| Offset | 欄位 | 型別 |
|---:|---|---|
| 0 | magic = `0x31464941` | u32 |
| 4 | version = `2` | u32 |
| 8 | nchips，本實驗為 16 | u32 |
| 12 | per_chip_control_ns | u32 |
| 16 | per_host_command_ns | u32 |
| 20 | reserved[3]，全部為 0 | 3 × u32 |

v3 保留完全相同的 800-byte 長度與 chip records，只改變 header 中的以下欄位：

| Offset | v3 欄位 | 型別 |
|---:|---|---|
| 4 | version = `3` | u32 |
| 20 | pcie_input_bytes，host 實際需要傳給 controller 的 input bytes | u32 |
| 24 | pcie_output_bytes，controller 傳回 host 的 output bytes | u32 |
| 28 | reserved = 0 | u32 |

其後固定 16 個 48-byte chip records `<QQIIQQII>`：

| Record offset | 欄位 | 型別 |
|---:|---|---|
| 0 | useful_bytes | u64 |
| 8 | physical_pages | u64 |
| 16 | read_rounds | u32 |
| 20 | cold_reads | u32 |
| 24 | input_bytes | u64 |
| 32 | output_bytes | u64 |
| 40 | matrix_command_count | u32 |
| 44 | reserved = 0 | u32 |

Kernel 支援 nchips 1–16。每 chip 限制 pages ≤ 2³⁰、useful_bytes ≤ pages × 16384、cold_reads ≤ read_rounds ≤ pages ≤ 4 × read_rounds；非零 pages 需要非零 rounds。v2 的每 chip input/output 各 ≤ 2⁴⁰ bytes、matrix command count ≤ 2²⁰，非零 useful bytes 必須有 matrix command；control/host latency 各 ≤ 10⁹ ns。這些界限讓單次 descriptor 的加總與乘法保持在 u64 範圍內，completion target 加法另外檢查 overflow。

v3 套用相同的 per-chip bounds，並要求 PCIe input/output 各不超過對應 per-chip bytes 的總和。PCIe 欄位是 u32；若將來單一 command 需要超過其上限的 host vector，應拆 command 或升級 ABI。

## Completion-time model

每 chip sensing / bandwidth service：

```text
nand_i = read_rounds_i × 9700 + cold_reads_i × 18300 ns
bandwidth_i = ceil(physical_pages_i × 16384 × 5 / 32) ns
flash = max_i(max(nand_i, bandwidth_i))
control = sum_i(matrix_command_count_i) × per_chip_control_ns
host = per_host_command_ns
```

這保留每 chip 6.4 decimal GB/s 的 bandwidth ceiling，cold round 為 28 µs，其後 round 為 9.7 µs。

將 chip i 指派到 channel `i % 8`。每次 vector phase 的 PCIe 上限為 8 bytes/ns、每 channel 上限為 2 bytes/ns；兩者在 phase 內重疊，取較慢者。v3 使用獨立 host PCIe byte count：

```text
input = max(ceil(pcie_input_bytes / 8),
            max_channel(ceil(sum_chip_on_channel(input_bytes_i) / 2)))
output = 對 pcie_output_bytes、per-chip output_bytes 使用相同公式
total = flash + control + host + input + output
completion_target = request_start + total
```

v2 為相容性保留 `pcie_input_bytes = sum_i(input_bytes_i)`、output 同理；其成本相當於 host 對 chips 分別傳送，可能高估共享 input 在 PCIe 上的流量。v3 才能明確區分 host 傳送一次與 ONFI fanout。Input reuse 只在同一 phase、同一 chip 的共用輸入 streams 間成立，沒有假設跨 host commands 保留 vector。

這是有順序 barrier 的成本模型；chip control 視為 controller 逐一發送，input phase、flash phase 與 output phase 相加。它沒有重現實體控制線、DMA 或計算電路。實際 `ioctl_wall_ns` 仍包含 guest driver、VM 排程與 CPU 執行成本，與 descriptor 的假設成本分開記錄。

## Counters

`/proc/nvmev/aif_stats` 保留四行 header 與原有 CSV schema。`model` 為 `metadata_only_int8_v1`、`metadata_only_int8_v2` 或 `metadata_only_int8_v3`，同一次 reset 後混用則為 `metadata_only_int8_mixed`。`model_ns` 是全部 completion model cost；每 chip 的 `service_ns` 仍只包含 flash service。

`/proc/nvmev/aif_v2_stats` 保留檔名，擴充為十行 `key integer`：`flash_ns`、`control_ns`、`host_ns`、`input_ns`、`output_ns`、`vector_input_bytes`、`vector_output_bytes`、`chip_matrix_commands`、`pcie_input_bytes`、`pcie_output_bytes`。其中 vector byte counts 代表 ONFI fanout，PCIe counters 代表 host/controller 邊界。它也累積 v1 的 flash cost，因此五個 latency components 的和永遠對應同次 replay 的 `model_ns`。向 `aif_stats` 寫入任意資料會同時清空兩組 counters；應只在 synchronous replay 之間 reset。

## 驗證紀錄

2026-10-19 的 guest module 已重建、重新載入並通過：舊 v1 client（cdw10=0）、新版 v2 client、同檔 v1/v2 mixed replay、invalid rounds/vector/matrix/control bounds、錯誤 cdw10 長度，以及 truncated workload。所有 kernel-rejected descriptors 均回覆 INVALID_FIELD，counter 維持 0。

三 chip synthetic v2 command 的預期值為 `28000 + 1500 + 2000 + 40 + 10 = 31550 ns`；kernel 各 component 完全一致。Input 共 102 bytes、output 共 24 bytes、6 個 chip matrix commands，也完全吻合。完整證據位於 NFS `runs/aif-v2-smoke/reload-and-smoke.log`、`negative-and-mixed.log` 與 fixtures。

v3 已重新編譯、載入並通過 fanout 對照：16 chips 各需要 100-byte input、20-byte output，host 只供應一次 100-byte input；kernel 回報 ONFI input 1600 bytes、PCIe input 100 bytes，input latency 100 ns，總成本 `28000 + 4000 + 2000 + 100 + 40 = 34140 ns`。相同 chip demand 的 v2 command 保留 1600-byte PCIe input，input latency 200 ns、總成本 34240 ns。PCIe input/output 超過 ONFI 總量或 reserved 非零均回覆 INVALID_FIELD，counters 維持 0。證據位於 NFS `runs/aif-v3-smoke/reload-and-smoke.log`。
