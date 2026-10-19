/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef NVMEV_AIF_H
#define NVMEV_AIF_H

#define AIF_OPCODE 0xc1
#define AIF_MAGIC 0x31464941U
#define AIF_MAX_CHIPS 16
#define AIF_V1_SIZE 400
#define AIF_V2_SIZE 800

/* Little endian wire ABI. Metadata only: no weights or GEMV outputs. */
struct aif_chip_demand {
	__le64 useful_bytes;
	__le64 physical_pages;
	__le32 read_rounds;
	__le32 cold_reads;
};
struct aif_command {
	__le32 magic;
	__le32 version;
	__le32 nchips;
	__le32 reserved;
	struct aif_chip_demand chip[AIF_MAX_CHIPS];
};

/* v2 adds explicit experimental controller and vector-transfer cost metadata. */
struct aif_chip_demand_v2 {
	__le64 useful_bytes;
	__le64 physical_pages;
	__le32 read_rounds;
	__le32 cold_reads;
	__le64 input_bytes;
	__le64 output_bytes;
	__le32 matrix_command_count;
	__le32 reserved;
};
struct aif_command_v2 {
	__le32 magic;
	__le32 version;
	__le32 nchips;
	__le32 per_chip_control_ns;
	__le32 per_host_command_ns;
	__le32 reserved[3];
	struct aif_chip_demand_v2 chip[AIF_MAX_CHIPS];
};

/* v3 separates one host transfer from the controller's per-chip fanout. */
struct aif_command_v3 {
	__le32 magic;
	__le32 version;
	__le32 nchips;
	__le32 per_chip_control_ns;
	__le32 per_host_command_ns;
	__le32 pcie_input_bytes;
	__le32 pcie_output_bytes;
	__le32 reserved;
	struct aif_chip_demand_v2 chip[AIF_MAX_CHIPS];
};

struct nvmev_request;
struct nvmev_result;
struct proc_dir_entry;
bool aif_process(struct nvmev_request *req, struct nvmev_result *ret);
void aif_proc_init(struct proc_dir_entry *parent);
void aif_proc_remove(struct proc_dir_entry *parent);
#endif
