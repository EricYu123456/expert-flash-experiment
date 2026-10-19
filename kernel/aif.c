// SPDX-License-Identifier: GPL-2.0-only
/* Trace-driven AiF timing extension; does not calculate matrix products. */
#include <linux/highmem.h>
#include <linux/io.h>
#include <linux/math64.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/spinlock.h>
#include "nvmev.h"
#include "aif.h"

struct aif_counter {
	u64 useful_bytes, physical_pages, read_rounds, cold_reads, service_ns, commands;
};
static struct aif_counter counters[AIF_MAX_CHIPS];
static u64 command_count, total_model_ns, total_active_chips;
struct aif_cost_counter {
	u64 flash_ns, control_ns, host_ns, input_ns, output_ns;
	u64 vector_input_bytes, vector_output_bytes, chip_matrix_commands;
	u64 pcie_input_bytes, pcie_output_bytes;
};
static struct aif_cost_counter cost_totals;
static unsigned int versions_seen;
static DEFINE_SPINLOCK(aif_lock);

static u64 vector_transfer_ns(u64 total, const struct aif_command_v2 *d,
			      unsigned int nchips, bool output)
{
	u64 duration = DIV_ROUND_UP_ULL(total, 8);
	unsigned int channel, i;
	for (channel = 0; channel < 8; channel++) {
		u64 bytes = 0;
		for (i = channel; i < nchips; i += 8)
			bytes += output ? le64_to_cpu(d->chip[i].output_bytes) :
				 le64_to_cpu(d->chip[i].input_bytes);
		duration = max(duration, DIV_ROUND_UP_ULL(bytes, 2));
	}
	return duration;
}

static u64 chip_service_ns(u64 pages, u64 rounds, u64 cold)
{
	u64 nand_ns = rounds * 9700ULL + cold * 18300ULL;
	u64 bandwidth_ns = DIV_ROUND_UP_ULL(pages * 16384ULL * 5ULL, 32ULL);
	return max(nand_ns, bandwidth_ns);
}

bool aif_process(struct nvmev_request *req, struct nvmev_result *ret)
{
	union {
		struct aif_command v1;
		struct aif_command_v2 v2;
		struct aif_command_v3 v3;
	} d;
	struct aif_cost_counter cost = {0};
	u64 paddr = le64_to_cpu(req->cmd->rw.prp1), duration = 0;
	void *mapped;
	unsigned int i, active = 0, version, nchips, payload_size;
	u32 control_ns = 0, host_ns = 0;
	u32 declared_size = le32_to_cpu(req->cmd->common.cdw10[0]);
	unsigned long flags;
	BUILD_BUG_ON(sizeof(struct aif_command) != AIF_V1_SIZE);
	BUILD_BUG_ON(sizeof(struct aif_command_v2) != AIF_V2_SIZE);
	BUILD_BUG_ON(sizeof(struct aif_command_v3) != AIF_V2_SIZE);
	BUILD_BUG_ON(AIF_V2_SIZE > PAGE_SIZE);

	/* The replay client guarantees a page aligned, single-page payload. */
	if ((paddr & (PAGE_SIZE - 1)) || !pfn_valid(paddr >> PAGE_SHIFT)) {
		ret->status = NVME_SC_INVALID_FIELD;
		return true;
	}
	mapped = kmap_atomic(pfn_to_page(paddr >> PAGE_SHIFT));
	/* Inspect only the common header before selecting a bounded payload size. */
	memcpy(&d, mapped, 12);
	version = le32_to_cpu(d.v1.version);
	nchips = le32_to_cpu(d.v1.nchips);
	payload_size = version == 1 ? AIF_V1_SIZE : AIF_V2_SIZE;
	if (le32_to_cpu(d.v1.magic) != AIF_MAGIC ||
	    (version != 1 && version != 2 && version != 3) || !nchips || nchips > AIF_MAX_CHIPS ||
	    (version == 1 && declared_size && declared_size != AIF_V1_SIZE) ||
	    (version >= 2 && declared_size != AIF_V2_SIZE)) {
		kunmap_atomic(mapped);
		ret->status = NVME_SC_INVALID_FIELD;
		return true;
	}
	memcpy(&d, mapped, payload_size);
	kunmap_atomic(mapped);
	if (version == 1 && le32_to_cpu(d.v1.reserved)) {
		ret->status = NVME_SC_INVALID_FIELD;
		return true;
	}
	if (version >= 2) {
		control_ns = le32_to_cpu(d.v2.per_chip_control_ns);
		host_ns = le32_to_cpu(d.v2.per_host_command_ns);
		if (control_ns > 1000000000U || host_ns > 1000000000U ||
		    (version == 2 && (d.v2.reserved[0] || d.v2.reserved[1] || d.v2.reserved[2])) ||
		    (version == 3 && d.v3.reserved)) {
			ret->status = NVME_SC_INVALID_FIELD;
			return true;
		}
		cost.host_ns = host_ns;
	}
	for (i = 0; i < nchips; i++) {
		u64 pages, useful, rounds, cold;
		if (version == 1) {
			pages = le64_to_cpu(d.v1.chip[i].physical_pages);
			useful = le64_to_cpu(d.v1.chip[i].useful_bytes);
			rounds = le32_to_cpu(d.v1.chip[i].read_rounds);
			cold = le32_to_cpu(d.v1.chip[i].cold_reads);
		} else {
			const struct aif_chip_demand_v2 *chip = &d.v2.chip[i];
			u64 input = le64_to_cpu(chip->input_bytes);
			u64 output = le64_to_cpu(chip->output_bytes);
			u64 matrices = le32_to_cpu(chip->matrix_command_count);
			pages = le64_to_cpu(chip->physical_pages);
			useful = le64_to_cpu(chip->useful_bytes);
			rounds = le32_to_cpu(chip->read_rounds);
			cold = le32_to_cpu(chip->cold_reads);
			/* These limits bound all sums/products below well within u64. */
			if (input > (1ULL << 40) || output > (1ULL << 40) ||
			    matrices > (1U << 20) || chip->reserved || (useful && !matrices)) {
				ret->status = NVME_SC_INVALID_FIELD;
				return true;
			}
			cost.vector_input_bytes += input;
			cost.vector_output_bytes += output;
			cost.chip_matrix_commands += matrices;
		}
		/* Bound descriptors before arithmetic; 16-KiB pages, 4 planes/chip. */
		if (pages > (1ULL << 30) || useful > pages * 16384ULL ||
		    cold > rounds || rounds > pages || pages > rounds * 4 ||
		    (pages && !rounds)) {
			ret->status = NVME_SC_INVALID_FIELD;
			return true;
		}
		/* 6.4 decimal GB/s per chip; max, not additive bandwidth penalty. */
		cost.flash_ns = max(cost.flash_ns, chip_service_ns(pages, rounds, cold));
		active += useful != 0;
	}
	if (version >= 2) {
		cost.pcie_input_bytes = version == 3 ? le32_to_cpu(d.v3.pcie_input_bytes) :
			cost.vector_input_bytes;
		cost.pcie_output_bytes = version == 3 ? le32_to_cpu(d.v3.pcie_output_bytes) :
			cost.vector_output_bytes;
		if (cost.pcie_input_bytes > cost.vector_input_bytes ||
		    cost.pcie_output_bytes > cost.vector_output_bytes) {
			ret->status = NVME_SC_INVALID_FIELD;
			return true;
		}
		cost.control_ns = cost.chip_matrix_commands * control_ns;
		cost.input_ns = vector_transfer_ns(cost.pcie_input_bytes, &d.v2, nchips, false);
		cost.output_ns = vector_transfer_ns(cost.pcie_output_bytes, &d.v2, nchips, true);
	}
	duration = cost.flash_ns + cost.control_ns + cost.host_ns +
		   cost.input_ns + cost.output_ns;
	if (duration > U64_MAX - req->nsecs_start) {
		ret->status = NVME_SC_INVALID_FIELD;
		return true;
	}
	spin_lock_irqsave(&aif_lock, flags);
	command_count++;
	total_model_ns += duration;
	total_active_chips += active;
	versions_seen |= 1U << (version - 1);
	cost_totals.flash_ns += cost.flash_ns;
	cost_totals.control_ns += cost.control_ns;
	cost_totals.host_ns += cost.host_ns;
	cost_totals.input_ns += cost.input_ns;
	cost_totals.output_ns += cost.output_ns;
	cost_totals.vector_input_bytes += cost.vector_input_bytes;
	cost_totals.vector_output_bytes += cost.vector_output_bytes;
	cost_totals.chip_matrix_commands += cost.chip_matrix_commands;
	cost_totals.pcie_input_bytes += cost.pcie_input_bytes;
	cost_totals.pcie_output_bytes += cost.pcie_output_bytes;
	for (i = 0; i < nchips; i++) {
		u64 service;
		if (version == 1) {
			counters[i].useful_bytes += le64_to_cpu(d.v1.chip[i].useful_bytes);
			counters[i].physical_pages += le64_to_cpu(d.v1.chip[i].physical_pages);
			counters[i].read_rounds += le32_to_cpu(d.v1.chip[i].read_rounds);
			counters[i].cold_reads += le32_to_cpu(d.v1.chip[i].cold_reads);
			service = chip_service_ns(le64_to_cpu(d.v1.chip[i].physical_pages),
				le32_to_cpu(d.v1.chip[i].read_rounds), le32_to_cpu(d.v1.chip[i].cold_reads));
		} else {
			counters[i].useful_bytes += le64_to_cpu(d.v2.chip[i].useful_bytes);
			counters[i].physical_pages += le64_to_cpu(d.v2.chip[i].physical_pages);
			counters[i].read_rounds += le32_to_cpu(d.v2.chip[i].read_rounds);
			counters[i].cold_reads += le32_to_cpu(d.v2.chip[i].cold_reads);
			service = chip_service_ns(le64_to_cpu(d.v2.chip[i].physical_pages),
				le32_to_cpu(d.v2.chip[i].read_rounds), le32_to_cpu(d.v2.chip[i].cold_reads));
		}
		counters[i].service_ns += service;
		counters[i].commands += service != 0;
	}
	spin_unlock_irqrestore(&aif_lock, flags);
	ret->nsecs_target = req->nsecs_start + duration;
	return true;
}

static int aif_stats_show(struct seq_file *m, void *v)
{
	struct aif_counter snapshot[AIF_MAX_CHIPS];
	u64 count, ns, active;
	unsigned int versions;
	const char *model;
	unsigned long flags;
	unsigned int i;
	spin_lock_irqsave(&aif_lock, flags);
	memcpy(snapshot, counters, sizeof(snapshot));
	count = command_count; ns = total_model_ns; active = total_active_chips;
	versions = versions_seen;
	spin_unlock_irqrestore(&aif_lock, flags);
	model = versions == 4 ? "metadata_only_int8_v3" :
		versions == 2 ? "metadata_only_int8_v2" :
		versions > 1 ? "metadata_only_int8_mixed" : "metadata_only_int8_v1";
	seq_printf(m, "model %s\ncommands %llu\nmodel_ns %llu\nactive_chip_commands %llu\n", model, count, ns, active);
	seq_puts(m, "chip,useful_bytes,physical_pages,read_rounds,cold_reads,service_ns,active_commands\n");
	for (i = 0; i < AIF_MAX_CHIPS; i++)
		seq_printf(m, "%u,%llu,%llu,%llu,%llu,%llu,%llu\n", i,
			snapshot[i].useful_bytes, snapshot[i].physical_pages,
			snapshot[i].read_rounds, snapshot[i].cold_reads,
			snapshot[i].service_ns, snapshot[i].commands);
	return 0;
}
static int aif_v2_stats_show(struct seq_file *m, void *v)
{
	struct aif_cost_counter cost;
	unsigned long flags;
	spin_lock_irqsave(&aif_lock, flags);
	cost = cost_totals;
	spin_unlock_irqrestore(&aif_lock, flags);
	seq_printf(m, "flash_ns %llu\ncontrol_ns %llu\nhost_ns %llu\ninput_ns %llu\noutput_ns %llu\nvector_input_bytes %llu\nvector_output_bytes %llu\nchip_matrix_commands %llu\n",
		cost.flash_ns, cost.control_ns, cost.host_ns, cost.input_ns,
		cost.output_ns, cost.vector_input_bytes, cost.vector_output_bytes,
		cost.chip_matrix_commands);
	seq_printf(m, "pcie_input_bytes %llu\npcie_output_bytes %llu\n",
		cost.pcie_input_bytes, cost.pcie_output_bytes);
	return 0;
}
static int aif_stats_open(struct inode *inode, struct file *file)
{
	return single_open(file, aif_stats_show, NULL);
}
static ssize_t aif_stats_write(struct file *f, const char __user *buf, size_t n, loff_t *pos)
{
	unsigned long flags;
	/* Reset only between synchronous replays. */
	spin_lock_irqsave(&aif_lock, flags);
	memset(counters, 0, sizeof(counters));
	memset(&cost_totals, 0, sizeof(cost_totals));
	command_count = total_model_ns = total_active_chips = 0;
	versions_seen = 0;
	spin_unlock_irqrestore(&aif_lock, flags);
	return n;
}
static const struct proc_ops aif_stats_ops = {
	.proc_open = aif_stats_open, .proc_read = seq_read,
	.proc_lseek = seq_lseek, .proc_release = single_release,
	.proc_write = aif_stats_write,
};
void aif_proc_init(struct proc_dir_entry *parent)
{
	proc_create("aif_stats", 0600, parent, &aif_stats_ops);
	proc_create_single("aif_v2_stats", 0400, parent, aif_v2_stats_show);
}
void aif_proc_remove(struct proc_dir_entry *parent)
{
	remove_proc_entry("aif_stats", parent);
	remove_proc_entry("aif_v2_stats", parent);
}
