"""Fixed placement, locality, shared bandwidth, and explicit schedule checks."""
import importlib.util
from pathlib import Path
import sys
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = load("plane_wave_model")
baseline = load("aif_model")


class PlaneWaveTests(unittest.TestCase):
    def test_complementary_two_page_pairs_halve_cold_latency(self):
        layout = m.build_layout({0: 2 * m.PAGE, 1: 2 * m.PAGE}, policy="pair")
        result = m.evaluate_chip(layout, [0, 1], include_timeline=True)
        self.assertEqual(result["service_ns"], m.COLD_NS)
        self.assertEqual(result["read_waves"], 1)
        self.assertEqual(result["cold_reads"], 2)
        self.assertEqual(result["plane_cold_reads"], 4)
        self.assertEqual(result["page_slot_fill"], 1)
        self.assertEqual(result["peak_contexts"], 2)
        self.assertEqual(result["conservative_service_ns"], 28_000 + 10_240)
        self.assertEqual([len(p) for p in result["plane_timeline"]], [1] * 4)

    def test_two_page_same_pair_conflict_cannot_be_dynamically_moved(self):
        layout = m.build_layout({0: 2 * m.PAGE, 2: 2 * m.PAGE}, policy="pair")
        first = {e: [s.page_address(p) for p in range(s.pages)] for e, s in layout.shards.items()}
        result = m.evaluate_chip(layout, [2, 0])
        self.assertEqual(result["service_ns"], 2 * m.COLD_NS)
        self.assertEqual(result["page_slot_fill"], 0.5)
        self.assertEqual(result["peak_contexts"], 1)
        self.assertEqual(first, {e: [s.page_address(p) for p in range(s.pages)] for e, s in layout.shards.items()})

    def test_one_plus_three_is_not_falsely_a_full_pair_wave(self):
        layout = m.build_layout({0: m.PAGE, 1: 3 * m.PAGE}, policy="pair")
        result = m.evaluate_chip(layout, [0, 1])
        self.assertEqual(result["service_ns"], m.COLD_NS + m.CR_NS)
        self.assertEqual(result["read_waves"], 2)
        self.assertEqual(result["page_slot_fill"], 0.5)
        self.assertEqual(result["physical_pages"], 4)

    def test_five_pages_show_extra_nand_cost_of_narrow_masks(self):
        for policy, waves in (("baseline", 2), ("pair", 3), ("single", 5)):
            with self.subTest(policy=policy):
                result = m.evaluate_chip(m.build_layout({0: 5 * m.PAGE}, policy=policy), [0])
                self.assertEqual(result["service_ns"], m.COLD_NS + (waves - 1) * m.CR_NS)
                self.assertEqual(result["physical_pages"], 5)
                self.assertEqual(result["read_waves"], waves)
                self.assertEqual(result["idle_wave_slots"], waves * 4 - 5)

    def test_block_crossing_resets_cold_per_plane_per_expert(self):
        layout = m.build_layout({0: 193 * 2 * m.PAGE, 1: 193 * 2 * m.PAGE}, policy="pair")
        self.assertTrue(layout.validate())
        result = m.evaluate_chip(layout, [0, 1], include_timeline=True)
        self.assertEqual(result["cold_reads"], 4)
        self.assertEqual(result["plane_cold_reads"], 8)
        self.assertEqual(result["nand_ns"], 2 * m.COLD_NS + 191 * m.CR_NS)
        for timeline in result["plane_timeline"]:
            self.assertEqual([x["block"] for x in timeline if x["stream_cold"]], [0, 1])
            self.assertEqual([x["wordline"] for x in timeline if x["stream_cold"]], [0, 0])
            self.assertTrue(all(x["cold"] for x in timeline))

    def test_identical_adjacent_experts_still_each_need_cold(self):
        layout = m.build_layout({0: m.PAGE, 4: m.PAGE}, policy="single")
        result = m.evaluate_chip(layout, [0, 4], include_timeline=True)
        self.assertEqual(result["nand_ns"], 2 * m.COLD_NS)
        self.assertTrue(all(x["cold"] for x in result["plane_timeline"][0]))
        self.assertEqual([x["wordline"] for x in result["plane_timeline"][0]], [0, 1])

    def test_long_full_stream_respects_shared_chip_ecc_bandwidth(self):
        layout = m.build_layout({0: m.PAGE * 4 * 192}, policy="baseline")
        result = m.evaluate_chip(layout, [0], include_timeline=True)
        self.assertEqual(result["service_ns"], 192 * 10_240)
        self.assertGreater(result["service_ns"], result["nand_ns"])
        self.assertEqual(result["conservative_service_ns"], 192 * m.COLD_NS + result["ecc_ns"])
        self.assertEqual(result["retained_cr_schedule_ns"], result["nand_ns"] + result["ecc_ns"])
        self.assertEqual(result["conservative_plane_cold_reads"], 192 * 4)
        drains = sorted((x["drain_start_ns"], x["drain_end_ns"]) for p in result["plane_timeline"] for x in p)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(drains, drains[1:])))

    def test_all_four_planes_share_one_bandwidth_limit(self):
        layout = m.build_layout({e: 192 * m.PAGE for e in range(4)}, policy="single")
        result = m.evaluate_chip(layout, range(4), ecc_bw=1_000_000_000)
        self.assertEqual(result["service_ns"], 4 * 192 * m.PAGE)
        self.assertEqual(result["physical_pages"], 4 * 192)
        self.assertEqual(result["peak_contexts"], 4)

    def test_baseline_exactly_restores_existing_demand_model(self):
        for block in (4, 192):
            counts = [1, m.PAGE - 1, m.PAGE, 2 * m.PAGE, 5 * m.PAGE,
                      (4 * block + 1) * m.PAGE, 13 * m.PAGE + 19]
            layout = m.build_layout(dict(enumerate(counts)), block_pages=block)
            self.assertTrue(layout.validate())
            demand = baseline.Demand()
            for nbytes in counts:
                demand.add_stream(nbytes, block, 1)
            result = m.evaluate_chip(layout, range(len(counts)))
            self.assertEqual(result["useful_bytes"], demand.useful)
            self.assertEqual(result["physical_pages"], demand.pages)
            self.assertEqual(result["read_waves"], demand.rounds)
            self.assertEqual(result["cold_reads"], demand.cold)
            self.assertEqual(result["service_ns"], demand.service_ns)

    def test_capacity_accounts_for_all_padding_without_duplicate_pages(self):
        layout = m.build_layout({0: 5 * m.PAGE - 7, 2: 3 * m.PAGE,
                                 4: 7 * m.PAGE, 1: m.PAGE}, policy="pair", block_pages=4)
        self.assertTrue(layout.validate())
        summary = layout.summary()
        self.assertEqual(summary["intra_page_padding_bytes"], 7)
        self.assertEqual(summary["reserved_plane_slots"], summary["physical_pages"] +
                         summary["stream_tail_slots"] + summary["boundary_padding_slots"] +
                         summary["final_block_tail_slots"])
        self.assertGreater(summary["boundary_padding_slots"], 0)

    def test_shared_x_does_not_remove_output_contexts(self):
        layout = m.build_layout({e: m.PAGE for e in range(4)}, policy="single")
        shared = m.evaluate_chip(layout, range(4), common_input=True)
        distinct = m.evaluate_chip(layout, range(4), common_input=False)
        self.assertEqual(shared["input_contexts_required"], 1)
        self.assertEqual(distinct["input_contexts_required"], 4)
        self.assertEqual(shared["accumulator_contexts_required"], 4)
        self.assertEqual(shared["service_ns"], distinct["service_ns"])

    def test_context_shortage_forces_serial_batches(self):
        layout = m.build_layout({e: m.PAGE for e in range(4)}, policy="single")
        self.assertEqual(m.evaluate_chip(layout, range(4), contexts=1)["service_ns"], 4 * m.COLD_NS)
        self.assertEqual(m.evaluate_chip(layout, range(4), contexts=2)["service_ns"], 2 * m.COLD_NS)

    def test_independent_addresses_must_not_be_called_same_wl(self):
        layout = m.build_layout({0: m.PAGE, 2: 2 * m.PAGE, 1: m.PAGE}, policy="pair")
        result = m.evaluate_chip(layout, [1, 2], include_timeline=True)
        self.assertFalse(result["wave_timeline"][0]["same_block_wordline"])
        self.assertTrue(result["flexible_address_required"])

    def test_plane_timeline_has_no_buffer_reuse_before_drain(self):
        layout = m.build_layout({e: (e + 1) * m.PAGE for e in range(8)}, policy="single", block_pages=3)
        result = m.evaluate_chip(layout, range(8), include_timeline=True)
        for timeline in result["plane_timeline"]:
            self.assertTrue(all(a["drain_end_ns"] <= b["nand_start_ns"] for a, b in zip(timeline, timeline[1:])))
        self.assertEqual(max(x["end_ns"] for x in result["wave_timeline"]), result["conservative_service_ns"])

    def test_partial_pages_do_not_inflate_fill(self):
        layout = m.build_layout({e: 1 for e in range(4)}, policy="single")
        result = m.evaluate_chip(layout, range(4))
        self.assertEqual(result["page_slot_fill"], 1)
        self.assertEqual(result["plane_slot_fill"], 1 / m.PAGE)
        self.assertEqual(result["physical_page_fill"], 1 / m.PAGE)

    def test_row_gemv_reconstruction_when_page_boundary_splits_rows(self):
        # 16-KiB pages split these 1,025-byte rows. A page is not one output
        # row: the shard's byte offset determines input/accumulator ownership.
        rows, columns = 33, 1025
        self.assertNotEqual(m.PAGE % columns, 0)
        matrices = {e: bytes((e * 23 + i * 7) % 256 for i in range(rows * columns))
                    for e in range(2)}
        x = [(i % 5) - 2 for i in range(columns)]

        def gemv(data):
            return [sum((data[row * columns + j] - 128) * x[j] for j in range(columns))
                    for row in range(rows)]

        for policy in ("baseline", "pair", "single"):
            with self.subTest(policy=policy):
                layout = m.build_layout({e: len(data) for e, data in matrices.items()}, policy=policy)
                physical = {}
                for e, data in matrices.items():
                    shard = layout.shards[e]
                    for p in range(shard.pages):
                        physical[shard.page_address(p)] = data[p * m.PAGE:(p + 1) * m.PAGE].ljust(m.PAGE, b"\x00")
                self.assertEqual(len(physical), sum(s.pages for s in layout.shards.values()))
                for e, data in matrices.items():
                    shard = layout.shards[e]
                    reconstructed = b"".join(physical[shard.page_address(p)] for p in range(shard.pages))[:shard.nbytes]
                    self.assertEqual(gemv(reconstructed), gemv(data))

    def test_phase_uses_slowest_chip_and_unassigned_chips_are_empty(self):
        a = m.build_layout({0: 2 * m.PAGE, 1: 2 * m.PAGE}, policy="pair")
        b = m.build_layout({0: 5 * m.PAGE})
        result = m.evaluate_phase([0, 1], [a, b, m.build_layout({})])
        self.assertEqual(result["service_ns"], 37_700)
        self.assertEqual(result["physical_pages"], 9)
        self.assertEqual(result["chips"][2]["service_ns"], 0)

    def test_overlapping_masks_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical or disjoint"):
            m.build_layout({0: 2 * m.PAGE, 1: 2 * m.PAGE}, policy="pair", masks={0: (0, 1), 1: (1, 2)})


if __name__ == "__main__":
    unittest.main()
