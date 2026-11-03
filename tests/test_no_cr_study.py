"""Ordinary-read replay invariants, checked against independent chip loops."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import aif_model as flash
import joint_plane_search as joint
import run_no_cr_study as study


class NoCrStudyTests(unittest.TestCase):
    def test_every_wave_costs_28us_even_across_multiple_blocks(self):
        for width in (1, 2, 4):
            for pages in (0, 1, 2, 3, 4, 191 * width, 192 * width,
                          192 * width + 1, 1400 * width + 3):
                value = study.ordinary_stream_cost(pages * flash.PAGE, width)
                waves = (pages + width - 1) // width
                self.assertEqual(value['nand_ns'], waves * 28000)
                self.assertEqual(value['cold_reads'], (waves + 191) // 192)
                self.assertEqual(value['matrix_commands'], value['cold_reads'])

    def test_block_changes_keep_commands_but_do_not_change_ordinary_sense(self):
        short_blocks = study.ordinary_stream_cost(193 * flash.PAGE * 4, 4, block_pages=64)
        long_blocks = study.ordinary_stream_cost(193 * flash.PAGE * 4, 4, block_pages=192)
        self.assertEqual(short_blocks['nand_ns'], long_blocks['nand_ns'])
        self.assertGreater(short_blocks['matrix_commands'], long_blocks['matrix_commands'])
        self.assertEqual(short_blocks['nand_ns'], 193 * 28000)

    def test_one_plus_three_and_two_plus_two_page_tails(self):
        # Two independent fixed pair lanes: 1+3 needs two epochs, 2+2 one.
        def serial(pages):
            return sum(study.ordinary_stream_cost(p * flash.PAGE, 4)['nand_ns'] for p in pages)
        def pairs(pages):
            return max(study.ordinary_stream_cost(p * flash.PAGE, 2)['nand_ns'] for p in pages)
        self.assertEqual(serial((1, 3)), 56000)
        self.assertEqual(pairs((1, 3)), 56000)
        self.assertEqual(serial((2, 2)), 56000)
        self.assertEqual(pairs((2, 2)), 28000)
        # Aggregating pages before ceil would incorrectly turn 1+3 into one.
        self.assertGreater(pairs((1, 3)), 28000)

    def test_changing_cr_only_changes_nand_and_derived_latency(self):
        rng = np.random.default_rng(9032)
        dims = dict(gate=(357, 71123), up=(357, 71123), down=(71123, 357))
        routes = np.stack([rng.choice(32, 8, replace=False) for _ in range(36)])
        for g in study.GROUPS:
            for width in study.WIDTHS:
                slots = rng.permutation(32)
                profiles = [joint.phase_profiles(dims, g, width, use_cr=cr) for cr in (True, False)]
                for on, off in zip(*profiles):
                    for key in on:
                        if key not in ('nand', 'use_cr'):
                            np.testing.assert_equal(on[key], off[key], err_msg=f'{g}/{width}/{key}')
                    np.testing.assert_array_equal(off['nand'], off['waves'] * 28000)
                counts = joint.counts_for(routes, slots, g, width)
                on, off = [joint.costs_from_counts(counts, p, g, width, 8) for p in profiles]
                for key in (*study.COMPONENTS, *study.COUNTERS, 'model_ns'):
                    if key not in ('nand_bound_ns', 'ordinary_read_epochs', 'flash_ns', 'model_ns'):
                        np.testing.assert_array_equal(on[key], off[key], err_msg=f'{g}/{width}/{key}')
                np.testing.assert_array_equal(off['ordinary_read_epochs'], off['read_waves'])
                np.testing.assert_array_equal(off['model_ns'] - on['model_ns'], off['flash_ns'] - on['flash_ns'])
                self.assertTrue((off['flash_ns'] >= on['flash_ns']).all())

    def test_ordinary_mode_keeps_ecc_overlap_instead_of_serializing_drain(self):
        dims = dict(gate=(512, 1024), up=(512, 1024), down=(1024, 512))
        slots = list(range(8))
        counts = joint.counts_for(np.array([[0, 1]]), slots, 16, 2)
        profiles = joint.phase_profiles(dims, 16, 2, use_cr=False)
        fast = joint.costs_from_counts(counts, profiles, 16, 2, 2)
        expected = sum(max(int(p['nand'].max()), 2 * int(p['ecc'].max())) for p in profiles)
        serialized = sum(int(p['nand'].max()) + 2 * int(p['ecc'].max()) for p in profiles)
        self.assertEqual(int(fast['flash_ns'][0]), expected)
        self.assertLess(int(fast['flash_ns'][0]), serialized)

    def test_scalar_chip_loop_matches_all_components_and_counters(self):
        rng = np.random.default_rng(1092)
        dimensions = (dict(gate=(257, 513), up=(257, 513), down=(513, 257)),
                      dict(gate=(8099, 65539), up=(8099, 65539), down=(65539, 8099)))
        for dims in dimensions:
            for g in study.GROUPS:
                for width in study.WIDTHS:
                    slots = rng.permutation(20).tolist()
                    selected = rng.choice(20, 6, replace=False).tolist()
                    counts = joint.counts_for(np.array([selected]), slots, g, width)
                    for cr in (False, True):
                        fast = joint.costs_from_counts(counts, joint.phase_profiles(dims, g, width, cr), g, width, 6)
                        scalar = study.scalar_event(selected, dims, g, width, slots, cr)
                        for key, expected in scalar.items():
                            self.assertEqual(int(fast[key][0]), expected, f'{g}/{width}/{cr}/{key}')

    def test_full_training_objective_and_swap_capacity(self):
        rng = np.random.default_rng(501)
        routes = np.stack([rng.choice(32, 8, replace=False) for _ in range(100)])
        unique, weights = np.unique(np.sort(routes, axis=-1), axis=0, return_counts=True)
        dims = dict(gate=(512, 65539), up=(512, 65539), down=(65539, 512))
        for g, width in ((4, 1), (8, 2), (16, 1), (2, 4)):
            initial = rng.permutation(32).tolist()
            profiles = joint.phase_profiles(dims, g, width, use_cr=False)
            learned, stats = joint.optimize_slots(unique, weights, 32, initial, g, width, profiles, 80, 71)
            initial_cost = joint.costs_from_counts(joint.counts_for(unique, initial, g, width), profiles, g, width, 8)
            learned_cost = joint.costs_from_counts(joint.counts_for(unique, learned, g, width), profiles, g, width, 8)
            self.assertEqual(stats['initial_model_ns_weighted'], int(initial_cost['model_ns'] @ weights))
            self.assertEqual(stats['final_model_ns_weighted'], int(learned_cost['model_ns'] @ weights))
            self.assertLessEqual(stats['final_model_ns_weighted'], stats['initial_model_ns_weighted'])
            self.assertEqual(sorted(learned), list(range(32)))
            self.assertEqual(stats['capacity_before'], stats['capacity_after'])

    def test_degenerate_dimensions_receive_the_whole_budget(self):
        rng = np.random.default_rng(44)
        unique, weights = np.unique(np.sort(np.stack([rng.choice(8, 2, replace=False) for _ in range(30)]), axis=1),
                                   axis=0, return_counts=True)
        dims = dict(gate=(512, 1024), up=(512, 1024), down=(1024, 512))
        old = dict(mapping=list(range(8)), bins={'1': [e % 4 for e in range(8)],
                                               '2': [e % 2 for e in range(8)]})
        _, _, mappings, stats = study.fit_layer_group((0, 16, dims, 8, unique, weights, old, 20, 10))
        for width in (1, 2):
            self.assertEqual(mappings['joint_group_plane'][str(width)], mappings['staged_group_then_plane'][str(width)])
            for method in ('joint', 'staged'):
                self.assertTrue(all(r['proposals_budget'] == 20 for r in stats[str(width)][method]['runs']))
        self.assertTrue(all(r['proposals_budget'] == 20 for r in stats['4']['group_runs']))


if __name__ == '__main__':
    unittest.main()
