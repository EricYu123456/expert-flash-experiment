"""Invariant checks for complete-cost joint placement and CR ablation."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import corpus_analysis as corpus
import joint_plane_search as joint
import run_plane_wave_study as previous


class JointPlacementTests(unittest.TestCase):
    def test_cr_width4_restores_old_full_model_and_fast_objective(self):
        rng = np.random.default_rng(713)
        routes = np.stack([rng.choice(32, 6, replace=False) for _ in range(36)]).reshape(2, 6, 3, 6)
        layers = [0, 3, 7]
        dims = {l: dict(gate=(257, 513), up=(257, 513), down=(513, 257)) for l in layers}
        mappings = {str(l): rng.permutation(32).tolist() for l in layers}
        for g in joint.GROUPS:
            old = corpus.evaluate_routes(routes, layers, dims, g, True, mappings)
            actual = joint.evaluate_routes(routes, layers, dims, g, 4, mappings)
            for k in (*joint.COMPONENTS, 'model_ns'):
                np.testing.assert_array_equal(actual[k], old[k])
            for width in joint.WIDTHS:
                for cr in (True, False):
                    p = joint.phase_profiles(dims[0], g, width, cr)
                    counts = joint.counts_for(routes[:, :, 0], mappings['0'], g, width)
                    np.testing.assert_array_equal(joint.model_ns_from_counts(counts, p, g, width, 6),
                                                   joint.costs_from_counts(counts, p, g, width, 6)['model_ns'].reshape(-1))

    def test_complete_objective_includes_physical_channel_vector_cost(self):
        dims = dict(gate=(4096, 8192), up=(4096, 8192), down=(8192, 4096))
        p = joint.phase_profiles(dims, 4, 2)
        together = np.array([[[1, 1], [0, 0], [1, 1], [0, 0]]])
        apart = np.array([[[1, 1], [1, 1], [0, 0], [0, 0]]])
        a = joint.costs_from_counts(together, p, 4, 2, 4)
        b = joint.costs_from_counts(apart, p, 4, 2, 4)
        np.testing.assert_array_equal(a['flash_ns'], b['flash_ns'])
        self.assertGreater(int(a['input_ns'][0]), int(b['input_ns'][0]))
        self.assertGreater(int(a['model_ns'][0]), int(b['model_ns'][0]))

    def test_joint_moves_groups_and_planes_and_beats_fixed_group_toy(self):
        unique = np.array([[0, 4, 8, 12], [0, 4, 8, 13], [0, 4, 9, 12]], dtype=np.int16)
        weights = np.array([20, 2, 2])
        dims = dict(gate=(128, 512), up=(128, 512), down=(512, 128))
        p = joint.phase_profiles(dims, 4, 2)
        initial = list(range(16))
        fixed, _ = joint.optimize_slots(unique, weights, 16, initial, 4, 2, p, 256, 91, False)
        learned, stats = joint.optimize_slots(unique, weights, 16, initial, 4, 2, p, 256, 91)
        self.assertLess(joint.score_slots(unique, weights, learned, 4, 2, p)[0],
                        joint.score_slots(unique, weights, fixed, 4, 2, p)[0])
        self.assertGreater(stats['accepted_cross_group'], 0)
        self.assertGreater(stats['accepted_cross_plane'], 0)
        self.assertGreater(stats['accepted_both'], 0)
        self.assertEqual(stats['capacity_before'], stats['capacity_after'])
        self.assertEqual(sorted(learned), initial)
        again, repeated = joint.optimize_slots(unique, weights, 16, initial, 4, 2, p, 256, 91)
        self.assertEqual(learned, again)
        self.assertEqual(stats, repeated)
        self.assertEqual(joint.score_slots(unique, weights, learned, 4, 2, p)[0],
                         stats['final_model_ns_weighted'])

    def test_v1_seed_recovers_each_fixed_capacity_and_bin(self):
        rng = np.random.default_rng(42)
        placement = rng.permutation(32)
        for g in joint.GROUPS:
            for width in joint.WIDTHS:
                groups, bins = joint.decode_slots(placement, g, width)
                bins = bins.copy()
                for group in range(16 // g):
                    ids = np.flatnonzero(groups == group)
                    bins[ids] = rng.permutation(bins[ids])
                slots = joint.slots_from_group_bins(placement, bins, g, width)
                got_groups, got_bins = joint.decode_slots(slots, g, width)
                np.testing.assert_array_equal(got_groups, groups)
                np.testing.assert_array_equal(got_bins, bins)

    def test_no_cr_changes_only_nand_coefficient_and_retains_ecc_overlap(self):
        dims = dict(gate=(10001, 513), up=(10001, 513), down=(513, 10001))
        for g in joint.GROUPS:
            for w in joint.WIDTHS:
                cr = joint.phase_profiles(dims, g, w, True)
                ordinary = joint.phase_profiles(dims, g, w, False)
                for a, b in zip(cr, ordinary):
                    for key in ('rows', 'useful', 'pages', 'waves', 'cold', 'ecc'):
                        np.testing.assert_array_equal(a[key], b[key])
                    np.testing.assert_array_equal(b['nand'], b['waves'] * 28000)
                counts = joint.counts_for(np.array([[0, 3, 5, 7]]), list(range(16)), g, w)
                a = joint.costs_from_counts(counts, cr, g, w, 4)
                b = joint.costs_from_counts(counts, ordinary, g, w, 4)
                for key in ('input_ns', 'output_ns', 'control_ns', 'host_ns', 'useful_bytes',
                            'physical_pages', 'matrix_commands', 'cold_reads'):
                    np.testing.assert_array_equal(a[key], b[key])
                expected = sum(max(int(counts.max()) * int(p['nand'].max()),
                                   int(counts.sum(axis=-1).max()) * int(p['ecc'].max()))
                               for p in ordinary)
                self.assertEqual(int(b['flash_ns'][0]), expected)

    def test_whole_rows_are_sharded_before_page_rounding(self):
        dims = dict(gate=(35, 17), up=(35, 17), down=(17, 35))
        p = joint.phase_profiles(dims, 16, 2)
        self.assertEqual(int(p[0]['useful'].sum()), 70 * 17)
        np.testing.assert_array_equal(p[0]['rows'], [5] * 6 + [4] * 10)
        self.assertTrue((p[0]['rows'] * 17 == p[0]['useful']).all())
        self.assertTrue((p[0]['pages'] == 1).all())
        self.assertNotEqual(int(p[0]['pages'].sum()), (70 * 17 + 16383) // 16384)


if __name__ == '__main__':
    unittest.main()
