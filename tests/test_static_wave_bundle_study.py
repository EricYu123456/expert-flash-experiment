"""Conservation, real overfetch and reference checks for static supermatrices."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import corpus_analysis as old
import static_wave_bundle_study as study


class StaticBundleTests(unittest.TestCase):
    def test_singletons_match_existing_model_with_uneven_rows_and_pages(self):
        rng = np.random.default_rng(211)
        routes = np.stack([rng.choice(16, 6, replace=False) for _ in range(31)])
        placement = rng.permutation(16)
        dims = dict(gate=(515, 259), up=(515, 259), down=(259, 515))
        for g in (1, 2, 4, 8, 16):
            result, conservative, metrics = study.layer_costs(routes, placement, g, dims, list(range(16)))
            reference = old.layer_costs(routes, placement, g, old.profiles(dims, g, True))
            for key in study.COMPONENTS:
                np.testing.assert_array_equal(result[key], reference[key], err_msg=f'g={g}, {key}')
            np.testing.assert_array_equal(metrics['read_weight_bytes'], metrics['selected_useful_bytes'])
            self.assertTrue(np.all(conservative['model_ns'] >= result['model_ns']))

    def test_unused_member_is_read_computed_and_returned(self):
        # At g=16, each original gate/up shard is 32KiB (two pages).
        dims = dict(gate=(256, 1024), up=(256, 1024), down=(1024, 256))
        placement = [0, 1]
        labels = [0, 0]
        singleton, _, singleton_metrics = study.layer_costs(np.array([[0]]), placement, 16, dims, [0, 1])
        bundle, _, metrics = study.layer_costs(np.array([[0]]), placement, 16, dims, labels)
        self.assertEqual(int(metrics['gate_returned_experts'][0]), 2)
        self.assertEqual(int(metrics['gate_selected_experts'][0]), 1)
        self.assertGreater(int(metrics['physical_pages'][0]), int(singleton_metrics['physical_pages'][0]))
        self.assertGreater(int(bundle['output_ns'][0]), int(singleton['output_ns'][0]))
        self.assertEqual(int(metrics['read_weight_bytes'][0]), 2 * 512 * 1024 + 1024 * 256)
        both, _, both_metrics = study.layer_costs(np.array([[0, 1]]), placement, 16, dims, labels)
        separate, _, _ = study.layer_costs(np.array([[0, 1]]), placement, 16, dims, [0, 1])
        self.assertEqual(int(both_metrics['gate_returned_experts'][0]), 2)
        self.assertLess(int(both['flash_ns'][0]), int(separate['flash_ns'][0]))
        self.assertLess(int(both['control_ns'][0]), int(separate['control_ns'][0]))

    def test_common_input_stacking_preserves_all_selected_output_values(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=19)
        matrices = [rng.normal(size=(7, 19)) for _ in range(4)]
        for g in (1, 2, 4):
            outputs = {i: [] for i in range(4)}
            for shard in range(g):
                start = shard * (7 // g) + min(shard, 7 % g)
                count = 7 // g + int(shard < 7 % g)
                # Actual storage order is chip-major, then expert-major.
                matrix = np.concatenate([w[start:start + count] for w in matrices], axis=0)
                returned = matrix @ x
                for expert in (0, 2):
                    outputs[expert].extend(returned[expert * count:(expert + 1) * count])
            for expert in (0, 2):
                np.testing.assert_allclose(outputs[expert], matrices[expert] @ x, atol=1e-12)

    def test_learning_is_deterministic_and_never_moves_experts_between_chip_groups(self):
        routes = np.array([[0, 4], [0, 4], [1, 5], [2, 6], [3, 7]] * 4)
        placement = list(range(16))
        first = study.fixed_bundles(routes, placement, 4, 2, 128, 31)
        self.assertEqual(first, study.fixed_bundles(routes, placement, 4, 2, 128, 31))
        labels, stats = first
        for label in set(labels):
            members = [e for e, b in enumerate(labels) if b == label]
            self.assertEqual(len(members), 2)
            self.assertEqual(len({placement[e] % 4 for e in members}), 1)
        self.assertLessEqual(stats['final_touched_bundles'], stats['initial_touched_bundles'])

    def test_bundles_cannot_span_chip_groups(self):
        dims = dict(gate=(256, 1024), up=(256, 1024), down=(1024, 256))
        with self.assertRaisesRegex(ValueError, 'fixed chip group'):
            study.layer_costs(np.array([[0, 1]]), [0, 1], 8, dims, [0, 0])


if __name__ == '__main__':
    unittest.main()
