"""Invariants for train-only fixed-plane trace replay and vectorized timing."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import aif_model as flash
import corpus_analysis as corpus
import plane_wave_model as plane
import run_plane_wave_study as study


class PlaneWaveStudyTests(unittest.TestCase):
    def test_width4_reproduces_existing_baseline_with_uneven_shards(self):
        rng = np.random.default_rng(932)
        routes = np.stack([rng.choice(20, 6, replace=False) for _ in range(30)]).reshape(2, 5, 3, 6)
        layers = [1, 3, 7]
        dimensions = {layer: dict(gate=(35, 17), up=(35, 17), down=(17, 35)) for layer in layers}
        for g in study.GROUPS:
            mappings = {str(layer): rng.permutation(20).tolist() for layer in layers}
            bins = {str(layer): [0] * 20 for layer in layers}
            new = study.evaluate_routes(routes, layers, dimensions, g, 4, mappings, bins)
            old = corpus.evaluate_routes(routes, layers, dimensions, g, True, mappings)
            for key in (*study.COMPONENTS, 'model_ns'):
                np.testing.assert_array_equal(new[key], old[key], err_msg=f'{g}/{key}')

    def test_vectorized_fixed_masks_match_scalar_for_partial_pages_and_uneven_shards(self):
        dims = dict(gate=(257, 513), up=(257, 513), down=(513, 257))
        selected = [0, 2, 3, 7, 9]
        placement = [3, 8, 1, 7, 5, 6, 0, 2, 4, 9]
        for g in (2, 8, 16):
            for width in study.WIDTHS:
                bins = [e % (4 // width) for e in range(10)]
                counts = study.counts_for(np.array([selected]), placement, bins, g, width)
                profiles = study.phase_profiles(dims, g, width)
                for timing in study.TIMINGS:
                    fast = study.costs_from_counts(counts, profiles, g, width, len(selected), timing)
                    scalar_time, scalar_waves, scalar_pages = 0, 0, 0
                    for p in profiles:
                        services = []
                        for group in range(16 // g):
                            resident = [e for e in range(10) if placement[e] % (16 // g) == group]
                            active = [e for e in selected if e in resident]
                            for shard in range(g):
                                masks = {e: tuple(range(bins[e] * width, (bins[e] + 1) * width)) for e in resident}
                                layout = plane.build_layout({e: int(p['useful'][shard]) for e in resident},
                                                            policy={4:'baseline', 2:'pair', 1:'single'}[width], masks=masks)
                                value = plane.evaluate_chip(layout, active, common_input=p['common'])
                                key = {'ideal_overlap':'service_ns',
                                       'retained_cr_schedule':'retained_cr_schedule_ns',
                                       'conservative_drain':'conservative_service_ns'}[timing]
                                services.append(value[key])
                                scalar_waves += value['read_waves']
                                scalar_pages += value['physical_pages']
                        scalar_time += max(services)
                    self.assertEqual(int(fast['flash_ns'][0]), scalar_time)
                    self.assertEqual(int(fast['read_waves'][0]), scalar_waves)
                    self.assertEqual(int(fast['physical_pages'][0]), scalar_pages)

    def test_mask_learning_preserves_chip_group_and_plane_capacities(self):
        rng = np.random.default_rng(19)
        routes = np.stack([rng.choice(32, 8, replace=False) for _ in range(200)])
        unique, weights = np.unique(np.sort(routes, axis=-1), axis=0, return_counts=True)
        placement = rng.permutation(32)
        profiles = study.phase_profiles(dict(gate=(512, 1024), up=(512, 1024), down=(1024, 512)), 8, 1)
        learned, stats = study.optimize_bins(unique, weights, 32, placement, 8, 1, profiles, 100, 43)
        initial = ((placement // 2) % 4).tolist()
        for group in range(2):
            residents = np.flatnonzero(placement % 2 == group)
            self.assertEqual(sorted(initial[e] for e in residents), sorted(learned[e] for e in residents))
        self.assertLessEqual(stats['final_flash_ns_weighted'], stats['initial_flash_ns_weighted'])
        repeated, _ = study.optimize_bins(unique, weights, 32, placement, 8, 1, profiles, 100, 43)
        self.assertEqual(learned, repeated)

    def test_wave_occupancy_improves_only_for_coactivated_disjoint_fixed_bins(self):
        dims = dict(gate=(256, 1024), up=(256, 1024), down=(1024, 256))
        for experts, bins, expect_less in (([0, 1], [0, 1], True), ([0, 1], [0, 0], False)):
            values = {}
            for width in (4, 2):
                active_bins = [0, 0] if width == 4 else bins
                counts = study.counts_for(np.array([experts]), [0, 1], active_bins, 16, width)
                values[width] = study.costs_from_counts(counts, study.phase_profiles(dims, 16, width),
                                                      16, width, len(experts))
            if expect_less:
                self.assertLess(values[2]['flash_ns'][0], values[4]['flash_ns'][0])
            else:
                self.assertGreaterEqual(values[2]['flash_ns'][0], values[4]['flash_ns'][0])
            self.assertEqual(values[2]['useful_bytes'][0], values[4]['useful_bytes'][0])
            self.assertEqual(values[2]['physical_pages'][0], values[4]['physical_pages'][0])

    def test_paired_interval_does_not_treat_tokens_as_independent_samples(self):
        prompt_means = np.array([100, 110, 120, 150], dtype=np.float64)
        self.assertEqual(study.paired_interval(prompt_means, prompt_means, samples=20), [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
