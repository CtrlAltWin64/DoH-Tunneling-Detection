import unittest

import numpy as np

from src.evasion import low_and_slow, timing_jitter
from src.feature_extraction import FEATURE_NAMES, prefix_feature_vector
from tests._helpers import make_flow


class Evasion(unittest.TestCase):
    def setUp(self):
        self.f = make_flow(60, label=1, fid="m1", seed=5)

    def test_low_and_slow_scales_iats_keeps_everything_else(self):
        g = low_and_slow(self.f, 5.0)
        np.testing.assert_allclose(np.diff(g.ts), 5.0 * np.diff(self.f.ts))
        self.assertEqual(g.ts[0], self.f.ts[0])
        np.testing.assert_array_equal(g.length, self.f.length)
        np.testing.assert_array_equal(g.direction, self.f.direction)
        self.assertEqual((g.label, g.flow_id, g.total_bytes), (self.f.label, self.f.flow_id, self.f.total_bytes))

    def test_features_are_recomputed_from_new_timestamps(self):
        g = low_and_slow(self.f, 5.0)
        a = dict(zip(FEATURE_NAMES, prefix_feature_vector(self.f.ts, self.f.length, self.f.direction, 20)))
        b = dict(zip(FEATURE_NAMES, prefix_feature_vector(g.ts, g.length, g.direction, 20)))
        self.assertAlmostEqual(b["duration"], 5.0 * a["duration"])
        self.assertAlmostEqual(b["iat_mean"], 5.0 * a["iat_mean"])
        self.assertEqual(b["bytes_total"], a["bytes_total"])      # size features untouched

    def test_jitter_preserves_order_sizes_and_is_nonnegative(self):
        g = timing_jitter(self.f, 1.0, 0.2, seed=1)
        self.assertTrue(np.all(np.diff(g.ts) >= 0))
        np.testing.assert_array_equal(g.length, self.f.length)
        np.testing.assert_array_equal(g.direction, self.f.direction)
        self.assertFalse(np.allclose(g.ts, self.f.ts))

    def test_jitter_is_reproducible_and_seed_dependent(self):
        a = timing_jitter(self.f, 0.5, 0.05, seed=1)
        b = timing_jitter(self.f, 0.5, 0.05, seed=1)
        c = timing_jitter(self.f, 0.5, 0.05, seed=2)
        np.testing.assert_array_equal(a.ts, b.ts)
        self.assertFalse(np.allclose(a.ts, c.ts))

    def test_jitter_independent_of_processing_order(self):
        other = make_flow(60, label=1, fid="m2", seed=6)
        first = timing_jitter(self.f, 0.5, 0.05, seed=3)
        timing_jitter(other, 0.5, 0.05, seed=3)                  # processing another flow in between changes nothing
        again = timing_jitter(self.f, 0.5, 0.05, seed=3)
        np.testing.assert_array_equal(first.ts, again.ts)

    def test_original_flow_not_mutated(self):
        before = self.f.ts.copy()
        low_and_slow(self.f, 3.0); timing_jitter(self.f, 0.5, 0.1, 0)
        np.testing.assert_array_equal(before, self.f.ts)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            low_and_slow(self.f, 0)
        with self.assertRaises(ValueError):
            timing_jitter(self.f, -1, 0, 0)


if __name__ == "__main__":
    unittest.main()
