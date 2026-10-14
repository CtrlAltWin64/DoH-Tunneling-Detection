import unittest

import numpy as np

from src.adaptive_detector import policy_outcomes, run_policy, select_threshold, sweep_thresholds, threshold_grid
from tests._helpers import make_curves


class ThresholdCrossing(unittest.TestCase):
    def test_spec_example_alerts_at_20_packets(self):
        c = make_curves([[0.40, 0.78, 0.82, 0.85]], [0])
        o = policy_outcomes(c, 0.75)
        self.assertTrue(o["alert"][0])
        self.assertEqual(o["alert_packets"][0], 20)
        self.assertEqual(o["observed_packets"][0], 20)

    def test_equality_counts_as_crossing(self):
        c = make_curves([[0.1, 0.75, 0.2, 0.2]], [1])
        self.assertEqual(policy_outcomes(c, 0.75)["alert_packets"][0], 20)

    def test_continues_when_threshold_not_reached_then_stops_at_first_crossing(self):
        c = make_curves([[0.1, 0.2, 0.9, 0.95]], [1])
        o = policy_outcomes(c, 0.8)
        self.assertEqual(o["alert_packets"][0], 30)               # skipped 10 and 20, stopped at 30, not 40

    def test_nondetection_recorded_as_such(self):
        c = make_curves([[0.1, 0.2, 0.3, 0.4]], [1])
        o = policy_outcomes(c, 0.8)
        self.assertFalse(o["alert"][0])
        self.assertTrue(np.isnan(o["alert_packets"][0]))          # not "benign", simply not detected
        self.assertEqual(o["observed_packets"][0], 40)            # observed up to Tmax
        self.assertTrue(np.isnan(o["latency_s"][0]))

    def test_decision_ignores_later_checkpoints_once_alerted(self):
        a = make_curves([[0.1, 0.9, 0.0, 0.0]], [1])
        b = make_curves([[0.1, 0.9, 1.0, 1.0]], [1])
        self.assertEqual(policy_outcomes(a, 0.8)["alert_packets"][0], policy_outcomes(b, 0.8)["alert_packets"][0])

    def test_fixed_window_is_single_checkpoint_policy(self):
        c = make_curves([[0.9, 0.9, 0.9, 0.2]], [1])
        self.assertTrue(policy_outcomes(c, 0.8)["alert"][0])                  # adaptive alerts at 10
        o = policy_outcomes(c, 0.8, col_idx=[3])                              # fixed 40 sees only p=0.2
        self.assertFalse(o["alert"][0])
        self.assertEqual(o["observed_packets"][0], 40)

    def test_run_policy_vectorised(self):
        P = np.array([[0.1, 0.2], [0.9, 0.1], [0.1, 0.9]])
        alert, k = run_policy(P, 0.5)
        self.assertEqual(list(alert), [False, True, True])
        self.assertEqual(list(k), [-1, 0, 1])


class SequentialFPR(unittest.TestCase):
    def setUp(self):
        # three benign flows
        self.c = make_curves([[0.40, 0.78, 0.82, 0.85],      # alerts at 20  -> FP (later probs don't undo it)
                              [0.10, 0.20, 0.30, 0.40],      # never
                              [0.80, 0.10, 0.10, 0.10]],     # alerts at 10  -> FP
                             [0, 0, 0])

    def test_fpr_counts_unique_flows_alerting_at_any_checkpoint(self):
        sw = sweep_thresholds(self.c, [0.75]).iloc[0]
        self.assertEqual(sw["fp"], 2)
        self.assertAlmostEqual(sw["fpr"], 2 / 3)

    def test_not_equal_to_single_checkpoint_fpr(self):
        per_cp_40 = sweep_thresholds(self.c, [0.75], col_idx=[3]).iloc[0]["fpr"]   # only flow 0 has p40>=.75
        self.assertAlmostEqual(per_cp_40, 1 / 3)
        self.assertGreater(sweep_thresholds(self.c, [0.75]).iloc[0]["fpr"], per_cp_40)

    def test_sweep_matches_bruteforce_policy_runs(self):
        rng = np.random.default_rng(0)
        P = np.sort(rng.random((60, 4)), axis=1) * rng.random((60, 1))
        y = rng.integers(0, 2, 60)
        c = make_curves(P, y)
        for tau in (0.05, 0.2, 0.4, 0.7):
            o = policy_outcomes(c, tau)
            row = sweep_thresholds(c, [tau]).iloc[0]
            self.assertEqual(row["fp"], int((o["alert"] & (o["label"] == 0)).sum()))
            self.assertAlmostEqual(row["coverage"], float((o["alert"] & (o["label"] == 1)).sum() / (o["label"] == 1).sum()))


class ThresholdSelection(unittest.TestCase):
    def make(self):
        rng = np.random.default_rng(1)
        ben = np.clip(rng.normal(0.3, 0.15, (200, 1)) + np.arange(4) * 0.03, 0, 1)
        mal = np.clip(rng.normal(0.7, 0.2, (200, 1)) + np.arange(4) * 0.03, 0, 1)
        return make_curves(np.vstack([ben, mal]), [0] * 200 + [1] * 200)

    def test_selected_threshold_satisfies_alpha_and_is_lowest_feasible(self):
        c = self.make()
        s = select_threshold(c, 0.01, 0.001)
        self.assertLessEqual(s["val_fpr"], 0.01)
        sw = s["sweep"]
        lower = sw[sw["tau"] < s["tau"]]
        self.assertTrue((lower["fpr"] > 0.01).all())              # every smaller candidate violates the constraint

    def test_selection_uses_only_the_passed_curves(self):
        c = self.make()
        a = select_threshold(c, 0.05, 0.001)["tau"]
        c2 = self.make()
        c2.P[c2.y == 1] = 0.0                                      # change malicious data -> threshold must not move
        self.assertEqual(a, select_threshold(c2, 0.05, 0.001)["tau"])    # FPR depends on benign flows only

    def test_resolution_warning_when_too_few_benign_flows(self):
        c = make_curves([[0.1, 0.2, 0.3, 0.4]] * 20 + [[0.9] * 4] * 20, [0] * 20 + [1] * 20)
        s = select_threshold(c, 0.01, 0.01)
        self.assertTrue(s["resolution_warning"])                  # floor(0.01*20) == 0 allowed false positives

    def test_infeasible_target_degrades_to_never_alert_sentinel(self):
        c = make_curves([[1.0] * 4] * 10 + [[1.0] * 4] * 10, [0] * 10 + [1] * 10)
        s = select_threshold(c, 0.01, 0.1)
        self.assertTrue(s["never_alerts"])
        self.assertEqual(s["val_fpr"], 0.0)

    def test_grid_ends_with_sentinel_above_one(self):
        g = threshold_grid(0.001)
        self.assertGreater(g[-1], 1.0)
        self.assertAlmostEqual(g[-2], 1.0)


if __name__ == "__main__":
    unittest.main()
