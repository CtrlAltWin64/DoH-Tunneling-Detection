import math
import unittest

import numpy as np

from src.adaptive_detector import policy_outcomes
from src.evaluation import clopper_pearson, expected_calibration_error, prefix_metrics, summarize
from tests._helpers import make_curves


class Confusion(unittest.TestCase):
    def setUp(self):
        # mal0 detected at 10, mal1 detected at 30, mal2 never, ben0 alerts at 20, ben1 never, ben2 never
        P = [[0.9, 0.9, 0.9, 0.9], [0.1, 0.2, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1],
             [0.1, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]]
        self.c = make_curves(P, [1, 1, 1, 0, 0, 0])
        self.m = summarize(policy_outcomes(self.c, 0.8), 40)

    def test_counts(self):
        m = self.m
        self.assertEqual((m["TP"], m["FN"], m["FP"], m["TN"]), (2, 1, 1, 2))

    def test_undetected_malicious_is_fn_and_undetected_benign_is_not_fp(self):
        self.assertEqual(self.m["FN"], 1)
        self.assertEqual(self.m["FP"], 1)                         # only the benign flow that alerted

    def test_rates(self):
        m = self.m
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertAlmostEqual(m["fpr"], 1 / 3)
        self.assertAlmostEqual(m["coverage"], 2 / 3)

    def test_coverage_equals_detected_over_malicious(self):
        self.assertAlmostEqual(self.m["coverage"], 2 / 3)

    def test_observation_cost_and_stopping_packets(self):
        m = self.m
        # malicious observed: 10, 30, 40(undetected)
        self.assertAlmostEqual(m["avg_packets_malicious"], (10 + 30 + 40) / 3)
        self.assertAlmostEqual(m["cost_malicious"], (10 + 30 + 40) / 3 / 40)
        # all flows: 10,30,40,20,40,40
        self.assertAlmostEqual(m["avg_packets_all"], (10 + 30 + 40 + 20 + 40 + 40) / 6)

    def test_latency_only_over_detected_malicious(self):
        # durations are 0.1*checkpoint -> 1.0 s and 3.0 s
        self.assertEqual(self.m["n_detected_for_latency"], 2)
        self.assertAlmostEqual(self.m["latency_mean_s"], 2.0)

    def test_no_alerts_gives_nan_precision_not_crash(self):
        m = summarize(policy_outcomes(self.c, 5.0), 40)
        self.assertTrue(math.isnan(m["precision"]))
        self.assertEqual(m["TP"], 0)
        self.assertEqual(m["f1"], 0.0)


class DEBD(unittest.TestCase):
    def setUp(self):
        # flow 0: alert at 10 packets, 200 bytes seen of 1000 total -> 20 %
        # flow 1: never detected, total 500 bytes -> counted as 100 %
        bytes_cum = np.array([[200, 400, 600, 800], [100, 200, 300, 400]], float)
        P = [[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]]
        c = make_curves(P, [1, 1], bytes_cum=bytes_cum, total_bytes=[1000, 500])
        self.m = summarize(policy_outcomes(c, 0.5), 40)

    def test_per_flow_mean_with_undetected_as_100(self):
        self.assertAlmostEqual(self.m["debd_mean_all_pct"], 60.0)

    def test_detected_only_mean(self):
        self.assertAlmostEqual(self.m["debd_mean_detected_pct"], 20.0)

    def test_byte_weighted_differs_from_per_flow_mean(self):
        self.assertAlmostEqual(self.m["debd_byte_weighted_all_pct"], 100 * (200 + 500) / 1500)
        self.assertNotAlmostEqual(self.m["debd_byte_weighted_all_pct"], self.m["debd_mean_all_pct"])

    def test_alert_packet_bytes_are_included(self):
        # alert at checkpoint 20 -> bytes up to and including packet 20 (400), not 200
        c = make_curves([[0.1, 0.9, 0.9, 0.9]], [1], bytes_cum=np.array([[200, 400, 600, 800]], float), total_bytes=[800])
        o = policy_outcomes(c, 0.5)
        self.assertEqual(o["bytes_pre"][0], 400)
        self.assertAlmostEqual(summarize(o, 40)["debd_mean_all_pct"], 50.0)


class Statistics(unittest.TestCase):
    def test_clopper_pearson_zero_events(self):
        ci = clopper_pearson(0, 100)
        self.assertEqual(ci["ci_low"], 0.0)
        self.assertAlmostEqual(ci["upper_one_sided"], 1 - 0.05 ** (1 / 100), places=6)
        self.assertAlmostEqual(ci["ci_high"], 1 - 0.025 ** (1 / 100), places=6)
        self.assertGreater(ci["upper_one_sided"], 0.0)            # zero observed FPR is not "zero true FPR"

    def test_clopper_pearson_brackets_point_estimate(self):
        ci = clopper_pearson(7, 100)
        self.assertLess(ci["ci_low"], 0.07)
        self.assertGreater(ci["ci_high"], 0.07)

    def test_prefix_metrics_perfect_and_ece(self):
        y = np.array([0, 0, 1, 1]); p = np.array([0.0, 0.1, 0.9, 1.0])
        m = prefix_metrics(y, p)
        self.assertEqual((m["precision"], m["recall"], m["f1"], m["fpr"], m["auc"]), (1, 1, 1, 0, 1))
        self.assertAlmostEqual(expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
