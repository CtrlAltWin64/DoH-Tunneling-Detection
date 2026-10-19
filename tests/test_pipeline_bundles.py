"""Training-path artifact and data-role regressions on small synthetic fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from config import Config
from src.model_bundle import load_cnn_bundle
from src.pipeline import run_experiment
from src.prefix_generation import build_prefix_table
from src.synthetic import make_synthetic_flows
from src.train_baseline import load_rf_bundle


class PipelineBundles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "run"
        cls.flows = make_synthetic_flows(100, 100, seed=41, short_frac=0.0)
        cls.cfg = Config(checkpoints=(5, 10), rf_n_estimators=8, calibration_folds=2,
                         cnn_conv_channels=8, cnn_gru_hidden=8, cnn_gru_layers=1,
                         cnn_epochs=2, cnn_batch_size=64, cnn_device="cpu",
                         low_and_slow_factors=(), jitter_settings=())
        run_experiment(cls.flows, cls.cfg, cls.out, synthetic=True, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_bundles_reproduce_saved_test_scores(self):
        models = {"rf": load_rf_bundle(self.out / "rf_bundle.pkl"),
                  "cnn": load_cnn_bundle(self.out / "cnn_gru_bundle.pt")}
        for name, bundle in models.items():
            filename = "predictions_test.csv" if name == "rf" else "cnn_gru_predictions_test.csv"
            saved = pd.read_csv(self.out / filename).set_index("flow_id")
            flows = [f for f in self.flows if f.flow_id in saved.index]
            for cp in self.cfg.checkpoints:
                if name == "rf":
                    table = build_prefix_table(flows, [cp])
                    actual = pd.Series(bundle.predict_proba(table), index=table["flow_id"])
                else:
                    values, ids = bundle.predict_proba(flows, cp)
                    actual = pd.Series(values, index=ids)
                np.testing.assert_allclose(actual.loc[saved.index], saved[f"p_{cp}"], atol=1e-7, rtol=1e-6)

    def test_manifest_metrics_stops_and_paths_match_saved_outputs(self):
        manifest = json.loads((self.out / "experiment.json").read_text(),
                              parse_constant=lambda x: self.fail(f"Invalid JSON number: {x}"))
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(json.loads((self.out / "run_status.json").read_text())["status"], "complete")
        for model, payload in manifest["models"].items():
            for path in payload["artifacts"].values():
                self.assertTrue((self.out / path).is_file(), path)
            frame = pd.read_csv(self.out / payload["artifacts"]["test_metrics"])
            for saved, exported in zip(frame.to_dict("records"), payload["test_metrics"]):
                self.assertEqual(saved["TP"], exported["TP"])
                counts = [r for r in payload["stopping_distribution"] if r["policy"] == saved["policy"]]
                self.assertEqual(sum(r["flows"] for r in counts), saved["n_flows"])
                self.assertEqual(sum(r["flows"] for r in counts if r["alert"] and r["label"] == 0), saved["FP"])
                observed = sum(r["flows"] * r["observed_packets"] for r in counts) / saved["n_flows"]
                self.assertAlmostEqual(observed, saved["avg_packets_all"])
        for path in manifest["figures"]:
            self.assertTrue((self.out / path).is_file())

    def test_failure_has_no_success_manifest(self):
        out = Path(self.tmp.name) / "failed"
        with patch("src.pipeline.fit_rf", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                run_experiment(self.flows, self.cfg, out, synthetic=True, log=lambda *a: None)
        self.assertEqual(json.loads((out / "run_status.json").read_text())["status"], "failed")
        self.assertFalse((out / "experiment.json").exists())

    def test_failure_before_training_is_marked(self):
        from src import pipeline
        with patch.object(pipeline, "_run_experiment", side_effect=ValueError("ineligible cohort")):
            with self.assertRaisesRegex(ValueError, "ineligible cohort"):
                run_experiment([], self.cfg, Path(self.tmp.name) / "ineligible", log=lambda *a: None)

    def test_training_roles_are_separate_from_threshold_validation_and_test(self):
        outer = pd.read_csv(self.out / "split_assignment.csv").set_index("flow_id")
        inner = pd.read_csv(self.out / "train_fit_cal_split.csv")
        self.assertFalse(inner["flow_id"].duplicated().any())
        self.assertEqual(set(inner["fit_or_cal"]), {"train_fit", "train_cal", "train_early_stop"})
        self.assertEqual(set(outer.loc[inner["flow_id"], "partition"]), {"train"})
        rec = json.loads((self.out / "run_record.json").read_text())
        self.assertEqual(rec["models"]["cnn_gru"]["early_stopping_partition"], "train_early_stop")
        history = json.loads((self.out / "cnn_gru_train_history.json").read_text())
        self.assertEqual(rec["models"]["cnn_gru"]["epochs_trained"], len(history["train_loss"]))
        manifest = json.loads((self.out / "split_manifest.json").read_text())
        self.assertEqual(manifest["seed"], self.cfg.seed)


if __name__ == "__main__":
    unittest.main()
