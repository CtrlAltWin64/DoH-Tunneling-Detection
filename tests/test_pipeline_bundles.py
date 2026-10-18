"""Training-path artifact and data-role regressions on small synthetic fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

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
