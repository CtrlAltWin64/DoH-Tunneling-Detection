"""Reload the real RF training/calibration outputs, without refitting on reload."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import Config
from src.calibration import fit_calibrator
from src.feature_extraction import FEATURE_NAMES
from src.prefix_generation import build_prefix_table
from src.synthetic import make_synthetic_flows
from src.train_baseline import fit_rf, load_rf_bundle, raw_proba, save_rf_bundle


class RfPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(checkpoints=(10, 20), rf_n_estimators=12, calibration_folds=2,
                         rf_min_samples_leaf=2, seed=19)
        cls.train = build_prefix_table(make_synthetic_flows(30, 30, seed=19), cls.cfg.checkpoints)
        cls.test = build_prefix_table(make_synthetic_flows(8, 8, seed=23), cls.cfg.checkpoints)
        cls.model = fit_rf(cls.train, cls.cfg)
        cls.calibrator = fit_calibrator(cls.train, cls.cfg)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "rf_bundle.pkl"
        save_rf_bundle(self.path, self.model, self.calibrator, self.cfg,
                       thresholds={"adaptive": .73, "fixed_20": .68})

    def test_actual_training_outputs_reload_raw_and_calibrated_predictions(self):
        bundle = load_rf_bundle(self.path)
        raw = raw_proba(self.model, self.test)
        calibrated = self.calibrator.transform(raw, self.test.checkpoint.to_numpy())
        np.testing.assert_allclose(bundle.predict_proba(self.test, calibrated=False), raw, rtol=0, atol=1e-14)
        np.testing.assert_allclose(bundle.predict_proba(self.test), calibrated, rtol=0, atol=1e-14)
        self.assertEqual(tuple(FEATURE_NAMES), bundle.feature_names)
        self.assertEqual(tuple(self.cfg.checkpoints), bundle.config["checkpoints"])
        self.assertEqual(.73, bundle.thresholds["adaptive"])
        self.assertIn("scikit-learn", bundle.versions)

    def test_saved_feature_order_is_used_after_dataframe_columns_reordered(self):
        bundle = load_rf_bundle(self.path)
        shuffled = self.test[list(reversed(self.test.columns))]
        np.testing.assert_allclose(bundle.predict_proba(shuffled), bundle.predict_proba(self.test), rtol=0, atol=1e-14)

    def test_inference_rejects_missing_features_and_unseen_calibration_checkpoint(self):
        bundle = load_rf_bundle(self.path)
        with self.assertRaisesRegex(ValueError, "missing RF inference columns"):
            bundle.predict_proba(self.test.drop(columns=[FEATURE_NAMES[0]]))
        unseen = self.test.copy()
        unseen["checkpoint"] = 99
        with self.assertRaisesRegex(ValueError, "checkpoints absent"):
            bundle.predict_proba(unseen)

    def test_loader_rejects_incompatible_version_and_feature_schema(self):
        with self.path.open("rb") as stream:
            original = pickle.load(stream)
        for key, value, message in [("format_version", 99, "format/version"),
                                    ("feature_names", list(reversed(FEATURE_NAMES)), "feature schema")]:
            payload = dict(original)
            payload[key] = value
            with self.path.open("wb") as stream:
                pickle.dump(payload, stream)
            with self.assertRaisesRegex(ValueError, message):
                load_rf_bundle(self.path)


if __name__ == "__main__":
    unittest.main()
