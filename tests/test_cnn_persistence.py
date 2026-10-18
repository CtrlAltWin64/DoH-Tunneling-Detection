"""Regression coverage for artifacts emitted by the actual training path."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from config import Config
from src.cnn_gru_model import (
    CnnGruClassifier, N_SEQ_FEATURES, SequenceNormalizer, load_model,
    score_flows_at_checkpoint, train_cnn_gru,
)
from src.synthetic import make_synthetic_flows


class TrainingArtifactPersistence(unittest.TestCase):
    def setUp(self):
        self.flows = make_synthetic_flows(20, 20, seed=81, short_frac=0.0)
        self.cfg = Config(
            checkpoints=(5, 10), cnn_epochs=2, cnn_patience=2,
            cnn_batch_size=128, cnn_conv_channels=8, cnn_gru_hidden=8,
            cnn_gru_layers=1, cnn_dropout=0.0, seed=81,
        )

    def test_real_training_checkpoint_loads_and_reproduces_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cnn_gru_best.pt"
            model, normalizer, history = train_cnn_gru(
                self.flows[:30], self.flows[30:], self.cfg.checkpoints,
                self.cfg, path, device="cpu",
            )
            restored, restored_normalizer = load_model(path)
            expected, ids = score_flows_at_checkpoint(self.flows[30:], 10, model, normalizer)
            actual, restored_ids = score_flows_at_checkpoint(self.flows[30:], 10, restored, restored_normalizer)
            self.assertEqual(ids, restored_ids)
            np.testing.assert_array_equal(expected, actual)
            self.assertFalse(restored.training)
            self.assertEqual(history["epochs_trained"], len(history["train_loss"]))
            self.assertEqual(history["best_val_loss"], min(history["val_loss"]))

    def test_small_improvement_saved_without_resetting_patience(self):
        # One validation batch per epoch; a tiny real improvement must be saved
        # even when too small to reset the configured patience counter.
        losses = iter([5e-7, 4e-7, 3e-7, 2e-7])

        class ControlledLoss(torch.nn.Module):
            def forward(self, logits, labels):
                if torch.is_grad_enabled():
                    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                return torch.tensor(next(losses), device=logits.device)

        cfg = SimpleNamespace(**{**vars(self.cfg), "cnn_epochs": 4, "cnn_min_delta": 1e-6})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            with mock.patch("src.cnn_gru_model.nn.BCEWithLogitsLoss", return_value=ControlledLoss()):
                _, _, history = train_cnn_gru(
                    self.flows[:30], self.flows[30:], cfg.checkpoints, cfg, path, device="cpu",
                )
            payload = torch.load(path, weights_only=False)
            self.assertEqual(history["epochs_trained"], 3)
            self.assertEqual(history["best_epoch"], 3)
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["val_loss"], min(history["val_loss"]))
            self.assertIn("model_kwargs", payload)

    def test_legacy_training_checkpoint_recovers_architecture(self):
        model = CnnGruClassifier(N_SEQ_FEATURES, 8, 3, 12, 2, 0.15)
        normalizer = SequenceNormalizer().fit(np.ones((12, N_SEQ_FEATURES), dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"model_state": model.state_dict(), "normalizer": normalizer.state_dict(),
                        "epoch": 1, "val_loss": 0.1}, path)
            (Path(tmp) / "run_record.json").write_text(json.dumps({"config": {"cnn_dropout": 0.15}}))
            restored, restored_normalizer = load_model(path)
            self.assertEqual(restored.gru.dropout, 0.15)
            expected, _ = score_flows_at_checkpoint(self.flows[:3], 5, model, normalizer)
            actual, _ = score_flows_at_checkpoint(self.flows[:3], 5, restored, restored_normalizer)
            np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
