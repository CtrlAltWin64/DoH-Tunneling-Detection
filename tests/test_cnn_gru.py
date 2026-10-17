"""Tests for the 1D-CNN + GRU model: shapes, checkpoint save/load, inference causality,
training smoke-test (tiny synthetic data, no GPU required), and integration with prefix scoring."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import Config
from src.cnn_gru_model import (
    CnnGruClassifier,
    N_SEQ_FEATURES,
    SequenceNormalizer,
    _build_sequence_array,
    build_sequence_dataset,
    load_model,
    save_model,
    score_flows_at_checkpoint,
    train_cnn_gru,
)
from src.cnn_gru_prefix import build_cnn_prefix_table, raw_proba_cnn
from src.feature_extraction import FEATURE_NAMES
from src.synthetic import make_synthetic_flows
from tests._helpers import make_flow

CFG_TINY = Config(
    rf_n_estimators=10, seed=7,
    cnn_epochs=3, cnn_patience=2, cnn_batch_size=16,
    cnn_conv_channels=16, cnn_gru_hidden=16, cnn_gru_layers=1,
    cnn_dropout=0.0,
    checkpoints=(10, 20),
    jitter_settings=((0.5, 0.05),),
    low_and_slow_factors=(2.0,),
)


class ModelShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CnnGruClassifier(
            n_features=N_SEQ_FEATURES,
            conv_channels=16, conv_kernel=3,
            gru_hidden=16, gru_layers=1, dropout=0.0,
        )
        self.model.eval()

    def test_forward_single_sample(self):
        x = torch.zeros(1, 10, N_SEQ_FEATURES)
        lengths = torch.tensor([10])
        logits = self.model(x, lengths)
        self.assertEqual(logits.shape, (1,))

    def test_forward_padded_batch(self):
        """Batch with different sequence lengths; padded inputs should give per-sample logits."""
        x = torch.zeros(4, 30, N_SEQ_FEATURES)
        lengths = torch.tensor([10, 20, 25, 30])
        logits = self.model(x, lengths)
        self.assertEqual(logits.shape, (4,))

    def test_output_is_scalar_per_sample(self):
        for bs in (1, 3, 8):
            x = torch.randn(bs, 15, N_SEQ_FEATURES)
            lengths = torch.full((bs,), 15)
            logits = self.model(x, lengths)
            self.assertEqual(logits.shape, (bs,))

    def test_checkpoint_lengths_10_20_30_40(self):
        """Model processes all required checkpoint lengths cleanly."""
        for t in (10, 20, 30, 40):
            x = torch.randn(2, t, N_SEQ_FEATURES)
            lengths = torch.tensor([t, t])
            logits = self.model(x, lengths)
            self.assertEqual(logits.shape, (2,))
            self.assertTrue(torch.isfinite(logits).all())

    def test_mixed_length_batch_invariance(self):
        """Causal conv and masked normalization ensure a sequence's prediction is invariant to padding and other batch elements."""
        torch.manual_seed(42)
        # Sequence A of length 15
        seq_a = torch.randn(15, N_SEQ_FEATURES)
        # Sequence B of length 35
        seq_b = torch.randn(35, N_SEQ_FEATURES)
        # Sequence C of length 10
        seq_c = torch.randn(10, N_SEQ_FEATURES)

        # Single-sample evaluation of sequence A
        x_solo = seq_a.unsqueeze(0)  # (1, 15, F)
        with torch.no_grad():
            logit_solo = self.model(x_solo, torch.tensor([15]))

        # Batched evaluation with padded tensors up to max_len=40
        x_batch = torch.zeros(3, 40, N_SEQ_FEATURES)
        x_batch[0, :35] = seq_b
        x_batch[1, :15] = seq_a
        x_batch[2, :10] = seq_c
        lengths = torch.tensor([35, 15, 10])

        with torch.no_grad():
            logits_batch = self.model(x_batch, lengths)

        # Sequence A is at index 1 in the batch
        torch.testing.assert_close(logits_batch[1], logit_solo[0], rtol=1e-5, atol=1e-5)

    def test_invalid_lengths_and_empty_batches_rejected(self):
        """Model strictly validates input dimensions and sequence lengths."""
        # 2D input instead of 3D
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, N_SEQ_FEATURES), torch.tensor([10, 10]))
        # Empty batch
        with self.assertRaises(ValueError):
            self.model(torch.zeros(0, 10, N_SEQ_FEATURES), torch.tensor([], dtype=torch.long))
        # Batch size mismatch
        with self.assertRaises(ValueError):
            self.model(torch.randn(3, 10, N_SEQ_FEATURES), torch.tensor([10, 10]))
        # Sequence length 0 or negative
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 10, N_SEQ_FEATURES), torch.tensor([0, 10]))
        # Sequence length exceeding tensor max_len
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 10, N_SEQ_FEATURES), torch.tensor([10, 15]))
        # Feature count mismatch
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 10, N_SEQ_FEATURES + 2), torch.tensor([10, 10]))

    def test_cpu_execution(self):
        """CPU execution runs without error."""
        cpu_model = self.model.to("cpu")
        x = torch.randn(2, 12, N_SEQ_FEATURES, device="cpu")
        lengths = torch.tensor([12, 8], device="cpu")
        with torch.no_grad():
            logits = cpu_model(x, lengths)
        self.assertEqual(logits.device.type, "cpu")
        self.assertEqual(logits.shape, (2,))


class SequenceNormalizerTests(unittest.TestCase):
    def test_fit_transform_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 8)).astype(np.float32)
        norm = SequenceNormalizer().fit(X)
        Xt = norm.transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=1e-5)

    def test_constant_feature_does_not_crash(self):
        X = np.ones((50, 4), dtype=np.float32)
        X[:, 0] = 0  # constant column
        norm = SequenceNormalizer().fit(X)
        Xt = norm.transform(X)
        self.assertTrue(np.all(np.isfinite(Xt)))

    def test_state_dict_roundtrip(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 8)).astype(np.float32)
        n1 = SequenceNormalizer().fit(X)
        n2 = SequenceNormalizer.from_state_dict(n1.state_dict())
        np.testing.assert_allclose(n1.transform(X), n2.transform(X), rtol=1e-5, atol=1e-6)

    def test_unfitted_raises(self):
        n = SequenceNormalizer()
        with self.assertRaises(RuntimeError):
            n.transform(np.zeros((5, 8)))


class SequenceBuilding(unittest.TestCase):
    def test_sequence_shape_equals_t(self):
        f = make_flow(50, seed=0)
        for t in (10, 20, 30):
            arr = _build_sequence_array(f, t)
            self.assertEqual(arr.shape, (t, N_SEQ_FEATURES))

    def test_causal_future_packets_dont_affect_prefix(self):
        f = make_flow(60, seed=1)
        arr_base = _build_sequence_array(f, 20)
        # Scramble everything after packet 20
        f2 = replace(f,
                     ts=f.ts.copy(),
                     length=f.length.copy(),
                     direction=f.direction.copy())
        f2.ts[20:] += 1e6
        f2.length[20:] = 1
        f2.direction[20:] = -f2.direction[20:]
        arr_scr = _build_sequence_array(f2, 20)
        np.testing.assert_array_equal(arr_base, arr_scr)

    def test_build_sequence_dataset_filters_short_flows(self):
        flows = [make_flow(5, fid="short"), make_flow(15, fid="long")]
        seqs, labels, fids = build_sequence_dataset(flows, t=10)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(fids[0], "long")

    def test_feature_count_matches_constant(self):
        self.assertEqual(N_SEQ_FEATURES, 8)


class CheckpointSaveLoad(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_model.pt"
            torch.manual_seed(0)
            model = CnnGruClassifier(
                n_features=N_SEQ_FEATURES,
                conv_channels=16, conv_kernel=3,
                gru_hidden=16, gru_layers=1, dropout=0.0,
            )
            rng = np.random.default_rng(0)
            norm = SequenceNormalizer().fit(rng.standard_normal((50, N_SEQ_FEATURES)))
            save_model(model, norm, path)
            model2, norm2 = load_model(path, device="cpu")

            # Forward pass should give identical results
            model.eval(); model2.eval()
            x = torch.randn(2, 10, N_SEQ_FEATURES)
            lengths = torch.tensor([10, 10])
            with torch.no_grad():
                out1 = model(x, lengths)
                out2 = model2(x, lengths)
            torch.testing.assert_close(out1, out2)

    def test_normalizer_preserved_after_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            rng = np.random.default_rng(2)
            X = rng.standard_normal((40, N_SEQ_FEATURES)).astype(np.float32)
            norm = SequenceNormalizer().fit(X)
            model = CnnGruClassifier(
                n_features=N_SEQ_FEATURES,
                conv_channels=8, conv_kernel=3,
                gru_hidden=8, gru_layers=1, dropout=0.0,
            )
            save_model(model, norm, path)
            _, norm2 = load_model(path, device="cpu")
            np.testing.assert_allclose(
                norm.transform(X), norm2.transform(X), rtol=1e-5
            )


class TrainingSmokeTest(unittest.TestCase):
    """Trains a tiny model on synthetic data and checks the interface contract."""

    @classmethod
    def setUpClass(cls):
        flows = make_synthetic_flows(n_benign=60, n_malicious=60, seed=3, short_frac=0.05)
        cls.train = [f for f in flows[:100] if len(f.ts) >= 10]
        cls.val = [f for f in flows[100:] if len(f.ts) >= 10]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ckpt = Path(cls.tmp.name) / "model.pt"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_train_returns_model_normalizer_history(self):
        model, normalizer, history = train_cnn_gru(
            self.train, self.val, CFG_TINY.checkpoints, CFG_TINY, self.ckpt, device="cpu"
        )
        self.assertIsInstance(model, CnnGruClassifier)
        self.assertIsInstance(normalizer, SequenceNormalizer)
        self.assertIn("best_epoch", history)
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertTrue(history["best_epoch"] >= 1)
        self.assertTrue(Path(self.ckpt).exists())

    def test_inference_returns_probabilities_in_0_1(self):
        model, normalizer, _ = train_cnn_gru(
            self.train, self.val, CFG_TINY.checkpoints, CFG_TINY,
            Path(self.tmp.name) / "m2.pt", device="cpu"
        )
        probs, fids = score_flows_at_checkpoint(
            self.val, 10, model, normalizer, device="cpu"
        )
        self.assertEqual(len(probs), len(fids))
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))


class PrefixTableIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flows = make_synthetic_flows(n_benign=60, n_malicious=60, seed=4, short_frac=0.0)
        cls.flows = flows
        cls.tmp = tempfile.TemporaryDirectory()
        ckpt = Path(cls.tmp.name) / "int_model.pt"
        model, normalizer, _ = train_cnn_gru(
            flows[:80], flows[80:], CFG_TINY.checkpoints, CFG_TINY, ckpt, device="cpu"
        )
        cls.model = model
        cls.normalizer = normalizer

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prefix_table_has_expected_columns(self):
        df = build_cnn_prefix_table(
            self.flows[:20], CFG_TINY.checkpoints,
            self.model, self.normalizer, device="cpu"
        )
        for col in ["flow_id", "label", "checkpoint", "cnn_gru_raw_score"]:
            self.assertIn(col, df.columns)
        for col in FEATURE_NAMES:
            self.assertIn(col, df.columns)

    def test_raw_proba_cnn_returns_floats_in_0_1(self):
        df = build_cnn_prefix_table(
            self.flows[:20], CFG_TINY.checkpoints,
            self.model, self.normalizer, device="cpu"
        )
        probs = raw_proba_cnn(df)
        self.assertEqual(len(probs), len(df))
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))

    def test_checkpoint_column_matches_requested_checkpoints(self):
        df = build_cnn_prefix_table(
            self.flows[:10], CFG_TINY.checkpoints,
            self.model, self.normalizer, device="cpu"
        )
        self.assertEqual(set(df["checkpoint"].unique()), set(CFG_TINY.checkpoints))

    def test_flow_isolation_consistent_prefix(self):
        """Scoring a single flow at t=10 should match the value in the prefix table."""
        fl = [f for f in self.flows if len(f.ts) >= 10][0:1]
        df = build_cnn_prefix_table(fl, (10,), self.model, self.normalizer, device="cpu")
        table_score = float(df[df["checkpoint"] == 10]["cnn_gru_raw_score"].iloc[0])
        direct_probs, _ = score_flows_at_checkpoint(fl, 10, self.model, self.normalizer, device="cpu")
        self.assertAlmostEqual(table_score, float(direct_probs[0]), places=5)


class SeparateCalibrationTests(unittest.TestCase):
    """Verifies calibration uses separate data from model fitting and test set is completely isolated."""

    def test_calibration_subset_disjoint_from_fitting_subset(self):
        import pandas as pd
        from src.dataset_split import split_train_calibration
        from src.cnn_gru_calibration import fit_calibrator_cnn

        flows = make_synthetic_flows(n_benign=50, n_malicious=50, seed=12, short_frac=0.0)
        train_flows = [f for f in flows if len(f.ts) >= 10]
        meta = pd.DataFrame([
            {"flow_id": f.flow_id, "label": f.label, "source_file": f.source_file}
            for f in train_flows
        ])

        assign = split_train_calibration(meta, group_by="flow", seed=42, cal_ratio=0.25)
        fit_fids = {f.flow_id for f in train_flows if assign[f.flow_id] == "train_fit"}
        cal_fids = {f.flow_id for f in train_flows if assign[f.flow_id] == "train_cal"}

        # Disjointness guarantee
        self.assertEqual(len(fit_fids & cal_fids), 0)
        self.assertTrue(len(fit_fids) > 0)
        self.assertTrue(len(cal_fids) > 0)

        # Both classes in both subsets
        fit_labels = {f.label for f in train_flows if f.flow_id in fit_fids}
        cal_labels = {f.label for f in train_flows if f.flow_id in cal_fids}
        self.assertEqual(fit_labels, {0, 1})
        self.assertEqual(cal_labels, {0, 1})

        # Fit calibrator on calibration subset
        cal_flows = [f for f in train_flows if f.flow_id in cal_fids]
        model = CnnGruClassifier(
            n_features=N_SEQ_FEATURES,
            conv_channels=8, conv_kernel=3,
            gru_hidden=8, gru_layers=1, dropout=0.0,
        )
        norm = SequenceNormalizer().fit(np.random.default_rng(0).standard_normal((30, N_SEQ_FEATURES)))
        calibrator, cal_df = fit_calibrator_cnn(
            cal_flows, (10, 20), model, norm, CFG_TINY, device="cpu"
        )
        self.assertEqual(set(cal_df["flow_id"].unique()), cal_fids)
        self.assertFalse(any(fid in fit_fids for fid in cal_df["flow_id"]))


if __name__ == "__main__":
    unittest.main()
