"""CNN-GRU prefix scoring for the pipeline.

Provides:
- build_cnn_prefix_table(): produces a prefix table in the same format as
  build_prefix_table() but with CNN-GRU sequence scores replacing RF features.
  The table is used directly by calibration.py and adaptive_detector.py.
- raw_proba_cnn(): extracts raw (pre-calibration) probabilities from the table.

The prefix table schema is compatible with the RF pipeline:
  flow_id, label, source_file, checkpoint, flow_total_packets, flow_total_bytes,
  + all FEATURE_NAMES from feature_extraction.py (for calibrator compatibility)
  + a special column "cnn_gru_raw_score" (the sigmoid probability from the model)

Calibration is applied by calibration.py to the "cnn_gru_raw_score" column.
For compatibility with the shared Calibrator class (which expects raw probabilities
from any model), the calibrator is fitted using cnn_gru_raw_score instead of RF proba.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cnn_gru_model import (
    CnnGruClassifier,
    SequenceNormalizer,
    build_sequence_dataset,
    score_flows_at_checkpoint,
)
from .feature_extraction import FEATURE_NAMES, META_COLUMNS, prefix_feature_vector


def build_cnn_prefix_table(
    flows,
    checkpoints,
    model: CnnGruClassifier,
    normalizer: SequenceNormalizer,
    device: str = "cpu",
    batch_size: int = 256,
) -> pd.DataFrame:
    """Build a prefix table scored by the CNN-GRU model.

    Returns a DataFrame with the same META_COLUMNS + FEATURE_NAMES schema as
    build_prefix_table(), PLUS a 'cnn_gru_raw_score' column.

    Only flows with >= t packets contribute a row at checkpoint t.
    """
    metas, feats, scores = [], [], []
    for t in checkpoints:
        probs, flow_ids = score_flows_at_checkpoint(
            flows, t, model, normalizer, device=device, batch_size=batch_size
        )
        # Build a fast lookup: flow_id -> index in flows
        fid_to_flow = {fl.flow_id: fl for fl in flows if len(fl.ts) >= t}
        for prob, fid in zip(probs, flow_ids):
            fl = fid_to_flow[fid]
            metas.append((fl.flow_id, fl.label, fl.source_file, t,
                          fl.total_packets, fl.total_bytes))
            feats.append(prefix_feature_vector(fl.ts, fl.length, fl.direction, t))
            scores.append(float(prob))

    if not metas:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_NAMES + ["cnn_gru_raw_score"])

    df = pd.DataFrame(metas, columns=META_COLUMNS)
    df = pd.concat([df, pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)], axis=1)
    df["cnn_gru_raw_score"] = scores
    return df


def raw_proba_cnn(df: pd.DataFrame) -> np.ndarray:
    """Extract raw CNN-GRU scores as a probability array (same API as train_baseline.raw_proba)."""
    return df["cnn_gru_raw_score"].to_numpy(dtype=float)
