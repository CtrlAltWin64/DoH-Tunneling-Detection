"""Shared Random Forest trained on progressive prefixes (10/20/30/40 packets).

Weighting issue: a long flow contributes up to four correlated prefix rows, a
short flow fewer. With ``flow_weighting=True`` every row gets weight
1 / (#rows of its flow in the training set), so each training FLOW carries
equal total weight regardless of how many checkpoints it supports.
The same fitted model (and the same features) is used at every checkpoint;
the fixed-window baseline simply reads its 40-packet probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .feature_extraction import FEATURE_NAMES


def flow_weights(flow_ids: pd.Series) -> np.ndarray:
    counts = flow_ids.map(flow_ids.value_counts()).to_numpy(dtype=float)
    w = 1.0 / counts
    return w * (len(w) / w.sum())          # mean weight 1



def make_rf(cfg) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.seed,
    )

def fit_rf(train_df: pd.DataFrame, cfg) -> RandomForestClassifier:
    if train_df.empty:
        raise ValueError("training prefix table is empty")
    if train_df["label"].nunique() < 2:
        raise ValueError("training data contains a single class")
    model = make_rf(cfg)
    w = flow_weights(train_df["flow_id"]) if cfg.flow_weighting else None
    model.fit(train_df[FEATURE_NAMES].to_numpy(), train_df["label"].to_numpy(), sample_weight=w)
    return model


def raw_proba(model, df: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(df[FEATURE_NAMES].to_numpy())[:, classes.index(1)]
