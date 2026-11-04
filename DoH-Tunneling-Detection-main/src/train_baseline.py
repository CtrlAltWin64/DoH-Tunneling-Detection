"""Shared Random Forest trained on progressive prefixes (10/20/30/40 packets).

Weighting issue: a long flow contributes up to four correlated prefix rows, a
short flow fewer. With ``flow_weighting=True`` every row gets weight
1 / (#rows of its flow in the training set), so each training FLOW carries
equal total weight regardless of how many checkpoints it supports.
The same fitted model (and the same features) is used at every checkpoint;
the fixed-window baseline simply reads its 40-packet probability.
"""
from __future__ import annotations

import pickle
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from .feature_extraction import FEATURE_NAMES


def flow_weights(flow_ids: pd.Series) -> np.ndarray:
    counts = flow_ids.map(flow_ids.value_counts()).to_numpy(dtype=float)
    w = 1.0 / counts
    return w * (len(w) / w.sum())          # mean weight 1



def make_rf(cfg) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight=cfg.rf_class_weight,
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


@dataclass
class RfBundle:
    """Fitted forest, calibration maps, and the metadata needed for inference."""

    model: RandomForestClassifier
    calibrator: object
    feature_names: tuple[str, ...]
    config: dict
    versions: dict
    thresholds: dict

    def predict_proba(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Return malicious-class probabilities with the saved feature ordering."""
        required = list(self.feature_names) + (["checkpoint"] if calibrated else [])
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(f"missing RF inference columns: {missing}")
        if df.empty:
            return np.empty(0, dtype=float)
        if calibrated:
            unknown = set(df["checkpoint"]) - set(self.config["checkpoints"])
            if unknown:
                raise ValueError(f"checkpoints absent from the fitted RF bundle: {sorted(unknown)}")
        classes = list(self.model.classes_)
        raw = self.model.predict_proba(df[list(self.feature_names)].to_numpy())[:, classes.index(1)]
        return self.calibrator.transform(raw, df["checkpoint"].to_numpy()) if calibrated else raw


def save_rf_bundle(path, model, calibrator, cfg, *, thresholds=None) -> Path:
    """Persist an actual fitted forest and calibrator in a versioned local bundle.

    This uses pickle to preserve sklearn estimators; only load trusted project
    artifacts. No training or calibration is performed by this helper.
    """
    check_is_fitted(model)
    if not isinstance(model, RandomForestClassifier) or model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError("RF bundle requires a fitted forest with the project's ordered features")
    if 1 not in model.classes_ or not callable(getattr(calibrator, "transform", None)):
        raise ValueError("RF bundle requires malicious-class predictions and a probability calibrator")
    payload = {
        "format": "doh-rf-bundle", "format_version": 1,
        "model": model, "calibrator": calibrator, "feature_names": list(FEATURE_NAMES),
        "config": asdict(cfg), "thresholds": dict(thresholds or {}),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "estimator_params": model.get_params(), "calibration": calibrator.describe(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_rf_bundle(path) -> RfBundle:
    """Load a trusted bundle written by :func:`save_rf_bundle`."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("format") != "doh-rf-bundle" or payload.get("format_version") != 1:
        raise ValueError("unsupported RF bundle format/version")
    required = {"model", "calibrator", "feature_names", "config", "versions", "thresholds"}
    if not required <= payload.keys():
        raise ValueError(f"incomplete RF bundle: missing {sorted(required - payload.keys())}")
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("RF bundle feature schema does not match this code version")
    model = payload["model"]
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("RF bundle does not contain a RandomForestClassifier")
    check_is_fitted(model)
    if model.n_features_in_ != len(payload["feature_names"]) or 1 not in model.classes_:
        raise ValueError("RF bundle estimator has incompatible features or classes")
    if not callable(getattr(payload["calibrator"], "transform", None)) or "checkpoints" not in payload["config"]:
        raise ValueError("RF bundle calibration/configuration is incomplete")
    return RfBundle(model, payload["calibrator"], tuple(payload["feature_names"]),
                    payload["config"], payload["versions"], payload["thresholds"])
