"""Leakage-safe probability calibration (Platt / sigmoid scaling).

Procedure (uses TRAINING flows only; validation/test never touch it):
  1. Out-of-fold (OOF) raw Random-Forest probabilities for the training prefix
     rows, with folds grouped by original flow (all prefixes of a flow in one fold).
  2. Fit a 1-feature logistic regression on logit(raw OOF prob) -> label,
     one per checkpoint (``calibration_per_checkpoint=True``) or one shared.
  3. The final RF (fit on all training rows) is calibrated with these maps.

Limitations (documented, not hidden):
  * OOF forests see 80 % of the training flows, the final forest 100 %, so the
    OOF probabilities are slightly pessimistic relative to the final model.
  * Calibration maps are constrained to be NON-DECREASING. If a fitted slope is <= 0 (this happens
    when the out-of-fold forest has no usable signal at a checkpoint, e.g. AUC <= 0.5), that map would
    invert the evidence, so it is replaced: per-checkpoint -> shared map -> identity (raw probability).
    Every replacement is listed under ``fallbacks`` in run_record.json.
  * A monotone calibration map does not change the ranking of flows at one
    checkpoint. With ONE shared map the alert decisions are identical to
    thresholding raw probabilities (only the meaning of tau changes). Per-
    checkpoint maps can change the cross-checkpoint behaviour of the policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .feature_extraction import FEATURE_NAMES
from .train_baseline import flow_weights, make_rf

_EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def oof_raw_probabilities(train_df: pd.DataFrame, cfg) -> np.ndarray:
    """Flow-grouped out-of-fold raw probabilities for every training prefix row."""
    X = train_df[FEATURE_NAMES].to_numpy()
    y = train_df["label"].to_numpy()
    groups = train_df["flow_id"].to_numpy()
    skf = StratifiedGroupKFold(n_splits=cfg.calibration_folds, shuffle=True, random_state=cfg.seed)
    oof = np.full(len(train_df), np.nan)
    for tr, te in skf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            raise ValueError("a calibration fold has a single class in its training part; need more data")
        m = make_rf(cfg)
        w = flow_weights(train_df["flow_id"].iloc[tr]) if cfg.flow_weighting else None
        m.fit(X[tr], y[tr], sample_weight=w)
        oof[te] = m.predict_proba(X[te])[:, list(m.classes_).index(1)]
    if np.isnan(oof).any():
        raise RuntimeError("internal error: some rows received no OOF prediction")
    return oof


class Calibrator:
    def __init__(self, method: str, per_checkpoint: bool):
        self.method, self.per_checkpoint = method, per_checkpoint
        self.models: dict = {}      # checkpoint -> fitted map (only maps with positive slope are kept)
        self.shared = None          # None = identity
        self.fallbacks: list = []

    @staticmethod
    def _increasing(model) -> bool:
        return float(model.coef_[0, 0]) > 0.0

    def fit(self, raw: np.ndarray, y: np.ndarray, checkpoints: np.ndarray):
        if self.method == "none":
            return self
        lr = lambda: LogisticRegression(C=1e6, max_iter=1000)
        shared = lr().fit(_logit(raw), y)
        if self._increasing(shared):
            self.shared = shared
        else:
            self.fallbacks.append("shared map had non-positive slope -> identity")
        if self.per_checkpoint:
            for t in np.unique(checkpoints):
                m = checkpoints == t
                if len(set(y[m])) < 2:
                    self.fallbacks.append(f"checkpoint {int(t)}: single class -> shared map")
                    continue
                model = lr().fit(_logit(raw[m]), y[m])
                if self._increasing(model):
                    self.models[int(t)] = model
                else:
                    self.fallbacks.append(f"checkpoint {int(t)}: non-positive slope -> shared map")
        return self

    def _map_for(self, t):
        return self.models.get(int(t), self.shared) if self.per_checkpoint else self.shared

    def transform(self, raw: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.method == "none":
            return raw
        out = np.empty(len(raw))
        for t in np.unique(checkpoints):
            m = checkpoints == t
            model = self._map_for(t)
            out[m] = raw[m] if model is None else model.predict_proba(_logit(raw[m]))[:, 1]
        return out

    def describe(self) -> dict:
        d = {"method": self.method, "per_checkpoint": self.per_checkpoint, "fallbacks": self.fallbacks}
        if self.shared is not None:
            d["shared_coef_intercept"] = [float(self.shared.coef_[0, 0]), float(self.shared.intercept_[0])]
        d["per_checkpoint_coef_intercept"] = {
            str(k): [float(v.coef_[0, 0]), float(v.intercept_[0])] for k, v in self.models.items()}
        return d


def fit_calibrator(train_df: pd.DataFrame, cfg) -> Calibrator:
    cal = Calibrator(cfg.calibration, cfg.calibration_per_checkpoint)
    if cfg.calibration == "none":
        return cal
    oof = oof_raw_probabilities(train_df, cfg)
    return cal.fit(oof, train_df["label"].to_numpy(), train_df["checkpoint"].to_numpy())
