"""CNN-GRU probability calibration using a strictly separated calibration subset.

Methodology:
1. The training partition is split into model-fitting (train_fit) and calibration (train_cal)
   subsets, keeping all prefixes of a flow together (grouped by flow or capture).
2. The CNN-GRU is trained ONLY on train_fit flows.
3. The trained CNN-GRU scores the held-out train_cal flows (never seen during backpropagation).
4. A monotonic Platt scaling calibrator is fitted on these genuine out-of-sample predictions.
5. The validation partition is evaluated using this frozen calibrator to select the sequential
   stopping threshold tau under the FPR constraint (alpha <= 0.01).
6. The test partition is evaluated once at the very end with frozen model, calibrator, and tau.

This strictly eliminates in-sample overconfidence bias and guarantees that test data
never touches model fitting, calibration, or threshold selection.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .calibration import Calibrator
from .cnn_gru_model import CnnGruClassifier, SequenceNormalizer
from .cnn_gru_prefix import build_cnn_prefix_table, raw_proba_cnn

logger = logging.getLogger(__name__)


def fit_calibrator_cnn(
    cal_flows,
    checkpoints,
    model: CnnGruClassifier,
    normalizer: SequenceNormalizer,
    cfg,
    device: str = "cpu",
) -> tuple[Calibrator, pd.DataFrame]:
    """Fit Platt calibrator on genuine out-of-sample predictions from held-out calibration flows.

    Parameters
    ----------
    cal_flows   : list of Flow objects held out from model fitting (train_cal)
    checkpoints : tuple of int (e.g. (10, 20, 30, 40))
    model       : trained CnnGruClassifier
    normalizer  : fitted SequenceNormalizer
    cfg         : Config
    device      : compute device ("cpu", "cuda")

    Returns
    -------
    calibrator : fitted Calibrator
    cal_df     : DataFrame with calibration prefix table and raw scores
    """
    cal = Calibrator(cfg.calibration, cfg.calibration_per_checkpoint)
    cal_df = build_cnn_prefix_table(
        cal_flows, checkpoints, model, normalizer, device=device
    )

    if cfg.calibration == "none":
        return cal, cal_df

    raw_scores = raw_proba_cnn(cal_df)
    y = cal_df["label"].to_numpy()
    cps = cal_df["checkpoint"].to_numpy()

    # Fit Platt scaling (logistic regression on logit(raw)) with monotonicity constraints
    cal.fit(raw_scores, y, cps)
    logger.info(
        f"[CNN-GRU calibration] Fitted on {len(cal_flows)} held-out calibration flows "
        f"({len(cal_df)} prefix rows). Fallbacks: {cal.fallbacks}"
    )
    return cal, cal_df
