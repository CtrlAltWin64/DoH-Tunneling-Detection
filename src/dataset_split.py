"""Flow-level and capture-level train/validation/test split with strict leakage prevention.

Splitting is done on ORIGINAL FLOWS (or on source-file groups), never on prefix
rows, so all prefixes of a flow share one partition. Uses scikit-learn's
StratifiedGroupKFold(5 folds, shuffled, seeded): fold 0 -> test, fold 1 ->
validation, folds 2-4 -> train. Class balance is approximately preserved.

Also provides:
- split_train_calibration(): splits training partition into model-fitting (train_fit)
  and calibration (train_cal) subsets with zero group overlap.
- create_split_manifest(): generates a complete audit record of flows, captures,
  class distributions, and group overlap verification per partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PARTITIONS = ("train", "val", "test")


def split_flows(flow_meta: pd.DataFrame, group_by: str, seed: int) -> dict[str, str]:
    """flow_meta needs columns flow_id, label, source_file (one row per flow).
    Returns {flow_id: 'train'|'val'|'test'}."""
    for c in ("flow_id", "label", "source_file"):
        if c not in flow_meta.columns:
            raise ValueError(f"flow_meta missing column {c!r}")
    if flow_meta["flow_id"].duplicated().any():
        raise ValueError("duplicate flow_id values - flow identifiers must be unique")
    meta = flow_meta.reset_index(drop=True)
    groups = meta["flow_id"] if group_by == "flow" else meta["source_file"]
    y = meta["label"].to_numpy()
    for lab in (0, 1):
        n_groups = groups[y == lab].nunique()
        if n_groups < 5:
            raise ValueError(
                f"class {lab} has only {n_groups} {group_by} group(s); at least 5 are needed for a "
                f"60/20/20 split" + (" - use group_by='flow'" if group_by != "flow" else ""))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = [te for _, te in skf.split(np.zeros(len(meta)), y, groups)]
    assign = {}
    for k, idx in enumerate(folds):
        part = "test" if k == 0 else "val" if k == 1 else "train"
        for fid in meta.loc[idx, "flow_id"]:
            assign[fid] = part

    # Strict integrity and leakage checks
    meta["partition"] = meta["flow_id"].map(assign)
    for part in PARTITIONS:
        labs = set(meta.loc[meta["partition"] == part, "label"])
        if labs != {0, 1}:
            raise ValueError(f"partition {part!r} does not contain both classes ({labs}); data too small/imbalanced")

    # Assert zero group overlap across partitions
    train_groups = set(meta.loc[meta["partition"] == "train", "flow_id" if group_by == "flow" else "source_file"])
    val_groups = set(meta.loc[meta["partition"] == "val", "flow_id" if group_by == "flow" else "source_file"])
    test_groups = set(meta.loc[meta["partition"] == "test", "flow_id" if group_by == "flow" else "source_file"])

    if train_groups & val_groups:
        raise RuntimeError(f"Data leakage detected: {len(train_groups & val_groups)} groups overlap between train and val")
    if train_groups & test_groups:
        raise RuntimeError(f"Data leakage detected: {len(train_groups & test_groups)} groups overlap between train and test")
    if val_groups & test_groups:
        raise RuntimeError(f"Data leakage detected: {len(val_groups & test_groups)} groups overlap between val and test")

    return assign


def split_train_calibration(
    train_meta: pd.DataFrame, group_by: str, seed: int, cal_ratio: float = 0.25
) -> dict[str, str]:
    """Split training flows into model-fitting (train_fit) and calibration (train_cal) subsets.

    Used by CNN-GRU to ensure the probability calibrator is fitted on genuine
    out-of-sample predictions rather than in-sample training predictions.

    Parameters
    ----------
    train_meta : DataFrame with columns flow_id, label, source_file for training flows only.
    group_by   : "flow" or "source_file"
    seed       : random seed
    cal_ratio  : target calibration fraction (default 0.25 -> ~75% fit, ~25% cal)

    Returns
    -------
    dict mapping flow_id -> 'train_fit' | 'train_cal'
    """
    for c in ("flow_id", "label", "source_file"):
        if c not in train_meta.columns:
            raise ValueError(f"train_meta missing column {c!r}")
    if train_meta["flow_id"].duplicated().any():
        raise ValueError("duplicate flow_id values in train_meta")

    meta = train_meta.reset_index(drop=True)
    groups = meta["flow_id"] if group_by == "flow" else meta["source_file"]
    y = meta["label"].to_numpy()

    # Determine number of folds (default 4 folds -> 1 fold = 25% calibration)
    n_splits = max(2, int(round(1.0 / cal_ratio))) if cal_ratio > 0 else 4
    for lab in (0, 1):
        n_groups = groups[y == lab].nunique()
        if n_groups < n_splits:
            # Fallback if too few groups for n_splits: use 2 splits if possible
            if n_groups >= 2:
                n_splits = 2
            else:
                raise ValueError(
                    f"class {lab} has only {n_groups} {group_by} group(s) in training data; "
                    f"cannot split into separate model-fitting and calibration subsets"
                )

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [te for _, te in skf.split(np.zeros(len(meta)), y, groups)]

    # Fold 0 is calibration, remaining folds are model-fitting
    assign = {}
    cal_idx = set(folds[0])
    for idx in range(len(meta)):
        fid = meta.loc[idx, "flow_id"]
        assign[fid] = "train_cal" if idx in cal_idx else "train_fit"

    meta["sub_part"] = meta["flow_id"].map(assign)
    fit_groups = set(meta.loc[meta["sub_part"] == "train_fit", "flow_id" if group_by == "flow" else "source_file"])
    cal_groups = set(meta.loc[meta["sub_part"] == "train_cal", "flow_id" if group_by == "flow" else "source_file"])

    if fit_groups & cal_groups:
        raise RuntimeError("Data leakage: overlapping groups between train_fit and train_cal")

    # Verify both classes in fit and cal subsets
    for sp in ("train_fit", "train_cal"):
        labs = set(meta.loc[meta["sub_part"] == sp, "label"])
        if labs != {0, 1}:
            raise ValueError(f"subset {sp!r} does not contain both classes ({labs})")

    return assign


def create_split_manifest(flow_meta: pd.DataFrame, assign: dict[str, str], group_by: str) -> dict:
    """Generate split audit report with flow counts, class counts, capture files, and overlap checks."""
    df = flow_meta.copy()
    df["partition"] = df["flow_id"].map(assign)
    manifest = {
        "group_by": group_by,
        "total_flows": len(df),
        "total_benign": int((df["label"] == 0).sum()),
        "total_malicious": int((df["label"] == 1).sum()),
        "partitions": {},
    }
    for part in PARTITIONS:
        sub = df[df["partition"] == part]
        ben = int((sub["label"] == 0).sum())
        mal = int((sub["label"] == 1).sum())
        files = sorted(sub["source_file"].unique().tolist())
        manifest["partitions"][part] = {
            "n_flows": len(sub),
            "n_benign": ben,
            "n_malicious": mal,
            "n_source_files": len(files),
            "source_files": files,
        }

    # Verify group disjointness
    p_groups = {
        p: set(df.loc[df["partition"] == p, "flow_id" if group_by == "flow" else "source_file"])
        for p in PARTITIONS
    }
    manifest["leakage_check"] = {
        "train_val_overlap": len(p_groups["train"] & p_groups["val"]),
        "train_test_overlap": len(p_groups["train"] & p_groups["test"]),
        "val_test_overlap": len(p_groups["val"] & p_groups["test"]),
        "verified_disjoint": bool(
            len(p_groups["train"] & p_groups["val"]) == 0
            and len(p_groups["train"] & p_groups["test"]) == 0
            and len(p_groups["val"] & p_groups["test"]) == 0
        ),
    }
    return manifest
