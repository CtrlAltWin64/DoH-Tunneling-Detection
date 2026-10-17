"""Flow-level and capture-level train/validation/test split with strict leakage prevention.

Splitting is done on ORIGINAL FLOWS (or on source-file capture groups), never on
prefix rows, so all prefixes of a flow share one partition. Uses scikit-learn's
StratifiedGroupKFold(5 folds, shuffled, seeded): fold 0 -> test, fold 1 ->
validation, folds 2-4 -> train (60% train, 20% val, 20% test).

Strict experimental protocol:
- Final test partition is strictly isolated: never used for fitting models, calibrators,
  early stopping, or threshold selection.
- Calibration subset (train_cal) is strictly held-out from model training (train_fit)
  with zero group overlap.
- Assertions verify that group overlap between all pairs of partitions is 0.
- Complete audit manifest records group IDs, classes, counts, seeds, and disjointness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PARTITIONS = ("train", "val", "test")


def split_flows(flow_meta: pd.DataFrame, group_by: str, seed: int) -> dict[str, str]:
    """Split flows into train/val/test partitions by flow or source_file group.

    Parameters
    ----------
    flow_meta : DataFrame with columns 'flow_id', 'label', 'source_file'
    group_by  : 'flow' or 'source_file'
    seed      : integer random seed for reproducibility

    Returns
    -------
    dict mapping flow_id -> 'train' | 'val' | 'test'
    """
    for c in ("flow_id", "label", "source_file"):
        if c not in flow_meta.columns:
            raise ValueError(f"flow_meta missing required column {c!r}")
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
                f"60/20/20 split" + (" - use group_by='flow'" if group_by != "flow" else "")
            )

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
            raise ValueError(
                f"partition {part!r} does not contain both classes ({labs}); data too small/imbalanced"
            )

    # Assert zero group overlap across partitions
    group_col = "flow_id" if group_by == "flow" else "source_file"
    train_groups = set(meta.loc[meta["partition"] == "train", group_col])
    val_groups = set(meta.loc[meta["partition"] == "val", group_col])
    test_groups = set(meta.loc[meta["partition"] == "test", group_col])

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
    group_by   : 'flow' or 'source_file'
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
    group_col = "flow_id" if group_by == "flow" else "source_file"
    fit_groups = set(meta.loc[meta["sub_part"] == "train_fit", group_col])
    cal_groups = set(meta.loc[meta["sub_part"] == "train_cal", group_col])

    if fit_groups & cal_groups:
        raise RuntimeError("Data leakage: overlapping groups between train_fit and train_cal")

    # Verify both classes in fit and cal subsets
    for sp in ("train_fit", "train_cal"):
        labs = set(meta.loc[meta["sub_part"] == sp, "label"])
        if labs != {0, 1}:
            raise ValueError(f"subset {sp!r} does not contain both classes ({labs})")

    return assign


def create_split_manifest(
    flow_meta: pd.DataFrame, assign: dict[str, str], group_by: str, seed: int | None = None
) -> dict:
    """Generate split audit report with flow counts, class counts, capture files, and overlap checks."""
    df = flow_meta.copy()
    df["partition"] = df["flow_id"].map(assign)
    group_col = "flow_id" if group_by == "flow" else "source_file"

    manifest = {
        "group_by": group_by,
        "seed": seed,
        "total_flows": len(df),
        "total_benign": int((df["label"] == 0).sum()),
        "total_malicious": int((df["label"] == 1).sum()),
        "total_groups": int(df[group_col].nunique()),
        "partitions": {},
    }
    for part in PARTITIONS:
        sub = df[df["partition"] == part]
        ben = int((sub["label"] == 0).sum())
        mal = int((sub["label"] == 1).sum())
        files = sorted(sub["source_file"].unique().tolist())
        groups_in_part = sorted(sub[group_col].unique().tolist())
        manifest["partitions"][part] = {
            "n_flows": len(sub),
            "n_benign": ben,
            "n_malicious": mal,
            "n_groups": len(groups_in_part),
            "n_source_files": len(files),
            "source_files": files,
        }

    # Verify group disjointness
    p_groups = {
        p: set(df.loc[df["partition"] == p, group_col])
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


def save_split_manifest(manifest: dict, out_path: str | Path) -> Path:
    """Save split manifest dictionary to JSON file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p
