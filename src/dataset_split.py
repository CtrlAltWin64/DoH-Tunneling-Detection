"""Flow-level train/validation/test split (60/20/20).

Splitting is done on ORIGINAL FLOWS (or on source-file groups), never on prefix
rows, so all prefixes of a flow share one partition. Uses scikit-learn's
StratifiedGroupKFold(5 folds, shuffled, seeded): fold 0 -> test, fold 1 ->
validation, folds 2-4 -> train. Class balance is approximately preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PARTITIONS = ("train", "val", "test")


def split_flows(flow_meta: pd.DataFrame, group_by: str, seed: int) -> dict:
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
    part_of = meta["flow_id"].map(assign).to_numpy()
    for part in PARTITIONS:
        labs = set(y[part_of == part])
        if labs != {0, 1}:
            raise ValueError(f"partition {part!r} does not contain both classes ({labs}); data too small/imbalanced")
    return assign
