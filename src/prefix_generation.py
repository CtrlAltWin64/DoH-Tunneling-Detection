"""Progressive prefix table: one row per (flow, checkpoint) the flow can supply.

* A flow with fewer than t packets simply has no row for checkpoint t
  (never padded with fabricated packets).
* Rows are produced in flow order, then checkpoint order -> deterministic.
* The original ``flow_id`` is kept on every row so that splitting can be done
  per flow (see dataset_split.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_extraction import FEATURE_NAMES, META_COLUMNS, prefix_feature_vector


def build_prefix_table(flows, checkpoints) -> pd.DataFrame:
    metas, feats = [], []
    for fl in flows:
        n_avail = len(fl.ts)
        for t in checkpoints:
            if n_avail < t:
                continue
            metas.append((fl.flow_id, fl.label, fl.source_file, t, fl.total_packets, fl.total_bytes))
            feats.append(prefix_feature_vector(fl.ts, fl.length, fl.direction, t))
    if not metas:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_NAMES)
    df = pd.DataFrame(metas, columns=META_COLUMNS)
    df = pd.concat([df, pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)], axis=1)
    return df


def eligible_counts(flows, checkpoints) -> pd.DataFrame:
    """Number of flows able to supply each checkpoint, per label."""
    rows = []
    for lab, name in ((0, "benign"), (1, "malicious")):
        sel = [f for f in flows if f.label == lab]
        row = {"label": name, "n_flows": len(sel)}
        for t in checkpoints:
            row[f"n_ge_{t}"] = sum(len(f.ts) >= t for f in sel)
        rows.append(row)
    return pd.DataFrame(rows)
