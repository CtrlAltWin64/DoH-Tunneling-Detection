"""Progressive prefix table: one row per (flow, checkpoint) the flow can supply.

Design principles:
- Uses the project's standard 4 checkpoints: (10, 20, 30, 40) packets by default.
- Causal: each row at checkpoint t uses ONLY the first t packets of that flow.
- Short-flow policy: A flow with fewer than t packets simply has no row for checkpoint t
  (never padded with fabricated packets or zero-filled).
- Deterministic: Rows are produced in flow order, then checkpoint order.
- Provenance: Preserves flow_id, label, source_file, checkpoint, total_packets, total_bytes.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .feature_extraction import FEATURE_NAMES, META_COLUMNS, prefix_feature_vector

DEFAULT_CHECKPOINTS: tuple[int, ...] = (10, 20, 30, 40)


def _validate_checkpoints(checkpoints: Sequence[int] | None) -> list[int]:
    """Validate and normalize checkpoint sizes."""
    if checkpoints is None:
        return list(DEFAULT_CHECKPOINTS)
    cps = sorted({int(t) for t in checkpoints})
    if not cps:
        raise ValueError("Checkpoints list cannot be empty")
    if cps[0] < 2:
        raise ValueError(f"All checkpoints must be >= 2 (inter-arrival times need two packets), got {cps[0]}")
    return cps


def generate_flow_prefixes(flow, checkpoints: Sequence[int] | None = None) -> list[tuple[tuple, np.ndarray]]:
    """Generate (meta, features) pairs for all valid checkpoints of a single flow.

    Parameters
    ----------
    flow : Flow object
    checkpoints : sequence of int checkpoints, e.g. (10, 20, 30, 40)

    Returns
    -------
    list of (meta_tuple, feature_vector) for each checkpoint where flow has >= t packets.
    """
    cps = _validate_checkpoints(checkpoints)
    n_avail = len(flow.ts)
    results = []
    for t in cps:
        if n_avail < t:
            continue
        meta = (flow.flow_id, flow.label, flow.source_file, t, flow.total_packets, flow.total_bytes)
        feat = prefix_feature_vector(flow.ts, flow.length, flow.direction, t)
        results.append((meta, feat))
    return results


def build_prefix_table(flows, checkpoints: Sequence[int] | None = None) -> pd.DataFrame:
    """Build progressive prefix DataFrame for all flows and checkpoints.

    Parameters
    ----------
    flows : sequence of Flow objects
    checkpoints : sequence of int checkpoints (default: (10, 20, 30, 40))

    Returns
    -------
    pd.DataFrame with META_COLUMNS + FEATURE_NAMES.
    """
    cps = _validate_checkpoints(checkpoints)
    metas, feats = [], []
    for fl in flows:
        for meta, feat in generate_flow_prefixes(fl, cps):
            metas.append(meta)
            feats.append(feat)

    if not metas:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_NAMES)

    df_meta = pd.DataFrame(metas, columns=META_COLUMNS)
    df_feat = pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)
    return pd.concat([df_meta, df_feat], axis=1)


def eligible_counts(flows, checkpoints: Sequence[int] | None = None) -> pd.DataFrame:
    """Number and fraction of flows able to supply each checkpoint, partitioned by label.

    Parameters
    ----------
    flows : sequence of Flow objects
    checkpoints : sequence of int checkpoints (default: (10, 20, 30, 40))

    Returns
    -------
    pd.DataFrame with counts per checkpoint and class.
    """
    cps = _validate_checkpoints(checkpoints)
    rows = []
    for lab, name in ((0, "benign"), (1, "malicious")):
        sel = [f for f in flows if f.label == lab]
        n_flows = len(sel)
        row = {"label": name, "n_flows": n_flows}
        for t in cps:
            cnt = sum(len(f.ts) >= t for f in sel)
            row[f"n_ge_{t}"] = cnt
            row[f"pct_ge_{t}"] = round(100.0 * cnt / n_flows, 2) if n_flows > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
