"""Causal metadata features.

STRICT CAUSAL RULE: every feature at checkpoint t is computed from packets
1..t of the flow ONLY (ts[:t], length[:t], direction[:t]). Nothing here ever
reads the packet arrays beyond index t, and flow-level totals
(``flow_total_bytes`` etc.) are stored only as evaluation metadata - they are
NOT in FEATURE_NAMES (a unit test enforces both properties).

Feature definitions (population std, ddof=0; "fwd" = direction +1):
  n_packets        t
  bytes_total      sum of the t packet lengths (IP-level bytes)
  fwd_packets / bwd_packets          packet counts per direction
  fwd_bytes / bwd_bytes              byte sums per direction
  fwd_packet_frac  fwd_packets / t
  len_mean/std/min/max               packet length statistics over the t packets
  fwd_len_mean / bwd_len_mean        mean length per direction (0 if none yet)
  iat_mean/std/min/max               statistics of the t-1 inter-arrival times
  duration         ts[t-1] - ts[0]  (elapsed time of the prefix, not of the flow)

Absolute timestamps are never used, only differences within the prefix.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "n_packets", "bytes_total", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "fwd_packet_frac", "len_mean", "len_std", "len_min", "len_max",
    "fwd_len_mean", "bwd_len_mean", "iat_mean", "iat_std", "iat_min", "iat_max", "duration",
]

# Meta columns that are carried with every prefix row but are NOT model inputs.
META_COLUMNS = ["flow_id", "label", "source_file", "checkpoint", "flow_total_packets", "flow_total_bytes"]


def prefix_feature_vector(ts, length, direction, t: int) -> np.ndarray:
    """Features of the first ``t`` packets. Raises if the flow is shorter (no padding)."""
    if t < 2:
        raise ValueError("prefix length must be >= 2 (inter-arrival times need two packets)")
    if len(ts) < t:
        raise ValueError(f"flow has {len(ts)} packets, cannot build a {t}-packet prefix")
    tt = np.asarray(ts[:t], dtype=np.float64)
    ll = np.asarray(length[:t], dtype=np.float64)
    fwd = np.asarray(direction[:t]) > 0
    bwd = ~fwd
    iat = np.diff(tt)
    n_f, n_b = int(fwd.sum()), int(bwd.sum())
    return np.array([
        t, ll.sum(), n_f, n_b, ll[fwd].sum(), ll[bwd].sum(),
        n_f / t, ll.mean(), ll.std(), ll.min(), ll.max(),
        ll[fwd].mean() if n_f else 0.0, ll[bwd].mean() if n_b else 0.0,
        iat.mean(), iat.std(), iat.min(), iat.max(), tt[-1] - tt[0],
    ], dtype=np.float64)
