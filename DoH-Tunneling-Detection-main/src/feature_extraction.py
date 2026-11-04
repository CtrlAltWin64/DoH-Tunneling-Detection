"""Causal metadata features for Random Forest and CNN-GRU classifiers.

STRICT CAUSAL RULE: every feature at checkpoint t is computed from packets
1..t of the flow ONLY (ts[:t], length[:t], direction[:t]). Nothing here ever
reads the packet arrays beyond index t, and flow-level totals
(``flow_total_bytes`` etc.) are stored only as evaluation metadata - they are
NOT in FEATURE_NAMES (a unit test enforces both properties).

Target-leakage prevention:
- Flow identifiers, capture filenames, labels, and future packets are never
  used as model input features.
- Absolute timestamps are never used: only inter-arrival times (differences)
  and elapsed duration within the observed prefix are computed.

1. Random Forest Prefix Features (18 features, float64, 1D array of shape (18,)):
   0. n_packets       : t (count of packets in the prefix, dimensionless)
   1. bytes_total     : cumulative IP total length of the t packets (bytes)
   2. fwd_packets     : count of packets in forward direction (+1) (count)
   3. bwd_packets     : count of packets in backward direction (-1) (count)
   4. fwd_bytes       : byte sum of forward packets (bytes)
   5. bwd_bytes       : byte sum of backward packets (bytes)
   6. fwd_packet_frac : fwd_packets / t (fraction in [0, 1], dimensionless)
   7. len_mean        : mean IP packet length over the t packets (bytes)
   8. len_std         : population standard deviation (ddof=0) of length (bytes)
   9. len_min         : minimum IP packet length over the t packets (bytes)
  10. len_max         : maximum IP packet length over the t packets (bytes)
  11. fwd_len_mean    : mean length of forward packets (bytes; 0.0 if no fwd packets)
  12. bwd_len_mean    : mean length of backward packets (bytes; 0.0 if no bwd packets)
  13. iat_mean        : mean of the t-1 inter-arrival times (seconds)
  14. iat_std         : population standard deviation of IATs (seconds)
  15. iat_min         : minimum IAT over the t-1 intervals (seconds)
  16. iat_max         : maximum IAT over the t-1 intervals (seconds)
  17. duration        : prefix elapsed time ts[t-1] - ts[0] (seconds)

2. CNN-GRU Sequential Packet Features (8 features per step, float32, 2D matrix (t, 8)):
   0. pkt_len         : IP total packet length (bytes)
   1. direction       : direction relative to initiator (+1.0 forward, -1.0 backward)
   2. iat             : inter-arrival time from previous packet; 0.0 for first packet (seconds)
   3. elapsed         : elapsed time since flow start, ts[i] - ts[0] (seconds)
   4. cum_pkts_fwd    : cumulative forward packet count up to packet i (count)
   5. cum_pkts_bwd    : cumulative backward packet count up to packet i (count)
   6. cum_bytes_fwd   : cumulative forward bytes up to packet i (bytes)
   7. cum_bytes_bwd   : cumulative backward bytes up to packet i (bytes)
"""
from __future__ import annotations

import numpy as np

# 18 Random Forest prefix features
FEATURE_NAMES: list[str] = [
    "n_packets", "bytes_total", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "fwd_packet_frac", "len_mean", "len_std", "len_min", "len_max",
    "fwd_len_mean", "bwd_len_mean", "iat_mean", "iat_std", "iat_min", "iat_max", "duration",
]

# 8 CNN-GRU sequential packet features
SEQ_FEATURE_NAMES: list[str] = [
    "pkt_len", "direction", "iat", "elapsed",
    "cum_pkts_fwd", "cum_pkts_bwd", "cum_bytes_fwd", "cum_bytes_bwd",
]
N_SEQ_FEATURES: int = len(SEQ_FEATURE_NAMES)

# Meta columns carried with every prefix row for evaluation and tracking, NOT model inputs
META_COLUMNS: list[str] = [
    "flow_id", "label", "source_file", "checkpoint", "flow_total_packets", "flow_total_bytes"
]


def prefix_feature_vector(ts, length, direction, t: int) -> np.ndarray:
    """Features of the first ``t`` packets for Random Forest.

    Parameters
    ----------
    ts : sequence of float timestamps (seconds)
    length : sequence of int IP total packet lengths (bytes)
    direction : sequence of int directions (+1 for forward, -1 for backward)
    t : prefix length (number of packets to observe)

    Returns
    -------
    np.ndarray, shape (18,), dtype np.float64

    Raises
    ------
    ValueError: if t < 2 or len(ts) < t.
    """
    if t < 2:
        raise ValueError("prefix length must be >= 2 (inter-arrival times need two packets)")
    if len(ts) < t:
        raise ValueError(f"flow has {len(ts)} packets, cannot build a {t}-packet prefix")

    tt = np.asarray(ts[:t], dtype=np.float64)
    ll = np.asarray(length[:t], dtype=np.float64)
    dd = np.asarray(direction[:t])
    fwd = dd > 0
    bwd = ~fwd
    iat = np.diff(tt)
    n_f, n_b = int(fwd.sum()), int(bwd.sum())

    return np.array([
        float(t),
        float(ll.sum()),
        float(n_f),
        float(n_b),
        float(ll[fwd].sum()) if n_f else 0.0,
        float(ll[bwd].sum()) if n_b else 0.0,
        float(n_f / t),
        float(ll.mean()),
        float(ll.std()),
        float(ll.min()),
        float(ll.max()),
        float(ll[fwd].mean()) if n_f else 0.0,
        float(ll[bwd].mean()) if n_b else 0.0,
        float(iat.mean()),
        float(iat.std()),
        float(iat.min()),
        float(iat.max()),
        float(tt[-1] - tt[0]),
    ], dtype=np.float64)


def extract_packet_sequence(ts, length, direction, t: int) -> np.ndarray:
    """Extract causal per-packet sequence matrix for CNN-GRU.

    Parameters
    ----------
    ts : sequence of float timestamps (seconds)
    length : sequence of int IP total packet lengths (bytes)
    direction : sequence of int directions (+1 for forward, -1 for backward)
    t : prefix length (number of packets to extract)

    Returns
    -------
    np.ndarray, shape (t, 8), dtype np.float32:
        Columns: [pkt_len, direction, iat, elapsed, cum_pkts_fwd, cum_pkts_bwd, cum_bytes_fwd, cum_bytes_bwd]

    Raises
    ------
    ValueError: if t < 1 or len(ts) < t.
    """
    if t < 1:
        raise ValueError("prefix length t must be >= 1 for sequence extraction")
    if len(ts) < t:
        raise ValueError(f"flow has {len(ts)} packets, cannot build a {t}-packet sequence")

    tt = np.asarray(ts[:t], dtype=np.float64)
    ln = np.asarray(length[:t], dtype=np.float64)
    dr = np.asarray(direction[:t], dtype=np.float64)

    fwd = dr > 0
    cum_fwd = np.cumsum(fwd).astype(np.float64)
    cum_bwd = np.cumsum(~fwd).astype(np.float64)
    cum_fwd_bytes = np.cumsum(np.where(fwd, ln, 0.0))
    cum_bwd_bytes = np.cumsum(np.where(~fwd, ln, 0.0))

    iat = np.concatenate([[0.0], np.diff(tt)])
    elapsed = tt - tt[0]

    mat = np.stack([
        ln,
        dr,
        iat,
        elapsed,
        cum_fwd,
        cum_bwd,
        cum_fwd_bytes,
        cum_bwd_bytes,
    ], axis=1).astype(np.float32)

    return mat
