"""Shared helpers for tests (synthetic data with known structure)."""
import numpy as np

from src.adaptive_detector import FlowCurves
from src.preprocessing import Flow


def make_flow(n=60, label=0, fid="f0", seed=0, source="a.pcap"):
    rng = np.random.default_rng(seed)
    ts = 100.0 + np.concatenate([[0.0], np.cumsum(rng.uniform(0.001, 0.2, n - 1))])
    length = rng.integers(60, 1500, n).astype(np.int64)
    direction = rng.choice([-1, 1], n).astype(np.int8)
    direction[0] = 1
    return Flow(fid, label, source, ts, length, direction, n, int(length.sum()))


def make_curves(P, y, cps=(10, 20, 30, 40), bytes_cum=None, total_bytes=None, dur=None):
    P = np.asarray(P, dtype=float)
    n, K = P.shape
    cps = np.asarray(cps[:K])
    if bytes_cum is None:
        bytes_cum = np.tile(cps * 100.0, (n, 1))
    if total_bytes is None:
        total_bytes = np.full(n, cps[-1] * 100.0)
    if dur is None:
        dur = np.tile(cps * 0.1, (n, 1))
    return FlowCurves(np.array([f"f{i}" for i in range(n)]), np.asarray(y), P,
                      np.asarray(dur, float), np.asarray(bytes_cum, float), np.asarray(total_bytes, float), cps)
