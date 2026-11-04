"""Offline timestamp perturbations (SIMULATED robustness only).

These transformations rewrite packet timestamps of already-captured packet-level
flows and then every feature is recomputed from the new timestamps (features are
always built from the arrays by prefix_feature_vector, never patched).
They are NOT real evasive traffic: no tool was re-run, TCP dynamics are not
re-simulated, and a real attacker could change packet sizes/counts too.
Only a packet-level source (PCAP-derived flows) is accepted; nothing is applied to
completed-flow CSV statistics.

Low-and-slow (time dilation): every inter-arrival time is multiplied by ``factor`` > 1.
   Changes: IAT statistics and prefix duration. Unchanged: sizes, directions, order, label.
Timing jitter: iat' = iat * exp(N(0, sigma^2)) + U(0, add_max_s).
   Changes: IAT statistics, duration. Unchanged: sizes, directions, order (iat' >= 0, so
   chronological order is preserved), label.
Randomness: one RNG per flow seeded from (seed, crc32(flow_id)) -> reproducible regardless
of processing order.
"""
from __future__ import annotations

import zlib
from dataclasses import replace

import numpy as np


def _rng(seed: int, flow_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(flow_id.encode())])


def _rebuild(ts: np.ndarray, new_iat: np.ndarray) -> np.ndarray:
    return ts[0] + np.concatenate([[0.0], np.cumsum(new_iat)])


def low_and_slow(flow, factor: float):
    if factor <= 0:
        raise ValueError("factor must be > 0")
    iat = np.diff(flow.ts)
    return replace(flow, ts=_rebuild(flow.ts, iat * factor))


def timing_jitter(flow, sigma: float, add_max_s: float, seed: int):
    if sigma < 0 or add_max_s < 0:
        raise ValueError("sigma and add_max_s must be >= 0")
    rng = _rng(seed, flow.flow_id)
    iat = np.diff(flow.ts)
    new = iat * np.exp(rng.normal(0.0, sigma, size=iat.shape)) + rng.uniform(0.0, add_max_s, size=iat.shape) \
        if add_max_s > 0 else iat * np.exp(rng.normal(0.0, sigma, size=iat.shape))
    return replace(flow, ts=_rebuild(flow.ts, new))
