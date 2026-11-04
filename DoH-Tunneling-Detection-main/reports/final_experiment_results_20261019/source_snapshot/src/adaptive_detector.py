"""Adaptive stopping policy, fixed-window policy and FPR-constrained threshold selection.

One abstraction covers both detectors:
  * adaptive detector  = policy over ALL checkpoints (10,20,30,40)
  * fixed-window at t  = the same policy restricted to the single checkpoint t

Policy for a flow with calibrated probabilities p_1..p_K at checkpoints c_1<..<c_K:
  stop at the first k with p_k >= tau  -> ALERT at c_k (bytes/time up to and incl. packet c_k)
  if no k qualifies                    -> NOT DETECTED by the observation limit; the flow was
                                          observed up to c_K (convention for cost/latency).
The decision for checkpoint k uses only p_1..p_k (causal): later checkpoints are
never consulted once an alert fired (tested in tests/test_stopping_rule.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FlowCurves:
    """Per-flow arrays over the checkpoints; only flows that can supply ALL checkpoints."""
    flow_ids: np.ndarray
    y: np.ndarray            # (n,)   1 = malicious
    P: np.ndarray            # (n,K)  calibrated probabilities
    dur: np.ndarray          # (n,K)  prefix duration (seconds) at each checkpoint
    bytes_cum: np.ndarray    # (n,K)  bytes observed up to each checkpoint
    total_bytes: np.ndarray  # (n,)   total counted bytes of the flow (DEBD denominator)
    cps: np.ndarray          # (K,)

    def subset(self, mask) -> "FlowCurves":
        return FlowCurves(self.flow_ids[mask], self.y[mask], self.P[mask], self.dur[mask],
                          self.bytes_cum[mask], self.total_bytes[mask], self.cps)


def curves_from_prefix_table(df: pd.DataFrame, proba: np.ndarray, checkpoints) -> FlowCurves:
    """Pivot the prefix table into per-flow curves; flows lacking any checkpoint are dropped."""
    cps = np.asarray(checkpoints)
    d = df[["flow_id", "label", "checkpoint", "duration", "bytes_total", "flow_total_bytes"]].copy()
    d["p"] = proba
    counts = d.groupby("flow_id")["checkpoint"].nunique()
    full = counts[counts == len(cps)].index
    d = d[d["flow_id"].isin(full)].sort_values(["flow_id", "checkpoint"], kind="stable")
    if d.empty:
        raise ValueError("no flow in this partition can supply all checkpoints")
    n = d["flow_id"].nunique()
    K = len(cps)
    g = d.groupby("flow_id", sort=True)
    first = g.first()
    return FlowCurves(
        flow_ids=first.index.to_numpy(),
        y=first["label"].to_numpy().astype(int),
        P=d["p"].to_numpy().reshape(n, K),
        dur=d["duration"].to_numpy().reshape(n, K),
        bytes_cum=d["bytes_total"].to_numpy().reshape(n, K),
        total_bytes=first["flow_total_bytes"].to_numpy().astype(float),
        cps=cps,
    )


def run_policy(P: np.ndarray, tau: float):
    """Return (alert[n] bool, k_stop[n] int) where k_stop = index of first crossing, or -1."""
    cross = P >= tau
    alert = cross.any(axis=1)
    k_stop = np.where(alert, cross.argmax(axis=1), -1)
    return alert, k_stop


def policy_outcomes(curves: FlowCurves, tau: float, col_idx=None) -> pd.DataFrame:
    """Per-flow outcome table for the policy restricted to checkpoint columns ``col_idx``."""
    idx = list(range(len(curves.cps))) if col_idx is None else list(col_idx)
    P = curves.P[:, idx]
    cps = curves.cps[idx]
    alert, k_stop = run_policy(P, tau)
    j = np.where(alert, k_stop, len(idx) - 1)          # observation index (undetected -> last)
    rows = np.arange(len(P))
    observed = cps[j]
    bytes_obs = curves.bytes_cum[:, idx][rows, j]
    dur_obs = curves.dur[:, idx][rows, j]
    # Distinguish early alert (< final checkpoint) vs terminal alert (= final checkpoint) vs unresolved
    is_terminal = alert & (k_stop == len(idx) - 1)
    is_early = alert & (k_stop < len(idx) - 1)
    decision = np.where(is_early, "early_alert", np.where(is_terminal, "terminal_alert", "unresolved"))
    return pd.DataFrame({
        "flow_id": curves.flow_ids,
        "label": curves.y,
        "alert": alert,
        "is_early": is_early,
        "is_terminal": is_terminal,
        "decision": decision,
        "alert_packets": np.where(alert, observed, np.nan),   # NaN = not detected by the limit
        "observed_packets": observed,                         # Tmax when undetected
        "latency_s": np.where(alert, dur_obs, np.nan),        # elapsed time to the alert packet
        "bytes_pre": bytes_obs,                               # incl. the packet at the stopping point
        "total_bytes": curves.total_bytes,
    })


def sweep_thresholds(curves: FlowCurves, thresholds, col_idx=None) -> pd.DataFrame:
    """Validation sweep: complete-policy FPR, coverage and cost for each candidate tau.

    FPR counts a benign flow once if it alerts at ANY checkpoint (never per checkpoint).
    """
    idx = list(range(len(curves.cps))) if col_idx is None else list(col_idx)
    P = curves.P[:, idx]
    cps = curves.cps[idx]
    mx = P.max(axis=1)                                     # alert at some checkpoint  <=>  max_k p_k >= tau
    ben, mal = curves.y == 0, curves.y == 1
    rows = []
    for tau in thresholds:
        alert = mx >= tau
        fp = int((alert & ben).sum())
        n_ben = int(ben.sum())
        n_mal = int(mal.sum())
        det = int((alert & mal).sum())
        first_k = np.where(alert[:, None] & (P >= tau), np.arange(len(idx))[None, :], len(idx)).min(axis=1)
        obs = np.where(alert, cps[np.minimum(first_k, len(idx) - 1)], cps[-1])
        rows.append({
            "tau": float(tau), "fp": fp, "n_benign": n_ben, "fpr": fp / n_ben,
            "coverage": det / n_mal if n_mal else np.nan,
            "mean_cost_malicious": float(obs[mal].mean() / cps[-1]) if n_mal else np.nan,
        })
    return pd.DataFrame(rows)


def threshold_grid(step: float) -> np.ndarray:
    g = np.round(np.arange(step, 1.0 + step / 2, step), 10)
    return np.concatenate([g, [1.0 + 1e-9]])                # last value = "never alert" sentinel


def select_threshold(curves: FlowCurves, alpha: float, step: float, col_idx=None) -> dict:
    """Lowest grid threshold whose COMPLETE sequential policy has validation FPR <= alpha.

    Lower tau => more alerts, higher coverage, earlier stops (monotone), so the lowest
    feasible tau is the best feasible choice on all three axes. Selection touches only the
    validation curves passed in.
    """
    sweep = sweep_thresholds(curves, threshold_grid(step), col_idx)
    feasible = sweep[sweep["fpr"] <= alpha]
    n_ben = int(sweep["n_benign"].iloc[0])
    info = {"alpha": alpha, "n_benign_val": n_ben, "sweep": sweep,
            "min_resolvable_fpr": 1.0 / n_ben if n_ben else np.nan}
    chosen = feasible.iloc[0]
    info.update(tau=float(chosen["tau"]), val_fp=int(chosen["fp"]), val_fpr=float(chosen["fpr"]),
                val_coverage=float(chosen["coverage"]),
                never_alerts=bool(chosen["tau"] > 1.0),
                resolution_warning=bool(int(np.floor(alpha * n_ben + 1e-12)) == 0))
    return info
