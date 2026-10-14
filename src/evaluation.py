"""Metrics. Positive class = malicious DoH tunnelling. Every definition is explicit.

Confusion counts (flow level, over all ELIGIBLE flows of the evaluated set):
  TP malicious flow that alerts          FN malicious flow that does NOT alert (incl. "not detected by the limit")
  FP benign flow that alerts (at ANY checkpoint)   TN benign flow that never alerts ("no-alert", not an FP)

precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN); FPR = FP/(FP+TN).
Detection coverage = detected malicious / eligible malicious. At flow level this equals
recall (undetected malicious flows are FN by definition); both are reported because the
project proposal defines both.

Observation cost: OC_i = packets observed / Tmax. Non-detected flows were observed up to
Tmax, so OC = 1. Reported over all flows and over malicious flows separately (benign flows
are never "done" under the adaptive rule, so the all-flow mean is dominated by them).

Detection latency: elapsed time between the first and the alert-triggering packet
(duration of the prefix at the stopping checkpoint). Defined only for detected flows; the
number of detected flows is reported next to it. Packet-count stopping and elapsed time are
different quantities (time can grow under low-and-slow while packet counts do not).

DEBD (Data Exposed Before Detection) -- a PROXY for pre-alert malicious-traffic exposure,
NOT a measurement of stolen data:
  Bpre   = IP-level bytes (headers + payload, BOTH directions) of the packets observed up to and
           INCLUDING the packet at the stopping checkpoint (they arrived before the decision),
           after retransmission removal.
  Btotal = all counted bytes of the flow in the capture (incomplete captures understate it and
           therefore overstate DEBD for short captures).
  DEBD_i = 100 * Bpre / Btotal, for malicious flows.
  Non-detected malicious flows get 100 % ("the whole flow passed unalerted") in the *_all
  variants; the *_detected variant excludes them and is only meaningful next to coverage.
  Mean per-flow DEBD weights every flow equally; byte-weighted DEBD = 100*sum(Bpre)/sum(Btotal)
  weights long/heavy flows more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> dict:
    """Exact binomial interval for k events in n trials, plus one-sided upper bound."""
    if n == 0:
        return {"ci_low": np.nan, "ci_high": np.nan, "upper_one_sided": np.nan}
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    up = 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    return {"ci_low": lo, "ci_high": hi, "upper_one_sided": up}


def _div(a, b):
    return float(a / b) if b else float("nan")


def summarize(out: pd.DataFrame, tmax: int) -> dict:
    """Metrics for one policy on one set of eligible flows (see module docstring)."""
    mal = out["label"].to_numpy() == 1
    ben = ~mal
    alert = out["alert"].to_numpy()
    tp, fn = int((alert & mal).sum()), int((~alert & mal).sum())
    fp, tn = int((alert & ben).sum()), int((~alert & ben).sum())
    det = alert & mal
    m = {
        "n_flows": len(out), "n_malicious": int(mal.sum()), "n_benign": int(ben.sum()),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "fpr": _div(fp, fp + tn),
        "coverage": _div(tp, tp + fn),
    }
    ci = clopper_pearson(fp, fp + tn)
    m.update({"fpr_ci95_low": ci["ci_low"], "fpr_ci95_high": ci["ci_high"], "fpr_upper95_one_sided": ci["upper_one_sided"]})
    obs = out["observed_packets"].to_numpy(dtype=float)
    m["avg_packets_all"] = float(obs.mean())
    m["avg_packets_malicious"] = float(obs[mal].mean()) if mal.any() else float("nan")
    m["cost_all"] = m["avg_packets_all"] / tmax
    m["cost_malicious"] = m["avg_packets_malicious"] / tmax
    lat = out.loc[det, "latency_s"]
    m["n_detected_for_latency"] = int(det.sum())
    m["latency_mean_s"] = float(lat.mean()) if len(lat) else float("nan")
    m["latency_median_s"] = float(lat.median()) if len(lat) else float("nan")
    if mal.any():
        bpre = out.loc[mal, "bytes_pre"].to_numpy(dtype=float)
        btot = out.loc[mal, "total_bytes"].to_numpy(dtype=float)
        a = alert[mal]
        per_flow_all = np.where(a, 100.0 * bpre / btot, 100.0)
        m["debd_mean_all_pct"] = float(per_flow_all.mean())
        m["debd_mean_detected_pct"] = float((100.0 * bpre[a] / btot[a]).mean()) if a.any() else float("nan")
        m["debd_byte_weighted_all_pct"] = float(100.0 * np.where(a, bpre, btot).sum() / btot.sum())
    else:
        m.update(debd_mean_all_pct=np.nan, debd_mean_detected_pct=np.nan, debd_byte_weighted_all_pct=np.nan)
    return m


def prefix_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """Prefix-level classification at a fixed probability threshold (Experiment 2)."""
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {
        "n": len(y), "precision": _div(tp, tp + fp), "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn), "fpr": _div(fp, fp + tn),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
