"""Matplotlib figures. Every figure is drawn from tables produced by real measurements;
if the needed columns are missing a ValueError is raised instead of drawing something misleading.
Colour-blind-safe palette + hatches so grayscale prints stay readable."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
HATCH = ["", "//", "..", "xx", "\\\\", "--"]


def _need(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"cannot plot: missing columns {miss}")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prf_vs_prefix(prefix_df, path, title_suffix=""):
    _need(prefix_df, ["checkpoint", "precision", "recall", "f1"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(["precision", "recall", "f1"]):
        ax.plot(prefix_df["checkpoint"], prefix_df[m], marker="o", color=COLORS[i], label=m)
    ax.set_xlabel("Prefix length (packets)"); ax.set_ylabel("Score (threshold 0.5, test flows)")
    ax.set_ylim(0, 1.02); ax.set_xticks(prefix_df["checkpoint"]); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Prefix-level precision / recall / F1" + title_suffix)
    _save(fig, path)


def plot_val_fpr_vs_threshold(sweeps: dict, taus: dict, alpha, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, sw) in enumerate(sweeps.items()):
        _need(sw, ["tau", "fpr"])
        s = sw[sw["tau"] <= 1.0]
        ax.plot(s["tau"], s["fpr"] * 100, color=COLORS[i % 6], label=name, lw=2 if name == "adaptive" else 1.2)
        if name in taus and taus[name] <= 1.0:
            ax.axvline(taus[name], color=COLORS[i % 6], ls=":", lw=1)
    ax.axhline(alpha * 100, color="k", ls="--", lw=1, label=f"target α = {alpha * 100:g}%")
    ax.set_xlabel("Threshold τ"); ax.set_ylabel("Validation FPR (% of benign flows)")
    ax.set_title("Complete-policy FPR vs threshold (validation)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, path)


def grouped_bars(df, label_col, metrics: dict, ylabel, title, path):
    """metrics: {column: legend label}. One group per row of df (policy/condition)."""
    _need(df, [label_col] + list(metrics))
    n, k = len(df), len(metrics)
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * n * k / 2 + 3), 4.2))
    x = np.arange(n)
    for j, (col, lab) in enumerate(metrics.items()):
        bars = ax.bar(x + (j - (k - 1) / 2) * w, df[col].to_numpy(dtype=float), w, label=lab,
                      color=COLORS[j % 6], hatch=HATCH[j % 6], edgecolor="white")
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(df[label_col], rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_robustness(rob_df, path, policies=("adaptive", "fixed_40")):
    """2x2 panel: F1, coverage, malicious observation cost, DEBD (all-flow convention) per condition."""
    _need(rob_df, ["condition_label", "policy", "f1", "coverage", "cost_malicious", "debd_mean_all_pct"])
    panels = [("f1", "F1-score"), ("coverage", "Detection coverage"),
              ("cost_malicious", "Observation cost (malicious flows)"), ("debd_mean_all_pct", "Mean DEBD (%, undetected = 100)")]
    conds = list(dict.fromkeys(rob_df["condition_label"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, ttl) in zip(axes.ravel(), panels):
        x = np.arange(len(conds)); w = 0.8 / len(policies)
        for j, pol in enumerate(policies):
            sub = rob_df[rob_df["policy"] == pol].set_index("condition_label").reindex(conds)
            ax.bar(x + (j - (len(policies) - 1) / 2) * w, sub[col].to_numpy(dtype=float), w, label=pol,
                   color=COLORS[j], hatch=HATCH[j], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=10); ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Simulated timing perturbations of malicious test flows (thresholds frozen on normal validation data)", fontsize=10)
    _save(fig, path)
