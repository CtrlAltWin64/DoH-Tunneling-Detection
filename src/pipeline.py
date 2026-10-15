"""Experiments 1-6 end to end. Order of operations (nothing earlier ever sees later data):

  split flows -> prefix tables -> fit RF + calibrator on TRAIN only (Experiment A)
                               -> fit CNN-GRU + calibrator on TRAIN only (Experiments B/C)
  -> score val/test -> select thresholds on VAL ONLY -> save thresholds.json
  -> evaluate frozen policies on TEST -> robustness on TEST (thresholds still frozen)

Experiment labelling:
  A   : Random Forest baseline (fixed-window + adaptive with RF)
  B   : Fixed-window CNN-GRU (evaluate at each checkpoint independently)
  C   : Adaptive CNN-GRU (sequential stopping policy over all checkpoints)
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from . import plots
from .adaptive_detector import curves_from_prefix_table, policy_outcomes, select_threshold
from .calibration import fit_calibrator
from .dataset_split import create_split_manifest, split_flows
from .evaluation import clopper_pearson, expected_calibration_error, prefix_metrics, summarize
from .evasion import low_and_slow, timing_jitter
from .feature_extraction import FEATURE_NAMES
from .prefix_generation import build_prefix_table, eligible_counts
from .train_baseline import fit_rf, raw_proba

logger = logging.getLogger(__name__)


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def environment_record() -> dict:
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return rec


def _score_rf(df, model, calibrator):
    raw = raw_proba(model, df)
    return raw, calibrator.transform(raw, df["checkpoint"].to_numpy())


def _policy_set(cps):
    """name -> column indices. adaptive uses every checkpoint; fixed_t uses only checkpoint t."""
    d = {"adaptive": None}
    for j, t in enumerate(cps):
        d[f"fixed_{t}"] = [j]
    return d


def _get_device(cfg) -> str:
    d = getattr(cfg, "cnn_device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


# ---------------------------------------------------------------------------
# CNN-GRU sub-pipeline
# ---------------------------------------------------------------------------

def _run_cnn_gru_experiment(
    by_part, tables_rf, cps, cfg, out, synthetic, log, device,
):
    """Train CNN-GRU on training flows and evaluate all policies.

    Returns dict with keys: thresholds, test_df, val_df, constraint_df, robustness_df, prefix_df
    """
    from .cnn_gru_calibration import fit_calibrator_cnn
    from .cnn_gru_model import load_model, train_cnn_gru
    from .cnn_gru_prefix import build_cnn_prefix_table, raw_proba_cnn
    from .dataset_split import split_train_calibration

    tmax = cps[-1]
    ckpt_path = out / "cnn_gru_best.pt"

    # Split training partition into model-fitting (train_fit) and calibration (train_cal)
    train_meta = pd.DataFrame([
        {"flow_id": fl.flow_id, "label": fl.label, "source_file": fl.source_file}
        for fl in by_part["train"]
    ])
    train_assign = split_train_calibration(
        train_meta, group_by=cfg.group_by, seed=cfg.seed, cal_ratio=cfg.calibration_split_ratio
    )
    train_fit_flows = [fl for fl in by_part["train"] if train_assign[fl.flow_id] == "train_fit"]
    train_cal_flows = [fl for fl in by_part["train"] if train_assign[fl.flow_id] == "train_cal"]
    train_meta["fit_or_cal"] = train_meta["flow_id"].map(train_assign)
    train_meta.to_csv(out / "train_fit_cal_split.csv", index=False)
    log(
        f"[CNN-GRU] Partitioned training set: {len(train_fit_flows)} model-fitting flows, "
        f"{len(train_cal_flows)} calibration flows (ratio={cfg.calibration_split_ratio})"
    )

    # Train or load CNN-GRU (on model-fitting flows only)
    log("[CNN-GRU] Training 1D-CNN + GRU on model-fitting flows ...")
    model, normalizer, history = train_cnn_gru(
        train_fit_flows, by_part["val"], cps, cfg, ckpt_path, device=device
    )
    log(f"[CNN-GRU] Trained {history['best_epoch']} epochs; best val_loss={history['best_val_loss']:.5f}")

    # Save training history
    (out / "cnn_gru_train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Fit probability calibrator on held-out train_cal flows (out-of-sample)
    log("[CNN-GRU] Fitting Platt calibrator on held-out calibration flows ...")
    cal, cal_df = fit_calibrator_cnn(train_cal_flows, cps, model, normalizer, cfg, device=device)
    log(f"[CNN-GRU] Calibrator fitted on {len(train_cal_flows)} held-out calibration flows (out-of-sample)")

    # Build prefix tables using CNN-GRU scores
    log("[CNN-GRU] Scoring prefix tables ...")
    cnn_tables = {
        p: build_cnn_prefix_table(by_part[p], cps, model, normalizer, device=device)
        for p in ("train", "val", "test")
    }

    # Score val + test
    cnn_scored = {}
    for p in ("val", "test"):
        raw = raw_proba_cnn(cnn_tables[p])
        cal_p = cal.transform(raw, cnn_tables[p]["checkpoint"].to_numpy())
        cnn_scored[p] = (raw, cal_p)

    cnn_curves = {
        p: curves_from_prefix_table(cnn_tables[p], cnn_scored[p][1], cps)
        for p in ("val", "test")
    }
    for p, c in cnn_curves.items():
        if len(set(c.y)) < 2:
            log(f"WARNING [CNN-GRU]: {p} eligible flows contain only one class — evaluation may be unreliable")

    # Calibration quality
    cal_rows = []
    d = cnn_tables["val"]
    raw_val = cnn_scored["val"][0]
    cal_val = cnn_scored["val"][1]
    for t in cps:
        m = (d["checkpoint"] == t).to_numpy()
        y = d.loc[m, "label"].to_numpy()
        cal_rows.append({
            "checkpoint": t, "n_rows": int(m.sum()),
            "brier_raw": float(np.mean((raw_val[m] - y) ** 2)),
            "brier_calibrated": float(np.mean((cal_val[m] - y) ** 2)),
            "ece_raw": expected_calibration_error(y, raw_val[m]),
            "ece_calibrated": expected_calibration_error(y, cal_val[m]),
        })
    pd.DataFrame(cal_rows).to_csv(
        out / "tables" / "cnn_gru_calibration_validation.csv", index=False
    )

    # Prefix-level metrics (Experiment B = fixed windows for CNN-GRU)
    pref = []
    for j, t in enumerate(cps):
        r = prefix_metrics(
            cnn_curves["test"].y, cnn_curves["test"].P[:, j], cfg.prefix_eval_threshold
        )
        pref.append({"checkpoint": t, **r})
    prefix_df = pd.DataFrame(pref)
    prefix_df.to_csv(out / "tables" / "cnn_gru_prefix_level_test.csv", index=False)

    # Threshold selection on VALIDATION (freeze before test)
    policies = _policy_set(cps)
    sel = {
        name: select_threshold(cnn_curves["val"], cfg.alpha, cfg.threshold_step, idx)
        for name, idx in policies.items()
    }
    frozen = {name: s["tau"] for name, s in sel.items()}
    for name, s in sel.items():
        s["sweep"].to_csv(
            out / "tables" / f"cnn_gru_threshold_sweep_val_{name}.csv", index=False
        )
    thr_payload = {name: {k: v for k, v in s.items() if k != "sweep"} for name, s in sel.items()}
    (out / "cnn_gru_thresholds.json").write_text(json.dumps(thr_payload, indent=2), encoding="utf-8")
    log("[CNN-GRU] Thresholds frozen: " + ", ".join(f"{k}={v:.3f}" for k, v in frozen.items()))

    # Evaluate frozen policies on val + test
    rows_test, rows_val, comp = [], [], []
    outcome_frames = {}
    for name, idx in policies.items():
        tau = frozen[name]
        ov = policy_outcomes(cnn_curves["val"], tau, idx)
        ot = policy_outcomes(cnn_curves["test"], tau, idx)
        outcome_frames[name] = ot
        mv, mt = summarize(ov, tmax), summarize(ot, tmax)
        rows_val.append({"policy": name, "tau": tau, **mv})
        rows_test.append({"policy": name, "tau": tau, **mt})
        vci = clopper_pearson(mv["FP"], mv["FP"] + mv["TN"])
        comp.append({
            "policy": name, "alpha": cfg.alpha, "tau": tau, "never_alerts": tau > 1.0,
            "n_benign_val": mv["n_benign"], "val_fp": mv["FP"], "val_fpr": mv["fpr"],
            "val_fpr_upper95": vci["upper_one_sided"], "val_satisfies_alpha": bool(mv["fpr"] <= cfg.alpha),
            "n_benign_test": mt["n_benign"], "test_fp": mt["FP"], "test_fpr": mt["fpr"],
            "test_fpr_ci95_low": mt["fpr_ci95_low"], "test_fpr_ci95_high": mt["fpr_ci95_high"],
            "test_fpr_upper95": mt["fpr_upper95_one_sided"], "test_fpr_within_alpha": bool(mt["fpr"] <= cfg.alpha),
            "resolution_warning": sel[name]["resolution_warning"],
        })
    test_df = pd.DataFrame(rows_test)
    val_df = pd.DataFrame(rows_val)
    constraint_df = pd.DataFrame(comp)
    test_df.to_csv(out / "tables" / "cnn_gru_fixed_vs_adaptive_test.csv", index=False)
    val_df.to_csv(out / "tables" / "cnn_gru_fixed_vs_adaptive_validation.csv", index=False)
    constraint_df.to_csv(out / "tables" / "cnn_gru_fpr_constraint_validation_vs_test.csv", index=False)

    # Per-flow prediction tables
    def pred_frame(c, tag):
        f = pd.DataFrame({"flow_id": c.flow_ids, "label": c.y})
        for j, t in enumerate(c.cps):
            f[f"p_{t}"] = c.P[:, j]
        for name in ("adaptive", f"fixed_{tmax}"):
            o = policy_outcomes(c, frozen[name], policies[name])
            f[f"{name}_alert"] = o["alert"].to_numpy()
            f[f"{name}_alert_packets"] = o["alert_packets"].to_numpy()
        return f

    pred_frame(cnn_curves["val"], "val").to_csv(
        out / "cnn_gru_predictions_val.csv", index=False
    )
    pred_frame(cnn_curves["test"], "test").to_csv(
        out / "cnn_gru_predictions_test.csv", index=False
    )

    # Robustness (Experiment C / evasion on CNN-GRU)
    test_flows_full = [f for f in by_part["test"] if len(f.ts) >= tmax]
    conditions = [("normal", "normal", lambda f: f)]
    for k in cfg.low_and_slow_factors:
        conditions.append((f"low_and_slow_x{k:g}", f"low-and-slow x{k:g}", lambda f, k=k: low_and_slow(f, k)))
    for s, a in cfg.jitter_settings:
        conditions.append((f"jitter_s{s:g}_a{a:g}", f"jitter σ={s:g}, +{a:g}s", lambda f, s=s, a=a: timing_jitter(f, s, a, cfg.seed)))

    rob_rows = []
    for cid, clabel, fn in conditions:
        fl = [fn(f) if f.label == 1 else f for f in test_flows_full]
        tab = build_cnn_prefix_table(fl, cps, model, normalizer, device=device)
        raw_rob = raw_proba_cnn(tab)
        cal_rob = cal.transform(raw_rob, tab["checkpoint"].to_numpy())
        c = curves_from_prefix_table(tab, cal_rob, cps)
        for name in ("adaptive", f"fixed_{tmax}"):
            m = summarize(policy_outcomes(c, frozen[name], policies[name]), tmax)
            rob_rows.append({"condition": cid, "condition_label": clabel, "policy": name,
                             "tau": frozen[name], **m})
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(out / "tables" / "cnn_gru_robustness_test.csv", index=False)

    return {
        "thresholds": frozen,
        "test": test_df,
        "validation": val_df,
        "constraint": constraint_df,
        "robustness": rob_df,
        "prefix": prefix_df,
        "model": model,
        "normalizer": normalizer,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    flows, cfg, out_dir, synthetic: bool = False, overwrite: bool = False, log=print
) -> dict:
    cfg.validate()
    cps = list(cfg.checkpoints)
    tmax = cps[-1]
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{out} already exists and is not empty; choose another --name or pass --overwrite"
        )
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    banner = (
        "SYNTHETIC DATA - pipeline check only, NOT results about DoH traffic\n\n"
        if synthetic else ""
    )

    # ------------------------------------------------------------------ Exp 1: dataset + split
    usable = [f for f in flows if len(f.ts) >= cps[0]]
    if not usable:
        raise ValueError(f"no flow has >= {cps[0]} packets")
    meta = pd.DataFrame({
        "flow_id": [f.flow_id for f in usable],
        "label": [f.label for f in usable],
        "source_file": [f.source_file for f in usable],
    })
    assign = split_flows(meta, cfg.group_by, cfg.seed)
    meta["partition"] = meta["flow_id"].map(assign)
    meta.to_csv(out / "split_assignment.csv", index=False)
    split_manifest = create_split_manifest(meta, assign, cfg.group_by)
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    by_part = {p: [f for f in usable if assign[f.flow_id] == p] for p in ("train", "val", "test")}

    stats = []
    for name, fl in (("all_flows", flows), ("train", by_part["train"]),
                     ("val", by_part["val"]), ("test", by_part["test"])):
        ec = eligible_counts(fl, cps)
        ec.insert(0, "partition", name)
        stats.append(ec)
    dataset_stats = pd.concat(stats, ignore_index=True)
    dataset_stats.to_csv(out / "tables" / "dataset_stats.csv", index=False)

    # RF prefix tables (used by both RF and for CNN-GRU comparison scaffolding)
    tables_rf = {p: build_prefix_table(by_part[p], cps) for p in by_part}
    for p, t in tables_rf.items():
        if t.empty:
            raise ValueError(f"{p} partition produced no prefix rows")
        bad = int((~np.isfinite(t[FEATURE_NAMES].to_numpy())).sum())
        if bad:
            raise ValueError(f"{bad} non-finite feature values in {p} prefixes")

    dup_seq = {}
    for f in usable:
        if len(f.ts) >= tmax:
            k = (f.label, f.length[:tmax].tobytes(), f.direction[:tmax].tobytes())
            dup_seq[k] = dup_seq.get(k, 0) + 1
    exp1 = {
        "n_flows_total": len(flows),
        "n_flows_usable_ge_first_checkpoint": len(usable),
        "n_flows_excluded_too_short": len(flows) - len(usable),
        "informational_flows_sharing_identical_first_tmax_length_direction_sequence":
            int(sum(v for v in dup_seq.values() if v > 1)),
        "group_by": cfg.group_by, "seed": cfg.seed,
    }
    log(f"[1] flows={len(flows)} usable={len(usable)}  train/val/test flows = "
        f"{len(by_part['train'])}/{len(by_part['val'])}/{len(by_part['test'])}")

    # ------------------------------------------------------------------ Experiment A: RF baseline
    result_rf = None
    if cfg.model_type in ("rf", "both"):
        model_rf = fit_rf(tables_rf["train"], cfg)
        calibrator_rf = fit_calibrator(tables_rf["train"], cfg)
        log("[A] RF + calibrator fitted on training flows only")

        scored_rf = {}
        for p in ("val", "test"):
            raw, cal = _score_rf(tables_rf[p], model_rf, calibrator_rf)
            scored_rf[p] = (raw, cal)
        curves_rf = {
            p: curves_from_prefix_table(tables_rf[p], scored_rf[p][1], cps)
            for p in ("val", "test")
        }
        for p in curves_rf:
            c = curves_rf[p]
            if len(set(c.y)) < 2:
                raise ValueError(f"RF {p}: eligible (>= {tmax} packets) flows contain a single class")
        n_elig = {
            p: {"benign": int((curves_rf[p].y == 0).sum()), "malicious": int((curves_rf[p].y == 1).sum())}
            for p in curves_rf
        }
        exp1["eligible_flows_for_all_checkpoints_rf"] = n_elig

        # Calibration quality (diagnostic)
        cal_rows = []
        d = tables_rf["val"]
        for t in cps:
            m = (d["checkpoint"] == t).to_numpy()
            y = d.loc[m, "label"].to_numpy()
            cal_rows.append({
                "checkpoint": t, "n_rows": int(m.sum()),
                "brier_raw": float(np.mean((scored_rf["val"][0][m] - y) ** 2)),
                "brier_calibrated": float(np.mean((scored_rf["val"][1][m] - y) ** 2)),
                "ece_raw": expected_calibration_error(y, scored_rf["val"][0][m]),
                "ece_calibrated": expected_calibration_error(y, scored_rf["val"][1][m]),
            })
        pd.DataFrame(cal_rows).to_csv(out / "tables" / "rf_calibration_validation.csv", index=False)
        # Keep backward-compat name too
        pd.DataFrame(cal_rows).to_csv(out / "tables" / "calibration_validation.csv", index=False)

        # Prefix-level classification (Exp 2 / RF)
        pref_rf = []
        for j, t in enumerate(cps):
            r = prefix_metrics(curves_rf["test"].y, curves_rf["test"].P[:, j], cfg.prefix_eval_threshold)
            pref_rf.append({"checkpoint": t, **r})
        prefix_df_rf = pd.DataFrame(pref_rf)
        prefix_df_rf.to_csv(out / "tables" / "prefix_level_test.csv", index=False)

        # Threshold selection (val only)
        policies = _policy_set(cps)
        sel_rf = {
            name: select_threshold(curves_rf["val"], cfg.alpha, cfg.threshold_step, idx)
            for name, idx in policies.items()
        }
        frozen_rf = {name: s["tau"] for name, s in sel_rf.items()}
        for name, s in sel_rf.items():
            s["sweep"].to_csv(
                out / "tables" / f"threshold_sweep_val_{name}.csv", index=False
            )
        thr_rf_payload = {
            name: {k: v for k, v in s.items() if k != "sweep"} for name, s in sel_rf.items()
        }
        (out / "thresholds.json").write_text(json.dumps(thr_rf_payload, indent=2), encoding="utf-8")
        log("[A] RF thresholds frozen: " + ", ".join(f"{k}={v:.3f}" for k, v in frozen_rf.items()))

        # Frozen policy evaluation
        rows_test_rf, rows_val_rf, comp_rf = [], [], []
        for name, idx in policies.items():
            tau = frozen_rf[name]
            ov = policy_outcomes(curves_rf["val"], tau, idx)
            ot = policy_outcomes(curves_rf["test"], tau, idx)
            mv, mt = summarize(ov, tmax), summarize(ot, tmax)
            rows_val_rf.append({"policy": name, "tau": tau, **mv})
            rows_test_rf.append({"policy": name, "tau": tau, **mt})
            vci = clopper_pearson(mv["FP"], mv["FP"] + mv["TN"])
            comp_rf.append({
                "policy": name, "alpha": cfg.alpha, "tau": tau, "never_alerts": tau > 1.0,
                "n_benign_val": mv["n_benign"], "val_fp": mv["FP"], "val_fpr": mv["fpr"],
                "val_fpr_upper95": vci["upper_one_sided"], "val_satisfies_alpha": bool(mv["fpr"] <= cfg.alpha),
                "n_benign_test": mt["n_benign"], "test_fp": mt["FP"], "test_fpr": mt["fpr"],
                "test_fpr_ci95_low": mt["fpr_ci95_low"], "test_fpr_ci95_high": mt["fpr_ci95_high"],
                "test_fpr_upper95": mt["fpr_upper95_one_sided"], "test_fpr_within_alpha": bool(mt["fpr"] <= cfg.alpha),
                "resolution_warning": sel_rf[name]["resolution_warning"],
            })
        test_df_rf = pd.DataFrame(rows_test_rf)
        val_df_rf = pd.DataFrame(rows_val_rf)
        comp_df_rf = pd.DataFrame(comp_rf)
        test_df_rf.to_csv(out / "tables" / "fixed_vs_adaptive_test.csv", index=False)
        val_df_rf.to_csv(out / "tables" / "fixed_vs_adaptive_validation.csv", index=False)
        comp_df_rf.to_csv(out / "tables" / "fpr_constraint_validation_vs_test.csv", index=False)

        # Per-flow predictions
        def pred_frame_rf(c, tag):
            f = pd.DataFrame({"flow_id": c.flow_ids, "label": c.y})
            for j, t in enumerate(c.cps):
                f[f"p_{t}"] = c.P[:, j]
            for name in ("adaptive", f"fixed_{tmax}"):
                o = policy_outcomes(c, frozen_rf[name], policies[name])
                f[f"{name}_alert"] = o["alert"].to_numpy()
                f[f"{name}_alert_packets"] = o["alert_packets"].to_numpy()
            return f

        pred_frame_rf(curves_rf["val"], "val").to_csv(out / "predictions_val.csv", index=False)
        pred_frame_rf(curves_rf["test"], "test").to_csv(out / "predictions_test.csv", index=False)

        # Robustness (RF)
        test_flows_full = [f for f in by_part["test"] if len(f.ts) >= tmax]
        conditions = [("normal", "normal", lambda f: f)]
        for k in cfg.low_and_slow_factors:
            conditions.append((f"low_and_slow_x{k:g}", f"low-and-slow x{k:g}", lambda f, k=k: low_and_slow(f, k)))
        for s, a in cfg.jitter_settings:
            conditions.append((f"jitter_s{s:g}_a{a:g}", f"jitter σ={s:g}, +{a:g}s",
                               lambda f, s=s, a=a: timing_jitter(f, s, a, cfg.seed)))
        rob_rows_rf = []
        for cid, clabel, fn in conditions:
            fl = [fn(f) if f.label == 1 else f for f in test_flows_full]
            tab = build_prefix_table(fl, cps)
            _, cal_r = _score_rf(tab, model_rf, calibrator_rf)
            c = curves_from_prefix_table(tab, cal_r, cps)
            for name in ("adaptive", f"fixed_{tmax}"):
                m = summarize(policy_outcomes(c, frozen_rf[name], policies[name]), tmax)
                rob_rows_rf.append({"condition": cid, "condition_label": clabel,
                                    "policy": name, "tau": frozen_rf[name], **m})
        rob_df_rf = pd.DataFrame(rob_rows_rf)
        rob_df_rf.to_csv(out / "tables" / "robustness_test.csv", index=False)

        chk = rob_df_rf[(rob_df_rf["condition"] == "normal") & (rob_df_rf["policy"] == "adaptive")].iloc[0]
        main_ad = test_df_rf[test_df_rf["policy"] == "adaptive"].iloc[0]
        consistent_rf = (int(chk["TP"]), int(chk["FP"])) == (int(main_ad["TP"]), int(main_ad["FP"]))
        if not consistent_rf:
            log("WARNING [RF]: robustness 'normal' condition differs from main test evaluation")

        result_rf = {
            "thresholds": frozen_rf, "test": test_df_rf, "validation": val_df_rf,
            "constraint": comp_df_rf, "robustness": rob_df_rf, "prefix": prefix_df_rf,
        }

    # ------------------------------------------------------------------ Experiments B+C: CNN-GRU
    result_cnn = None
    if cfg.model_type in ("cnn_gru", "both"):
        device = _get_device(cfg)
        log(f"[CNN-GRU] Using device: {device}")
        result_cnn = _run_cnn_gru_experiment(
            by_part, tables_rf, cps, cfg, out, synthetic, log, device
        )

    # ------------------------------------------------------------------ Figures
    fig = out / "figures"

    if result_rf is not None:
        plots.plot_prf_vs_prefix(result_rf["prefix"], fig / "rf_prf_vs_prefix_length.png", " RF (test)")
        plots.plot_prf_vs_prefix(result_rf["prefix"], fig / "prf_vs_prefix_length.png", " RF (test)")
        plots.plot_val_fpr_vs_threshold(
            {k: sel_rf[k]["sweep"] for k in sel_rf}, frozen_rf, cfg.alpha,
            fig / "rf_val_fpr_vs_threshold.png"
        )
        plots.plot_val_fpr_vs_threshold(
            {k: sel_rf[k]["sweep"] for k in sel_rf}, frozen_rf, cfg.alpha,
            fig / "val_fpr_vs_threshold.png"
        )
        two_rf = result_rf["test"][result_rf["test"]["policy"].isin(["adaptive", f"fixed_{tmax}"])]
        plots.grouped_bars(two_rf, "policy", {"cost_malicious": "malicious flows", "cost_all": "all flows"},
                           "Normalised obs. cost (pkts/Tmax)", "RF Observation cost (test)",
                           fig / "rf_observation_cost.png")
        plots.grouped_bars(two_rf, "policy",
                           {"debd_mean_all_pct": "mean, undetected=100%",
                            "debd_mean_detected_pct": "mean, detected only",
                            "debd_byte_weighted_all_pct": "byte-weighted, undetected=100%"},
                           "DEBD proxy (%)", "RF Pre-alert exposure proxy (test)",
                           fig / "rf_debd_comparison.png")
        plots.grouped_bars(two_rf, "policy", {"coverage": "coverage (= recall)", "fpr": "FPR"},
                           "Fraction", "RF Coverage and FPR (test)", fig / "rf_detection_coverage.png")
        plots.grouped_bars(two_rf, "policy", {"coverage": "coverage (= recall)", "fpr": "FPR"},
                           "Fraction", "RF Coverage and FPR (test)", fig / "detection_coverage.png")
        plots.plot_robustness(result_rf["robustness"], fig / "rf_robustness.png")
        plots.plot_robustness(result_rf["robustness"], fig / "robustness.png")
        # Observation cost (backward compat)
        plots.grouped_bars(two_rf, "policy", {"cost_malicious": "malicious flows", "cost_all": "all flows"},
                           "Normalised obs. cost (pkts/Tmax)", "Observation cost (test)",
                           fig / "observation_cost.png")
        plots.grouped_bars(two_rf, "policy", {"latency_mean_s": "mean", "latency_median_s": "median"},
                           "Detection latency (s, detected malicious flows)", "Detection latency RF (test)",
                           fig / "detection_latency.png")
        plots.grouped_bars(two_rf, "policy",
                           {"debd_mean_all_pct": "mean, undetected=100%",
                            "debd_mean_detected_pct": "mean, detected only",
                            "debd_byte_weighted_all_pct": "byte-weighted, undetected=100%"},
                           "DEBD proxy (%)", "Pre-alert exposure proxy (test)",
                           fig / "debd_comparison.png")

    if result_cnn is not None:
        plots.plot_prf_vs_prefix(result_cnn["prefix"], fig / "cnn_gru_prf_vs_prefix_length.png", " CNN-GRU (test)")
        two_cnn = result_cnn["test"][result_cnn["test"]["policy"].isin(["adaptive", f"fixed_{tmax}"])]
        plots.grouped_bars(two_cnn, "policy", {"cost_malicious": "malicious flows", "cost_all": "all flows"},
                           "Normalised obs. cost (pkts/Tmax)", "CNN-GRU Observation cost (test)",
                           fig / "cnn_gru_observation_cost.png")
        plots.grouped_bars(two_cnn, "policy",
                           {"debd_mean_all_pct": "mean, undetected=100%",
                            "debd_mean_detected_pct": "mean, detected only",
                            "debd_byte_weighted_all_pct": "byte-weighted, undetected=100%"},
                           "DEBD proxy (%)", "CNN-GRU Pre-alert exposure proxy (test)",
                           fig / "cnn_gru_debd_comparison.png")
        plots.grouped_bars(two_cnn, "policy", {"coverage": "coverage (= recall)", "fpr": "FPR"},
                           "Fraction", "CNN-GRU Coverage and FPR (test)", fig / "cnn_gru_detection_coverage.png")
        plots.plot_robustness(result_cnn["robustness"], fig / "cnn_gru_robustness.png")

        if result_rf is not None:
            _plot_rf_vs_cnn_comparison(result_rf, result_cnn, tmax, fig)

    # ------------------------------------------------------------------ Reproducibility record
    model_record: dict = {}
    if result_rf is not None:
        model_record["rf"] = {
            "type": "RandomForestClassifier",
            "n_estimators": cfg.rf_n_estimators,
            "min_samples_leaf": cfg.rf_min_samples_leaf,
            "class_weight": cfg.rf_class_weight,
            "flow_weighting": cfg.flow_weighting,
            "frozen_thresholds": result_rf["thresholds"],
        }
    if result_cnn is not None:
        model_record["cnn_gru"] = {
            "type": "1D-CNN+GRU",
            "conv_channels": cfg.cnn_conv_channels,
            "conv_kernel": cfg.cnn_conv_kernel,
            "gru_hidden": cfg.cnn_gru_hidden,
            "gru_layers": cfg.cnn_gru_layers,
            "dropout": cfg.cnn_dropout,
            "epochs_trained": result_cnn["history"].get("best_epoch"),
            "best_val_loss": result_cnn["history"].get("best_val_loss"),
            "device": _get_device(cfg),
            "frozen_thresholds": result_cnn["thresholds"],
        }

    record = {
        "config": asdict(cfg),
        "environment": environment_record(),
        "synthetic": synthetic,
        "feature_names_rf": FEATURE_NAMES,
        "experiment1": exp1,
        "models": model_record,
        "evasion_note": (
            "offline timestamp perturbation of malicious TEST flows only; "
            "simulated, not real attack traffic"
        ),
    }
    (out / "run_record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ Summary markdown
    # Determine primary result set for the backward-compat summary
    primary_rf = result_rf
    primary_cnn = result_cnn

    rf_note = "" if primary_rf is None else _make_rf_summary_section(
        primary_rf, dataset_stats, exp1, cfg, cps, tmax
    )
    cnn_note = "" if primary_cnn is None else _make_cnn_summary_section(primary_cnn, tmax)

    md = (banner
          + f"# Experiment {out.name}\n\nMeasured values only; interpretation is for the authors.\n\n"
          + f"## Dataset statistics\n{md_table(dataset_stats)}\n\n"
          + rf_note + cnn_note
          + "## Conventions\nDEBD is a proxy for pre-alert malicious-traffic exposure (IP-level bytes, "
            "both directions, including the alert packet, retransmissions removed, "
            "undetected malicious = 100%). Coverage equals recall at flow level.\n")
    (out / "summary.md").write_text(md, encoding="utf-8")
    log(f"[done] results in {out}")

    # Return combined result (backward-compat keys from RF if available, else CNN)
    primary = result_rf if result_rf is not None else result_cnn
    return {
        "thresholds": primary["thresholds"],
        "test": primary["test"],
        "validation": primary["validation"],
        "constraint": primary["constraint"],
        "robustness": primary["robustness"],
        "prefix": primary["prefix"],
        "out_dir": str(out),
        "rf": result_rf,
        "cnn_gru": result_cnn,
    }


def _plot_rf_vs_cnn_comparison(result_rf, result_cnn, tmax, fig: Path):
    """Side-by-side comparison plots (RF vs CNN-GRU adaptive)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    metrics = ["coverage", "fpr", "cost_malicious", "debd_mean_all_pct"]
    labels = ["Coverage (recall)", "FPR", "Obs. cost (malicious)", "DEBD mean (%)"]

    def _get_adaptive(df):
        return df[df["policy"] == "adaptive"].iloc[0] if "adaptive" in df["policy"].values else None

    rf_ad = _get_adaptive(result_rf["test"])
    cnn_ad = _get_adaptive(result_cnn["test"])
    if rf_ad is None or cnn_ad is None:
        return

    fig_obj, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, col, label in zip(axes, metrics, labels):
        vals = [rf_ad[col], cnn_ad[col]]
        bars = ax.bar(["RF", "CNN-GRU"], vals, color=COLORS[:2], edgecolor="white")
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig_obj.suptitle("Adaptive policy: RF vs CNN-GRU (test set)", fontsize=11)
    fig_obj.tight_layout()
    fig_obj.savefig(fig / "rf_vs_cnn_adaptive_comparison.png", dpi=150)
    plt.close(fig_obj)


def _make_rf_summary_section(result_rf, dataset_stats, exp1, cfg, cps, tmax):
    policies = _policy_set(cps)
    comp_cols = ["policy", "tau", "n_benign_val", "val_fp", "val_fpr",
                 "val_fpr_upper95", "n_benign_test", "test_fp", "test_fpr", "test_fpr_upper95"]
    main_cols = ["policy", "tau", "precision", "recall", "f1", "fpr", "coverage",
                 "avg_packets_malicious", "cost_malicious", "latency_mean_s",
                 "debd_mean_all_pct", "debd_byte_weighted_all_pct"]
    rob_cols = ["condition_label", "policy", "precision", "recall", "f1", "fpr",
                "coverage", "cost_malicious", "latency_mean_s", "debd_mean_all_pct"]
    test_df = result_rf["test"]
    constraint_df = result_rf["constraint"]
    rob_df = result_rf["robustness"]
    return (
        f"## [Experiment A] Random Forest — Prefix-level classification (threshold {cfg.prefix_eval_threshold})\n"
        f"{md_table(result_rf['prefix'])}\n\n"
        f"## [Experiment A] RF — FPR constraint (alpha={cfg.alpha})\n"
        f"{md_table(constraint_df[comp_cols])}\n\n"
        f"## [Experiment A] RF — Fixed-window vs Adaptive (test)\n"
        f"{md_table(test_df[main_cols])}\n\n"
        f"## [Experiment A] RF — Simulated robustness (test)\n"
        f"{md_table(rob_df[rob_cols])}\n\n"
    )


def _make_cnn_summary_section(result_cnn, tmax):
    comp_cols = ["policy", "tau", "n_benign_val", "val_fp", "val_fpr",
                 "val_fpr_upper95", "n_benign_test", "test_fp", "test_fpr", "test_fpr_upper95"]
    main_cols = ["policy", "tau", "precision", "recall", "f1", "fpr", "coverage",
                 "avg_packets_malicious", "cost_malicious", "latency_mean_s",
                 "debd_mean_all_pct", "debd_byte_weighted_all_pct"]
    rob_cols = ["condition_label", "policy", "precision", "recall", "f1", "fpr",
                "coverage", "cost_malicious", "latency_mean_s", "debd_mean_all_pct"]
    return (
        "## [Experiments B+C] CNN-GRU — Prefix-level classification\n"
        f"{md_table(result_cnn['prefix'])}\n\n"
        "## [Experiments B+C] CNN-GRU — FPR constraint\n"
        f"{md_table(result_cnn['constraint'][comp_cols])}\n\n"
        "## [Experiment B] CNN-GRU Fixed-window vs [Experiment C] Adaptive (test)\n"
        f"{md_table(result_cnn['test'][main_cols])}\n\n"
        "## [Experiment C] CNN-GRU — Simulated robustness (test)\n"
        f"{md_table(result_cnn['robustness'][rob_cols])}\n\n"
    )
