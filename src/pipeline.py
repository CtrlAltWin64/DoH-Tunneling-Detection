"""Experiments 1-6 end to end. Order of operations (nothing earlier ever sees later data):

  split flows -> prefix tables -> fit RF + calibrator on TRAIN only
  -> score val/test -> select thresholds on VAL ONLY -> save thresholds.json
  -> evaluate frozen policies on TEST -> robustness on TEST (thresholds still frozen)
"""
from __future__ import annotations

import json
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

from . import plots
from .adaptive_detector import curves_from_prefix_table, policy_outcomes, select_threshold
from .calibration import fit_calibrator
from .dataset_split import split_flows
from .evaluation import clopper_pearson, expected_calibration_error, prefix_metrics, summarize
from .evasion import low_and_slow, timing_jitter
from .feature_extraction import FEATURE_NAMES
from .prefix_generation import build_prefix_table, eligible_counts
from .train_baseline import fit_rf, raw_proba


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def environment_record() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _score(df, model, calibrator):
    raw = raw_proba(model, df)
    return raw, calibrator.transform(raw, df["checkpoint"].to_numpy())


def _policy_set(cps):
    """name -> column indices. adaptive uses every checkpoint; fixed_t uses only checkpoint t."""
    d = {"adaptive": None}
    for j, t in enumerate(cps):
        d[f"fixed_{t}"] = [j]
    return d


def run_experiment(flows, cfg, out_dir, synthetic: bool = False, overwrite: bool = False, log=print) -> dict:
    cfg.validate()
    cps = list(cfg.checkpoints)
    tmax = cps[-1]
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} already exists and is not empty; choose another --name or pass --overwrite")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    banner = "SYNTHETIC DATA - pipeline check only, NOT results about DoH traffic\n\n" if synthetic else ""

    # ------------------------------------------------------------- Experiment 1: dataset + split
    usable = [f for f in flows if len(f.ts) >= cps[0]]
    if not usable:
        raise ValueError(f"no flow has >= {cps[0]} packets")
    meta = pd.DataFrame({"flow_id": [f.flow_id for f in usable], "label": [f.label for f in usable],
                         "source_file": [f.source_file for f in usable]})
    assign = split_flows(meta, cfg.group_by, cfg.seed)
    meta["partition"] = meta["flow_id"].map(assign)
    meta.to_csv(out / "split_assignment.csv", index=False)
    by_part = {p: [f for f in usable if assign[f.flow_id] == p] for p in ("train", "val", "test")}

    stats = []
    for name, fl in (("all_flows", flows), ("train", by_part["train"]), ("val", by_part["val"]), ("test", by_part["test"])):
        ec = eligible_counts(fl, cps)
        ec.insert(0, "partition", name)
        stats.append(ec)
    dataset_stats = pd.concat(stats, ignore_index=True)
    dataset_stats.to_csv(out / "tables" / "dataset_stats.csv", index=False)

    tables = {p: build_prefix_table(by_part[p], cps) for p in by_part}
    for p, t in tables.items():
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
    exp1 = {"n_flows_total": len(flows), "n_flows_usable_ge_first_checkpoint": len(usable),
            "n_flows_excluded_too_short": len(flows) - len(usable),
            "informational_flows_sharing_identical_first_tmax_length_direction_sequence": int(sum(v for v in dup_seq.values() if v > 1)),
            "group_by": cfg.group_by, "seed": cfg.seed}
    log(f"[1] flows={len(flows)} usable={len(usable)}  train/val/test flows = "
        f"{len(by_part['train'])}/{len(by_part['val'])}/{len(by_part['test'])}")

    # ------------------------------------------------------------- training + calibration (TRAIN only)
    model = fit_rf(tables["train"], cfg)
    calibrator = fit_calibrator(tables["train"], cfg)
    log("[2] random forest + calibrator fitted on training flows only")

    scored = {}
    for p in ("val", "test"):
        raw, cal = _score(tables[p], model, calibrator)
        scored[p] = (raw, cal)
    curves = {p: curves_from_prefix_table(tables[p], scored[p][1], cps) for p in ("val", "test")}
    for p in curves:
        c = curves[p]
        if len(set(c.y)) < 2:
            raise ValueError(f"{p}: eligible (>= {tmax} packets) flows contain a single class")
    n_elig = {p: {"benign": int((curves[p].y == 0).sum()), "malicious": int((curves[p].y == 1).sum())} for p in curves}
    exp1["eligible_flows_for_all_checkpoints"] = n_elig

    # calibration quality on validation rows (diagnostic only)
    cal_rows = []
    d = tables["val"]
    for t in cps:
        m = (d["checkpoint"] == t).to_numpy()
        y = d.loc[m, "label"].to_numpy()
        cal_rows.append({"checkpoint": t, "n_rows": int(m.sum()),
                         "brier_raw": float(np.mean((scored["val"][0][m] - y) ** 2)),
                         "brier_calibrated": float(np.mean((scored["val"][1][m] - y) ** 2)),
                         "ece_raw": expected_calibration_error(y, scored["val"][0][m]),
                         "ece_calibrated": expected_calibration_error(y, scored["val"][1][m])})
    pd.DataFrame(cal_rows).to_csv(out / "tables" / "calibration_validation.csv", index=False)

    # ------------------------------------------------------------- Experiment 2: prefix-level classification
    pref = []
    for j, t in enumerate(cps):
        r = prefix_metrics(curves["test"].y, curves["test"].P[:, j], cfg.prefix_eval_threshold)
        pref.append({"checkpoint": t, **r})
    prefix_df = pd.DataFrame(pref)
    prefix_df.to_csv(out / "tables" / "prefix_level_test.csv", index=False)

    # ------------------------------------------------------------- thresholds on VALIDATION only, then freeze
    policies = _policy_set(cps)
    sel = {name: select_threshold(curves["val"], cfg.alpha, cfg.threshold_step, idx) for name, idx in policies.items()}
    frozen = {name: s["tau"] for name, s in sel.items()}
    for name, s in sel.items():
        s["sweep"].to_csv(out / "tables" / f"threshold_sweep_val_{name}.csv", index=False)
    thresholds_json = {name: {k: v for k, v in s.items() if k != "sweep"} for name, s in sel.items()}
    (out / "thresholds.json").write_text(json.dumps(thresholds_json, indent=2))   # frozen BEFORE any test evaluation
    log("[3] thresholds frozen on validation: " + ", ".join(f"{k}={v:.3f}" for k, v in frozen.items()))

    # ------------------------------------------------------------- Experiments 3-5: frozen policies on val + test
    rows_test, rows_val, comp = [], [], []
    outcome_frames = {}
    for name, idx in policies.items():
        tau = frozen[name]
        ov = policy_outcomes(curves["val"], tau, idx)
        ot = policy_outcomes(curves["test"], tau, idx)
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
            "resolution_warning": sel[name]["resolution_warning"]})
    test_df, val_df, comp_df = pd.DataFrame(rows_test), pd.DataFrame(rows_val), pd.DataFrame(comp)
    test_df.to_csv(out / "tables" / "fixed_vs_adaptive_test.csv", index=False)
    val_df.to_csv(out / "tables" / "fixed_vs_adaptive_validation.csv", index=False)
    comp_df.to_csv(out / "tables" / "fpr_constraint_validation_vs_test.csv", index=False)

    # per-flow predictions
    def pred_frame(c, tag):
        f = pd.DataFrame({"flow_id": c.flow_ids, "label": c.y})
        for j, t in enumerate(c.cps):
            f[f"p_{t}"] = c.P[:, j]
        for name in ("adaptive", f"fixed_{tmax}"):
            o = policy_outcomes(c, frozen[name], policies[name])
            f[f"{name}_alert"] = o["alert"].to_numpy()
            f[f"{name}_alert_packets"] = o["alert_packets"].to_numpy()
        return f
    pred_frame(curves["val"], "val").to_csv(out / "predictions_val.csv", index=False)
    pred_frame(curves["test"], "test").to_csv(out / "predictions_test.csv", index=False)

    # ------------------------------------------------------------- Experiment 6: simulated robustness (TEST only)
    conditions = [("normal", "normal", lambda f: f)]
    for k in cfg.low_and_slow_factors:
        conditions.append((f"low_and_slow_x{k:g}", f"low-and-slow x{k:g}", lambda f, k=k: low_and_slow(f, k)))
    for s, a in cfg.jitter_settings:
        conditions.append((f"jitter_s{s:g}_a{a:g}", f"jitter σ={s:g}, +{a:g}s", lambda f, s=s, a=a: timing_jitter(f, s, a, cfg.seed)))
    test_flows = [f for f in by_part["test"] if len(f.ts) >= tmax]
    rob_rows = []
    for cid, clabel, fn in conditions:
        fl = [fn(f) if f.label == 1 else f for f in test_flows]    # only malicious flows are perturbed
        tab = build_prefix_table(fl, cps)
        _, cal = _score(tab, model, calibrator)
        c = curves_from_prefix_table(tab, cal, cps)
        for name in ("adaptive", f"fixed_{tmax}"):
            m = summarize(policy_outcomes(c, frozen[name], policies[name]), tmax)
            rob_rows.append({"condition": cid, "condition_label": clabel, "policy": name, "tau": frozen[name], **m})
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(out / "tables" / "robustness_test.csv", index=False)
    chk = rob_df[(rob_df["condition"] == "normal") & (rob_df["policy"] == "adaptive")].iloc[0]
    main_ad = test_df[test_df["policy"] == "adaptive"].iloc[0]
    consistent = (int(chk["TP"]), int(chk["FP"])) == (int(main_ad["TP"]), int(main_ad["FP"]))
    if not consistent:
        log("WARNING: robustness 'normal' condition differs from the main test evaluation - investigate")

    # ------------------------------------------------------------- figures
    fig = out / "figures"
    plots.plot_prf_vs_prefix(prefix_df, fig / "prf_vs_prefix_length.png", " (test)")
    plots.plot_val_fpr_vs_threshold({k: v["sweep"] for k, v in sel.items()}, frozen, cfg.alpha, fig / "val_fpr_vs_threshold.png")
    two = test_df[test_df["policy"].isin(["adaptive", f"fixed_{tmax}"])]
    plots.grouped_bars(two, "policy", {"cost_malicious": "malicious flows", "cost_all": "all flows"},
                       "Normalised observation cost (packets/Tmax)", "Observation cost (test)", fig / "observation_cost.png")
    plots.grouped_bars(two, "policy", {"latency_mean_s": "mean", "latency_median_s": "median"},
                       "Detection latency (s, detected malicious flows)", "Detection latency (test)", fig / "detection_latency.png")
    plots.grouped_bars(two, "policy", {"debd_mean_all_pct": "mean, undetected=100%", "debd_mean_detected_pct": "mean, detected only",
                                       "debd_byte_weighted_all_pct": "byte-weighted, undetected=100%"},
                       "DEBD proxy (%)", "Pre-alert exposure proxy (test)", fig / "debd_comparison.png")
    plots.grouped_bars(two, "policy", {"coverage": "coverage (= recall)", "fpr": "FPR"}, "Fraction", "Coverage and FPR (test)",
                       fig / "detection_coverage.png")
    plots.plot_robustness(rob_df, fig / "robustness.png")

    # ------------------------------------------------------------- reproducibility record + summary
    record = {"config": asdict(cfg), "environment": environment_record(), "synthetic": synthetic,
              "feature_names": FEATURE_NAMES, "experiment1": exp1,
              "model": {"type": "RandomForestClassifier", "n_estimators": cfg.rf_n_estimators,
                        "min_samples_leaf": cfg.rf_min_samples_leaf, "class_weight": cfg.rf_class_weight,
                        "flow_weighting": cfg.flow_weighting},
              "calibration": calibrator.describe(), "frozen_thresholds": frozen,
              "robustness_normal_matches_main_test": consistent,
              "evasion_note": "offline timestamp perturbation of malicious TEST flows only; simulated, not real attack traffic"}
    (out / "run_record.json").write_text(json.dumps(record, indent=2, default=str))

    comp_cols = ["policy", "tau", "n_benign_val", "val_fp", "val_fpr", "val_fpr_upper95", "n_benign_test", "test_fp", "test_fpr", "test_fpr_upper95"]
    main_cols = ["policy", "tau", "precision", "recall", "f1", "fpr", "coverage", "avg_packets_malicious", "cost_malicious",
                 "latency_mean_s", "debd_mean_all_pct", "debd_byte_weighted_all_pct"]
    rob_cols = ["condition_label", "policy", "precision", "recall", "f1", "fpr", "coverage", "cost_malicious", "latency_mean_s", "debd_mean_all_pct"]
    md = (banner + f"# Experiment {out.name}\n\nMeasured values only; interpretation is for the authors to write.\n\n"
          f"## Dataset statistics (flows able to supply >= t packets)\n{md_table(dataset_stats)}\n\n"
          f"## Prefix-level classification, test flows with >= {tmax} packets, threshold {cfg.prefix_eval_threshold}\n{md_table(prefix_df)}\n\n"
          f"## FPR constraint (alpha={cfg.alpha}) - validation vs test\n{md_table(comp_df[comp_cols])}\n\n"
          f"## Fixed-window vs adaptive (test)\n{md_table(test_df[main_cols])}\n\n"
          f"## Simulated timing perturbations (test; malicious flows perturbed, thresholds frozen)\n{md_table(rob_df[rob_cols])}\n\n"
          "## Conventions\nDEBD is a proxy for pre-alert malicious-traffic exposure (IP-level bytes, both directions, "
          "including the alert packet, retransmissions removed, undetected malicious = 100%). Coverage equals recall at flow level. "
          "Non-detected flows count as FN (malicious) or no-alert (benign) and are observed up to Tmax.\n")
    (out / "summary.md").write_text(md)
    log(f"[done] results in {out}")
    return {"thresholds": frozen, "test": test_df, "validation": val_df, "constraint": comp_df, "robustness": rob_df,
            "prefix": prefix_df, "out_dir": str(out)}
