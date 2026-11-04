"""Versioned, finite JSON summary for offline consumers of new experiments.

This module does not run models or serve an API. CSVs retain full precision and
per-flow detail; paths in the manifest are relative to its experiment directory.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_result_manifest(out_dir):
    """Called only after the pipeline has saved all metrics, models and figures."""
    out = Path(out_dir)
    record = json.loads((out / "run_record.json").read_text(encoding="utf-8"))
    split = pd.read_csv(out / "split_assignment.csv")
    sources = {p: set(g.source_file) for p, g in split.groupby("partition")}
    overlaps = {f"{a}__{b}": sorted(sources[a] & sources[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    models = {}
    cps = record["config"]["checkpoints"]
    for model in record["models"]:
        prefix = "" if model == "rf" else "cnn_gru_"
        table = f"tables/{prefix}fixed_vs_adaptive_test.csv"
        pred_path = f"{prefix}predictions_test.csv"
        pred = pd.read_csv(out / pred_path)
        thresholds = record["models"][model]["frozen_thresholds"]
        stops = []
        for policy, tau in thresholds.items():
            selected = cps if policy == "adaptive" else [int(policy.split("_")[1])]
            crossed = pred[[f"p_{cp}" for cp in selected]].to_numpy() >= tau
            alerted = crossed.any(axis=1)
            observed = np.asarray(selected)[np.where(alerted, crossed.argmax(axis=1), len(selected) - 1)]
            frame = pd.DataFrame({"label": pred.label, "alert": alerted, "observed_packets": observed})
            counts = frame.groupby(["label", "alert", "observed_packets"]).size().reset_index(name="flows")
            stops.extend({"policy": policy, **r} for r in counts.to_dict("records"))
        models[model] = {
            "thresholds": thresholds,
            "test_metrics": pd.read_csv(out / table).to_dict("records"),
            "stopping_distribution": stops,
            "artifacts": {
                "test_metrics": table, "test_predictions": pred_path,
                "validation_metrics": f"tables/{prefix}fixed_vs_adaptive_validation.csv",
                "fpr_constraints": f"tables/{prefix}fpr_constraint_validation_vs_test.csv",
                "simulated_robustness": f"tables/{prefix}robustness_test.csv",
                "bundle": "rf_bundle.pkl" if model == "rf" else "cnn_gru_bundle.pt",
            },
        }
    root = Path(__file__).resolve().parents[1]
    source_hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [root / "config.py", root / "main.py", *sorted((root / "src").glob("*.py"))]}
    payload = {
        "schema_version": 1, "experiment_id": out.name,
        "completion_marker": "run_status.json",
        "synthetic": record["synthetic"], "config": record["config"],
        "environment": record["environment"], "source_sha256": source_hashes,
        "dataset": record["experiment1"],
        "split": {"group_by": record["config"]["group_by"], "capture_overlaps": overlaps,
                  "capture_disjoint": not any(overlaps.values()),
                  "assignment": "split_assignment.csv", "manifest": "split_manifest.json",
                  "eligibility_table": "tables/dataset_stats.csv"},
        "models": models,
        "figures": [p.relative_to(out).as_posix() for p in sorted((out / "figures").glob("*.png"))],
        "conventions": {
            "positive_label": 1, "fpr_denominator": "eligible benign flows (FP + TN)",
            "cohort": "flows reaching every configured checkpoint",
            "cost": "observed packets / largest configured checkpoint",
            "latency": "seconds from first retained packet to alert; detected attacks only",
            "uncertainty": "exact binomial FPR bounds assume independent flows",
            "missing_numeric_value": "null", "robustness": record["evasion_note"],
        },
    }
    path = out / "experiment.json"
    path.write_text(json.dumps(_finite(payload), indent=2, allow_nan=False), encoding="utf-8")
    return path
