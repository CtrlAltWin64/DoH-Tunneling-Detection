"""Check the archived seven-capture report's aggregate evidence without training.

This is arithmetic and file-integrity verification, not a replay from predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta


def verify(report_dir):
    root = Path(report_dir)
    checks = []

    def check(name, actual, expected):
        passed = bool(np.isclose(actual, expected, atol=1e-10, rtol=1e-8, equal_nan=True))
        checks.append({"check": name, "passed": passed})

    full = pd.read_csv(root / "tables/main_performance.csv")
    for i, r in full.iterrows():
        tp, fp, tn, fn = (r[k] for k in ("TP", "FP", "TN", "FN"))
        expected = {"eligible": tp + fp + tn + fn, "n_benign": fp + tn,
                    "n_malicious": tp + fn, "fpr_numerator": fp, "fpr_denominator": fp + tn,
                    "fnr_numerator": fn, "fnr_denominator": fn + tp,
                    "accuracy": (tp + tn) / (tp + fp + tn + fn),
                    "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                    "fpr": fp / (fp + tn), "fnr": fn / (tp + fn),
                    "f1": 2 * tp / (2 * tp + fp + fn),
                    "outer_test_population": r.eligible + r.excluded}
        for k, v in expected.items():
            check(f"main/{i}/{k}", r[k], v)
        for metric, k, n in (("fpr", fp, fp + tn), ("fnr", fn, fn + tp),
                              ("recall", tp, tp + fn), ("precision", tp, tp + fp),
                              ("accuracy", tp + tn, tp + tn + fp + fn)):
            bounds = (0 if k == 0 else beta.ppf(.025, k, n-k+1),
                      1 if k == n else beta.ppf(.975, k+1, n-k),
                      1 if k == n else beta.ppf(.95, k+1, n-k))
            for suffix, bound in zip(("ci95_low", "ci95_high", "upper95_one_sided"), bounds):
                check(f"main/{i}/{metric}_{suffix}", r[f"{metric}_{suffix}"], bound)
    common = full[full.population == "common_ge20"].set_index(["model", "policy"])
    compact = pd.read_csv(root / "tables/main_performance_compact.csv").set_index(["model", "policy"])
    for key, r in compact.iterrows():
        for col in compact:
            check(f"compact/{key}/{col}", r[col], common.loc[key, col])
    cost = pd.read_csv(root / "tables/observation_cost_latency.csv").set_index(["model", "policy"])
    stop = pd.read_csv(root / "tables/stopping_distribution.csv")
    for key, group in stop.groupby(["model", "policy"]):
        n = group.flows.sum()
        check(f"stops/{key}/n", n, common.loc[key, "eligible"])
        mean = float((group.flows * group.checkpoint).sum() / n)
        check(f"cost/{key}/mean", mean, cost.loc[key, "mean_packets"])
        check(f"cost/{key}/fraction", mean / 20, cost.loc[key, "cost_fraction_of_fixed20"])
        check(f"cost/{key}/reduction", (1 - mean / 20) * 100, cost.loc[key, "reduction_pct"])
        for i, r in group.iterrows():
            check(f"stops/{i}/denominator", r.denominator, n)
            check(f"stops/{i}/percent", r.percent, 100 * r.flows / n)
            check(f"stops/{i}/accounting", r.flows, r.alerts + r.unresolved)
    eligibility = pd.read_csv(root / "tables/checkpoint_eligibility.csv")
    for i, r in eligibility.iterrows():
        check(f"eligibility/{i}", r.total, r.eligible + r.excluded)
    images = []
    for path in sorted((root / "figures").glob("*")):
        if path.suffix == ".png":
            with Image.open(path) as im:
                im.verify()
        elif path.suffix == ".svg":
            ET.parse(path)
        else:
            continue
        images.append(path.relative_to(root).as_posix())
    historical = json.loads((root / "metrics/artifact_hashes.json").read_text())
    inventory = {"exact_match": [], "line_endings_only": [], "different": [], "missing": []}
    for name, expected in historical.items():
        path = root / name.replace("\\", "/")
        if not path.exists():
            inventory["missing"].append(name)
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() == expected:
            state = "exact_match"
        elif hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest() == expected:
            state = "line_endings_only"
        else:
            state = "different"
        inventory[state].append(name)
    return {"scope": "Aggregate arithmetic, PNG/SVG integrity and historical hash inventory only; no score/model replay.",
            "checks": len(checks), "passed": sum(c["passed"] for c in checks),
            "failed": sum(not c["passed"] for c in checks),
            "failures": [c for c in checks if not c["passed"]],
            "figures_validated": images, "historical_hash_inventory": inventory}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports/final_experiment_results_20261019")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("--output already exists; choose a new file")
    report = verify(args.report_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("checks", "passed", "failed")}))
    raise SystemExit(bool(report["failed"]))
