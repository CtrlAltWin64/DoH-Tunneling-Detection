"""Independently audit persisted experiment evidence without training or rewriting it.

Run from the repository root with Python 3.14. Output goes beside this script.
Historical metrics are recomputed from saved probabilities plus the processed flow
metadata, using explicit NumPy formulas rather than the production evaluators.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
CHECKS = []


def check(where, observed, expected, atol=1e-8):
    if isinstance(observed, (float, int, np.number)) and isinstance(expected, (float, int, np.number)):
        ok = bool(np.isclose(observed, expected, atol=atol, rtol=1e-10, equal_nan=True))
    else:
        ok = observed == expected
    CHECKS.append({"location": where, "passed": ok, "observed": observed, "expected": expected})


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def divide(a, b):
    return float(a / b) if b else float("nan")


def intervals(fp, n):
    return {
        "fpr_ci95_low": 0.0 if fp == 0 else float(beta.ppf(.025, fp, n - fp + 1)),
        "fpr_ci95_high": 1.0 if fp == n else float(beta.ppf(.975, fp + 1, n - fp)),
        "fpr_upper95_one_sided": 1.0 if fp == n else float(beta.ppf(.95, fp + 1, n - fp)),
    }


def count_metrics(y, alert, observed, tmax):
    mal, ben = y == 1, y == 0
    tp, fn = int(sum(mal & alert)), int(sum(mal & ~alert))
    fp, tn = int(sum(ben & alert)), int(sum(ben & ~alert))
    early = int(sum(mal & alert & (observed < tmax)))
    terminal = int(sum(mal & alert & (observed == tmax)))
    return {
        "n_flows": len(y), "n_malicious": int(sum(mal)), "n_benign": int(sum(ben)),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn, "TP_early": early, "TP_terminal": terminal,
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn), "fpr": divide(fp, fp + tn),
        "coverage": divide(tp, tp + fn), "early_detection_rate": divide(early, tp + fn),
        "terminal_detection_rate": divide(terminal, tp + fn),
        "avg_packets_all": float(observed.mean()), "avg_packets_malicious": float(observed[mal].mean()),
        "cost_all": float(observed.mean() / tmax), "cost_malicious": float(observed[mal].mean() / tmax),
        "n_detected_for_latency": tp, **intervals(fp, fp + tn),
    }


def independent_outcomes(pred, cps, policy, tau, flow_map):
    selected = cps if policy == "adaptive" else [int(policy.split("_")[1])]
    scores = pred[[f"p_{c}" for c in selected]].to_numpy()
    crossed = scores >= tau
    alert = crossed.any(axis=1)
    stop = np.where(alert, crossed.argmax(axis=1), len(selected) - 1)
    observed = np.asarray(selected)[stop]
    y = pred.label.to_numpy()
    result = count_metrics(y, alert, observed, max(cps))
    if flow_map is not None:
        selected_flows = [flow_map[fid] for fid in pred.flow_id]
        dur = np.array([float(f.ts[int(t) - 1] - f.ts[0]) for f, t in zip(selected_flows, observed)])
        before = np.array([float(np.sum(f.length[:int(t)])) for f, t in zip(selected_flows, observed)])
        total = np.array([f.total_bytes for f in selected_flows], dtype=float)
        detected = (y == 1) & alert
        mal = y == 1
        result.update(
            latency_mean_s=float(dur[detected].mean()) if detected.any() else math.nan,
            latency_median_s=float(np.median(dur[detected])) if detected.any() else math.nan,
            debd_mean_all_pct=float(np.where(alert[mal], 100 * before[mal] / total[mal], 100).mean()),
            debd_mean_detected_pct=float((100 * before[detected] / total[detected]).mean()) if detected.any() else math.nan,
            debd_byte_weighted_all_pct=float(100 * np.where(alert[mal], before[mal], total[mal]).sum() / total[mal].sum()),
        )
        if policy in ("adaptive", f"fixed_{max(cps)}"):
            check(f"predictions/{policy}/bytes", np.allclose(pred[f"{policy}_bytes_pre"], before), True)
            check(f"predictions/{policy}/total_bytes", np.allclose(pred[f"{policy}_total_bytes"], total), True)
            check(f"predictions/{policy}/latency", np.allclose(pred[f"{policy}_latency_s"], np.where(alert, dur, np.nan), equal_nan=True), True)
    if policy in ("adaptive", f"fixed_{max(cps)}"):
        check(f"predictions/{policy}/alert", bool(np.array_equal(pred[f"{policy}_alert"], alert)), True)
        check(f"predictions/{policy}/observed_packets", bool(np.array_equal(pred[f"{policy}_observed_packets"], observed)), True)
    return result


def prefix_metrics(y, score):
    pred = score >= .5
    tp, fp = int(sum(pred & (y == 1))), int(sum(pred & (y == 0)))
    fn, tn = int(sum(~pred & (y == 1))), int(sum(~pred & (y == 0)))
    bins = np.clip(np.digitize(score, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ece = sum(float((bins == b).mean()) * abs(float(score[bins == b].mean() - y[bins == b].mean())) for b in range(10) if (bins == b).any())
    return {"n":len(y), "precision":divide(tp,tp+fp), "recall":divide(tp,tp+fn), "f1":divide(2*tp,2*tp+fp+fn),
            "fpr":divide(fp,fp+tn), "auc":float(roc_auc_score(y,score)), "brier":float(np.mean((score-y)**2)), "ece":ece}


def verify_dataset_stats(path, flows, split):
    if flows is None:
        return
    assignment = dict(zip(split.flow_id, split.partition))
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        label = 0 if row.label == "benign" else 1
        cohort = [f for f in flows if f.label == label and (row.partition == "all_flows" or assignment.get(f.flow_id) == row.partition)]
        check(str(path.relative_to(ROOT)) + f"/{row.partition}/{row.label}/n_flows", len(cohort), int(row.n_flows))
        for col in frame:
            if col.startswith("n_ge_"):
                cp = int(col.split("_")[-1])
                n = sum(len(f.ts) >= cp for f in cohort)
                check(str(path.relative_to(ROOT)) + f"/{row.partition}/{row.label}/{col}", n, int(row[col]))
                check(str(path.relative_to(ROOT)) + f"/{row.partition}/{row.label}/pct_ge_{cp}", round(100 * n / len(cohort), 2), row[f"pct_ge_{cp}"])


def run(output=None):
    output = Path(output) if output else OUT / "evidence_verification.json"
    if output.exists():
        raise FileExistsError(f"Preserve {output}; choose a new --output path")
    flows = pickle.loads((ROOT / "data/processed/flows.pkl").read_bytes())
    flow_map = {f.flow_id:f for f in flows}
    raw = sorted((ROOT / "data/raw").rglob("*.pcap*"))
    report = {"created_utc":datetime.now(timezone.utc).isoformat(), "workspace":str(ROOT),
        "method":"Independent equations; no training, raw-capture reprocessing, or historical artifact writes. Floats compared at absolute 1e-8 plus relative 1e-10.",
        "raw_captures":[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in raw],
        "flow_counts": {"total":len(flows),"by_label":dict(Counter(str(f.label) for f in flows)),"by_source":dict(Counter(f.source_file for f in flows))},
        "prefix_eligibility":{}, "experiments":{}, "limitations":[]}
    check("processed/unique_flow_ids", len(flow_map), len(flows))
    for cp in [2,4,5,8,10,15,20,30,40]:
        report["prefix_eligibility"][str(cp)] = {str(label):sum(f.label==label and len(f.ts)>=cp for f in flows) for label in [0,1]}
    processed_report=json.loads((ROOT / "data/processed/flows_report.json").read_text())
    check("flows_report/flows_total",len(flows),processed_report["flows_total"])
    check("flows_report/flows_built",len(flows),processed_report["counters"]["flows_built"])
    check("flows_report/source_files", sorted(set(f.source_file for f in flows)),sorted(x["file"] for x in processed_report["files_used"]))
    check("raw/distinct_hashes", len({p["sha256"] for p in report["raw_captures"]}),len(raw))
    # This verifier targets the historical schema, not arbitrary future runs.
    names = ("exp_ufu_real_20261018", "exp_ufu_real_cp10_20",
             "exp_ufu_real_cp2_4_8_10_15", "member3_smoke_20261017")
    for experiment in (ROOT / "data/results" / name for name in names):
        if not experiment.is_dir():
            raise FileNotFoundError(f"Restore the historical result directory: {experiment}")
        print(f"Verifying {experiment.name}", flush=True)
        record_path=experiment / "run_record.json"
        x={"path":str(experiment),"files":{},"models":{},"duplicate_figures":[],"figure_provenance":[],"full_score_reverification":False}
        report["experiments"][experiment.name]=x
        for ext in ["csv","png","json","pt"]:
            paths=list(experiment.rglob(f"*.{ext}"))
            x["files"][ext]=len(paths)
        x["files"]["table_csv"]=len(list((experiment/"tables").glob("*.csv")))
        split=pd.read_csv(experiment/"split_assignment.csv")
        manifest=json.loads((experiment/"split_manifest.json").read_text())
        check(experiment.name+"/split_unique_ids",split.flow_id.nunique(),len(split))
        check(experiment.name+"/split_total",len(split),manifest["total_flows"])
        x["split_partition_counts"]={name:{"total":len(g),"benign":int(sum(g.label==0)),"malicious":int(sum(g.label==1)),"sources":sorted(g.source_file.unique())} for name,g in split.groupby("partition")}
        for part,vals in x["split_partition_counts"].items():
            for metric,col in [("total","n_flows"),("benign","n_benign"),("malicious","n_malicious")]:
                check(experiment.name+"/manifest/"+part+"/"+col, vals[metric],manifest["partitions"][part][col])
        x["sources_present_in_all_partitions"]=sorted(set.intersection(*(set(g.source_file) for _,g in split.groupby("partition"))))
        if not record_path.exists():
            x["status"]="partial: no run_record, model scores, or figures"
            verify_dataset_stats(experiment/"tables/dataset_stats.csv",flows,split)
            continue
        record=json.loads(record_path.read_text())
        synthetic=record["synthetic"]
        cps=record["config"]["checkpoints"]
        x.update(status="synthetic smoke" if synthetic else "completed exploratory real-data run", checkpoints=cps, synthetic=synthetic,
                 recorded_config=record["config"], full_score_reverification=not synthetic)
        verify_dataset_stats(experiment/"tables/dataset_stats.csv",None if synthetic else flows,split)
        if not synthetic:
            expected_ids={f.flow_id for f in flows if len(f.ts)>=min(cps)}
            check(experiment.name+"/split_eligible_ids",set(split.flow_id)==expected_ids,True)
            signatures=Counter((tuple(f.length[:max(cps)]),tuple(f.direction[:max(cps)])) for f in flows if f.flow_id in expected_ids)
            duplicate_members=sum(n for n in signatures.values() if n>1)
            duplicate_extra=sum(n-1 for n in signatures.values())
            x["identical_length_direction_prefixes"]={"members_in_repeated_groups":duplicate_members,"additional_duplicate_rows":duplicate_extra,"unique_signatures":len(signatures)}
            signatures_partitions=defaultdict(set)
            for row in split.itertuples():
                f=flow_map[row.flow_id]
                signatures_partitions[(tuple(f.length[:max(cps)]),tuple(f.direction[:max(cps)]))].add(row.partition)
            x["signatures_shared_train_and_test"]=sum({"train","test"} <= parts for parts in signatures_partitions.values())
        hist=json.loads((experiment/"cnn_gru_train_history.json").read_text())
        x["training_history"]={"epochs_in_history":len(hist["train_loss"]),"reported_epochs_trained":record["models"]["cnn_gru"]["epochs_trained"],
            "saved_best_epoch":hist["best_epoch"],"saved_best_val_loss":hist["best_val_loss"],"true_min_epoch":int(np.argmin(hist["val_loss"]))+1,"true_min_val_loss":min(hist["val_loss"])}
        prediction_cache={}
        for model,prefix in [("rf",""),("cnn_gru","cnn_gru_")]:
            m={"frozen_thresholds":{},"normal_metrics":{},"robustness_conditions":[],"validation_sweep_rows_verified":0}
            x["models"][model]=m
            thresholds=json.loads((experiment/f"{prefix}thresholds.json").read_text())
            for split_name,table_suffix in [("val","validation"),("test","test")]:
                pred=pd.read_csv(experiment/f"{prefix}predictions_{split_name}.csv")
                prediction_cache[model,split_name]=pred
                check(experiment.name+f"/{model}/{split_name}/unique_ids", pred.flow_id.nunique(),len(pred))
                check(experiment.name+f"/{model}/{split_name}/finite_probability",bool(np.isfinite(pred[[f"p_{c}" for c in cps]]).all().all()),True)
                check(experiment.name+f"/{model}/{split_name}/probability_range",bool(((pred[[f"p_{c}" for c in cps]]>=0)&(pred[[f"p_{c}" for c in cps]]<=1)).all().all()),True)
                if not synthetic:
                    expected=set(r.flow_id for r in split.itertuples() if r.partition==split_name and len(flow_map[r.flow_id].ts)>=max(cps))
                    check(experiment.name+f"/{model}/{split_name}/eligible_ids",set(pred.flow_id)==expected,True)
                    check(experiment.name+f"/{model}/{split_name}/labels",all(flow_map[r.flow_id].label==r.label for r in pred.itertuples()),True)
                table=pd.read_csv(experiment/f"tables/{prefix}fixed_vs_adaptive_{table_suffix}.csv")
                m["normal_metrics"][split_name]={}
                for row in table.to_dict("records"):
                    policy=row["policy"];tau=thresholds[policy]["tau"]
                    m["frozen_thresholds"][policy]=tau
                    check(experiment.name+f"/{model}/{split_name}/{policy}/tau",tau,row["tau"])
                    values=independent_outcomes(pred,cps,policy,tau,None if synthetic else flow_map)
                    m["normal_metrics"][split_name][policy]=values
                    for k,v in values.items():
                        check(experiment.name+f"/{model}/{split_name}/{policy}/{k}",v,row[k])
                if split_name=="test":
                    for row in pd.read_csv(experiment/f"tables/{prefix}prefix_level_test.csv").to_dict("records"):
                        values=prefix_metrics(pred.label.to_numpy(),pred[f"p_{int(row['checkpoint'])}"].to_numpy())
                        for k,v in values.items():check(experiment.name+f"/{model}/prefix/{int(row['checkpoint'])}/{k}",v,row[k])
                else:
                    for policy,tinfo in thresholds.items():
                        sweep=pd.read_csv(experiment/f"tables/{prefix}threshold_sweep_val_{policy}.csv")
                        columns=cps if policy=="adaptive" else [int(policy.split("_")[1])]
                        probabilities=pred[[f"p_{c}" for c in columns]].to_numpy()
                        y=pred.label.to_numpy();ben=y==0;mal=y==1
                        for row in sweep.itertuples():
                            crossing=probabilities>=row.tau;alert=crossing.any(axis=1)
                            observed=np.asarray(columns)[np.where(alert,crossing.argmax(axis=1),len(columns)-1)]
                            actual={"fp":int(sum(alert&ben)),"n_benign":int(sum(ben)),"fpr":float((alert&ben).sum()/ben.sum()),"coverage":float((alert&mal).sum()/mal.sum()),"mean_cost_malicious":float(observed[mal].mean()/max(columns))}
                            for k,v in actual.items():check(experiment.name+f"/{model}/sweep/{policy}/{row.tau}/{k}",v,getattr(row,k))
                        m["validation_sweep_rows_verified"]+=len(sweep)
                        chosen=sweep[sweep.fpr<=record["config"]["alpha"]].iloc[0]
                        for k,source in [("tau","tau"),("val_fp","fp"),("n_benign_val","n_benign"),("val_fpr","fpr"),("val_coverage","coverage")]:
                            check(experiment.name+f"/{model}/threshold/{policy}/{k}",chosen[source],tinfo[k])
            rob=pd.read_csv(experiment/f"tables/{prefix}robustness_test.csv")
            m["robustness_conditions"]=list(rob.condition.unique())
            m["robustness_note"]="Non-normal perturbation probabilities were not persisted; aggregate arithmetic can be checked but fitted-model responses cannot be replayed from saved scores."
            for row in rob.to_dict("records"):
                tag=experiment.name+f"/{model}/robustness/{row['condition']}/{row['policy']}"
                tp,fn,fp,tn=(row[k] for k in ["TP","FN","FP","TN"])
                vals={"n_flows":tp+fn+fp+tn,"n_malicious":tp+fn,"n_benign":fp+tn,"precision":divide(tp,tp+fp),"recall":divide(tp,tp+fn),"coverage":divide(tp,tp+fn),"f1":divide(2*tp,2*tp+fp+fn),"fpr":divide(fp,fp+tn),"early_detection_rate":divide(row["TP_early"],tp+fn),"terminal_detection_rate":divide(row["TP_terminal"],tp+fn),"n_detected_for_latency":tp,"cost_all":row["avg_packets_all"]/max(cps),"cost_malicious":row["avg_packets_malicious"]/max(cps),**intervals(fp,fp+tn)}
                for k,v in vals.items():check(tag+"/"+k,v,row[k])
                if row["condition"]=="normal":
                    for k,v in m["normal_metrics"]["test"][row["policy"]].items():check(tag+"/normal/"+k,v,row[k])
            m["calibration_reverification_limitation"]="Calibration CSVs include unpersisted raw probabilities and shorter validation flows excluded from saved policy predictions; historical calibrated scores alone cannot independently reproduce every Brier/ECE entry."
        for split_name in ["val","test"]:
            a,b=prediction_cache["rf",split_name],prediction_cache["cnn_gru",split_name]
            check(experiment.name+f"/{split_name}/model_population_equal",a.flow_id.tolist(),b.flow_id.tolist())
            x[f"{split_name}_model_probability_differences"]={str(c):int(sum(a[f"p_{c}"].to_numpy()!=b[f"p_{c}"].to_numpy())) for c in cps}
        figures=defaultdict(list)
        for p in sorted((experiment/"figures").glob("*.png")):
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                item={"path":str(p.relative_to(ROOT)),"sha256":sha(p),"pixels":list(im.size)}
            figures[item["sha256"]].append(p.name)
            x["figure_provenance"].append(item)
        x["duplicate_figures"]=[v for v in figures.values() if len(v)>1]
        x["historical_rf_model_persisted"]=False
        x["historical_cnn_calibrator_persisted"]=False
    report["limitations"]=[
        "Five local raw captures are present; File02 and File03 are absent from data/raw and the processed artifact.",
        "All real-data outer partitions share all five source captures. No unseen-capture generalization test exists.",
        "Current raw/processed hashes establish present identity only; historical run records did not save hashes to prove training-time identity.",
        "Historical RF trees and CNN calibrators are absent. Calibrated saved scores can be audited, but complete historical fitted prediction pipelines cannot be reloaded.",
        "Perturbed per-flow scores and raw calibration predictions are absent. Their reported aggregate arithmetic, not all fitted-model outputs, is independently verifiable.",
        "Preprocessing packet counters were not independently recomputed by re-reading full captures in this audit.",
        "Figure existence, PNG integrity, and artifact hashes are checked; document embedding/caption validation is performed separately.",
        "No saved SHAP/Captum package-execution artifacts were present at audit start; native IG diagnostic prose is not a persisted attribution dataset.",
    ]
    report["verification"]={"checks":len(CHECKS),"passed":sum(x["passed"] for x in CHECKS),"failed":sum(not x["passed"] for x in CHECKS),"failures":[x for x in CHECKS if not x["passed"]]}
    report["artifact_hashes"]={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"data").rglob("*")) if p.is_file() and p.suffix in {".csv",".json",".pkl",".pt"}}
    def clean(value):
        if isinstance(value, dict):return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)):return [clean(v) for v in value]
        if isinstance(value, np.generic):return clean(value.item())
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean(report),indent=2),encoding="utf-8")
    print(json.dumps(clean(report["verification"]),indent=2))
    return 1 if report["verification"]["failed"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New verification JSON path; existing files are refused")
    raise SystemExit(run(parser.parse_args().output))
