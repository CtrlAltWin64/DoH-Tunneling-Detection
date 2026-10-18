"""Command-line entry point.

  python main.py inspect  [--raw-dir DIR]              look at the raw files first (no assumptions)
  python main.py build    [--raw-dir DIR] [--max-packets-per-file N]
                                                         PCAP -> flows (data/processed/flows.pkl)
  python main.py run      --name NAME [--alpha 0.01] [--group-by flow|source_file]
                           [--model rf|cnn_gru|both] [--overwrite]
                                                         Experiments A/B/C on data/processed/flows.pkl
  python main.py all      --name NAME                   build + run
  python main.py dry-run  [--name NAME]                  full pipeline on SYNTHETIC data (code check only)
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from config import Config
from src import inspect_dataset as insp
from src.pipeline import run_experiment
from src.preprocessing import build_flows, load_flows, save_flows
from src.synthetic import make_synthetic_flows


def _cfg(args) -> Config:
    cfg = Config()
    kw = {}
    if getattr(args, "raw_dir", None):
        kw["raw_dir"] = args.raw_dir
    if getattr(args, "alpha", None) is not None:
        kw["alpha"] = args.alpha
    if getattr(args, "group_by", None):
        kw["group_by"] = args.group_by
    if getattr(args, "seed", None) is not None:
        kw["seed"] = args.seed
    if getattr(args, "max_packets_per_file", None):
        kw["max_packets_per_file"] = args.max_packets_per_file
    if getattr(args, "name", None):
        kw["experiment_name"] = args.name
    for key in ("processed_dir", "results_dir", "checkpoints"):
        value = getattr(args, key, None)
        if value is not None:
            kw[key] = tuple(value) if key == "checkpoints" else value
    if getattr(args, "model", None):
        kw["model_type"] = args.model
    # CNN-GRU overrides (optional)
    if getattr(args, "cnn_epochs", None) is not None:
        kw["cnn_epochs"] = args.cnn_epochs
    if getattr(args, "cnn_device", None):
        kw["cnn_device"] = args.cnn_device
    c = replace(cfg, **kw)
    if not c.pcap_label_rules:
        c = replace(c, pcap_label_rules=c.get_preset_label_rules())
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("inspect", "build", "run", "all", "dry-run"):
        p = sub.add_parser(name)
        p.add_argument("--raw-dir")
        p.add_argument("--processed-dir", help="Directory containing flows.pkl (or build destination)")
        p.add_argument("--results-dir", help="Parent directory for new experiment outputs")
        p.add_argument("--checkpoints", type=int, nargs="+", help="Increasing packet checkpoints, e.g. 10 20")
        p.add_argument("--name")
        p.add_argument("--alpha", type=float)
        p.add_argument("--group-by", choices=["flow", "source_file"])
        p.add_argument("--seed", type=int)
        p.add_argument("--max-packets-per-file", type=int)
        p.add_argument("--overwrite", action="store_true")
        p.add_argument(
            "--model", choices=["rf", "cnn_gru", "both"], default=None,
            help="Which model(s) to train: rf = Random Forest only (Experiment A), "
                 "cnn_gru = CNN-GRU only (Experiments B/C), both = all (default)",
        )
        p.add_argument("--cnn-epochs", "--cnn_epochs", dest="cnn_epochs", type=int, help="Override CNN-GRU training epochs")
        p.add_argument("--cnn-device", "--cnn_device", dest="cnn_device", choices=["auto", "cpu", "cuda"],
                       help="Override CNN-GRU compute device")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = _cfg(args)
    cfg.validate()

    if args.cmd == "inspect":
        rep = insp.inspect_dir(cfg.raw_dir)
        insp.print_report(rep)
        print("\nsaved:", insp.save_report(rep, cfg.processed_dir))
        return 0

    if args.cmd in ("build", "all"):
        flows, report = build_flows(cfg)
        path = save_flows(flows, report, cfg.processed_dir)
        print(f"built {len(flows)} flows -> {path}\ncounters: {report['counters']}")

    if args.cmd in ("run", "all"):
        if not args.name:
            ap.error("--name is required for run/all")
        flows = load_flows(cfg.processed_dir)
        out = Path(cfg.results_dir) / cfg.experiment_name
        run_experiment(flows, cfg, out, synthetic=False, overwrite=args.overwrite)

    if args.cmd == "dry-run":
        print(
            "*** SYNTHETIC DATA: this only checks that the code runs; "
            "it says nothing about DoH traffic ***"
        )
        flows = make_synthetic_flows(n_benign=600, n_malicious=600, seed=cfg.seed)
        out = Path(cfg.results_dir) / (args.name or "synthetic_smoke_test")
        # Lightweight settings for synthetic smoke test
        small = replace(
            cfg,
            rf_n_estimators=60,
            cnn_epochs=args.cnn_epochs if args.cnn_epochs is not None else 5,
            cnn_patience=3,
            experiment_name=out.name,
        )
        run_experiment(flows, small, out, synthetic=True, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
