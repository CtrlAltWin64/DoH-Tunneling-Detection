"""Central configuration for the adaptive DoH-stopping project.

Every experiment records ``asdict(Config)`` next to its results, so all
choices (seed, checkpoints, FPR target, label rules, ...) are reproducible.

The only dataset-specific settings you MUST review are in the "PCAP input"
block (``pcap_label_rules``, ``doh_server_ips``). They are intentionally
empty: they depend on the real folder layout, which is verified with
``python main.py inspect`` rather than assumed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # ---------------- paths (relative to the project folder) ----------------
    raw_dir: str = os.environ.get("DOH_RAW_DIR", str(PROJECT_ROOT / "data" / "raw"))
    processed_dir: str = str(PROJECT_ROOT / "data" / "processed")
    results_dir: str = str(PROJECT_ROOT / "data" / "results")
    experiment_name: str = "exp_default"

    # ---------------- PCAP input (MUST be reviewed for the real dataset) ----
    # Ordered mapping: path keyword -> "benign" | "malicious" | "exclude".
    # A PCAP's path (relative to raw_dir, lower-cased) is matched against the
    # keywords in order; the first hit wins. PCAPs matching nothing are skipped.
    # Example (ONLY after checking your folder names with `main.py inspect`):
    #   {"benign": "benign", "malicious": "malicious", "non-doh": "exclude"}
    pcap_label_rules: dict = field(default_factory=dict)
    pcap_extensions: tuple = (".pcap",)
    # Keep only TCP flows that use this port (DoH = HTTPS on 443). None = no filter.
    doh_port: int | None = 443
    # Optional allow-list of DoH resolver IPs. The dataset page says its CSVs are
    # labelled by server IP, so supplying the resolver IPs here is the most
    # reliable way to separate DoH from other HTTPS traffic. Empty = no filter.
    doh_server_ips: list = field(default_factory=list)
    flow_idle_timeout_s: float = 120.0      # idle gap that ends a flow
    max_stored_packets: int = 200           # per-flow packet arrays are capped (totals are still counted)
    drop_zero_payload_packets: bool = False  # True = ignore pure ACK/SYN/FIN packets
    max_packets_per_file: int | None = None  # smoke-test limiter

    # ---------------- prefixes ----------------
    checkpoints: tuple = (10, 20, 30, 40)

    # ---------------- split ----------------
    seed: int = 42
    # "flow": every flow is its own group (stratified flow-level split).
    # "source_file": all flows of one PCAP stay together (stricter, tests generalisation
    # across capture sessions; needs >= 5 PCAP files per class).
    group_by: str = "flow"

    # ---------------- model ----------------
    rf_n_estimators: int = 200
    rf_min_samples_leaf: int = 3
    rf_class_weight: str | None = "balanced_subsample"
    flow_weighting: bool = True  # each flow contributes equal total weight (1/#prefixes of that flow)

    # ---------------- calibration ----------------
    calibration: str = "sigmoid"           # "sigmoid" (Platt) or "none"
    calibration_per_checkpoint: bool = True
    calibration_folds: int = 5             # flow-grouped, inside the training partition only

    # ---------------- stopping / FPR ----------------
    alpha: float = 0.01                    # FPR target for the complete sequential policy
    threshold_step: float = 0.001
    prefix_eval_threshold: float = 0.5     # only for the prefix-level table (Experiment 2)

    # ---------------- robustness (offline, simulated) ----------------
    low_and_slow_factors: tuple = (2.0, 5.0, 10.0)
    # (sigma of multiplicative log-normal noise, max additive uniform delay in seconds)
    jitter_settings: tuple = ((0.25, 0.0), (0.5, 0.05), (1.0, 0.2))

    def validate(self) -> None:
        cps = list(self.checkpoints)
        if not cps or cps != sorted(set(cps)) or cps[0] < 2:
            raise ValueError("checkpoints must be strictly increasing integers >= 2")
        if self.max_stored_packets < cps[-1]:
            raise ValueError("max_stored_packets must be >= the largest checkpoint")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.group_by not in ("flow", "source_file"):
            raise ValueError("group_by must be 'flow' or 'source_file'")
        if self.calibration not in ("sigmoid", "none"):
            raise ValueError("calibration must be 'sigmoid' or 'none'")
        bad = {v for v in self.pcap_label_rules.values()} - {"benign", "malicious", "exclude"}
        if bad:
            raise ValueError(f"pcap_label_rules values must be benign/malicious/exclude, got {bad}")
