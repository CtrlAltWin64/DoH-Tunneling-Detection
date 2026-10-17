"""Central configuration for the adaptive DoH-stopping project.

Every experiment records ``asdict(Config)`` next to its results, so all
choices (seed, checkpoints, FPR target, label rules, ...) are reproducible.

Dataset support:
- Default dataset: UFU-DoH-EXF (DNS-360 Dataset, released 2026, AWS capture 2024-2026,
  controlled DoH exfiltration vs benign Tranco queries on port 443).
- Legacy/Alternative: CIRA-CIC-DoHBrw-2020.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


DATASET_LABEL_RULES: dict[str, dict[str, str]] = {
    "ufu_doh_exf": {
        "benign": "benign",
        "exfil": "malicious",
        "malicious": "malicious",
    },
    "cira_cic_dohbrw_2020": {
        "benign": "benign",
        "malicious": "malicious",
        "non-doh": "exclude",
    },
}


@dataclass
class Config:
    # ---------------- paths (relative to the project folder) ----------------
    raw_dir: str = os.environ.get("DOH_RAW_DIR", str(PROJECT_ROOT / "data" / "raw"))
    processed_dir: str = str(PROJECT_ROOT / "data" / "processed")
    results_dir: str = str(PROJECT_ROOT / "data" / "results")
    experiment_name: str = "exp_default"

    # ---------------- Dataset & PCAP input ----------------------------------
    # Target dataset: "ufu_doh_exf" (post-2022) or "cira_cic_dohbrw_2020"
    dataset_name: str = "ufu_doh_exf"

    # Ordered mapping: path keyword -> "benign" | "malicious" | "exclude".
    # Intentionally empty by default to force review via `main.py inspect`.
    # Presets are available in DATASET_LABEL_RULES.
    pcap_label_rules: dict = field(default_factory=dict)
    pcap_extensions: tuple = (".pcap", ".pcapng")
    # Keep only TCP flows that use this port (DoH = HTTPS on 443). None = no filter.
    doh_port: int | None = 443
    # Optional allow-list of DoH resolver IPs. When specified, only flows
    # communicating with these resolver IPs are included.
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
    # "source_file": all flows of one PCAP stay together (stricter capture-level split;
    # tests generalisation across capture sessions; needs >= 5 PCAP files per class).
    group_by: str = "flow"

    # ---------------- model selection -------
    # "rf"      : Experiment A -- Random Forest baseline
    # "cnn_gru" : Experiments B/C -- 1D-CNN + GRU hybrid
    # "both"    : trains and evaluates both models in the same run
    model_type: str = "both"

    # ---------------- Random Forest ---------
    rf_n_estimators: int = 200
    rf_min_samples_leaf: int = 3
    rf_class_weight: str | None = "balanced_subsample"
    flow_weighting: bool = True  # each flow contributes equal total weight (1/#prefixes of that flow)

    # ---------------- CNN-GRU ---------------
    cnn_conv_channels: int = 64       # output channels for both Conv1d layers
    cnn_conv_kernel: int = 3          # kernel width; must be <= smallest checkpoint
    cnn_gru_hidden: int = 64          # GRU hidden size
    cnn_gru_layers: int = 2           # stacked GRU layers
    cnn_dropout: float = 0.2          # dropout between GRU layers (only if gru_layers > 1)
    cnn_epochs: int = 30              # max training epochs
    cnn_patience: int = 7             # early-stopping patience (val-loss)
    cnn_batch_size: int = 128         # mini-batch size
    cnn_lr: float = 1e-3              # Adam learning rate
    cnn_device: str = "auto"          # "auto", "cpu", or "cuda"

    # ---------------- calibration ----------------
    calibration: str = "sigmoid"           # "sigmoid" (Platt) or "none"
    calibration_per_checkpoint: bool = True
    calibration_folds: int = 5             # flow-grouped, inside the training partition only (for RF)
    calibration_split_ratio: float = 0.25  # fraction of training partition reserved for CNN-GRU calibration

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
        if self.model_type not in ("rf", "cnn_gru", "both"):
            raise ValueError("model_type must be 'rf', 'cnn_gru', or 'both'")
        if not 0.05 <= self.calibration_split_ratio <= 0.5:
            raise ValueError("calibration_split_ratio must be between 0.05 and 0.5")
        if self.dataset_name not in ("ufu_doh_exf", "cira_cic_dohbrw_2020", "custom"):
            raise ValueError(f"Unknown dataset_name {self.dataset_name!r}")
        bad = {v for v in self.pcap_label_rules.values()} - {"benign", "malicious", "exclude"}
        if bad:
            raise ValueError(f"pcap_label_rules values must be benign/malicious/exclude, got {bad}")

    def get_preset_label_rules(self) -> dict[str, str]:
        """Return preset label rules for the configured dataset_name."""
        return dict(DATASET_LABEL_RULES.get(self.dataset_name, {}))

