# Adaptive Stopping for Early Detection of Evasive DoH Tunneling

**Information Security Research Project**  
Department of Computer Science & Engineering, MIT Manipal.  
Guided by: Sudhina Kumar GK.

---

## 1. Executive Summary & Honest Experimental Status

This project implements and evaluates an **Adaptive Sequential Stopping** framework for early detection of evasive DNS-over-HTTPS (DoH) tunneling and data exfiltration. The central research question investigated by measurement is:

> *Can adaptive sequential stopping on ordered packet prefixes detect evasive DoH tunnels earlier, reduce observation cost, and limit pre-alert data exposure under a strict 1% False Positive Rate (FPR) constraint, compared to standard fixed-window classifiers?*

### Honest Validation Status

| Dimension | Verification Status | Details |
|---|---|---|
| **Unit & Integration Tests** | **113 tests passing** (`pytest`) | Comprehensive coverage: model shapes, causal convolutions, masked batch normalization, variable-length batching, padding invariance, separate calibration, split leakage checks, sequential stopping, DEBD, PCAP/PCAPNG parsing, and end-to-end pipeline. |
| **Hybrid Model Architecture** | **Fully implemented & validated** | 1D-CNN + GRU with causal left-padded Conv1d, Masked BatchNorm, sequence packing, and linear head. |
| **Probability Calibration** | **Methodologically corrected** | Platt scaling fitted on a strictly separated held-out calibration partition (`train_cal`), eliminating in-sample overconfidence. |
| **Dataset Selection** | **Investigated & configured** | Targeted post-2022 dataset: **UFU-DoH-EXF** (DNS-360 Dataset, published SBSeg Sept 2026, AWS capture 2024–2026). Legacy support for CIRA-CIC-DoHBrw-2020 is preserved. See [DATASET_AND_PREPARATION.md](DATASET_AND_PREPARATION.md) for full report. |
| **Real PCAP Execution** | **Not yet executed on real captures** | Raw PCAP files from UFU-DoH-EXF or CIRA-CIC-DoHBrw-2020 are not hosted in this local repository. The pipeline adapter, label rules, and flow parser are fully functional and tested on synthetic PCAPs. |
| **Synthetic Smoke Test** | **Run successfully** | Full end-to-end execution on synthetic flows generating complete models, thresholds, tables, figures, and manifests under `data/results/synthetic_smoke_test/`. |

---

## 2. Dataset Investigation & Selection (Post-2022 Compliance)

Academic requirements mandate a dataset collected or released **after 2022** and a modern hybrid model rather than Random Forest alone.

### Primary Dataset: UFU-DoH-EXF (DNS-360 Dataset)
* **Authors:** Cristiano L. M. Borges and Prof. Rodrigo S. Miani (Faculty of Computing, Federal University of Uberlândia – UFU, Brazil).
* **Publication Venue & Date:** Proceedings of the *XXVI Simpósio Brasileiro de Cibersegurança (SBSeg)*, **September 2026**.
* **Traffic Capture Period:** Captured between **2024 and 2026** in an Amazon Web Services (AWS) cloud testbed.
* **Testbed Topology:** Deployed in AWS featuring a FortiGate virtual appliance router/chokepoint, an Ubuntu victim machine generating benign DoH traffic via the Tranco top-1M domain snapshot, and a Kali Linux attacker node executing controlled DoH exfiltration attacks.
* **Traffic & Captures:** Over 1,048,575 DoH flows (TCP/443). Raw packet captures (`.pcap` and `.pcapng`) are stored in `data/PCAP/DoH/` using structured naming:
  - `DoH_Benign_*.pcap` → Benign DoH traffic (top-1M domain lookups).
  - `DoH_Exfiltration_*.pcap` → Malicious DoH data exfiltration tunnels.
* **Reconstruction Feasibility:** Because full `.pcap`/`.pcapng` files are provided, true packet-level prefixes (packet lengths, inter-arrival times, directions) can be causally extracted.
* **Capture-Group Splitting:** Each capture file represents an independent collection session, enabling `group_by="source_file"` capture-level splitting to prevent cross-capture session leakage.
* **Local Storage Note:** Raw captures are hosted in the authors' research repository. They must be placed under `data/raw/` before executing real-data runs.
* **Detailed Member 1 Specification:** See [DATASET_AND_PREPARATION.md](DATASET_AND_PREPARATION.md) for full packet preprocessing pipeline, PCAP/PCAPNG parser details, prefix feature documentation, and split leakage proofs.

### Legacy Dataset: CIRA-CIC-DoHBrw-2020
* Captured in **2020** (pre-2022). Supported as an alternative baseline via path-keyword rules (`benign`, `malicious`, `non-doh`).

---

## 3. Architecture & Methodology

### A. 1D-CNN + GRU Hybrid Classifier
The sequence model processes ordered packet metadata prefixes:
1. **Input:** Sequence matrix $(B, L, 8)$ of ordered packet metadata:
   - `pkt_len`: IP total packet length.
   - `direction`: Packet direction relative to flow initiator (+1 forward, -1 backward).
   - `iat`: Inter-arrival time from previous packet (0 for first packet).
   - `elapsed`: Elapsed time since flow start.
   - `cum_pkts_fwd`, `cum_pkts_bwd`: Cumulative packet counts by direction.
   - `cum_bytes_fwd`, `cum_bytes_bwd`: Cumulative byte counts by direction.
2. **Causal 1D Convolutions:** Convolutions pad strictly on the left by $(K - 1)$ with zero right-padding. Packet $t$ never observes information from packet $t+1$, preserving strict temporal causality.
3. **Masked Batch Normalization (`MaskedBatchNorm1d`):** Mean and variance are computed solely across valid sequence positions ($\sum_i L_i$). Padded positions are explicitly masked to zero, ensuring that a sequence's prediction is mathematically invariant to whether it is evaluated alone or batched alongside sequences of differing lengths.
4. **GRU Temporal Aggregation:** 2-layer stacked GRU with `pack_padded_sequence` skipping padded steps. The hidden state from the last valid step $h_{L_i}$ is extracted.
5. **Classification Head:** Linear layer projecting to scalar malicious logit.

### B. Separate Probability Calibration
Rather than calibrating on in-sample training predictions (which causes overconfidence), the training partition is split into:
- **`train_fit` (75%):** Used exclusively for training the CNN-GRU weights.
- **`train_cal` (25%):** Held-out flows scored by the trained model. A monotonic Platt scaling calibrator is fitted on these true out-of-sample scores.
- **Outer Test Isolation:** Outer validation (`val`) is used solely for sequential threshold selection, and test (`test`) is evaluated only after all policies and thresholds are frozen.

### C. Sequential Stopping Policy & 1% FPR Constraint
At checkpoints $t \in \{10, 20, 30, 40\}$ packets:
- An alert is declared at the *first* checkpoint $k$ where calibrated probability $p_k \ge \tau$.
- If no checkpoint crosses $\tau$, the flow is unresolved/not detected and observed up to $t_{max} = 40$.
- **FPR Target ($\alpha = 0.01$):** Complete sequential policy is swept on validation benign flows. The lowest feasible $\tau$ satisfying validation sequential $FPR \le \alpha$ is frozen.
- **Confidence Bounds:** Clopper-Pearson exact 95% binomial confidence intervals are reported.

### D. Short-Flow & Observation Metrics
- **Eligible vs Full Population:** Experiments 3–6 compare fixed-window and adaptive policies on flows with $\ge 40$ packets to maintain identical comparison baselines. Short flows ($<40$ packets) excluded from the 40-packet comparison are reported in `dataset_stats.csv`.
- **Early vs Terminal Decisions:** Decisions are explicitly partitioned into:
  - `early_alert`: Alerted before the final checkpoint ($t \in \{10, 20, 30\}$).
  - `terminal_alert`: Alerted only at the terminal checkpoint ($t = 40$).
  - `unresolved`: Did not cross threshold by $t = 40$.
- **DEBD (Data Exposed Before Detection):** Pre-alert byte exposure proxy: $100 \times \frac{B_{pre}}{B_{total}}$.

---

## 4. Repository Structure

```
GitRepo/
├── .gitignore                     # Git ignore rules for caches, venvs, and captures
├── config.py                      # Central configuration (UFU-DoH-EXF settings, alpha, seeds)
├── main.py                        # CLI entry point (inspect, build, run, dry-run)
├── requirements.txt               # Dependency specifications (numpy, torch, scikit-learn, etc.)
├── README.md                      # Comprehensive project documentation
├── DATASET_AND_PREPARATION.md     # Member 1 technical report & UFU-DoH-EXF data preparation
├── data/
│   ├── raw/                       # Place raw PCAP captures here
│   ├── processed/                 # Generated flows.pkl and flows_report.json
│   └── results/                   # Experiment output directories
│       └── synthetic_smoke_test/  # Separated synthetic pipeline verification results
├── src/
│   ├── adaptive_detector.py       # Sequential stopping policy and threshold selection
│   ├── calibration.py             # Monotonic Platt scaling for Random Forest
│   ├── cnn_gru_calibration.py     # Out-of-sample Platt calibration for CNN-GRU
│   ├── cnn_gru_model.py           # Causal CNN-GRU, Masked BatchNorm, SequenceNormalizer
│   ├── cnn_gru_prefix.py          # Prefix scoring and prefix table construction for CNN-GRU
│   ├── dataset_split.py           # Flow/capture splitting, train_fit/train_cal split, manifest
│   ├── evaluation.py              # Metrics, confusion matrix, DEBD, Clopper-Pearson CIs
│   ├── evasion.py                 # Offline timing perturbation (low-and-slow, jitter)
│   ├── feature_extraction.py      # Causal RF prefix features
│   ├── inspect_dataset.py         # PCAP folder inspection without prior assumptions
│   ├── pipeline.py                # Full leakage-safe pipeline (Experiments A, B, C)
│   ├── plots.py                   # Generation of publication-quality figures
│   ├── prefix_generation.py       # RF prefix table generation
│   ├── preprocessing.py           # PCAP decoding into bidirectional flows
│   ├── synthetic.py               # Synthetic flow and PCAP fixture generator
│   └── train_baseline.py          # Random Forest baseline trainer
└── tests/                         # 113 automated unit & integration tests
    ├── test_cnn_gru.py            # CNN-GRU shapes, causality, padding invariance, calibration
    ├── test_evasion.py            # Timing perturbation properties
    ├── test_metrics.py            # Confusion metrics, latency, DEBD, Clopper-Pearson
    ├── test_pcap_end_to_end.py    # PCAP inspect -> build -> pipeline integration
    ├── test_pipeline.py           # Leakage, threshold freezing, reproducibility
    ├── test_prefixes.py           # Causal packet preprocessing and feature independence
    ├── test_splitting.py          # Flow/capture splitting, disjointness, manifests
    └── test_stopping_rule.py      # Threshold sweep, sequential FPR, first-crossing logic
```

---

## 5. Execution Guide (Windows PowerShell)

### Step 1: Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Full Automated Test Suite
```powershell
python -m pytest tests/ -v
```
*Expected: 113 passed tests.*

### Step 3: Run End-to-End Synthetic Smoke Test
```powershell
python main.py dry-run --model both --overwrite
```
*Executes Random Forest (Experiment A) and CNN-GRU (Experiments B/C) on synthetic flows. Results are written to `data/results/synthetic_smoke_test/`.*

### Step 4: Real Dataset Execution (When PCAPs are Available)
1. Download UFU-DoH-EXF PCAPs and place them into `data/raw/`.
2. Inspect the raw capture files:
   ```powershell
   python main.py inspect --raw-dir data/raw
   ```
3. Extract bidirectional flows:
   ```powershell
   python main.py build --raw-dir data/raw
   ```
4. Run full experimental pipeline:
   ```powershell
   python main.py run --name exp_ufu_doh --alpha 0.01 --model both --group-by source_file --overwrite
   ```

---

## 6. Generated Artifacts & Manifests

Every experiment run generates:
- **`split_manifest.json`:** Audit log of partitions, class counts, capture files, and verified 0 group overlap.
- **`train_fit_cal_split.csv`:** Record of training flows assigned to model fitting vs calibration.
- **`cnn_gru_best.pt`:** Trained model weights and fitted `SequenceNormalizer`.
- **`thresholds.json` & `cnn_gru_thresholds.json`:** Frozen $\tau$ values, validation FPR, and resolution checks.
- **`tables/`:** 24 CSV tables including `dataset_stats.csv`, `fixed_vs_adaptive_test.csv`, `robustness_test.csv`.
- **`figures/`:** 19 figures including ROC curves, observation cost bars, DEBD comparisons, and robustness plots.
- **`summary.md`:** Consolidated markdown report of measured values.