# Adaptive Stopping for Early Detection of Evasive DoH Tunneling

Information Security mini-project — Anusuya M (240953480), Vemba Kumar R (240953502), Tailam Praneeth (240953532);
guide: Sudhina Kumar GK; School of Computer Engineering, MIT Manipal.

This code answers one question **by measurement, not by assumption**: can adaptive per-flow stopping reduce
detection latency, observation cost and pre-alert exposure at an acceptable FPR, compared with a fixed-window detector?
A negative result is a valid outcome.

## Honest status (read this first)

| Item | Status |
|---|---|
| Unit + integration tests (74, `unittest`) | **Run and passing** on synthetic data and on synthetic PCAP files written by the tests |
| Full pipeline on SYNTHETIC flows (`dry-run`) | **Run successfully** — proves the code runs; says nothing about DoH |
| PCAP → inspect → build → pipeline on SYNTHETIC PCAPs | **Run successfully** (`tests/test_pcap_end_to_end.py`) |
| **Real CIRA-CIC-DoHBrw-2020 data** | **NOT run.** Nobody has executed this code on the real dataset yet. Expect to debug dataset-specific details (labels, folder names, DoH vs non-DoH traffic) |
| Performance on very large PCAPs | Parser speed measured at ≈150k packets/s on synthetic input; real-file memory/time not tested |

No experimental result about DoH exists in this repository. `data/results/` ships empty.

## A. Directory structure

```
adaptive-doh-detection/
├── README.md  requirements.txt  config.py  main.py
├── data/{raw,processed,results}/
├── src/
│   ├── inspect_dataset.py     look at the raw files before assuming anything
│   ├── preprocessing.py       PCAP -> bidirectional flows (timestamp, IP length, direction); labels from path rules
│   ├── feature_extraction.py  causal prefix features (documented in the file header)
│   ├── prefix_generation.py   one row per (flow, checkpoint); no padding
│   ├── dataset_split.py       60/20/20 split at FLOW (or source-file) level
│   ├── train_baseline.py      shared Random Forest + per-flow weighting
│   ├── calibration.py         flow-grouped out-of-fold Platt scaling (train only)
│   ├── adaptive_detector.py   stopping policy, fixed-window policy, FPR-constrained threshold selection
│   ├── evasion.py             offline low-and-slow / timing-jitter perturbation
│   ├── evaluation.py          all metric definitions (precision, recall, F1, FPR, coverage, cost, latency, DEBD, CIs)
│   ├── plots.py               the 7 required figures
│   ├── pipeline.py            Experiments 1-6 in the leakage-safe order
│   └── synthetic.py           synthetic data + tiny pcap writer (dry-run and tests only)
└── tests/                     74 tests (unittest)
```

## C. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, pandas, scikit-learn, scipy, matplotlib
```
Developed and tested here with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18, matplotlib 3.11.
Scapy is **not** needed (a small classic-pcap parser is included). `pytest` is optional; the tests use `unittest`.

## D. Dataset placement and configuration

**What the official page states** (https://www.unb.ca/cic/datasets/dohbrw-2020.html, checked while writing this):
CSV files generated with DoHMeter and raw PCAPs both exist; labels are Non-DoH, Benign-DoH (Chrome/Firefox) and
Malicious-DoH (dns2tcp, DNSCat2, Iodine); the CSVs are "labeled flow-wise based on the IP addresses of the servers";
traffic was captured between a DoH proxy and the DoH servers. The page does **not** list the PCAPs as a download or give a
folder structure, and gives no flow-timeout value.

**What that means for this project (the critical feasibility rule):**
the CSV files are completed-flow statistics. They **cannot** be used to build packet prefixes, and full-flow statistics
must not be passed off as "first 10 packets". Genuine prefixes need the **PCAPs**. If you only have CSVs, `inspect` will
say so and `build` will stop; obtain the PCAPs from the dataset authors/mirror (not verified here which download contains them).
Do not train this pipeline on the CSVs.

Steps:
1. Put the PCAP files anywhere under `data/raw/` (or set `DOH_RAW_DIR` / `--raw-dir`). Classic `.pcap` only; for pcapng run
   `editcap -F pcap in.pcapng out.pcap`.
2. `python main.py inspect` — prints folders, pcap headers, CSV columns, label-like columns and a verdict.
3. Edit `config.py`:
   * `pcap_label_rules` — **required, deliberately empty**. Example once you have seen your folder names:
     `{"benign": "benign", "malicious": "malicious", "non-doh": "exclude"}` (keyword in the relative path → label; first match wins; unmatched files are skipped and listed).
   * `doh_server_ips` — **strongly recommended**: the resolver IPs (AdGuard, Cloudflare, Google, Quad9 per the page). Without it,
     every TCP/443 flow is kept, which may put non-DoH HTTPS flows into the benign class and contaminate it.
   * `flow_idle_timeout_s` (default 120 s, *our* choice, not the dataset's), `drop_zero_payload_packets` (default False).
4. Decide `group_by` (`flow` default; `source_file` is stricter — run both as a sensitivity check).

## E. Commands

```bash
python -m unittest discover -s tests -t . -v         # 74 correctness tests (≈15 s)
python main.py dry-run                                # SYNTHETIC end-to-end check; results in data/results/dry_run_synthetic
python main.py inspect                                # real data: inspect first
python main.py build                                  # PCAP -> data/processed/flows.pkl (+ flows_report.json)
python main.py run --name exp_alpha1 --alpha 0.01     # Experiments 1-6
python main.py all --name exp_alpha1                  # build + run
python main.py build --max-packets-per-file 200000    # quick smoke test on big files
```
`run` refuses to overwrite an existing experiment folder unless `--overwrite` is given.

## F. Expected outputs (`data/results/<name>/`) and how to verify

| File | Check |
|---|---|
| `thresholds.json` | written **before** test evaluation; τ per policy (adaptive, fixed_10/20/30/40), validation FP/FPR, `resolution_warning` |
| `tables/dataset_stats.csv` | flows able to supply ≥10/20/30/40 packets, per label and partition — check class balance and how many flows are too short |
| `tables/prefix_level_test.csv` | Experiment 2 (precision/recall/F1/FPR/AUC/Brier/ECE per prefix length) |
| `tables/fixed_vs_adaptive_test.csv` (+`_validation`) | Experiments 3–5; `fpr_constraint_validation_vs_test.csv` has val/test FP counts, FPR, Clopper-Pearson bounds |
| `tables/robustness_test.csv` | Experiment 6 |
| `tables/calibration_validation.csv`, `threshold_sweep_val_*.csv` | diagnostics / plot data |
| `figures/*.png` | prf_vs_prefix_length, val_fpr_vs_threshold, observation_cost, detection_latency, debd_comparison, detection_coverage, robustness |
| `predictions_{val,test}.csv`, `split_assignment.csv` | per-flow probabilities/alerts; which partition each flow is in |
| `run_record.json` | config, seed, versions, features, model, calibration (+fallbacks), frozen thresholds |
| `summary.md` | all tables, measured values only |

Sanity checks to do on real data: (1) `dataset_stats` counts look plausible vs the dataset paper; (2) `run_record.json →
robustness_normal_matches_main_test` is `true`; (3) validation FPR ≤ α for every policy **or** `never_alerts`/`val_satisfies_alpha` tells you it was infeasible;
(4) prefix-level AUC should rise with prefix length — if AUC at 10 packets is already ≈1.0, suspect a leak (e.g. benign/malicious captured from different servers or ports, so that endpoint metadata is implicitly encoded in packet sizes) and investigate before trusting any result.

## Method decisions (and what they change)

* **Flow key / direction / length** — sorted TCP 5-tuple; +1 = direction of first packet seen; length = IP total length; retransmissions (end-seq ≤ highest seen) dropped; idle-timeout and SYN-after-FIN start new flows. Only TCP (DoH over HTTP/3 is not parsed).
* **Features** — 18 causal features from packets 1..t only (sizes, directions, per-direction counts/bytes, IAT statistics, prefix duration). Absolute timestamps and flow totals are never features. Tested by scrambling post-t packets.
* **Eligibility** — Experiments 3–6 use flows with ≥ 40 packets (the largest checkpoint) so both policies see the same flows. Flows shorter than 40 packets are **excluded from the comparison**; this is a selection effect, reported in `dataset_stats.csv`. Training also uses the shorter prefixes of shorter flows (≥10).
* **Shared RF weighting** — rows from one flow are correlated and long flows contribute more rows. Each training flow gets equal total weight (`flow_weighting=True`); `rf_class_weight="balanced_subsample"` handles class imbalance.
* **Calibration** — Platt scaling on **out-of-fold, flow-grouped** training predictions, per checkpoint. *Methodological safeguard found during testing:* with weak signal at short prefixes the fitted slope can become negative and **invert the evidence**; maps are therefore constrained to be non-decreasing (fallback: shared map → identity), recorded in `run_record.json`. A monotone map does not change the ranking at one checkpoint; it changes the meaning of τ and, with per-checkpoint maps, the cross-checkpoint trade-off.
* **Threshold** — grid step 0.001; for each candidate the *complete sequential policy* is run on benign validation flows (a flow is an FP if it alerts at any checkpoint); the lowest feasible τ is chosen (lower τ ⇒ more alerts, earlier stops, so it dominates other feasible τ). Fixed-window detectors get their own τ under the same α. Frozen before the test set is touched. With few benign validation flows α=1 % cannot be resolved (`resolution_warning`); zero observed FPR is reported with an upper confidence bound, never as a guarantee. Choosing the lowest feasible τ sits right at the validation noise edge, so **test FPR can exceed α** — that is a measured outcome, report it.
* **At equal τ the adaptive policy alerts on a superset of what fixed-40 alerts on**, so its validation FPR is ≥ fixed-40's; the adaptive τ is therefore ≥ the fixed-40 τ. Whether earlier stopping pays for that is exactly what the experiments measure.
* **Metrics** — see `src/evaluation.py` header. Coverage equals recall at flow level (undetected malicious = FN). Non-detected flows are observed up to Tmax (cost 1). Latency exists only for detected flows. All-flow cost is dominated by benign flows that never alert; use the malicious-flow cost for the early-warning claim.
* **DEBD** — a *proxy* for pre-alert malicious-traffic exposure, not stolen data: IP-level bytes, both directions, including the alert packet, retransmissions removed, denominator = all counted bytes of the flow in the capture, undetected malicious = 100 %. Reported as per-flow mean (undetected = 100 %), detected-only mean (biased toward easy flows; read next to coverage) and byte-weighted. Truncated captures overstate DEBD.
* **Robustness** — offline timestamp perturbation of the **malicious test flows only** (benign flows untouched, so FPs at the frozen τ cannot change): low-and-slow = all IATs × factor; jitter = IAT × lognormal(σ) + U(0, a) with per-flow seeded RNG; every feature recomputed from the new timestamps; packet order, sizes and labels unchanged. This is a **simulation** — not real attack traffic, not proof of robustness, and models only timing changes. The model is never trained on perturbed flows. Time dilation also inflates absolute latency by construction.
* **Related work** — this repository makes no novelty claim for DoH detection, Random Forests or early stopping. The synopsis cites WEDoHTool (early identification of DoH tunnel-tool traffic) as the closest prior direction; verify every reference yourselves before the report, and describe the combination as the *intended* contribution.

## G. Known limitations / open issues

* Never run on the real dataset; label rules, resolver IPs and directory layout are yours to supply and verify.
* Whether the benign PCAPs contain only DoH is unverified; contaminating the benign class with non-DoH HTTPS flows would invalidate results.
* The dataset's flows are generated by controlled tools and servers — good detection may reflect tool/server artefacts rather than behaviour (consider `group_by=source_file`).
* The flow-level split does not remove correlation between flows from the same session/host.
* Packet lengths are IP-level; an out-of-order (non-retransmitted) TCP segment may be mistaken for a retransmission and dropped.
* Latency, DEBD and cost depend on the 10/20/30/40 checkpoint grid; finer checkpoints could change conclusions.
* Single split and single seed: no variance estimate yet. For the paper, repeat over several seeds and report spread.
* Threshold grid and RF settings are fixed defaults; no hyper-parameter tuning was done (and none on test).
* Report/paper text is not generated here: write it after real results exist.
#   A d a p t i v e - d o H - d e t e c t i o n  
 