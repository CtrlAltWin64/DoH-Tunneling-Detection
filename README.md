# Adaptive Stopping for Early Detection of DoH Tunneling

Information Security research project, Department of Computer Science & Engineering,
MIT Manipal. Guided by Sudhina Kumar GK.

**Research question:** Can adaptive per-flow stopping detect DoH tunneling earlier
than fixed-window detection while maintaining an approximately 1% false-positive rate?

## Current research status

Real-capture experiments completed on 19 October 2026. These are **preliminary
within-capture measurements**, not evidence of unseen-capture generalization or
resistance to evasive traffic.

Seven captures directly under `data/` produced **140,193 flows: 2,095 benign and
138,098 malicious**. RF and CNN-GRU were evaluated at 10 and 20 packets with the
existing first-crossing adaptive policy. The common test cohort contains **4,137
malicious and 275 benign flows** reaching 20 packets.

| Model | Policy | TP | FP | TN | FN | Mean packets |
|---|---|---:|---:|---:|---:|---:|
| RF | Adaptive | 4,137 | 3 | 272 | 0 | 10.617 |
| RF | Fixed 10 | 4,137 | 3 | 272 | 0 | 10 |
| RF | Fixed 20 | 4,137 | 3 | 272 | 0 | 20 |
| CNN-GRU | Adaptive | 4,137 | 0 | 275 | 0 | 10.623 |
| CNN-GRU | Fixed 10 | 4,137 | 0 | 275 | 0 | 10 |
| CNN-GRU | Fixed 20 | 4,137 | 0 | 275 | 0 | 20 |

Adaptive stopping saved approximately 47% of observations versus fixed 20, but
showed **no measured benefit over fixed 10**. RF adaptive and fixed 20 flag different
benign flows despite equal aggregate counts. RF FPR was 3/275 = **1.09%**. CNN's
zero false positives gives a one-sided 95% exact upper FPR bound of **1.083%**,
assuming independent flows.

Limitations:

- All seven captures occur across train, validation and test; only flows are disjoint.
- Only about 15% of malicious flows reach 20 packets; the common comparison excludes the rest.
- No malicious flow reaches 30 or 40 packets, making the default four-checkpoint comparison unsuitable here.
- Filename/scenario labels and TCP/443 filtering do not independently authenticate DoH.
- No labeled genuine slow/evasive traffic evaluation was possible.
- Capture dependence, possible session fingerprints and parser limitations remain.

## Research outputs

- [Results and interpretation](reports/final_experiment_results_20261019/results_summary.md)
- [Artifact index](reports/final_experiment_results_20261019/artifact_index.md)
- [Full performance CSV](reports/final_experiment_results_20261019/tables/main_performance.csv)
- [PNG and editable SVG figures](reports/final_experiment_results_20261019/figures/index.md)
- [Methodology and limitations](reports/final_experiment_results_20261019/methodology_and_limitations.md)
- [Reproduction commands](reports/final_experiment_results_20261019/reproduction.md)
- [Dataset preparation](DATASET_AND_PREPARATION.md)
- [Git packaging and local artifacts](GIT_READINESS.md)

Verification recorded for this run: **113 tests**, **160 report checks**, **16 model
reload/timing checks**, and **100 saved synthetic checks** passed. Synthetic checks
are software evidence only. Historical real-run artifacts described by the earlier
audit were unavailable; these findings come from fresh extraction and training.

## Setup

The observed experiment environment used Python 3.13.12 on Windows and CPU PyTorch.
Exact library versions are in the saved [environment record](reports/final_experiment_results_20261019/metrics/environment.json).
Requirements specify broad minimums, not a historical environment lock.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

For a synthetic check, choose a new run name each time: the dry-run CLI overwrites
same-named synthetic outputs.

```powershell
.\.venv\Scripts\python.exe main.py dry-run --model both --name smoke_new
```

## Reproduce the real study

Captures are intentionally excluded from Git. Place the seven files listed in
[`data/README.md`](data/README.md) directly in `data/`, verify their hashes, and follow
[the fresh-run commands](reports/final_experiment_results_20261019/reproduction.md).
The study harness uses nonrecursive capture discovery, explicit scenario labels,
the supported `(10,20)` schedule and an explicitly preliminary flow split.

The generic CLI offers `inspect`, `build`, `run`, `all`, and `dry-run`:

```powershell
.\.venv\Scripts\python.exe main.py inspect --raw-dir data
```

Generic `build` intentionally requires explicit `pcap_label_rules` in `Config`.
The incoming code now defaults to `(5,10,15,20)`; it is **not** the configuration
used for the reported `(10,20)` study. Use the recorded source snapshot and study
harness for exact reproduction, rather than assuming the updated source is identical.

A clean clone includes aggregate reports, tables and figures, but not processed
flows, per-flow predictions or fitted models. Regenerating reports from scores
requires restoring the local bundle or running extraction/training first.

## Implementation

- `src/preprocessing.py`: packet parsing and bidirectional TCP flows.
- `src/feature_extraction.py`, `src/prefix_generation.py`: causal prefix features.
- `src/train_baseline.py`, `src/calibration.py`: RF and training-only calibration.
- `src/cnn_gru_model.py`, `src/cnn_gru_calibration.py`: CNN-GRU and held-out calibration.
- `src/dataset_split.py`: grouped splits and manifests.
- `src/adaptive_detector.py`: validation-tuned first-crossing decisions.
- `src/evaluation.py`: confusion counts, cost, latency and exact FPR intervals.
- `reports/final_experiment_results_20261019/`: study harness, source snapshot and evidence.

The study harness separates CNN fitting, early stopping, calibration, threshold
validation and test roles. The incoming generic pipeline also separates early
stopping inside training. The recorded study predates these incoming source changes;
its source snapshot preserves the implementation actually evaluated.

## Next research steps

Obtain independent capture sessions and provenance, prespecify a policy for short
flows, evaluate genuinely slower attacks, and compare with the earliest fixed-window
baseline under comparable FPR constraints. Publication claims require stronger
independent evaluation and a careful novelty comparison with prior work.

## Incoming work and experiment provenance

The merged upstream code adds model persistence, parser changes, a `(5,10,15,20)`
default schedule and a three-fold fallback for capture splitting when fewer than
five groups per class exist. The five-group restriction documented in the recorded
study refers to its historical source snapshot, not the updated splitter.

The [incoming README](reports/upstream_readme_fc46862.md) reports a separate
capture-isolated `paper_results_v2` experiment. Its raw predictions, split manifest
and model artifacts are absent locally, so those reported figures are not combined
with or treated as independent verification of the study summarized above.

The incoming frontend and Graphify artifacts are preserved. They do not change the
provenance of the saved research measurements. The 113-test count above belongs to
the recorded study; the merged source has additional tests.

For exact historical reproduction, use `source_snapshot/config.py` and
`source_snapshot/src/` in an isolated workspace as described in the reproduction
guide. Fresh runs against the merged implementation are new experiments.
