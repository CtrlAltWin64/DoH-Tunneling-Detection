# Adaptive Stopping for Early Detection of DoH Tunneling

This project asks whether a detector can identify malicious DNS-over-HTTPS traffic
from ordered packet prefixes while observing fewer packets than a fixed window,
under a validation-selected false-positive-rate (FPR) target.

The pipeline extracts bidirectional TCP/443 flows, constructs causal features,
trains Random Forest (RF) and CNN-GRU models, calibrates scores, selects thresholds
on validation flows, and evaluates frozen first-crossing and fixed-window policies
on test flows. Timestamp perturbations are **offline simulations**, not measured
evasive attacks.

**Start with [RESULTS_GUIDE.md](RESULTS_GUIDE.md) to collect report material.**
It separates the seven-capture archived report from two locally replayable
five-capture studies. All are preliminary within-capture evaluations, without
evidence of unseen-session generalization or population FPR below 1%.

## Layout

```text
README.md / RESULTS_GUIDE.md   Setup / exact results navigation and reproduction
DATASET_AND_PREPARATION.md     Dataset conventions and historical protocol
config.py / main.py            Configuration / CLI
src/                          Research code, persistence, JSON result export
tests/                        Unit, regression and synthetic integration tests
scripts/                      Aggregate-report verification
data/raw/                     Local captures (ignored)
data/processed/                Local flows and inspection reports (ignored)
data/results/                  Local experiments and their models (ignored)
reports/final_experiment_results_20261019/  Archived report and source snapshot
reports/phase10_verification/  Historical software audit evidence
reports/cleanup_verification_20261019/      Current verification evidence
reports/local_archive/         Preserved local clutter (ignored)
graphify-out/                  Generated development maps, not research results
frontend/                     Separate React/TypeScript mock UI
```

Model bundles stay with their experiment. A clean clone contains aggregate reports,
but not captures, processed flows, prediction scores or fitted models.

## Install and test (PowerShell, from the repository root)

The source requires Python 3.10+ syntax; choose a version with compatible dependency
wheels. Cleanup was checked on Windows CPU with Python 3.14.2; the archived
seven-capture study used Python 3.13.12. Recorded environments accompany both.
Requirements are minimum versions, not an exact historical lock.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

`requirements.txt` supplies NumPy, pandas, scikit-learn, SciPy, Matplotlib and
PyTorch (currently imported even for RF-only runs). Development requirements add
pytest and Pillow. The suite also supports
`python -m unittest discover -s tests -t . -q`.
No GPU, external packet parser, Node.js, Graphify, SHAP or Captum is required.

## Synthetic pipeline check

```powershell
$run = 'smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
.\.venv\Scripts\python.exe main.py dry-run --name $run --model both --checkpoints 10 20 --cnn-epochs 2 --cnn-device cpu
```

This trains small models on generated flows and checks evaluation, persistence and
plotting. **Its metrics are software evidence only.** Results go to
`data/results/<run>/`; existing nonempty directories are refused by default.

## Inspect, build and train on captures

Review [data/README.md](data/README.md) for the current five-capture layout.
The CLI applies UFU filename presets (`benign`, `exfil`, `malicious`) when
`Config.pcap_label_rules` is empty. Review the ordered rules and inspection before
building: filename labels are not independently verified ground truth. Custom
rules and settings not exposed by the CLI are configured in `config.py`.

```powershell
$run = 'exploratory_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
$processed = "data/processed/$run"
.\.venv\Scripts\python.exe main.py inspect --raw-dir data/raw --processed-dir $processed
.\.venv\Scripts\python.exe main.py build --raw-dir data/raw --processed-dir $processed
if ($LASTEXITCODE -ne 0) { throw 'Preprocessing failed' }
.\.venv\Scripts\python.exe main.py run --processed-dir $processed --name $run --model both --checkpoints 10 20 --group-by flow --seed 42 --cnn-device cpu
```

`inspect` samples headers/packets and saves inspection JSON; it does not establish
full-file validity. `build` recursively parses captures into `flows.pkl` and
`flows_report.json`, refusing existing flow files unless `--overwrite` is explicit.
`run` **trains new models**. `all` combines build + run with the same flags and
requires `--name`. Full real-data training was not rerun during cleanup.

To reuse the existing five-capture processed dataset, skip preprocessing and pass
`--processed-dir data/processed` with a fresh run name. That is a new experiment
against current code, not exact reproduction of a historical run.
`--raw-dir` defaults to repository `data/raw` (or `DOH_RAW_DIR`, if set).
Processed/results defaults are repository-rooted; explicit relative arguments use
the working directory. `--results-dir` selects another result parent.

The current schedule defaults to `(5,10,15,20)`; the commands explicitly use
`(10,20)`. Direct prefix helpers retain a legacy `(10,20,30,40)` default, so pass
checkpoints explicitly. Saved datasets have no malicious flows reaching 30/40.

## Outputs and model reuse

New runs write `run_status.json`, `run_record.json`, `experiment.json`, split
assignments/manifests, thresholds, prediction CSVs, metric tables, PNG figures,
`summary.md` and the selected model bundles. Only `status: complete` in
`run_status.json` marks success. Failed/running directories contain diagnostic
partial outputs. Prefer a new run name to `--overwrite`.

Saved-model inference uses Python APIs without fitting:
`src.train_baseline.load_rf_bundle` loads `rf_bundle.pkl`, and
`src.model_bundle.load_cnn_bundle` loads `cnn_gru_bundle.pt`.
[RESULTS_GUIDE.md](RESULTS_GUIDE.md#loading-new-model-bundles) gives an executable
example and saved-score verification commands. Only load trusted project pickles
and checkpoints. Historical `cnn_gru_best.pt` files lack a complete calibrator
bundle; raw checkpoint loading is not calibrated prediction replay.

## Evaluation limits and frontend boundary

All prefixes of a flow stay together. `--group-by source_file` additionally holds
out captures and requires at least three groups per class and both classes in
each partition. The local dataset has only two benign capture groups, so this
mode is blocked here. Inner CNN role splits can fall back to flows within outer
training captures; inspect `cnn_training_split.json`.

Main policy comparisons require every checkpoint and exclude short flows.
FPR counts false-alert benign **flows**, not packets/prefix rows. Zero false
positives in a small correlated sample does not establish sub-1% population FPR.
TCP/443 and scenario filenames do not authenticate DoH. See
[dataset limitations](DATASET_AND_PREPARATION.md).

The [frontend](frontend/README.md) has its own Node build and visibly synthetic
preview data. Its future offline file contract is documented there. It remains
disconnected from Python; no API, live training or deployment is implemented.
