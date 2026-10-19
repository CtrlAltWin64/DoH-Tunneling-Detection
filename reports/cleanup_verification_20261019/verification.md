# Final cleanup verification — 19 October 2026

Work was performed directly in the existing checkout, initially clean on `main`
at `c38aac2` (origin: `https://github.com/CtrlAltWin64/DoH-Tunneling-Detection.git`).
The latest preceding commits were `9ed2bd8`, `fc46862`, `3e3b535` and `8a2881a`.
No rebase, merge, cherry-pick or sequencer operation was active. A stale
`REBASE_HEAD` remained; Git metadata was left untouched. No staging, commits,
pushes, resets, history rewrites or branch operations were performed.

## Changes

- Rewrote the root [README](../../README.md) and added the detailed
  [results guide](../../RESULTS_GUIDE.md), separating actual local five-capture
  runs from the archived seven-capture report and synthetic evidence.
- Updated dataset, reproduction, packaging and frontend documentation. Added
  current-status notes to historical reports without rewriting their findings.
- Kept established source, data, experiment and model paths. Models remain with
  their runs. Preserved 72 tracked clutter/cache files in ignored
  `reports/local_archive/`, checked against Git content. See `archive_inventory.json`.
- Added CLI preflight checks before preprocessing, default processed-data overwrite
  refusal, configuration validation and concise CLI error messages. The RF now
  honors configured `rf_class_weight`; the default behavior is unchanged.
- Added running/failed/complete status records and version-1 finite JSON manifests
  containing metrics, stopping counts, provenance and relative artifact paths.
  The frontend remains disconnected and visibly synthetic.
- CNN-only runs now reject single-class evaluation cohorts, matching RF behavior.
  Robustness plots use the actual terminal checkpoint instead of hardcoded 40.
- Added a read-only aggregate-report checker. The older saved-evidence checker
  now supports fresh output paths, refuses overwrites and targets only its four
  historical runs.

## Commands executed and outcomes

Commands below ran from the repository root unless noted. CPU thread environment
variables `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` were used for test/smoke runs.

| Exact command / operation | Observed outcome |
|---|---|
| `python -m venv --system-site-packages .venv` | Initial ensurepip failed on Windows sandbox temp permissions. Recovery completed a local environment with `include-system-site-packages=false`. |
| `.\.venv\Scripts\python.exe -m ensurepip --upgrade` followed by `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` | Successful outside sandbox; installed declared dependencies in the local venv. |
| `.\.venv\Scripts\python.exe -m pip check` | No broken requirements. |
| `python -m unittest discover -s tests -t . -q` | Initial sandbox run failed on temporary-file permissions; authorized retry passed **130 tests in 39.858 s**. |
| `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider` | Final fresh-environment suite: **130 passed, 1 warning in 87.08 s**. Warning explicitly reports inner capture-to-flow split fallback. |
| `python main.py dry-run --model both --checkpoints 10 20 --cnn-epochs 2 --cnn-device cpu --results-dir reports/cleanup_verification_20261019/smoke --name both_models` | Completed RF + two-epoch CNN synthetic run; complete status, models, calibrated predictions, CSVs, plots and strict JSON emitted. **Software check only.** |
| `.\.venv\Scripts\python.exe main.py inspect --raw-dir data/raw --processed-dir reports/cleanup_verification_20261019/inspection` | All five captures sampled successfully, 2,000 packets each; inspection JSON saved separately. |
| `.\.venv\Scripts\python.exe main.py build --raw-dir data/raw --processed-dir reports/cleanup_verification_20261019/scratch/bounded_build --max-packets-per-file 2000` | **10,000 packets read, 355 flows**, five packet-limit counters; 2,858 non-443, five malformed and eight UDP packets skipped. Bounded real parser check only, not a real training/evaluation experiment. |
| `python reports/phase10_verification/verify_saved_evidence.py --output reports/cleanup_verification_20261019/saved_evidence.json` | **144,113 / 144,113 checks passed**: saved real scores, thresholds, normal metrics and flow metadata; synthetic counts; robustness aggregate arithmetic; image integrity. No fitting. |
| `python scripts/verify_report_tables.py --output reports/cleanup_verification_20261019/report_tables.json` | **480 / 480 checks passed**, 14 PNG/SVG files parsed. Before report-document edits: 94 original hashes matched, one earlier packaging script differed, 23 original large artifacts absent. |
| `.\.venv\Scripts\python.exe scripts/verify_report_tables.py --output reports/cleanup_verification_20261019/report_tables_final.json` | **480 / 480 checks passed** after documentation changes. 91 original hashes match; three updated documents and previously changed `verify_inputs.py` differ; 23 large artifacts absent. |
| `npm.cmd ci --ignore-scripts --no-audit --no-fund` in `frontend/` | 72 packages installed using the unchanged lockfile. |
| `npm.cmd run build` in `frontend/` | Initial sandbox `spawn EPERM`; retry passed TypeScript + Vite build, 1,577 modules transformed. No frontend integration. |
| `.\.venv\Scripts\python.exe main.py --help` and `main.py all --help` | Documentation flags and command names confirmed. |
| Actual CLI refusal checks, recorded in `cli_guards.json` | Five expected exit-2 cases: missing `all --name`, existing processed dataset, existing smoke run, path-valued name, zero packet limit. |
| Saved-model inference via `load_rf_bundle`, `load_cnn_bundle` | **Eight complete synthetic score replays passed**, validation/test × RF/CNN × checkpoints 10/20, 216 flows each. Max difference < 7e-16. No retraining. |
| `load_model` on both historical `cnn_gru_best.pt` files | Raw checkpoints loaded and produced finite sample scores. **Does not replace missing historical calibration.** |
| RF configuration override check | `Config(rf_class_weight=None)` reaches the actual estimator. |
| `git diff --check` | Passed after correcting line endings only in edited text. |

Saved-model checks are in `model_checks.json`; the interpreter/package environment
is in `environment.json`. Full local logs are `unittest.log`, `pytest.log`,
`smoke.log`, `inspect.log` and `bounded_build.log`; logs/scratch/smoke binaries are
ignored. Test logs are not research results. Historical phase10 logs were retained.
The initial sandbox failures were environment permissions, not passing checks;
all required retries succeeded. No automatic approval rejection remains unresolved.

The archived FPR-interval figure was visually inspected against its saved counts,
and the new synthetic robustness figure visibly shows both `adaptive` and
`fixed_20`. Historical figures were not regenerated. JSON manifest/count/path
consistency and failed-run handling are covered by the new integration regressions.

## Preservation and path checks

`preserved_artifacts.json` records 247 original capture, processed, prediction,
model, table, figure, metadata and graph artifacts. Final byte hashes agree for
all 247. No original capture/model/result was moved or overwritten. Historical
source snapshots and scientific numeric outputs are unchanged.

`path_and_preservation_checks.json` records verified documentation links, expanded
shared result filenames, expected absences and preservation results. New Git-visible
files were reviewed: documentation, source/test changes and compact verification
JSON only. Captures, environments, model binaries, frontend dependencies, temporary
logs and local archives remain ignored. The 72 tracked deletions all have preserved
local counterparts; no useful report evidence was silently discarded.

## Remaining limitations

1. File02/File03 are missing. The seven-capture report's processed, prediction,
   trace and fitted-model directories are missing (23 historical artifacts).
   Aggregate arithmetic is verifiable; complete historical score/model replay is not.
2. The original five-capture runs lack RF trees and CNN calibrators. Their saved
   scores can be checked, but their complete fitted pipelines cannot be reconstructed.
3. Two benign local capture groups fail the current capture-isolated outer split.
   All existing real studies share capture sessions across partitions. No unseen-
   capture performance or sub-1% population FPR claim is established.
4. Historical perturbed per-flow scores are missing; only aggregate robustness
   arithmetic was checked. Timestamp dilation is simulation, not genuine evasion.
5. Parser high-water retransmission filtering, short-flow exclusions, scenario
   labels and possible session fingerprints remain research limitations. Full raw
   capture re-extraction and expensive real-data training were not repeated.
6. Exact historical harness commands were checked against code, not executed
   without their missing inputs. Graphify maps remain historical development aids.

## Final logical layout

```text
README.md / RESULTS_GUIDE.md / DATASET_AND_PREPARATION.md / GIT_READINESS.md
config.py / main.py / requirements*.txt
src/                         research implementation and result export
tests/                       software regression and integration tests
scripts/                     aggregate-report verification
data/raw, processed, results/ local dataset and experiments
reports/final_experiment_results_20261019/  archived preliminary report
reports/phase10_verification/              historical audit evidence
reports/cleanup_verification_20261019/      this verification record
reports/local_archive/                     ignored preserved clutter
graphify-out/                generated development graphs and snapshots
frontend/                    independent synthetic preview
```
