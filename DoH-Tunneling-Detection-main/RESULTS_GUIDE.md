# Results collection and reproduction guide

Inventory verified on **19 October 2026** in this checkout. Paths below are relative
to the repository root. **Do not merge the two five-capture studies with the
seven-capture report.** They have different populations, source versions and model
artifacts. No experiment is designated definitive merely because its folder is
named `final` or has a newer date.

## 1. Choose the evidence set

| Existing directory | Contents and report use | Caveats / verification now |
|---|---|---|
| [reports/final_experiment_results_20261019/](reports/final_experiment_results_20261019/) | Seven-capture `(10,20)` study: publication-friendly CSV/Markdown tables, PNG/SVG figures, methodology, configuration and frozen source. Best organized package for an explicitly preliminary report. | Aggregate arithmetic and image/file integrity checked now. Its processed data, predictions, traces and fitted models are **absent**. Historical 160 report / 16 model-replay checks are retained records, not newly executed replay. |
| [data/results/exp_ufu_real_cp10_20/](data/results/exp_ufu_real_cp10_20/) | Completed five-capture `(10,20)` study, seed 42; RF/CNN saved scores, tables, plots and CNN checkpoint. Common test cohort: 4,132 malicious + 131 benign flows. | Normal validation/test metrics and threshold sweeps independently replayed from local scores and flow metadata. No complete historical RF/CNN inference bundle. Ignored/local, absent from a clean clone. |
| [data/results/exp_ufu_real_cp2_4_8_10_15/](data/results/exp_ufu_real_cp2_4_8_10_15/) | Separate five-capture `(2,4,8,10,15)` study, seed 42. Common test cohort: 27,620 malicious + 127 benign flows. Useful for checkpoint sensitivity with its own exclusions. | Same replay scope and persistence limits. Do not compare raw costs across horizons without noting Tmax=15 versus 20. Ignored/local. |
| [data/results/exp_ufu_real_20261018/](data/results/exp_ufu_real_20261018/) | Split assignments/manifest and `tables/dataset_stats.csv`. | **Incomplete**: no run record, predictions or model results. Not a completed experimental finding. |
| [data/results/member3_smoke_20261017/](data/results/member3_smoke_20261017/) | Historical synthetic scores, tables and plots. | `run_record.json` says `synthetic: true`. Software evidence only; synthetic flow timing/bytes cannot be replayed from the real processed dataset. |
| [reports/cleanup_verification_20261019/](reports/cleanup_verification_20261019/) | This cleanup's independent replay, aggregate checks, preservation inventory and verification record. | Software/audit evidence. Its `smoke/` directory contains new synthetic checks, not real research findings. |
| [reports/phase10_verification/](reports/phase10_verification/) | Earlier environment/checkpoint diagnostics, successful and failed test logs, independent saved-evidence checker. | Historical logs describe their own source/environment. Some failed because of Windows temporary-directory permissions. |
| [graphify-out/](graphify-out/) | Current and dated development graphs, HTML, JSON and architecture reports. | Generated navigation evidence, not classifier metrics. Maps predate cleanup source edits and may be stale. |

All three real-data evidence sets use **flow grouping**. All source captures overlap
train/validation/test in each study. There is no locally verified unseen-capture
generalization result. The old root README referenced `paper_results_v2` and an
`upstream_readme_fc46862.md`; neither exists in this checkout. Do not cite their
unavailable results as verified here.

## 2. Seven-capture report: exact files

The directory prefix for every file in this section is
`reports/final_experiment_results_20261019/`. Every linked file exists. CSVs contain
full precision; Markdown is a rounded reading view. Use CSVs as the numeric source,
and distinguish the `population` column in the full main table.

| File | Contents / use in the report | Important caveat |
|---|---|---|
| [results_summary.md](reports/final_experiment_results_20261019/results_summary.md) | Narrative interpretation and headline comparisons. | Historical narrative; current artifact availability is in this guide. |
| [tables/main_performance.csv](reports/final_experiment_results_20261019/tables/main_performance.csv) | Authoritative stored metric table: 10 rows; confusion counts, accuracy, precision, recall, F1, FPR/FNR, exclusions, denominators and exact intervals. | Filter `population == common_ge20` for the six comparable RF/CNN policy rows. Four `native_prefix` rows use checkpoint-specific populations and reuse common-validation thresholds. |
| [tables/main_performance_compact.csv](reports/final_experiment_results_20261019/tables/main_performance_compact.csv) | Six common-cohort comparison rows, suitable for a compact report table. | Derived subset, not a separate experiment. |
| [tables/observation_cost_latency.csv](reports/final_experiment_results_20261019/tables/observation_cost_latency.csv) | Mean/median/p90/p95 packets; cost relative to fixed 20; packet reduction; detected-attack latency. | Packet cost is over all common-cohort flows; latency conditions on detection. |
| [tables/matched_detection_latency.csv](reports/final_experiment_results_20261019/tables/matched_detection_latency.csv) | Paired adaptive/fixed-20 latency for shared true positives. | Selection is limited to attacks detected by both policies. |
| [tables/stopping_distribution.csv](reports/final_experiment_results_20261019/tables/stopping_distribution.csv) | Counts/percentages at 10 and 20, alerts and unresolved flows. | Unresolved terminal observations are not detections. |
| [tables/early_stopping_errors.csv](reports/final_experiment_results_20261019/tables/early_stopping_errors.csv) | Decision disagreements against tuned fixed 20 and same-threshold terminal scoring. | Equal aggregate FP counts can refer to different benign flows. |
| [tables/ranking_metrics.csv](reports/final_experiment_results_20261019/tables/ranking_metrics.csv) | ROC-AUC / average precision at 10 and 20. | Ranking metrics are not operating-point FPR guarantees. |
| [tables/curve_coordinates.csv](reports/final_experiment_results_20261019/tables/curve_coordinates.csv) | Saved ROC/PR coordinates for the matching figure. | Derived coordinates, not individual probabilities. |
| [tables/checkpoint_eligibility.csv](reports/final_experiment_results_20261019/tables/checkpoint_eligibility.csv) | Per-capture/class eligible and excluded counts at 10/20/30/40, including ALL totals. | No malicious flows reach 30/40; do not fabricate those comparisons. |
| [tables/capture_inventory.csv](reports/final_experiment_results_20261019/tables/capture_inventory.csv) | Seven capture filenames, classes, byte sizes, hashes and flow counts. | Hash identity does not establish provenance or independence. |
| [tables/split_class_counts.csv](reports/final_experiment_results_20261019/tables/split_class_counts.csv) | Train/validation/test class counts. | The split is by flow. |
| [tables/capture_partition_counts.csv](reports/final_experiment_results_20261019/tables/capture_partition_counts.csv) | Source-file counts across partitions. | Shows capture overlap directly. |
| [tables/cnn_training_role_counts.csv](reports/final_experiment_results_20261019/tables/cnn_training_role_counts.csv) | Model-fitting, calibration and early-stop role counts. | These are inner training roles, separate from outer threshold validation. |
| [tables/within_capture_performance.csv](reports/final_experiment_results_20261019/tables/within_capture_performance.csv) | Per-capture policy diagnostics. | Per-capture slices of a shared-capture test, not held-out-capture experiments. |
| [tables/first_packet_length_distribution.csv](reports/final_experiment_results_20261019/tables/first_packet_length_distribution.csv) | Potential class/session fingerprint diagnostic. | Distribution differences do not establish what the model learned. |
| [tables/synthetic_saved_verification.csv](reports/final_experiment_results_20261019/tables/synthetic_saved_verification.csv) | Historical 100-check synthetic verification table. | Original `data/results/synthetic_smoke_test` is absent here; not fresh verification or a real result. |
| [metrics/config.json](reports/final_experiment_results_20261019/metrics/config.json) and [metrics/environment.json](reports/final_experiment_results_20261019/metrics/environment.json) | Study configuration, seed, schedule and package versions. | Historical absolute paths describe another workspace. |
| [metrics/split_manifest.json](reports/final_experiment_results_20261019/metrics/split_manifest.json) and [metrics/capture_overlap.json](reports/final_experiment_results_20261019/metrics/capture_overlap.json) | Split summaries and explicit capture overlap. | Flow-disjoint flags do not imply capture independence. |
| [metrics/capture_split_blocker.json](reports/final_experiment_results_20261019/metrics/capture_split_blocker.json) | Historical five-group splitter failure. | Current splitter has a three-group fallback; the local five-capture set still fails with only two benign groups. |
| [metrics/preprocessing.json](reports/final_experiment_results_20261019/metrics/preprocessing.json) and [metrics/pcapng_structure.json](reports/final_experiment_results_20261019/metrics/pcapng_structure.json) | Extraction counters and historical full PCAPNG structure checks. | Not re-extracted during cleanup. |
| [metrics/rf_thresholds.json](reports/final_experiment_results_20261019/metrics/rf_thresholds.json) and [metrics/cnn_thresholds.json](reports/final_experiment_results_20261019/metrics/cnn_thresholds.json) | Frozen policy thresholds. | Validation choices, not test-tuned settings. |
| [metrics/rf_calibration.json](reports/final_experiment_results_20261019/metrics/rf_calibration.json), [metrics/cnn_calibration.json](reports/final_experiment_results_20261019/metrics/cnn_calibration.json), [metrics/cnn_history.json](reports/final_experiment_results_20261019/metrics/cnn_history.json) | Calibration maps and CNN training history. | Insufficient to reconstruct missing RF trees or CNN weights. |
| [methodology_and_limitations.md](reports/final_experiment_results_20261019/methodology_and_limitations.md) and [source_snapshot/](reports/final_experiment_results_20261019/source_snapshot/) | Methodology and actual historical source. | Current merged code is different; frozen source remains unchanged. |
| [metrics/report_verification.json](reports/final_experiment_results_20261019/metrics/report_verification.json), [metrics/model_replay_verification.json](reports/final_experiment_results_20261019/metrics/model_replay_verification.json) | Previously recorded 160 report and 16 model/trace checks. | Retained evidence, not a claim those missing models were loaded now. |
| [metrics/artifact_hashes.json](reports/final_experiment_results_20261019/metrics/artifact_hashes.json), [metrics/git_packaging_hashes.json](reports/final_experiment_results_20261019/metrics/git_packaging_hashes.json) | Original artifact hashes and later packaging hashes. | 23 original data/model/prediction artifacts are missing. Initially 94 existing artifacts matched; after three documentation updates, 91 still match. `verify_inputs.py` changed during earlier packaging. Numeric artifacts and frozen source are unchanged. |

Validation threshold sweeps are the six existing files:
`reports/final_experiment_results_20261019/metrics/rf_adaptive_threshold_sweep.csv`,
`reports/final_experiment_results_20261019/metrics/rf_fixed_10_threshold_sweep.csv`,
`reports/final_experiment_results_20261019/metrics/rf_fixed_20_threshold_sweep.csv`,
`reports/final_experiment_results_20261019/metrics/cnn_adaptive_threshold_sweep.csv`,
`reports/final_experiment_results_20261019/metrics/cnn_fixed_10_threshold_sweep.csv`, and
`reports/final_experiment_results_20261019/metrics/cnn_fixed_20_threshold_sweep.csv`.

### Figures paired with their tables

All seven figures have both PNG and editable SVG versions in the same directory.
Use SVG for vector placement, PNG when required by the report editor. The linked
PNG paths below and their `.svg` counterparts exist.

| Exact PNG path | Matching table(s) / caption information |
|---|---|
| [reports/final_experiment_results_20261019/figures/precision_recall_f1.png](reports/final_experiment_results_20261019/figures/precision_recall_f1.png) | `tables/main_performance.csv`, common_ge20; six tuned test policies. |
| [reports/final_experiment_results_20261019/figures/confusion_matrices.png](reports/final_experiment_results_20261019/figures/confusion_matrices.png) | Same main table; counts, not normalized rates. |
| [reports/final_experiment_results_20261019/figures/fpr_confidence_intervals.png](reports/final_experiment_results_20261019/figures/fpr_confidence_intervals.png) | Same main table; exact FPR intervals and sample denominators. |
| [reports/final_experiment_results_20261019/figures/observation_cost_and_latency.png](reports/final_experiment_results_20261019/figures/observation_cost_and_latency.png) | `tables/observation_cost_latency.csv`; qualify detection-conditioned latency. |
| [reports/final_experiment_results_20261019/figures/stopping_distribution.png](reports/final_experiment_results_20261019/figures/stopping_distribution.png) | `tables/stopping_distribution.csv`; include unresolved terminal flows. |
| [reports/final_experiment_results_20261019/figures/checkpoint_coverage.png](reports/final_experiment_results_20261019/figures/checkpoint_coverage.png) | `tables/checkpoint_eligibility.csv`; whole extracted population, not just the common test cohort. |
| [reports/final_experiment_results_20261019/figures/roc_precision_recall.png](reports/final_experiment_results_20261019/figures/roc_precision_recall.png) | `tables/curve_coordinates.csv` and `tables/ranking_metrics.csv`; common cohort. |

The [figure index](reports/final_experiment_results_20261019/figures/index.md) previews
the saved plots. No real evasive-traffic experiment or genuine rate-controlled
attack figure exists in this package.

## 3. Five-capture local studies: exact navigation

These two directories have the same file layout. Paths in the following table
use the complete run directories, so that tables cannot accidentally be mixed.

| Contents | `(10,20)` study | `(2,4,8,10,15)` study |
|---|---|---|
| Narrative | [summary.md](data/results/exp_ufu_real_cp10_20/summary.md) | [summary.md](data/results/exp_ufu_real_cp2_4_8_10_15/summary.md) |
| RF authoritative test table | [fixed_vs_adaptive_test.csv](data/results/exp_ufu_real_cp10_20/tables/fixed_vs_adaptive_test.csv) | [fixed_vs_adaptive_test.csv](data/results/exp_ufu_real_cp2_4_8_10_15/tables/fixed_vs_adaptive_test.csv) |
| CNN authoritative test table | [cnn_gru_fixed_vs_adaptive_test.csv](data/results/exp_ufu_real_cp10_20/tables/cnn_gru_fixed_vs_adaptive_test.csv) | [cnn_gru_fixed_vs_adaptive_test.csv](data/results/exp_ufu_real_cp2_4_8_10_15/tables/cnn_gru_fixed_vs_adaptive_test.csv) |
| RF FPR counts/bounds | [fpr_constraint_validation_vs_test.csv](data/results/exp_ufu_real_cp10_20/tables/fpr_constraint_validation_vs_test.csv) | [fpr_constraint_validation_vs_test.csv](data/results/exp_ufu_real_cp2_4_8_10_15/tables/fpr_constraint_validation_vs_test.csv) |
| CNN FPR counts/bounds | [cnn_gru_fpr_constraint_validation_vs_test.csv](data/results/exp_ufu_real_cp10_20/tables/cnn_gru_fpr_constraint_validation_vs_test.csv) | [cnn_gru_fpr_constraint_validation_vs_test.csv](data/results/exp_ufu_real_cp2_4_8_10_15/tables/cnn_gru_fpr_constraint_validation_vs_test.csv) |
| RF per-flow scores/stops | [predictions_test.csv](data/results/exp_ufu_real_cp10_20/predictions_test.csv) | [predictions_test.csv](data/results/exp_ufu_real_cp2_4_8_10_15/predictions_test.csv) |
| CNN per-flow scores/stops | [cnn_gru_predictions_test.csv](data/results/exp_ufu_real_cp10_20/cnn_gru_predictions_test.csv) | [cnn_gru_predictions_test.csv](data/results/exp_ufu_real_cp2_4_8_10_15/cnn_gru_predictions_test.csv) |
| Run identity/configuration | [run_record.json](data/results/exp_ufu_real_cp10_20/run_record.json) | [run_record.json](data/results/exp_ufu_real_cp2_4_8_10_15/run_record.json) |
| Split IDs and labels | [split_assignment.csv](data/results/exp_ufu_real_cp10_20/split_assignment.csv) | [split_assignment.csv](data/results/exp_ufu_real_cp2_4_8_10_15/split_assignment.csv) |
| Split summary | [split_manifest.json](data/results/exp_ufu_real_cp10_20/split_manifest.json) | [split_manifest.json](data/results/exp_ufu_real_cp2_4_8_10_15/split_manifest.json) |
| Eligibility | [dataset_stats.csv](data/results/exp_ufu_real_cp10_20/tables/dataset_stats.csv) | [dataset_stats.csv](data/results/exp_ufu_real_cp2_4_8_10_15/tables/dataset_stats.csv) |
| CNN raw checkpoint | [cnn_gru_best.pt](data/results/exp_ufu_real_cp10_20/cnn_gru_best.pt) | [cnn_gru_best.pt](data/results/exp_ufu_real_cp2_4_8_10_15/cnn_gru_best.pt) |

Each directory also contains `predictions_val.csv`, `cnn_gru_predictions_val.csv`,
`thresholds.json`, `cnn_gru_thresholds.json`, `cnn_gru_train_history.json` and
`train_fit_cal_split.csv`. Their `tables/` directories contain
`fixed_vs_adaptive_validation.csv`, `cnn_gru_fixed_vs_adaptive_validation.csv`,
`prefix_level_test.csv`, `cnn_gru_prefix_level_test.csv`, `robustness_test.csv` and
`cnn_gru_robustness_test.csv`. Validation sweeps use
`threshold_sweep_val_adaptive.csv` / `cnn_gru_threshold_sweep_val_adaptive.csv` and
`threshold_sweep_val_fixed_<checkpoint>.csv` / CNN-prefixed equivalents, for each
checkpoint in that run's name. These are actual shared names within both directories.

The prediction columns `p_<checkpoint>` are calibrated probabilities. Adaptive
and terminal fixed-policy columns include alert flags, observed/alert packets,
latency and byte totals in these historical files. They provide per-flow stopping
evidence; confusion counts are in the policy tables. Current new-run prediction
CSVs have a smaller schema: do not assume historical extra columns are universal.

Useful existing figures under **each run's `figures/`** are
`rf_prf_vs_prefix_length.png`, `cnn_gru_prf_vs_prefix_length.png`,
`rf_observation_cost.png`, `cnn_gru_observation_cost.png`,
`detection_latency.png`, `cnn_gru_detection_latency.png`,
`rf_debd_comparison.png`, `cnn_gru_debd_comparison.png`,
`rf_val_fpr_vs_threshold.png`, `cnn_gru_val_fpr_vs_threshold.png`,
`rf_vs_cnn_adaptive_comparison.png`, `rf_robustness.png`, and
`cnn_gru_robustness.png`. The last two show **simulated** perturbations only.
The current plotting code fixes a hardcoded `fixed_40` default at pipeline call
sites; preserved historical plots were not regenerated or silently corrected.

Four byte-identical compatibility pairs exist in each local completed run:
`detection_coverage.png` / `rf_detection_coverage.png`,
`prf_vs_prefix_length.png` / `rf_prf_vs_prefix_length.png`,
`robustness.png` / `rf_robustness.png`, and
`val_fpr_vs_threshold.png` / `rf_val_fpr_vs_threshold.png`.
`calibration_validation.csv` is a compatibility RF table alongside
`rf_calibration_validation.csv`. Prefer the explicit model names. Duplicates are
preserved because historical references may depend on them.

## 4. Collect material for the academic report

1. Choose one study and record its directory, seed, checkpoints, grouping and
   population in the draft. For the organized seven-capture report, begin with
   its `results_summary.md`, `metrics/config.json` and methodology document.
2. Export the six `common_ge20` rows from the seven-capture full main CSV, or use
   its compact CSV. For a local five-capture study, use its RF and CNN policy CSVs.
   Keep confusion counts, eligible benign denominator and exclusions beside rates.
3. Select graphs from that same directory using the mapping above. Do not pair a
   five-capture table with a seven-capture plot. Record units and the cohort in
   each caption. All-flow observation cost and detection-conditioned latency have
   different denominators.
4. Use the selected study's eligibility, split and capture tables for methodology.
   The seven-capture package extracted 140,193 flows (2,095 benign / 138,098 malicious).
   Local `data/processed/flows.pkl` has 139,076 (978 / 138,098). They are not substitutes.
5. Confirm uncertainty in the full CSV or FPR constraint table. Confusion matrices
   use TP/FP/TN/FN from the same policy/cohort. Use paired latency for a matched
   comparison and stopping distributions for observation savings.
6. Check `synthetic` in the run record (new runs also put it in status/manifest).
   The archived seven-capture package is a real-capture study documented by its
   capture manifest, not by a generic `run_record.json`. Synthetic verification
   tables and the UI preview are never experimental performance evidence.
7. Check `group_by` and actual source overlap, not just a `verified_disjoint` flag.
   These existing studies share all captures across outer partitions. Report them
   as within-capture exploratory results and describe short-flow exclusions.

For the seven-capture common cohort, RF adaptive FPR is 3/275 = 1.0909%; CNN is
0/275, with a one-sided 95% upper bound of 1.0834%. Adaptive costs about 10.62
packets versus 20, but fixed 10 already achieves the same headline classification
counts with lower cost. This evidence does not show an advantage over fixed 10.
The two local five-capture studies have different denominators; do not replace
these counts with their zero-FP headline results.

## 5. Verify or regenerate without overwriting evidence

### Independently check available results (no training or figure writes)

These commands write new verification JSON files and refuse an existing output:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
.\.venv\Scripts\python.exe reports/phase10_verification/verify_saved_evidence.py --output "reports/verification_$stamp/saved_evidence.json"
.\.venv\Scripts\python.exe scripts/verify_report_tables.py --output "reports/verification_$stamp/report_tables.json"
```

The first recomputes local historical metrics, threshold sweeps and trace fields
from saved probabilities plus real processed flow metadata. It also checks
synthetic count metrics and robustness aggregate arithmetic. It does not replay
perturbed model scores or recreate missing models. It targets only the four named
historical runs and leaves newer directories alone. The second checks the seven-capture aggregate tables,
confidence bounds, cost/stopping identities, PNG/SVG parsing and hash inventory;
it cannot verify absent per-flow records.

Cleanup outputs are
[saved_evidence.json](reports/cleanup_verification_20261019/saved_evidence.json)
(144,113 passed checks) and
[report_tables.json](reports/cleanup_verification_20261019/report_tables.json)
(480 passed checks, 14 figures). The post-documentation hash inventory is in
[report_tables_final.json](reports/cleanup_verification_20261019/report_tables_final.json). Neither result implies capture independence.

### Regenerate the seven-capture tables and figures from saved predictions

**Currently blocked:** restore the original package's `processed/` and
`predictions/` directories first. The exact 23 absent artifacts (including models)
are enumerated in `report_tables.json`. `data/processed/flows.pkl` is the wrong
five-capture population. The original `build_report.py` derives both tables and
all seven figures from saved prefix probabilities and traces; it does not train.
It is not a plot-only command and cannot run from aggregate tables alone.

After restoring the correct bundle, copy it and run the reporting script in the
copy so the completed package remains unchanged:

```powershell
$study = 'reports/final_experiment_results_20261019'
$copy = 'reports/report_replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
if (Test-Path $copy) { throw 'Choose a fresh directory' }
if (!(Test-Path "$study/processed/flow_metadata.csv") -or !(Test-Path "$study/predictions/rf_test_prefixes.csv")) { throw 'Restore the original processed/prediction bundle first' }
Copy-Item -LiteralPath $study -Destination $copy -Recurse
.\.venv\Scripts\python.exe "$copy/build_report.py"
```

For historical numerical behavior, use the source snapshot and recorded package
versions in the isolated workspace described in
[reproduction.md](reports/final_experiment_results_20261019/reproduction.md).
Running a copied script against current merged source is a report recomputation
under changed source, which must be disclosed. `verify_models.py` runs inference
and checks saved scores/timing; it additionally requires the missing model bundles
and writes verification records in the copy. No real training was run during cleanup.

### New experiments versus exact historical reproduction

The README's `build`, `run`, `all` and `dry-run` commands use current source and
create new experiments. Existing five-capture predictions can be audited exactly;
their historical RF trees and CNN calibrators were never saved, so full historical
fitted inference cannot be recreated from current files. New training is not a
retroactive verification of those fitted models.

The seven-capture harness has `prepare`, `rf` and `cnn` stages; all are expensive
extraction/training, not saved-score evaluation. Its frozen-source procedure is in
the reproduction document. It needs File02 and File03, which are absent locally,
and all seven verified captures directly inside the isolated workspace's `data/`.
The generic CLI instead discovers `data/raw/` recursively. Never combine both
layouts and duplicate the same capture content.

### Loading new model bundles

This executable example uses the new synthetic check's existing bundles to verify
inference wiring. It performs no fitting and makes no real-data claim:

```powershell
@'
from pathlib import Path
from src.synthetic import make_synthetic_flows
from src.prefix_generation import build_prefix_table
from src.train_baseline import load_rf_bundle
from src.model_bundle import load_cnn_bundle
run = Path('reports/cleanup_verification_20261019/smoke/both_models')
flows = [f for f in make_synthetic_flows(600, 600, seed=42) if len(f.ts) >= 20][:8]
rf = load_rf_bundle(run / 'rf_bundle.pkl')
cnn = load_cnn_bundle(run / 'cnn_gru_bundle.pt', device='cpu')
print(rf.predict_proba(build_prefix_table(flows, [10])))
print(cnn.predict_proba(flows, 10))
'@ | .\.venv\Scripts\python.exe -
```

For a new real run, use its bundle directory and `load_flows` from its matching
processed dataset. Bundle thresholds are available as `.thresholds`; probability
inference alone does not apply an adaptive decision. Evaluate ordered checkpoint
scores with `curves_from_prefix_table`, `policy_outcomes` and `summarize`, or use the
saved policy tables. New bundle round trips are covered by
`tests/test_pipeline_bundles.py`. Historical CNN raw checkpoints can now load via
`src.cnn_gru_model.load_model`, but missing calibration still prevents complete
historical inference replay. Source and library versions matter for pickle loading.

## 6. Metric definitions and research-integrity notes

- **Eligibility:** prefix training includes every available configured prefix.
  Main policy comparisons retain only flows supplying all checkpoints; missing
  prefixes are not fabricated or scored as benign. Fixed-window rows in that
  comparison use the same long-flow population. Native-prefix rows in the
  seven-capture report are separate. Flows shorter than the first checkpoint are
  excluded before splitting. No operational short-flow decision policy is validated.
- **FPR:** FP/(FP+TN), with one event per eligible benign flow, counting an alert
  at any checkpoint. The sequential rule uses the first score >= threshold.
  No-alert malicious flows are FN; no-alert benign flows are TN. Coverage equals
  recall. Prefix-row counts are not independent benign trials.
- **Thresholds:** lowest grid threshold satisfying complete-policy empirical
  validation FPR <= alpha; grid step is recorded (0.001 in these studies), with
  a >1 sentinel that never alerts. Thresholds freeze before test. Each fixed
  window has its own threshold. A validation target is not a test/population guarantee.
- **Observation cost:** observed packets / global Tmax for main tables, averaged
  over all flows and malicious flows separately. Adaptive no-alert flows observe
  Tmax. Fixed-t flows observe t even without an alert, so cost is t/Tmax.
  Validation sweep `mean_cost_malicious` instead divides by the selected policy's
  terminal checkpoint; fixed-policy sweep costs therefore equal 1. Do not mix them.
- **Latency:** first retained packet to alert packet in seconds; reported only
  among detected malicious flows. It excludes inference runtime and unknown attack
  onset. Paired comparisons use shared true positives. Under timing dilation,
  packet cost and elapsed latency can change differently.
- **DEBD:** IP-byte pre-alert exposure proxy, both directions, including alert
  packet, relative to counted full-flow bytes. Undetected attacks get 100% in
  all-flow versions. Mean-per-flow and byte-weighted versions differ. It is not
  a measurement of stolen payload; parser/capture completeness affects it.
- **Uncertainty:** Clopper–Pearson two-sided 95% intervals and one-sided 95% upper
  bounds are computed from binomial event counts. Zero FP with 131 or 127 benign
  flows yields upper bounds about 2.26% / 2.33%. At least 299 independent zero-FP
  benign trials are needed for a one-sided upper bound below 1%. Correlated flows
  from shared sessions do not supply that independence; intervals are optimistic.
- **Provenance and confounding:** filename/scenario labels, TCP/443 filtering,
  extreme imbalance, repeated prefix signatures and session fingerprints limit
  interpretation. The current parser still uses a TCP sequence high-water heuristic
  that can discard reordered unseen payload, and only stores 200 retained packets.
  Historical parser findings refer to historical source. See the dataset notes.
- **Simulation and missing evidence:** local robustness tables perturb malicious
  timestamps offline; benign traffic is unchanged. Perturbed per-flow scores are
  not saved for historical runs, so only aggregate arithmetic is replayable.
  No independently labeled rate-controlled attack study, held-out-capture result,
  capture-level confidence interval or multi-seed confirmation exists here.

## 7. Audit evidence and future frontend use

[reports/full_project_audit.md](reports/full_project_audit.md) is preserved with a
current-status preface; old line numbers, branch names and missing-artifact claims
are historical. [reports/cleanup_verification_20261019/verification.md](reports/cleanup_verification_20261019/verification.md)
records current checks and blockers. `preserved_artifacts.json` records hashes of
247 data/report/graph artifacts protected during cleanup. Historical logs remain
under `reports/phase10_verification/`; new temporary logs and smoke binaries are ignored.

New completed runs emit version-1 `experiment.json`, a strict JSON summary of
configuration, provenance, test metrics, stopping distributions, split information
and relative artifact paths. Read it only with `run_status.json` marked complete.
Undefined numbers are `null`. The full contract is in
[frontend/README.md](frontend/README.md#offline-result-contract-version-1).
The frontend remains a standalone synthetic preview and does not load this file yet.
