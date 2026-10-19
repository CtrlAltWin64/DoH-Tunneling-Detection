# Traceglass — Adaptive DoH Detection UI

A standalone React + TypeScript + Vite frontend concept for the Adaptive Stopping for Early Detection of Evasive DNS-over-HTTPS (DoH) Tunneling project.

## Isolation guarantee

This folder is designed to be copied into the repository as `frontend/`. It does not require or modify the Python research pipeline, tests, dataset files, root README, root requirements, or model code. All current screen values and flow rows are synthetic UI-preview data. No PCAPs are loaded, no model is called, and no research metrics should be presented as measured results.

## Run locally

Requires Node.js 20 or later.

```bash
npm ci
npm run dev
```

Open the localhost URL printed by Vite. To run a production compile check:

```bash
npm run build
```

## Main screens

- **Overview:** synthetic KPIs, adaptive-versus-fixed progression chart, stopping-rule illustration, and recent flow decisions.
- **Flow explorer:** filter sample flows, select a row, and inspect a sample decision trace.
- **Evasion lab:** interactive low-and-slow / timing-jitter UI with sliders; changes only the illustration and does not execute a detector.
- **Experiment runs:** sample run cards and filters.

## Design tokens

The visual theme uses graphite and deep forest surfaces with acid-mint, cyber-teal, aqua-foam and cool-blue accents. The layout uses translucent glass panels, subtle borders, radial lighting, a responsive sidebar, and reduced-motion support.

## Before connecting real data

The future boundary is offline experiment artifacts. No API endpoints or live
execution are implemented. Use actual test metrics; distinguish threshold
validation, test evaluation and simulated timing perturbations.

## Offline result contract (version 1)

New Python pipeline runs automatically write `experiment.json` in their run
directory (normally `data/results/<name>/`). `src/result_export.py` defines this
contract; `tests/test_pipeline_bundles.py` checks exported counts against CSVs.
The current UI does not read it. An existing example from software validation is
`reports/cleanup_verification_20261019/smoke/both_models/experiment.json`.

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1`; reject unsupported versions. |
| `experiment_id` | Run directory name. |
| `completion_marker` | `run_status.json`; require its `status` to be `complete`. `running` or `failed` directories are incomplete even if old files remain after explicit overwrite. |
| `synthetic` | Boolean; prominently label synthetic software checks. |
| `config`, `environment`, `source_sha256` | Seed, checkpoints, models, versions, timestamp, producing source hashes. Hashes do not certify capture provenance. |
| `dataset` | Extracted/usable/excluded flow counts and recorded grouping. |
| `split` | Grouping, capture overlap/disjointness, assignment/manifest/eligibility paths. |
| `models.rf`, `models.cnn_gru` | Selected models only: frozen thresholds, test-policy metrics, stopping distribution and artifact paths. |
| `stopping_distribution` | Within each model: policy, label (0 benign / 1 malicious), boolean alert, observed packet count, number of flows. Includes terminal no-alert observations. |
| `artifacts` | Within each model: metrics, scores, FPR constraints, simulated robustness and bundle paths. |
| `figures` | Relative PNG paths. |
| `conventions` | Cohort, FPR denominator, units, confidence-bound assumption and simulation qualifier. |

Artifact paths are relative to the run directory. A future reader should resolve
only contained relative paths. Configurations may contain local absolute paths;
these are metadata, not fetch URLs. Undefined metrics are JSON `null`, never
`NaN`/`Infinity`; do not turn them into zero rates. Scores are probabilities,
checkpoints are packet counts, latency is seconds and cost is a fraction of Tmax.
Confidence bounds assume independent flows, not capture-level generalization.

Per-flow scores remain in CSVs. The manifest is not a streaming detector API and
contains no frontend commands. Historical runs predate this schema and remain
documented in [RESULTS_GUIDE.md](../RESULTS_GUIDE.md); do not fabricate missing
historical data from the mock UI.

Independent build checked on 19 October 2026: `npm ci --ignore-scripts --no-audit
--no-fund`, then `npm run build`. No Python/backend integration was added.
