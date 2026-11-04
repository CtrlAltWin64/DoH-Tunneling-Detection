# Reproduction of the archived seven-capture study

This package preserves the historical experiment. Current availability and exact
table/figure navigation are in [RESULTS_GUIDE.md](../../RESULTS_GUIDE.md).
The local five-capture dataset is different. File02/File03 and this package's
processed/prediction/model directories are missing, so full replay is blocked.

For a read-only aggregate check, run from the repository root:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
.\.venv\Scripts\python.exe scripts/verify_report_tables.py --output "reports/verification_$stamp/report_tables.json"
```

This checks arithmetic and file integrity, not absent scores or model inference.

## Historical source and a fresh workspace

The original study used the files in `source_snapshot/`, not today's merged
implementation. The old five-group splitter restriction and CNN checkpoint
selection behavior belong to that snapshot. Install the package versions in
`metrics/environment.json` for closest reproduction; platform-level floating
point and training differences can still occur.

The following preparation commands copy the frozen implementation and harness
into a new workspace. They do not run training and refuse an existing destination:

```powershell
$root = (Get-Location).Path
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$study = Join-Path $root 'reports/final_experiment_results_20261019'
$workspace = Join-Path $root ('reproductions/study_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path $workspace) { throw 'Choose a fresh workspace' }
New-Item -ItemType Directory -Path "$workspace/reports/study","$workspace/data" -Force | Out-Null
Copy-Item -LiteralPath "$study/source_snapshot/config.py" -Destination $workspace
Copy-Item -LiteralPath "$study/source_snapshot/src" -Destination $workspace -Recurse
Copy-Item -Path "$study/*.py" -Destination "$workspace/reports/study"
```

Restore exactly the seven capture filenames listed in `metrics/capture_manifest.csv`
directly under the new workspace's `data/`, then compare SHA-256 and byte sizes.
The harness discovers `data/*.pcap*` nonrecursively; do not place extra or duplicate
captures there. It assigns benign filenames label 0 and other captures label 1,
so its input whitelist must be checked. Four benign / three malicious files are
expected. The current five available files can be copied from `data/raw/`, but the
missing File02/File03 must be restored from the original data source.

After restoring and verifying all seven captures and preparing a matching Python
environment, execute sequentially **inside that new workspace**:

```powershell
Push-Location $workspace
try {
    & $python -u reports/study/run_experiments.py prepare
    if ($LASTEXITCODE -ne 0) { throw 'prepare failed' }
    & $python reports/study/verify_inputs.py
    if ($LASTEXITCODE -ne 0) { throw 'input verification failed' }
    & $python -u reports/study/run_experiments.py rf
    if ($LASTEXITCODE -ne 0) { throw 'RF training failed' }
    & $python -u reports/study/run_experiments.py cnn
    if ($LASTEXITCODE -ne 0) { throw 'CNN training failed' }
    & $python reports/study/build_report.py
    if ($LASTEXITCODE -ne 0) { throw 'report generation failed' }
    & $python reports/study/verify_models.py
    if ($LASTEXITCODE -ne 0) { throw 'model verification failed' }
} finally {
    Pop-Location
}
```

`prepare` extracts flows/splits; `rf` and `cnn` train and score. `build_report.py`
regenerates tables and figures from probabilities and traces without training.
`verify_models.py` reloads the historical tuple-format pickle bundles, runs inference
and checks scores/timestamps; it also writes derived diagnostics. The historical
bundle format differs from the new generic pipeline's versioned bundles.

These stages were inspected against the scripts but were **not executed during
cleanup**, because the historical inputs/bundles are unavailable and expensive
real training was not needed. The current `$python` example points to the local
environment; replace it with a matching historical environment for closest replay.

## Reusing a restored original fitted bundle

To regenerate without training, first restore this package's original `processed/`,
`predictions/` and, for inference verification, `models/`. Copy the complete
restored package into the fresh workspace at `reports/study/` alongside frozen
source, and run only `build_report.py` and `verify_models.py`. The result guide also
shows a safe copy-based report command for current source. Never run report-writing
scripts directly over the preserved original package.

The historical synthetic replay expected `data/results/synthetic_smoke_test/`.
It is absent here. `verify_inputs.py` records a skip when those synthetic scores
are missing; a skip is not a fresh verification. The original records of 113 tests,
160 report checks, 16 model checks and 100 synthetic checks describe the original
environment. Current cleanup verification is recorded separately.
