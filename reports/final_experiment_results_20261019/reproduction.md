# Reproduction

From `D:\DoH-Tunneling-Detection-main`, using the existing environment:

```powershell
.\.venv\Scripts\python.exe reports/final_experiment_results_20261019/build_report.py
.\.venv\Scripts\python.exe reports/final_experiment_results_20261019/verify_models.py
```

The first command regenerates metrics and figures from saved real predictions; it does not train. The second checks fitted bundle predictions and timestamps. These regenerate derived reports/check records; preserve a copy if editing the reporting scripts.

To rerun full extraction/training without overwriting this package, create a NEW directory under reports, copy run_experiments.py, verify_inputs.py, build_report.py and verify_models.py into it, and copy source_snapshot/config.py and source_snapshot/src into the root of an isolated workspace with the same seven captures under data. Install the versions recorded in metrics/environment.json. Then, replacing RESULT_DIR with the new relative path, run SEQUENTIALLY:

```powershell
.\.venv\Scripts\python.exe -u RESULT_DIR/run_experiments.py prepare
.\.venv\Scripts\python.exe RESULT_DIR/verify_inputs.py
.\.venv\Scripts\python.exe -u RESULT_DIR/run_experiments.py rf
.\.venv\Scripts\python.exe -u RESULT_DIR/run_experiments.py cnn
.\.venv\Scripts\python.exe RESULT_DIR/build_report.py
.\.venv\Scripts\python.exe RESULT_DIR/verify_models.py
```

For synthetic saved-evidence checks, also retain data/results/synthetic_smoke_test. No rerun of synthetic training is required. Model bundles are local trusted Python pickles; source and library versions matter. Model scores, raw probabilities, calibration maps, full policy threshold sweeps, per-flow stopping traces, capture hashes and split roles are saved. CPU floating-point/training behavior across versions can differ; reproduce the recorded environment for closest agreement.

Software validation used:
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp reports/final_experiment_results_20261019/test_temp
```
113 tests passed; this validates software fixtures, not unseen-capture performance. The initial rf.log records a premature launch before split_assignment.csv existed; rf_run.log records the completed retry.

Exact commands to create a fresh run using the unchanged source in this workspace:

```powershell
Set-Location 'D:\DoH-Tunneling-Detection-main'
$resultDir = 'reports/final_experiment_results_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $resultDir -ErrorAction Stop
Copy-Item -Path 'reports/final_experiment_results_20261019/*.py' -Destination $resultDir
& '.\.venv\Scripts\python.exe' -u "$resultDir/run_experiments.py" prepare
& '.\.venv\Scripts\python.exe' "$resultDir/verify_inputs.py"
& '.\.venv\Scripts\python.exe' -u "$resultDir/run_experiments.py" rf
& '.\.venv\Scripts\python.exe' -u "$resultDir/run_experiments.py" cnn
& '.\.venv\Scripts\python.exe' "$resultDir/build_report.py"
& '.\.venv\Scripts\python.exe' "$resultDir/verify_models.py"
```

Check each command exits successfully before continuing. Existing model bundles and processed datasets are guarded against overwrite.
