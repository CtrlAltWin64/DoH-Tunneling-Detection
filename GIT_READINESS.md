# Git packaging

The outer repository is the research workspace. The nested
`DoH-Tunneling-Detection/` directory is an independent checkout and is ignored by
the outer repository; its files and Git history are preserved.

## Included in a normal commit

- Source, tests, requirements and documentation.
- Study/reproduction scripts and the source snapshot used for the recorded run.
- Aggregate CSV/Markdown tables and high-resolution PNG/editable SVG figures.
- Configurations, capture hashes, threshold sweeps and verification summaries.

## Local only

- All packet captures, including the seven inputs directly in `data/`.
- Virtual environments, Python caches and local environment files.
- `data/raw/`, `data/processed/`, `data/results/` contents except `.gitkeep`.
- Study `processed/`, `predictions/`, `models/`, `test_temp/` directories and run logs.
- The independent nested checkout.

Ignoring files does not delete them. The captures were untracked when these rules
were added, so no index removal or history rewrite was needed. This preparation
does not commit, push or stage changes.

## Reproducibility from a clone

The committed report can be read without large artifacts. To recompute it,
obtain the exact captures and rerun the saved study scripts, or restore the local
processed/model/prediction bundle. Historical `artifact_hashes.json` describes
the complete original local bundle, including files intentionally absent from Git;
it is not a claim that every listed file is committed. Later packaging script
changes are recorded separately in `metrics/git_packaging_hashes.json`.

The saved synthetic verification table is included. Replaying those synthetic checks
requires the original local `data/results/synthetic_smoke_test` scores. Real extraction
and training do not require synthetic artifacts. `verify_inputs.py` explicitly skips
the synthetic replay when those scores are absent; a skip is not fresh verification.

`requirements.txt` contains broad minimum versions. `requirements-dev.txt` adds
pytest and Pillow for tests and artifact checks. The observed research environment
is recorded in `reports/final_experiment_results_20261019/metrics/environment.json`;
minimum-compatible versions do not guarantee bitwise replay.

## Before committing

```powershell
git status --short
git diff --check
git check-ignore data/File01_DoH_Benign.pcapng data/File10_DoH_Exfiltration.pcap
git add --dry-run .
```

Inspect candidate files before staging. Never use `git add -f` for large datasets.
A standard `git add .` respects these rules. No code license or dataset redistribution
permission is inferred or added by this packaging step.
