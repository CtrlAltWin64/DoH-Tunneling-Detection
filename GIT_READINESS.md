# Repository packaging

Source, tests, configuration, dependency files, documentation and report-generating
scripts are tracked. Aggregate research tables, figures, source snapshots and
audit evidence stay in their established locations.

## Local only

- Captures and `data/raw/`, `data/processed/`, `data/results/` contents except placeholders.
- Historical study `processed/`, `predictions/`, `models/`, logs and test temporaries.
- Virtual environments, caches, credentials and environment files.
- Frontend dependencies, `dist/` and TypeScript build state.
- New verification scratch files, raw test logs and synthetic model outputs.
- `reports/local_archive/`: preserved machine-local clutter.

During cleanup, the tracked `powershell.cmd`, accidental root terminal-help file
whose name starts with `earch`, Graphify cache, `.graphify_root` and
`.cloud-cta-shown` were moved intact under `reports/local_archive/`, retaining
their relative layout. Their old tracked paths appear as deletions for review;
the ignored copies are still on disk. No Git index changes, commits, pushes,
history rewrites or branch operations were performed.

Useful Graphify JSON/HTML/reports and dated snapshots remain in `graphify-out/`.
They are generated development maps, not research evidence, and were not rebuilt.
Existing historical test logs remain tracked as audit evidence. New temporary
logs are ignored. Inaccessible old `reports/phase10_verification/test_tmp/`
directories were neither removed nor changed.

The initial checkout was clean on `main` at `c38aac2`. No rebase/merge/sequencer
operation was active; a leftover `REBASE_HEAD` alone did not indicate a rebase.
The older nested-checkout rule is retained, but no nested checkout exists here.

## Reproducibility

A clone includes readable aggregate evidence, not the large artifacts required
for inference or score replay. [RESULTS_GUIDE.md](RESULTS_GUIDE.md) distinguishes
what exists locally from the absent seven-capture bundles. Original artifact hashes
describe the historical full bundle, not a promise that every artifact is committed.

`requirements.txt` contains minimum versions; the historical and current observed
environments are recorded with their experiments/verification. Exact source and
library versions matter for serialized models.

Before staging, inspect `git status --short` and `git diff --check`.
Use `git add --dry-run .` to inspect candidates if desired. Do not force-add
captures or local archives. Review deletions knowing their local backups exist.
No license or dataset redistribution permission is inferred by cleanup.
