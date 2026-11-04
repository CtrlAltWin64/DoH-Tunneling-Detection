# Research results package

> Historical package index. Current availability differs: the large processed,
> prediction, trace and model directories are absent from this checkout. Tables,
> figures and recorded verification summaries remain. Older five-capture score
> files are present separately under `data/results/`. See the current
> [results guide](../../RESULTS_GUIDE.md) before interpreting retention claims below.

- [Main report](results_summary.md)
- [Methodology and limitations](methodology_and_limitations.md)
- [Reproduction commands](reproduction.md)
- [Main performance table](tables/main_performance_compact.md) · [Full CSV with confidence intervals and denominators](tables/main_performance.csv)
- [Observation cost and latency](tables/observation_cost_latency.md)
- [Paired latency](tables/matched_detection_latency.md)
- [Early stopping errors](tables/early_stopping_errors.md)
- [Stopping distribution](tables/stopping_distribution.md)
- [Dataset eligibility](tables/checkpoint_eligibility.md)
- [Capture inventory](tables/capture_inventory.md)
- [Split counts](tables/split_class_counts.md)
- [All figures, PNG and SVG](figures/index.md)

Verification: 113 existing tests passed, 160 report checks passed, 16 complete model-reload/timing checks passed, and 100 saved synthetic metric checks passed. Synthetic evidence is software validation only. Source snapshot, capture hashes, fitted model/calibrator bundles, raw/calibrated scores, split assignments and stopping traces are retained. Historical real-run artifacts mentioned in the earlier audit were absent.

All findings are preliminary within-capture results. No commit or push was performed.
