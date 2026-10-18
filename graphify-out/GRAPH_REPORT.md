# Graph Report - DoH-Tunneling-Detection  (2026-10-19)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 617 nodes · 1467 edges · 44 communities (23 shown, 21 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8a2881ad`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- numpy
- build_prefix_table
- App.tsx
- split_flows
- test_prefixes.py
- build_flows_from_pcap
- verify_saved_evidence.py
- test_cnn_gru.py
- Config
- compilerOptions
- PcapReader
- main.py
- CnnGruClassifier
- train_cnn_gru
- Calibrator
- SequenceNormalizer
- calibration.py
- test_pipeline_bundles.py
- load_rf_bundle
- ModelShapes
- Pipeline
- plots.py
- score_flows_at_checkpoint
- MaskedBatchNorm1d
- json
- preprocessing.py
- train_baseline.py
- make_synthetic_flows
- Confusion
- fit_calibrator_cnn
- Adaptive Sequential Stopping
- UFU-DoH-EXF Dataset
- Traceglass Dashboard
- Traceglass UI
- Full Project Audit Report
- matplotlib
- numpy
- pandas
- scikit-learn
- scipy
- torch
- UFU-DoH-EXF Dataset

## God Nodes (most connected - your core abstractions)
1. `run_experiment()` - 44 edges
2. `SequenceNormalizer` - 32 edges
3. `CnnGruClassifier` - 25 edges
4. `Config` - 25 edges
5. `policy_outcomes()` - 25 edges
6. `build_prefix_table()` - 24 edges
7. `_run_cnn_gru_experiment()` - 23 edges
8. `make_flow()` - 21 edges
9. `make_synthetic_flows()` - 21 edges
10. `train_cnn_gru()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `PcapPreprocessing` --uses--> `PcapFormatError`  [INFERRED]
  tests/test_prefixes.py → src/preprocessing.py
- `PcapPreprocessing` --uses--> `PcapReader`  [INFERRED]
  tests/test_prefixes.py → src/preprocessing.py
- `CheckpointSaveLoad` --uses--> `SequenceNormalizer`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `ModelShapes` --uses--> `CnnGruClassifier`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `SeparateCalibrationTests` --uses--> `CnnGruClassifier`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Data Science and ML Stack** — requirements_numpy, requirements_pandas, requirements_scikit_learn, requirements_scipy, requirements_matplotlib, requirements_torch [INFERRED 0.90]

## Communities (44 total, 21 thin omitted)

### Community 0 - "numpy"
Cohesion: 0.05
Nodes (30): curves_from_prefix_table(), FlowCurves, policy_outcomes(), run_policy(), select_threshold(), sweep_thresholds(), threshold_grid(), clopper_pearson() (+22 more)

### Community 1 - "build_prefix_table"
Cohesion: 0.07
Nodes (14): low_and_slow(), _rebuild(), _rng(), timing_jitter(), extract_packet_sequence(), prefix_feature_vector(), build_prefix_table(), eligible_counts() (+6 more)

### Community 2 - "App.tsx"
Cohesion: 0.06
Nodes (44): dependencies, lucide-react, react, react-dom, vite, @vitejs/plugin-react, devDependencies, @types/react (+36 more)

### Community 3 - "split_flows"
Cohesion: 0.11
Nodes (6): create_split_manifest(), save_split_manifest(), split_flows(), split_train_calibration(), FlowLevelSplit, meta_of()

### Community 4 - "test_prefixes.py"
Cohesion: 0.16
Nodes (3): _tcp_frame(), write_pcap(), _write_pcapng_file()

### Community 5 - "build_flows_from_pcap"
Cohesion: 0.13
Nodes (4): build_flows_from_pcap(), Flow, parse_tcp_packet(), PcapPreprocessing

### Community 6 - "verify_saved_evidence.py"
Cohesion: 0.19
Nodes (10): check(), count_metrics(), divide(), independent_outcomes(), intervals(), prefix_metrics(), run(), clean() (+2 more)

### Community 7 - "test_cnn_gru.py"
Cohesion: 0.16
Nodes (3): _build_sequence_array(), build_sequence_dataset(), raw_proba_cnn()

### Community 8 - "Config"
Cohesion: 0.18
Nodes (5): Config, inspect_dir(), make_pcap_tree(), PcapEndToEnd, _write_single_pcapng()

### Community 9 - "compilerOptions"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+9 more)

### Community 10 - "PcapReader"
Cohesion: 0.19
Nodes (4): inspect_pcap(), convert_pcapng_to_pcap(), PcapFormatError, PcapReader

### Community 11 - "main.py"
Cohesion: 0.19
Nodes (6): _cfg(), main(), build_flows(), load_flows(), save_flows(), Cli

### Community 12 - "CnnGruClassifier"
Cohesion: 0.21
Nodes (5): CnnGruClassifier, load_model(), save_model(), CheckpointSaveLoad, TrainingArtifactPersistence

### Community 13 - "train_cnn_gru"
Cohesion: 0.12
Nodes (4): collate_sequences(), SequenceDataset, train_cnn_gru(), TrainingSmokeTest

### Community 14 - "Calibrator"
Cohesion: 0.17
Nodes (3): Calibrator, _logit(), CalibrationMonotonicity

### Community 16 - "calibration.py"
Cohesion: 0.21
Nodes (6): fit_calibrator(), oof_raw_probabilities(), fit_rf(), flow_weights(), make_rf(), Weighting

### Community 17 - "test_pipeline_bundles.py"
Cohesion: 0.20
Nodes (3): CnnBundle, load_cnn_bundle(), PipelineBundles

### Community 18 - "load_rf_bundle"
Cohesion: 0.20
Nodes (4): load_rf_bundle(), raw_proba(), RfBundle, RfPersistence

### Community 21 - "plots.py"
Cohesion: 0.36
Nodes (6): grouped_bars(), _need(), plot_prf_vs_prefix(), plot_robustness(), plot_val_fpr_vs_threshold(), _save()

### Community 22 - "score_flows_at_checkpoint"
Cohesion: 0.23
Nodes (3): score_flows_at_checkpoint(), build_cnn_prefix_table(), PrefixTableIntegration

### Community 24 - "json"
Cohesion: 0.27
Nodes (4): create_dataset_manifest(), inspect_csv(), print_report(), save_report()

### Community 25 - "preprocessing.py"
Cohesion: 0.25
Nodes (3): _ip_str(), match_label_rule(), _State

### Community 27 - "make_synthetic_flows"
Cohesion: 0.33
Nodes (3): make_synthetic_flows(), CalibrationLeakage, split()

## Knowledge Gaps
- **15 isolated node(s):** `@types/react`, `@types/react-dom`, `typescript`, `Traceglass Dashboard`, `UFU-DoH-EXF Dataset` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 227 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_experiment()` connect `numpy` to `build_prefix_table`, `split_flows`, `test_prefixes.py`, `Config`, `main.py`, `calibration.py`, `test_pipeline_bundles.py`, `Pipeline`, `plots.py`, `json`, `train_baseline.py`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `SequenceNormalizer` (e.g. with `fit_calibrator_cnn()` and `build_cnn_prefix_table()`) actually correct?**
  _`SequenceNormalizer` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `@types/react`, `@types/react-dom`, `typescript` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.05446727185857621 - nodes in this community are weakly interconnected._
- **Why does `Config` connect `Config` to `test_prefixes.py`, `build_flows_from_pcap`, `test_cnn_gru.py`, `main.py`, `CnnGruClassifier`, `test_pipeline_bundles.py`, `load_rf_bundle`, `make_synthetic_flows`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `CnnGruClassifier` (e.g. with `fit_calibrator_cnn()` and `build_cnn_prefix_table()`) actually correct?**
  _`CnnGruClassifier` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Should `build_prefix_table` be split into smaller, more focused modules?**
  _Cohesion score 0.06623376623376623 - nodes in this community are weakly interconnected._