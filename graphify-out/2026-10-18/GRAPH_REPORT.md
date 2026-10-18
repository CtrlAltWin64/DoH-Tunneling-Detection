# Graph Report - DoH-Tunneling-Detection  (2026-10-18)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 560 nodes · 1309 edges · 26 communities (16 shown, 10 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.9)
- Token cost: 1,963 input · 275 output

## Graph Freshness
- Built from commit: `8a2881ad`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Project Configuration
- Adaptive Policy Evaluation
- Model Training and Calibration
- PCAP Processing Utilities
- Frontend Dependencies
- Flow Prefix Generation
- CNN-GRU Model Architecture
- Data Partitioning Logic
- TypeScript Configuration
- Sequence Normalization
- Experiment Execution Framework
- Model Training Pipeline
- Model Validation Tests
- CNN Prefix Scoring
- Visualization and Plotting
- Neural Network Layers
- Performance Metrics
- Model Persistence
- Statistical Analysis
- Integration Testing
- Calibration Validation
- Model Comparison Plots
- Dataset Definitions
- User Interface

## God Nodes (most connected - your core abstractions)
1. `run_experiment()` - 47 edges
2. `policy_outcomes()` - 31 edges
3. `SequenceNormalizer` - 29 edges
4. `build_prefix_table()` - 25 edges
5. `make_flow()` - 24 edges
6. `make_curves()` - 23 edges
7. `_run_cnn_gru_experiment()` - 23 edges
8. `CnnGruClassifier` - 22 edges
9. `summarize()` - 18 edges
10. `build_cnn_prefix_table()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `PcapPreprocessing` --uses--> `Config`  [INFERRED]
  tests/test_prefixes.py → config.py
- `TrainingSmokeTest` --uses--> `CnnGruClassifier`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `TrainingSmokeTest` --uses--> `SequenceNormalizer`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `ModelShapes` --uses--> `CnnGruClassifier`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `SeparateCalibrationTests` --uses--> `CnnGruClassifier`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py

## Import Cycles
- None detected.

## Communities (26 total, 10 thin omitted)

### Community 0 - "Project Configuration"
Cohesion: 0.06
Nodes (23): Config, _cfg(), main(), create_split_manifest(), save_split_manifest(), create_dataset_manifest(), inspect_csv(), inspect_dir() (+15 more)

### Community 1 - "Adaptive Policy Evaluation"
Cohesion: 0.06
Nodes (14): curves_from_prefix_table(), FlowCurves, policy_outcomes(), run_policy(), select_threshold(), sweep_thresholds(), threshold_grid(), summarize() (+6 more)

### Community 2 - "Model Training and Calibration"
Cohesion: 0.06
Nodes (18): Calibrator, fit_calibrator(), _logit(), oof_raw_probabilities(), low_and_slow(), _rebuild(), _rng(), timing_jitter() (+10 more)

### Community 3 - "PCAP Processing Utilities"
Cohesion: 0.07
Nodes (11): build_flows_from_pcap(), convert_pcapng_to_pcap(), parse_tcp_packet(), PcapFormatError, PcapReader, _State, _tcp_frame(), write_pcap() (+3 more)

### Community 4 - "Frontend Dependencies"
Cohesion: 0.06
Nodes (44): dependencies, lucide-react, react, react-dom, vite, @vitejs/plugin-react, devDependencies, @types/react (+36 more)

### Community 5 - "Flow Prefix Generation"
Cohesion: 0.09
Nodes (9): build_prefix_table(), eligible_counts(), generate_flow_prefixes(), _validate_checkpoints(), make_flow(), SequenceBuilding, CalibrationLeakage, split() (+1 more)

### Community 6 - "CNN-GRU Model Architecture"
Cohesion: 0.17
Nodes (3): _build_sequence_array(), build_sequence_dataset(), score_flows_at_checkpoint()

### Community 7 - "Data Partitioning Logic"
Cohesion: 0.17
Nodes (3): split_flows(), split_train_calibration(), FlowLevelSplit

### Community 8 - "TypeScript Configuration"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+9 more)

### Community 10 - "Experiment Execution Framework"
Cohesion: 0.19
Nodes (8): _get_device(), _make_cnn_summary_section(), _make_rf_summary_section(), md_table(), _policy_set(), run_experiment(), pred_frame_rf(), quiet()

### Community 11 - "Model Training Pipeline"
Cohesion: 0.15
Nodes (4): collate_sequences(), SequenceDataset, train_cnn_gru(), TrainingSmokeTest

### Community 13 - "CNN Prefix Scoring"
Cohesion: 0.21
Nodes (3): build_cnn_prefix_table(), raw_proba_cnn(), PrefixTableIntegration

### Community 14 - "Visualization and Plotting"
Cohesion: 0.36
Nodes (6): grouped_bars(), _need(), plot_prf_vs_prefix(), plot_robustness(), plot_val_fpr_vs_threshold(), _save()

### Community 16 - "Performance Metrics"
Cohesion: 0.29
Nodes (3): _div(), expected_calibration_error(), prefix_metrics()

### Community 17 - "Model Persistence"
Cohesion: 0.38
Nodes (4): CnnGruClassifier, load_model(), save_model(), CheckpointSaveLoad

### Community 18 - "Statistical Analysis"
Cohesion: 0.22
Nodes (5): clopper_pearson(), _run_cnn_gru_experiment(), pred_frame(), _timing_summary(), Statistics

## Knowledge Gaps
- **5 isolated node(s):** `@types/react`, `@types/react-dom`, `typescript`, `UFU-DoH-EXF Dataset`, `Traceglass UI`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 203 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_experiment()` connect `Experiment Execution Framework` to `Project Configuration`, `Adaptive Policy Evaluation`, `Model Training and Calibration`, `Flow Prefix Generation`, `Data Partitioning Logic`, `Visualization and Plotting`, `Performance Metrics`, `Statistical Analysis`, `Training and Selection`, `Model Comparison Plots`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `SequenceNormalizer` (e.g. with `fit_calibrator_cnn()` and `build_cnn_prefix_table()`) actually correct?**
  _`SequenceNormalizer` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `@types/react`, `@types/react-dom`, `typescript` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Project Configuration` be split into smaller, more focused modules?**
  _Cohesion score 0.05740740740740741 - nodes in this community are weakly interconnected._
- **Why does `policy_outcomes()` connect `Adaptive Policy Evaluation` to `Project Configuration`, `Experiment Execution Framework`, `Model Training and Calibration`, `Statistical Analysis`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Should `Adaptive Policy Evaluation` be split into smaller, more focused modules?**
  _Cohesion score 0.06116700201207243 - nodes in this community are weakly interconnected._
- **Why does `ModelShapes` connect `Model Validation Tests` to `Model Persistence`, `CNN-GRU Model Architecture`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._