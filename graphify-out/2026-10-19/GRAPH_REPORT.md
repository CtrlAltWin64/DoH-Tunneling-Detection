# Graph Report - DoH-Tunneling-Detection  (2026-10-19)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 563 nodes · 1310 edges · 27 communities (17 shown, 10 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8a2881ad`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- policy_outcomes
- numpy
- pipeline.py
- App.tsx
- calibration.py
- test_pipeline.py
- pandas
- build_flows_from_pcap
- Pipeline
- test_cnn_gru.py
- compilerOptions
- SequenceNormalizer
- test_prefixes.py
- preprocessing.py
- train_cnn_gru
- build_flows
- ModelShapes
- build_cnn_prefix_table
- inspect_dataset.py
- MaskedBatchNorm1d
- CnnGruClassifier
- fit_calibrator_cnn
- Adaptive Sequential Stopping Framework
- Dataset Definitions
- Traceglass Dashboard
- Traceglass UI

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
- `CheckpointSaveLoad` --uses--> `SequenceNormalizer`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `SeparateCalibrationTests` --uses--> `SequenceNormalizer`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `TrainingSmokeTest` --uses--> `SequenceNormalizer`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py
- `PcapPreprocessing` --uses--> `PcapFormatError`  [INFERRED]
  tests/test_prefixes.py → src/preprocessing.py
- `TrainingSmokeTest` --uses--> `CnnGruClassifier`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py

## Import Cycles
- None detected.

## Communities (27 total, 10 thin omitted)

### Community 0 - "policy_outcomes"
Cohesion: 0.06
Nodes (16): curves_from_prefix_table(), FlowCurves, policy_outcomes(), run_policy(), select_threshold(), sweep_thresholds(), threshold_grid(), summarize() (+8 more)

### Community 1 - "numpy"
Cohesion: 0.07
Nodes (14): low_and_slow(), _rebuild(), _rng(), timing_jitter(), extract_packet_sequence(), prefix_feature_vector(), build_prefix_table(), eligible_counts() (+6 more)

### Community 2 - "pipeline.py"
Cohesion: 0.07
Nodes (24): split_train_calibration(), clopper_pearson(), _div(), expected_calibration_error(), prefix_metrics(), environment_record(), _get_device(), _make_cnn_summary_section() (+16 more)

### Community 3 - "App.tsx"
Cohesion: 0.06
Nodes (44): dependencies, lucide-react, react, react-dom, vite, @vitejs/plugin-react, devDependencies, @types/react (+36 more)

### Community 4 - "calibration.py"
Cohesion: 0.08
Nodes (11): Calibrator, fit_calibrator(), _logit(), oof_raw_probabilities(), fit_rf(), flow_weights(), make_rf(), CalibrationLeakage (+3 more)

### Community 5 - "test_pipeline.py"
Cohesion: 0.13
Nodes (7): Config, _cfg(), main(), print_report(), Flow, make_synthetic_flows(), CNNPipelineSeparation

### Community 6 - "pandas"
Cohesion: 0.12
Nodes (5): create_split_manifest(), save_split_manifest(), split_flows(), FlowLevelSplit, meta_of()

### Community 7 - "build_flows_from_pcap"
Cohesion: 0.12
Nodes (4): build_flows_from_pcap(), PcapReader, _State, PcapPreprocessing

### Community 9 - "test_cnn_gru.py"
Cohesion: 0.17
Nodes (3): _build_sequence_array(), build_sequence_dataset(), score_flows_at_checkpoint()

### Community 10 - "compilerOptions"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+9 more)

### Community 12 - "test_prefixes.py"
Cohesion: 0.19
Nodes (5): convert_pcapng_to_pcap(), parse_tcp_packet(), _tcp_frame(), write_pcap(), _write_pcapng_file()

### Community 13 - "preprocessing.py"
Cohesion: 0.21
Nodes (4): _ip_str(), load_flows(), PcapFormatError, save_flows()

### Community 14 - "train_cnn_gru"
Cohesion: 0.15
Nodes (4): collate_sequences(), SequenceDataset, train_cnn_gru(), TrainingSmokeTest

### Community 15 - "build_flows"
Cohesion: 0.20
Nodes (5): build_flows(), match_label_rule(), make_pcap_tree(), PcapEndToEnd, _write_single_pcapng()

### Community 17 - "build_cnn_prefix_table"
Cohesion: 0.21
Nodes (3): build_cnn_prefix_table(), raw_proba_cnn(), PrefixTableIntegration

### Community 18 - "inspect_dataset.py"
Cohesion: 0.24
Nodes (5): create_dataset_manifest(), inspect_csv(), inspect_dir(), inspect_pcap(), save_report()

### Community 20 - "CnnGruClassifier"
Cohesion: 0.38
Nodes (4): CnnGruClassifier, load_model(), save_model(), CheckpointSaveLoad

## Knowledge Gaps
- **8 isolated node(s):** `@types/react`, `@types/react-dom`, `typescript`, `UFU-DoH-EXF Dataset`, `Traceglass UI` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 206 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_experiment()` connect `pipeline.py` to `policy_outcomes`, `numpy`, `calibration.py`, `test_pipeline.py`, `pandas`, `Pipeline`, `preprocessing.py`, `build_flows`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `SequenceNormalizer` (e.g. with `fit_calibrator_cnn()` and `build_cnn_prefix_table()`) actually correct?**
  _`SequenceNormalizer` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `@types/react`, `@types/react-dom`, `typescript` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `policy_outcomes` be split into smaller, more focused modules?**
  _Cohesion score 0.05921325051759834 - nodes in this community are weakly interconnected._
- **Why does `policy_outcomes()` connect `policy_outcomes` to `numpy`, `pipeline.py`, `pandas`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.06821787414066631 - nodes in this community are weakly interconnected._
- **Why does `ModelShapes` connect `ModelShapes` to `test_cnn_gru.py`, `CnnGruClassifier`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._