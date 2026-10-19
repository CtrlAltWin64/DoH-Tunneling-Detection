# Graph Report - DoH-Tunneling-Detection  (2026-10-19)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 910 nodes · 2248 edges · 68 communities (36 shown, 32 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 45 edges (avg confidence: 0.92)
- Token cost: 4,247 input · 789 output

## Graph Freshness
- Built from commit: `c38aac28`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Adaptive Stopping Policies
- Model Calibration Utilities
- Evasion and Robustness Simulation
- Project Infrastructure and Metrics
- Frontend Dependencies
- CNN-GRU Model Calibration
- Dataset Inspection and Configuration
- PCAP Preprocessing and Flows
- Evaluation Metrics and Partitioning
- Flow Curve Generation
- PCAP Parsing Tests
- Flow Loading and Labeling
- TypeScript Configuration
- CNN Prefix Table Integration
- Experiment Pipeline Execution
- Dataset Manifest Generation
- CNN-GRU Experiment Evaluation
- Random Forest Baseline Training
- Model Checkpoint Persistence
- Sequence Normalization
- Dataset Splitting Logic
- Flow Partitioning Tests
- Random Forest Bundle Persistence
- CLI Entry Point
- CNN-GRU Prefix Scoring
- Streaming PCAP Reader
- Model Shape Validation
- Experiment Orchestration
- Sequence Dataset Building
- Model Verification and Bundling
- CNN-GRU Training Tests
- Probability Calibration Fitting
- Performance Visualization Plots
- CNN-GRU Model Architecture
- Model Training and Normalization
- Prefix Table Generation
- Research Audit and Results
- Causal Convolution Forward Pass
- Synthetic Data Generation
- Sequence Building Tests
- Flow-Level Data Splitting
- Central Project Configuration
- CNN Out-of-Sample Calibration
- Timestamp Perturbation Evasion
- PCAPNG Conversion Reader
- Masked Batch Normalization
- CLI Integration Tests
- Performance Metric Tables
- Sequence Dataset Interface
- Model Comparison Plots
- Adaptive Comparison Visualization
- Latency and Cost Analysis
- Stopping Distribution Analysis
- Dataset Inventory
- Traceglass Dashboard
- Detection User Interface
- Repository Packaging
- Classification Performance Charts
- Matplotlib Library
- Numpy Library
- Pandas Library
- Scikit-Learn Library
- Scipy Library
- PyTorch Library

## God Nodes (most connected - your core abstractions)
1. `_run_experiment()` - 35 edges
2. `run_experiment()` - 34 edges
3. `run_experiment()` - 33 edges
4. `SequenceNormalizer` - 31 edges
5. `policy_outcomes()` - 28 edges
6. `build_prefix_table()` - 27 edges
7. `Config` - 26 edges
8. `CnnGruClassifier` - 24 edges
9. `_run_cnn_gru_experiment()` - 23 edges
10. `train_cnn_gru()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `PipelineBundles` --uses--> `Config`  [INFERRED]
  tests/test_pipeline_bundles.py → config.py
- `PcapPreprocessing` --uses--> `Config`  [INFERRED]
  tests/test_prefixes.py → config.py
- `PcapPreprocessing` --uses--> `PcapFormatError`  [INFERRED]
  tests/test_prefixes.py → src/preprocessing.py
- `PcapPreprocessing` --uses--> `PcapReader`  [INFERRED]
  tests/test_prefixes.py → src/preprocessing.py
- `ModelShapes` --uses--> `CnnGruClassifier`  [INFERRED]
  tests/test_cnn_gru.py → src/cnn_gru_model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Dataset Documentation and Inventory** — dataset_and_preparation_md, data_readme_md, reports_full_project_audit_md [EXTRACTED 0.90]
- **Efficiency and Latency Analysis** — reports_final_experiment_results_20261019_tables_observation_cost_latency, reports_final_experiment_results_20261019_figures_observation_cost_and_latency, reports_final_experiment_results_20261019_figures_stopping_distribution [EXTRACTED 0.95]
- **Performance Evaluation Suite** — reports_final_experiment_results_20261019_tables_main_performance, reports_final_experiment_results_20261019_figures_precision_recall_f1, reports_final_experiment_results_20261019_figures_roc_precision_recall [EXTRACTED 0.95]
- **Research Reporting and Verification Flow** — readme_md, results_guide_md, reports_final_experiment_results_20261019_results_summary_md, reports_cleanup_verification_20261019_verification_md [EXTRACTED 0.95]
- **Data Science and ML Stack** — requirements_numpy, requirements_pandas, requirements_scikit_learn, requirements_scipy, requirements_matplotlib, requirements_torch [INFERRED 0.90]

## Communities (68 total, 32 thin omitted)

### Community 0 - "Adaptive Stopping Policies"
Cohesion: 0.07
Nodes (13): FlowCurves, policy_outcomes(), run_policy(), select_threshold(), sweep_thresholds(), threshold_grid(), summarize(), make_curves() (+5 more)

### Community 1 - "Model Calibration Utilities"
Cohesion: 0.05
Nodes (13): Calibrator, fit_calibrator(), _logit(), oof_raw_probabilities(), run_experiment(), make_synthetic_flows(), PipelineBundles, CalibrationLeakage (+5 more)

### Community 2 - "Evasion and Robustness Simulation"
Cohesion: 0.07
Nodes (13): low_and_slow(), _rebuild(), _rng(), timing_jitter(), extract_packet_sequence(), prefix_feature_vector(), build_prefix_table(), eligible_counts() (+5 more)

### Community 3 - "Project Infrastructure and Metrics"
Cohesion: 0.08
Nodes (26): check(), ci(), main(), md(), metrics(), save(), table(), check() (+18 more)

### Community 4 - "Frontend Dependencies"
Cohesion: 0.06
Nodes (44): dependencies, lucide-react, react, react-dom, vite, @vitejs/plugin-react, devDependencies, @types/react (+36 more)

### Community 5 - "CNN-GRU Model Calibration"
Cohesion: 0.11
Nodes (9): _logit(), CnnGruClassifier, build_cnn_prefix_table(), extract_packet_sequence(), prefix_feature_vector(), fit_rf(), flow_weights(), make_rf() (+1 more)

### Community 6 - "Dataset Inspection and Configuration"
Cohesion: 0.11
Nodes (9): Config, create_dataset_manifest(), inspect_csv(), inspect_dir(), inspect_pcap(), save_report(), make_pcap_tree(), PcapEndToEnd (+1 more)

### Community 7 - "PCAP Preprocessing and Flows"
Cohesion: 0.17
Nodes (5): convert_pcapng_to_pcap(), Flow, parse_tcp_packet(), _tcp_frame(), write_pcap()

### Community 8 - "Evaluation Metrics and Partitioning"
Cohesion: 0.16
Nodes (13): split_flows(), clopper_pearson(), _div(), expected_calibration_error(), prefix_metrics(), summarize(), _get_device(), _make_cnn_summary_section() (+5 more)

### Community 9 - "Flow Curve Generation"
Cohesion: 0.15
Nodes (10): curves_from_prefix_table(), FlowCurves, policy_outcomes(), run_policy(), select_threshold(), sweep_thresholds(), threshold_grid(), _run_cnn_gru_experiment() (+2 more)

### Community 10 - "PCAP Parsing Tests"
Cohesion: 0.13
Nodes (4): build_flows_from_pcap(), _State, PcapPreprocessing, _write_pcapng_file()

### Community 11 - "Flow Loading and Labeling"
Cohesion: 0.15
Nodes (9): build_flows(), build_flows_from_pcap(), _ip_str(), load_flows(), match_label_rule(), parse_tcp_packet(), save_flows(), _State (+1 more)

### Community 12 - "TypeScript Configuration"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+9 more)

### Community 13 - "CNN Prefix Table Integration"
Cohesion: 0.16
Nodes (4): fit_calibrator_cnn(), build_cnn_prefix_table(), raw_proba_cnn(), PrefixTableIntegration

### Community 14 - "Experiment Pipeline Execution"
Cohesion: 0.21
Nodes (10): environment_record(), environment_record(), _get_device(), _make_cnn_summary_section(), _make_rf_summary_section(), md_table(), _policy_set(), _run_experiment() (+2 more)

### Community 15 - "Dataset Manifest Generation"
Cohesion: 0.17
Nodes (7): create_dataset_manifest(), inspect_csv(), inspect_dir(), inspect_pcap(), print_report(), save_report(), PcapFormatError

### Community 16 - "CNN-GRU Experiment Evaluation"
Cohesion: 0.18
Nodes (7): clopper_pearson(), _div(), expected_calibration_error(), prefix_metrics(), _run_cnn_gru_experiment(), pred_frame(), Statistics

### Community 17 - "Random Forest Baseline Training"
Cohesion: 0.17
Nodes (4): fit_rf(), flow_weights(), make_rf(), RfBundle

### Community 18 - "Model Checkpoint Persistence"
Cohesion: 0.21
Nodes (5): CnnGruClassifier, load_model(), save_model(), CheckpointSaveLoad, SeparateCalibrationTests

### Community 19 - "Sequence Normalization"
Cohesion: 0.21
Nodes (3): _pos_weight(), SequenceNormalizer, SequenceNormalizerTests

### Community 20 - "Dataset Splitting Logic"
Cohesion: 0.18
Nodes (4): create_split_manifest(), save_split_manifest(), split_train_calibration(), meta_of()

### Community 22 - "Random Forest Bundle Persistence"
Cohesion: 0.19
Nodes (4): load_rf_bundle(), raw_proba(), save_rf_bundle(), RfPersistence

### Community 23 - "CLI Entry Point"
Cohesion: 0.23
Nodes (6): _cfg(), main(), build_flows(), load_flows(), match_label_rule(), save_flows()

### Community 27 - "Experiment Orchestration"
Cohesion: 0.23
Nodes (6): dump(), prepare(), sha(), train(), save_split_manifest(), curves_from_prefix_table()

### Community 28 - "Sequence Dataset Building"
Cohesion: 0.18
Nodes (4): _build_sequence_array(), build_sequence_dataset(), collate_sequences(), score_flows_at_checkpoint()

### Community 30 - "CNN-GRU Training Tests"
Cohesion: 0.18
Nodes (3): train_cnn_gru(), TrainingSmokeTest, TrainingArtifactPersistence

### Community 31 - "Probability Calibration Fitting"
Cohesion: 0.24
Nodes (3): Calibrator, fit_calibrator(), oof_raw_probabilities()

### Community 32 - "Performance Visualization Plots"
Cohesion: 0.33
Nodes (6): grouped_bars(), _need(), plot_prf_vs_prefix(), plot_robustness(), plot_val_fpr_vs_threshold(), _save()

### Community 33 - "CNN-GRU Model Architecture"
Cohesion: 0.18
Nodes (3): collate_sequences(), MaskedBatchNorm1d, SequenceDataset

### Community 34 - "Model Training and Normalization"
Cohesion: 0.29
Nodes (4): load_model(), save_model(), SequenceNormalizer, train_cnn_gru()

### Community 35 - "Prefix Table Generation"
Cohesion: 0.24
Nodes (4): build_prefix_table(), eligible_counts(), generate_flow_prefixes(), _validate_checkpoints()

### Community 36 - "Research Audit and Results"
Cohesion: 0.22
Nodes (9): Dataset and Preparation, Adaptive Stopping for Early Detection of DoH Tunneling, Final Cleanup Verification, Checkpoint Coverage Chart, Confusion Matrices Chart, FPR Confidence Intervals Chart, Research Results — Preliminary Within-Capture Evaluation, Full-project Research Audit (+1 more)

### Community 38 - "Synthetic Data Generation"
Cohesion: 0.28
Nodes (4): Flow, make_synthetic_flows(), _tcp_frame(), write_pcap()

### Community 39 - "Sequence Building Tests"
Cohesion: 0.28
Nodes (3): _build_sequence_array(), build_sequence_dataset(), SequenceBuilding

### Community 43 - "Timestamp Perturbation Evasion"
Cohesion: 0.43
Nodes (4): low_and_slow(), _rebuild(), _rng(), timing_jitter()

### Community 47 - "Performance Metric Tables"
Cohesion: 0.40
Nodes (5): FPR Confidence Intervals Chart, ROC and Precision-Recall Curves, Capture Inventory Table, Checkpoint Eligibility Table, Main Performance Metrics Table

## Knowledge Gaps
- **27 isolated node(s):** `@types/react`, `@types/react-dom`, `typescript`, `Traceglass Dashboard`, `matplotlib` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 341 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Dataset Inspection and Configuration` to `Model Calibration Utilities`, `PCAP Preprocessing and Flows`, `PCAP Parsing Tests`, `Random Forest Bundle Persistence`, `CLI Entry Point`, `CNN-GRU Prefix Scoring`, `Experiment Orchestration`, `Model Verification and Bundling`, `CNN-GRU Training Tests`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **What connects `@types/react`, `@types/react-dom`, `typescript` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Adaptive Stopping Policies` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._
- **Why does `build_flows_from_pcap()` connect `PCAP Parsing Tests` to `CNN-GRU Model Calibration`, `PCAP Preprocessing and Flows`, `Flow Loading and Labeling`, `CLI Entry Point`, `Streaming PCAP Reader`, `Experiment Orchestration`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `Model Calibration Utilities` be split into smaller, more focused modules?**
  _Cohesion score 0.05388471177944862 - nodes in this community are weakly interconnected._
- **Why does `run_experiment()` connect `Model Calibration Utilities` to `Adaptive Stopping Policies`, `Evasion and Robustness Simulation`, `Project Infrastructure and Metrics`, `Dataset Inspection and Configuration`, `PCAP Preprocessing and Flows`, `Experiment Pipeline Execution`, `CNN-GRU Experiment Evaluation`, `Random Forest Baseline Training`, `Dataset Splitting Logic`, `Flow Partitioning Tests`, `Random Forest Bundle Persistence`, `CLI Entry Point`, `Experiment Orchestration`, `Model Verification and Bundling`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `Evasion and Robustness Simulation` be split into smaller, more focused modules?**
  _Cohesion score 0.07450980392156863 - nodes in this community are weakly interconnected._