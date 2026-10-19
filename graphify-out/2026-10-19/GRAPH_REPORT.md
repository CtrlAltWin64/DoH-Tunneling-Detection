# Graph Report - DoH-Tunneling-Detection  (2026-10-19)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 910 nodes · 2248 edges · 68 communities (36 shown, 32 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 45 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c38aac28`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- policy_outcomes
- run_experiment
- build_prefix_table
- verify_saved_evidence.py
- App.tsx
- numpy
- Config
- pathlib
- source_snapshot/src/pipeline.py
- _run_cnn_gru_experiment
- build_flows_from_pcap
- source_snapshot/src/preprocessing.py
- compilerOptions
- build_cnn_prefix_table
- src/pipeline.py
- source_snapshot/src/inspect_dataset.py
- _run_cnn_gru_experiment
- src/train_baseline.py
- CnnGruClassifier
- SequenceNormalizer
- test_splitting.py
- split_flows
- test_rf_persistence.py
- main.py
- test_cnn_gru.py
- PcapReader
- ModelShapes
- run_experiments.py
- source_snapshot/src/cnn_gru_model.py
- test_pipeline_bundles.py
- train_cnn_gru
- Calibrator
- source_snapshot/src/plots.py
- src/cnn_gru_model.py
- SequenceNormalizer
- build_prefix_table
- Research Results — Preliminary Within-Capture Evaluation
- Tensor
- source_snapshot/src/synthetic.py
- build_sequence_dataset
- source_snapshot/src/dataset_split.py
- source_snapshot/config.py
- fit_calibrator_cnn
- source_snapshot/src/evasion.py
- PcapReader
- MaskedBatchNorm1d
- Cli
- Main Performance Metrics Table
- SequenceDataset
- _plot_rf_vs_cnn_comparison
- _plot_rf_vs_cnn_comparison
- Observation Cost and Latency Chart
- Adaptive Stopping Distribution Chart
- Local Dataset Inventory
- Traceglass Dashboard
- Traceglass — Adaptive DoH Detection UI
- Repository Packaging
- Precision, Recall, and F1 Chart
- matplotlib
- numpy
- pandas
- scikit-learn
- scipy
- torch

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

### Community 0 - "policy_outcomes"
Cohesion: 0.07
Nodes (13): FlowCurves, policy_outcomes(), run_policy(), select_threshold(), sweep_thresholds(), threshold_grid(), summarize(), make_curves() (+5 more)

### Community 1 - "run_experiment"
Cohesion: 0.05
Nodes (13): Calibrator, fit_calibrator(), _logit(), oof_raw_probabilities(), run_experiment(), make_synthetic_flows(), PipelineBundles, CalibrationLeakage (+5 more)

### Community 2 - "build_prefix_table"
Cohesion: 0.07
Nodes (13): low_and_slow(), _rebuild(), _rng(), timing_jitter(), extract_packet_sequence(), prefix_feature_vector(), build_prefix_table(), eligible_counts() (+5 more)

### Community 3 - "verify_saved_evidence.py"
Cohesion: 0.08
Nodes (26): check(), ci(), main(), md(), metrics(), save(), table(), check() (+18 more)

### Community 4 - "App.tsx"
Cohesion: 0.06
Nodes (44): dependencies, lucide-react, react, react-dom, vite, @vitejs/plugin-react, devDependencies, @types/react (+36 more)

### Community 5 - "numpy"
Cohesion: 0.11
Nodes (9): _logit(), CnnGruClassifier, build_cnn_prefix_table(), extract_packet_sequence(), prefix_feature_vector(), fit_rf(), flow_weights(), make_rf() (+1 more)

### Community 6 - "Config"
Cohesion: 0.11
Nodes (9): Config, create_dataset_manifest(), inspect_csv(), inspect_dir(), inspect_pcap(), save_report(), make_pcap_tree(), PcapEndToEnd (+1 more)

### Community 7 - "pathlib"
Cohesion: 0.17
Nodes (5): convert_pcapng_to_pcap(), Flow, parse_tcp_packet(), _tcp_frame(), write_pcap()

### Community 8 - "source_snapshot/src/pipeline.py"
Cohesion: 0.16
Nodes (13): split_flows(), clopper_pearson(), _div(), expected_calibration_error(), prefix_metrics(), summarize(), _get_device(), _make_cnn_summary_section() (+5 more)

### Community 9 - "_run_cnn_gru_experiment"
Cohesion: 0.15
Nodes (10): curves_from_prefix_table(), FlowCurves, policy_outcomes(), run_policy(), select_threshold(), sweep_thresholds(), threshold_grid(), _run_cnn_gru_experiment() (+2 more)

### Community 10 - "build_flows_from_pcap"
Cohesion: 0.13
Nodes (4): build_flows_from_pcap(), _State, PcapPreprocessing, _write_pcapng_file()

### Community 11 - "source_snapshot/src/preprocessing.py"
Cohesion: 0.15
Nodes (9): build_flows(), build_flows_from_pcap(), _ip_str(), load_flows(), match_label_rule(), parse_tcp_packet(), save_flows(), _State (+1 more)

### Community 12 - "compilerOptions"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+9 more)

### Community 13 - "build_cnn_prefix_table"
Cohesion: 0.16
Nodes (4): fit_calibrator_cnn(), build_cnn_prefix_table(), raw_proba_cnn(), PrefixTableIntegration

### Community 14 - "src/pipeline.py"
Cohesion: 0.21
Nodes (10): environment_record(), environment_record(), _get_device(), _make_cnn_summary_section(), _make_rf_summary_section(), md_table(), _policy_set(), _run_experiment() (+2 more)

### Community 15 - "source_snapshot/src/inspect_dataset.py"
Cohesion: 0.17
Nodes (7): create_dataset_manifest(), inspect_csv(), inspect_dir(), inspect_pcap(), print_report(), save_report(), PcapFormatError

### Community 16 - "_run_cnn_gru_experiment"
Cohesion: 0.18
Nodes (7): clopper_pearson(), _div(), expected_calibration_error(), prefix_metrics(), _run_cnn_gru_experiment(), pred_frame(), Statistics

### Community 17 - "src/train_baseline.py"
Cohesion: 0.17
Nodes (4): fit_rf(), flow_weights(), make_rf(), RfBundle

### Community 18 - "CnnGruClassifier"
Cohesion: 0.21
Nodes (5): CnnGruClassifier, load_model(), save_model(), CheckpointSaveLoad, SeparateCalibrationTests

### Community 19 - "SequenceNormalizer"
Cohesion: 0.21
Nodes (3): _pos_weight(), SequenceNormalizer, SequenceNormalizerTests

### Community 20 - "test_splitting.py"
Cohesion: 0.18
Nodes (4): create_split_manifest(), save_split_manifest(), split_train_calibration(), meta_of()

### Community 22 - "test_rf_persistence.py"
Cohesion: 0.19
Nodes (4): load_rf_bundle(), raw_proba(), save_rf_bundle(), RfPersistence

### Community 23 - "main.py"
Cohesion: 0.23
Nodes (6): _cfg(), main(), build_flows(), load_flows(), match_label_rule(), save_flows()

### Community 27 - "run_experiments.py"
Cohesion: 0.23
Nodes (6): dump(), prepare(), sha(), train(), save_split_manifest(), curves_from_prefix_table()

### Community 28 - "source_snapshot/src/cnn_gru_model.py"
Cohesion: 0.18
Nodes (4): _build_sequence_array(), build_sequence_dataset(), collate_sequences(), score_flows_at_checkpoint()

### Community 30 - "train_cnn_gru"
Cohesion: 0.18
Nodes (3): train_cnn_gru(), TrainingSmokeTest, TrainingArtifactPersistence

### Community 31 - "Calibrator"
Cohesion: 0.24
Nodes (3): Calibrator, fit_calibrator(), oof_raw_probabilities()

### Community 32 - "source_snapshot/src/plots.py"
Cohesion: 0.33
Nodes (6): grouped_bars(), _need(), plot_prf_vs_prefix(), plot_robustness(), plot_val_fpr_vs_threshold(), _save()

### Community 33 - "src/cnn_gru_model.py"
Cohesion: 0.18
Nodes (3): collate_sequences(), MaskedBatchNorm1d, SequenceDataset

### Community 34 - "SequenceNormalizer"
Cohesion: 0.29
Nodes (4): load_model(), save_model(), SequenceNormalizer, train_cnn_gru()

### Community 35 - "build_prefix_table"
Cohesion: 0.24
Nodes (4): build_prefix_table(), eligible_counts(), generate_flow_prefixes(), _validate_checkpoints()

### Community 36 - "Research Results — Preliminary Within-Capture Evaluation"
Cohesion: 0.22
Nodes (9): Dataset and Preparation, Adaptive Stopping for Early Detection of DoH Tunneling, Final Cleanup Verification, Checkpoint Coverage Chart, Confusion Matrices Chart, FPR Confidence Intervals Chart, Research Results — Preliminary Within-Capture Evaluation, Full-project Research Audit (+1 more)

### Community 38 - "source_snapshot/src/synthetic.py"
Cohesion: 0.28
Nodes (4): Flow, make_synthetic_flows(), _tcp_frame(), write_pcap()

### Community 39 - "build_sequence_dataset"
Cohesion: 0.28
Nodes (3): _build_sequence_array(), build_sequence_dataset(), SequenceBuilding

### Community 43 - "source_snapshot/src/evasion.py"
Cohesion: 0.43
Nodes (4): low_and_slow(), _rebuild(), _rng(), timing_jitter()

### Community 47 - "Main Performance Metrics Table"
Cohesion: 0.40
Nodes (5): FPR Confidence Intervals Chart, ROC and Precision-Recall Curves, Capture Inventory Table, Checkpoint Eligibility Table, Main Performance Metrics Table

## Knowledge Gaps
- **27 isolated node(s):** `@types/react`, `@types/react-dom`, `typescript`, `Traceglass Dashboard`, `matplotlib` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 341 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `run_experiment`, `pathlib`, `build_flows_from_pcap`, `test_rf_persistence.py`, `main.py`, `test_cnn_gru.py`, `run_experiments.py`, `test_pipeline_bundles.py`, `train_cnn_gru`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **What connects `@types/react`, `@types/react-dom`, `typescript` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `policy_outcomes` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._
- **Why does `build_flows_from_pcap()` connect `build_flows_from_pcap` to `numpy`, `pathlib`, `source_snapshot/src/preprocessing.py`, `main.py`, `PcapReader`, `run_experiments.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `run_experiment` be split into smaller, more focused modules?**
  _Cohesion score 0.05388471177944862 - nodes in this community are weakly interconnected._
- **Why does `run_experiment()` connect `run_experiment` to `policy_outcomes`, `build_prefix_table`, `verify_saved_evidence.py`, `Config`, `pathlib`, `src/pipeline.py`, `_run_cnn_gru_experiment`, `src/train_baseline.py`, `test_splitting.py`, `split_flows`, `test_rf_persistence.py`, `main.py`, `run_experiments.py`, `test_pipeline_bundles.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `build_prefix_table` be split into smaller, more focused modules?**
  _Cohesion score 0.07450980392156863 - nodes in this community are weakly interconnected._