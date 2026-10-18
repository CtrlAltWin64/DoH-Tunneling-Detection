# Full-project research audit: Adaptive DoH tunneling detection

**Audit date:** 18 October 2026. **Scope:** all three members' research pipeline, tests, documentation, saved experiments, dataset assets, and the frontend's relationship to research results.

**Repository:** `C:\Users\vemba\Desktop\DoH-Tunneling-Detection`.

This is an audit, not an implementation change. Findings distinguish **confirmed defects**, **observed limitations**, **hypotheses requiring investigation**, and **recommendations**. Source references below are repository-relative paths with one-based line numbers; experiment paths are repository-relative unless stated otherwise.

The updated master task supersedes the earlier requested report name. Only this report is created; `reports/feasibility_audit.md` is not created. Evidence from checks already completed earlier in this same audit is retained. No training, preprocessing, or CNN calibration operation was resumed after the master instruction. In particular, the earlier in-memory calibration diagnostic had already finished, in 3.64 seconds, before that instruction.

## 1. Executive summary and readiness verdict

**The implementation supports useful exploratory experiments, but the current real-data results do not establish unseen-capture generalization, a population FPR below 1%, or a general advantage of adaptive stopping over well-chosen fixed windows. It is not ready to be presented as a completed, statistically validated detection study.**

Important positive findings:

- Both completed real-data experiments have distinct RF and CNN-GRU probability tables. Their identical headline metrics in the `(10, 20)` experiment are supported by different probabilities, different thresholds, and different robustness responses. There is no evidence that the two result tables were simply reused across models.
- Recomputing normal validation/test policy metrics from saved probabilities and actual flow metadata reproduces both models' tables in both completed runs, within `1e-8` numerical tolerance. Saved validation threshold choices also satisfy the implemented complete-policy empirical-FPR rule.
- The pipeline separates CNN fitting, early stopping, calibration, threshold validation, and test **by flow** in the completed runs. The normalizer matches training-fit data. Test labels are not used to choose weights or thresholds in the inspected execution path.
- The existing suite completed with **128 passed and 2 subtests passed in 135.94 seconds**. This includes synthetic integration checks, not validation of the real-data research conclusions.
- A small native-PyTorch Integrated Gradients diagnostic works on the actual saved CNN-GRU. Attributions are finite, repeated results match exactly, and the explained input uses only the selected prefix.

Principal blockers:

1. **All five processed captures occur in train, validation, and test.** There are zero held-out capture sessions in these completed runs. Flow disjointness is not capture independence.
2. The new files are **two benign PCAPNG captures**, not additional attacks. Available captures increase from **2 benign / 3 malicious to 4 benign / 3 malicious**. This still cannot satisfy the current splitter's five-captures-per-class requirement, nor the five distinct CNN research roles with both classes independently represented.
3. At terminal checkpoint 20, only **20,722 / 138,098 malicious flows (15.01%)** are eligible. At checkpoints 30 and 40, **no malicious flow** is eligible. The default `(10,20,30,40)` configuration cannot support this dataset's intended two-class policy experiment.
4. Zero observed adaptive false positives among 131 or 127 benign test flows does **not** establish population FPR below 1%. Even an optimistic independent-flow binomial calculation gives one-sided 95% upper bounds of **2.26%** and **2.33%**; shared captures undermine the independence assumption further.
5. Training-generated CNN checkpoints cannot be loaded by the public `load_model()` function without manual reconstruction. RF estimators and CNN calibrators are not persisted. This blocks a complete deployable/reproducible model bundle and direct RF SHAP on the reported fitted model.

**Recommended direction:** preserve these results as exploratory within-capture measurements; settle the capture-isolated protocol and short-flow population first; repair artifact persistence and documented parser edge cases; then run a new, explicitly approved real-data experiment. For XAI, use TreeSHAP for a persisted RF and prefix-specific Integrated Gradients for CNN-GRU, with GradientSHAP as a later compatibility test.

## 2. Repository state and Graphify architecture overview

### Preserved state

Verified branch: `member3-adaptive-evaluation`.

Verified HEAD: `8a2881a Merge remote repository and preserve local changes`.

The following modifications existed before this audit and remain untouched:

```text
 M README.md
 M main.py
 M src/adaptive_detector.py
 M src/pipeline.py
 M src/plots.py
 M tests/test_evasion.py
 M tests/test_metrics.py
 M tests/test_pipeline.py
 M tests/test_stopping_rule.py
?? DoH_dataset/
?? frontend/
?? graphify-out/
?? powershell.cmd
```

`.venv/`, Python/test caches, `data/raw/`, `data/processed/`, and experiment outputs under `data/results/` are ignored by existing rules. Two pre-existing inaccessible pytest temporary directories produce Git/search permission warnings; they were not removed. `powershell.cmd` is an existing wrapper invoking Windows PowerShell; it was not changed or used to bypass permissions.

The audit interpreter is `.venv\Scripts\python.exe`, Python **3.14.2**, on Windows. Observed installed versions: NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, Matplotlib 3.11.2, PyTorch 2.14.1, pytest 9.1.1. CUDA is unavailable. Neither `shap` nor `captum` is installed. No dependency was installed or changed. The saved real-run environment reports NumPy 2.5.1, so the current environment is not an exact historical lock.

### Graph-first navigation

Inspected `graphify-out/manifest.json`, `graph.json`, and `GRAPH_REPORT.md` before targeted source investigation. The graph contains **563 nodes, 1,310 edges, 27 communities**, with 97% extracted and 3% inferred edges. All 38 manifest source/document hashes matched their files; no dangling graph edge endpoints were found. The recorded commit matches HEAD. No graph refresh was necessary. Graph metadata does not authenticate experiment binaries or prove behavior.

High-connectivity functions guided the review: `run_experiment` (47 edges), `policy_outcomes` (31), `SequenceNormalizer` (29), `build_prefix_table` (25), and `_run_cnn_gru_experiment` (23). Graph traversal used an in-memory NetworkX fallback without changing Graphify artifacts.

| Responsibility | Actual execution path and interfaces | Evidence |
|---|---|---|
| Member 1 | `Config` → inspection → `PcapReader` → TCP parsing → bidirectional `Flow` objects → prefix features/tables → grouped flow assignments | `config.py:36`; `src/inspect_dataset.py:52`; `src/preprocessing.py:49,62,278,373,473`; `src/feature_extraction.py`; `src/prefix_generation.py`; `src/dataset_split.py:28` |
| Member 2, RF | Prefix table, ordered 18-feature matrix → RF → training-only OOF or capture-held-out calibration → checkpoint probabilities | `src/train_baseline.py:31,42`; `src/calibration.py:39,59`; `src/pipeline.py:388` |
| Member 2, CNN | `(batch,time,8)` prefix sequences + lengths → training-fit normalization → causal convolutions/masked normalization/packed GRU → scalar logit → sigmoid → separate calibration | `src/cnn_gru_model.py:42,92,224,263,361,521`; `src/cnn_gru_calibration.py:29`; `src/cnn_gru_prefix.py` |
| Member 3 | Model-specific probabilities → common `FlowCurves` → validation threshold selection → frozen first-crossing/fixed policies → test summaries → perturbed test inference → tables/plots | `src/adaptive_detector.py:23,38,76,142`; `src/evaluation.py:57`; `src/evasion.py:35,42`; `src/pipeline.py:103,317`; `src/plots.py` |
| Integration | CLI/configuration and output preservation; synthetic fixture generator; documentation and UI concept | `main.py:26,51`; `src/synthetic.py`; `README.md`; `frontend/README.md`; `frontend/src/App.tsx` |

Shared `Flow` fields are ID, label, source file, timestamp/length/direction arrays, total packets and total bytes. Full-flow totals remain metadata for eligibility/exposure; the RF uses an explicit feature allow-list and the CNN constructs prefix-only arrays. Source filename and full-flow byte totals are not direct classifier inputs.

## 3. Newly added files and all available capture assets

The two additions were identified by comparing actual directory contents with the processed manifest and `flows_report.json`, not by guessing from dates:

| New file: exact path | Type | Bytes | Existing processed data? |
|---|---|---:|---|
| `C:\Users\vemba\Desktop\DoH-Tunneling-Detection\DoH_dataset\File02_DoH_Benign.pcapng` | Raw PCAPNG capture, benign scenario | 287,519,176 | No |
| `C:\Users\vemba\Desktop\DoH-Tunneling-Detection\DoH_dataset\File03_DoH_Benign.pcapng` | Raw PCAPNG capture, benign scenario | 267,662,572 | No |

Full-file SHA-256 hashes:

```text
File02: 7a08c377aa3e80d021c1957cfbda59274b14020d0bc9b10b595b870bd3a52b08
File03: b9befb14e70b3223bad444668e447a934370af940ed0e3a977e84888cdc1930f
```

All seven unique captures are in `DoH_dataset/`. Five are also present, byte-for-byte identically, under `data/raw/Benign/` or `data/raw/Malicious/`:

| Capture | Class | Bytes | Existing processed flow count |
|---|---|---:|---:|
| File01_DoH_Benign.pcapng | Benign | 364,378,452 | 411 |
| File02_DoH_Benign.pcapng | Benign | 287,519,176 | Not processed |
| File03_DoH_Benign.pcapng | Benign | 267,662,572 | Not processed |
| File04_DoH_Benign.pcapng | Benign | 209,847,788 | 567 |
| File06_DoH_Exfiltration.pcap | Malicious | 40,634,319 | 9,742 |
| File07_DoH_Exfiltration.pcap | Malicious | 66,230,439 | 15,945 |
| File10_DoH_Exfiltration.pcap | Malicious | 467,388,156 | 112,411 |

Hashes for the other five unique captures:

```text
File01 dbd442b161c574b9c96358eaff2d1a1c607110739e158650b547addf2d0feffc
File04 79b9f8306178c80f73e015eacd4662ba64cc2105698466cf653d51c318f92c25
File06 2c8a984b5d0c7fc1b64ea1fd9cf22e4ec5ac14ccf6d38cf078b19a8ab4d22fb0
File07 b1ac3ec1589248dfdf2f6b4fa39a52b813ecdcf8d24326d632356f11229de7c7
File10 4893303e0fdc179e6df036cdde68250ccf6ed6947df02b5eeef9340c75fb5274
```

There are **1,703,660,902 bytes of unique capture content** (about 1.59 GiB), plus 1,148,479,154 bytes of duplicate original captures in `data/raw/`. Do not recursively ingest both directories together: path-based flow IDs would not prevent duplicate capture content from becoming separate examples. No archive extraction or download was needed.

### Compatibility and independence evidence

The repository's `inspect_pcap(path, max_packets=2000)` parsed all seven initial samples as Ethernet link type 1. Benign files are PCAPNG; malicious files are classic PCAP. These samples use microsecond timestamps. They establish initial compatibility, not whole-file structural validity.

First observed timestamps for File02 and File03 are respectively `1749352438.211475` and `1749749191.139657`. Their 2,000-packet sample spans are 36.252 and 36.592 seconds. They are distinct byte content recorded at different times. **Neither unique hashes nor different dates prove independent hosts, tools, configurations, campaigns, or statistical sampling units.** Session provenance still needs documentation.

Before the updated master instruction, a bounded in-memory parse of 50,000 packets from each addition produced:

| Capture sample | Built flows | Eligible at 2 | 4/5/8/10/15 | 20 | 30/40 |
|---|---:|---:|---:|---:|---:|
| File02 | 21 | 21 | 15 | 14 | 7 |
| File03 | 8 | 8 | 5 | 5 | 4 |

File02 recorded 31,858 non-443 packets, 7 malformed packets and 63 UDP packets skipped; File03 recorded 38,688 non-443, 5 malformed and 45 UDP packets skipped. Both explicitly reached the packet limit. These are **sample-only, boundary-flushed flows**, not full-file benign totals. Do not extrapolate their class balance or flow-length distribution to the complete additions. No processed file was saved.

### Provenance and label limits

The upstream dataset documents scenario-derived benign/exfiltration labels and preserving capture filenames. This supports the naming rules for the seven files, but does not independently certify the local bytes or individual application protocols. The upstream description distinguishes a custom HTTPS DoH exfiltration client from `dnscat2` for Do53; the local document's attribution of `iodine`, `dns2tcp`, and `dnscat2` to the DoH captures is not substantiated by that description. Local publication venue/date claims also need a verified bibliographic reference. [Upstream dataset repository](https://github.com/CborgesUFU/DNS-360-Dataset), [upstream PCAP documentation](https://raw.githubusercontent.com/CborgesUFU/DNS-360-Dataset/main/data/PCAP/README.md).

## 4. Dataset balance, checkpoint support, and split feasibility

### Existing processed population

`data/processed/flows.pkl` contains **139,076 flows: 978 benign and 138,098 malicious**. Benign prevalence is only **0.7032%**. Large malicious flow counts do not replace missing independent capture sessions. File10 contributes most malicious flows.

The saved report records 4,469,725 input packets, 818,682 skipped for port mismatch, 845 malformed, 3,613 UDP, 3,806 non-IP, 523 other non-TCP, 53 dropped retransmissions and 27 flows sorted for out-of-order timestamps. Flow counts reconcile with per-source counts and saved dataset tables. This is a reconciliation of saved processing, not a new full-capture replay.

Benign total-packet distribution: minimum 2, lower quartile 2, median 26, upper quartile 3,010, 95th percentile 3,299, maximum 3,424. Malicious: minimum 12, lower quartile 15, median 17, upper quartile 18, 95th percentile 22, maximum 29. Stored arrays cap at 200 packets; total counts/bytes continue beyond the cap.

Each eligible flow contributes exactly one prefix row at each checkpoint. Counts recalculated from stored arrays:

| Checkpoint | Benign prefixes/eligible flows | Malicious prefixes/eligible flows | Excluded benign | Excluded malicious |
|---:|---:|---:|---:|---:|
| 2 | 978 | 138,098 | 0 | 0 |
| 4 | 661 | 138,098 | 317 | 0 |
| 5 | 661 | 138,098 | 317 | 0 |
| 8 | 659 | 138,098 | 319 | 0 |
| 10 | 659 | 138,098 | 319 | 0 |
| 15 | 658 | 138,094 | 320 | 4 |
| 20 | 654 | 20,722 | 324 | 117,376 |
| 30 | 465 | 0 | 513 | 138,098 |
| 40 | 435 | 0 | 543 | 138,098 |

All 18 RF features and eight CNN features can be computed from two packets onward. At two packets, IAT variance is zero because only one interval exists; several aggregate/cumulative features are redundant. Both saved CNN models produced finite diagnostic logits at 2, 4, 5, 8, 10, 15, 20, 30, and 40 on eligible sample inputs. This establishes shape support, **not validated performance or calibration at checkpoints on which that model was not trained**. Left padding permits a two-packet input despite the stale config comment that kernel width must not exceed the smallest checkpoint (`config.py:92`).

### Checkpoint choice

| Candidate | Data support and judgment |
|---|---|
| `(10,20)` | Implemented and completed; terminal cohort excludes 84.99% of malicious flows. Suitable only for an explicitly long-flow conditional analysis. |
| `(5,10,15,20)` | Mechanically supported, but terminal 20 gives exactly the same severe selection problem. More intermediate checkpoints do not recover excluded short attacks. No completed local run for this set was found. |
| `(2,4,8,10,15)` | Best coverage among the three proposed sets: terminal 15 retains 99.9971% of malicious flows, but only 67.28% of benign flows. Already has a completed exploratory run. Very early predictions may exploit handshake/session fingerprints. |
| Default `(10,20,30,40)` | No malicious terminal-eligible examples. The pipeline's two-class check rejects the intended evaluation. Do not use as the real-data default without changing the protocol. |

Choose a checkpoint schedule **before a fresh confirmatory test**, using training-data flow lengths and the research question, not the best existing test score. A terminal horizon of 15 is better supported here than 20 or 40. Retain 2/4 only with explicit handshake/host-confounding analysis; a prespecified post-handshake candidate such as 5/10/15 is worth comparing on development captures. No schedule has been changed by this audit.

The complete-policy population is the intersection of all requested checkpoints (`src/adaptive_detector.py:38`). All fixed windows and the adaptive policy then use those same flows. This is internally fair, but excludes short flows even if an early checkpoint could have detected them. Missing prefixes are not padded into fabricated observations. Whole-population coverage requires a separate policy for flows ending before the terminal horizon and explicit reporting of those cases.

### Actual role distributions

Counts below are **benign / malicious**. Outer counts exclude flows shorter than the first checkpoint; CNN inner roles partition only outer training. In flow-grouped runs, RF instead fits on all outer training and calibrates with grouped OOF scores.

| Role | `(10,20)` | `(2,4,8,10,15)` |
|---|---:|---:|
| Outer train | 395 / 82,858 | 587 / 82,858 |
| CNN train-fit | 222 / 46,607 | 331 / 46,607 |
| CNN train-calibration | 99 / 20,715 | 146 / 20,715 |
| CNN early stopping | 74 / 15,536 | 110 / 15,536 |
| Outer threshold validation | 132 / 27,620 | 195 / 27,620 |
| Outer test | 132 / 27,620 | 196 / 27,620 |
| Terminal-eligible train | 391 / 12,307 | 407 / 82,855 |
| Terminal-eligible validation | 132 / 4,283 | 124 / 27,619 |
| Terminal-eligible test | 131 / 4,132 | 127 / 27,620 |

CNN training-role prefix counts:

| Run and checkpoint | Train-fit B/M | Calibration B/M | Early stopping B/M |
|---|---:|---:|---:|
| 10/20 run, t=10 | 222 / 46,607 | 99 / 20,715 | 74 / 15,536 |
| 10/20 run, t=20 | 218 / 6,919 | 99 / 3,122 | 74 / 2,266 |
| 2–15 run, t=2 | 331 / 46,607 | 146 / 20,715 | 110 / 15,536 |
| 2–15 run, t=4 | 228 / 46,607 | 107 / 20,715 | 74 / 15,536 |
| 2–15 run, t=8 | 227 / 46,607 | 107 / 20,715 | 74 / 15,536 |
| 2–15 run, t=10 | 227 / 46,607 | 107 / 20,715 | 74 / 15,536 |
| 2–15 run, t=15 | 227 / 46,605 | 107 / 20,714 | 73 / 15,536 |

These small benign role counts limit calibration and threshold reliability despite the large total row counts. Prefix rows from one flow are correlated; they are not additional independent benign trials.

### Capture isolation

`src/dataset_split.py:28` uses five-fold `StratifiedGroupKFold` and explicitly requires at least five groups in each class. Current processed data has 2 benign / 3 malicious source groups. Adding File02 and File03 gives only 4 / 3. A direct splitter check failed on the original groups; a source-count feasibility check including the two additions still failed. It does not require full processing to establish this group-count impossibility.

All five original captures appear in all three outer partitions of both completed real runs. Flow IDs are unique/disjoint, and the union of CNN inner roles equals outer training, but capture context is shared. A manifest's `verified_disjoint: true` is conditional on `group_by: flow` and does not certify source isolation.

The current CNN protocol needs fitting, early stopping, calibration, threshold validation and test with both classes: at least five independent captures per class is a **necessary minimum**, not assurance that size-imbalanced folds will work well. At least one more benign and two more malicious captures are needed even to reach that numerical minimum; substantially more diverse sessions are preferable.

A defensible small-data alternative requires an approved protocol change: an explicitly exploratory leave-capture-out evaluation, fewer separate development roles (for example prespecified epochs instead of a separate early-stop capture), and carefully nested/cross-fitted development predictions. With only three attack captures, do not promise five independent roles or precise generalization uncertainty. Never silently switch to flow splitting and retain an unseen-session claim.

## 5. Member 1: preprocessing, labels, and feature pipeline

### Verified behavior

- Classic PCAP magic values distinguish endianness and micro/nanosecond timestamps. PCAPNG supports section/interface/enhanced-packet blocks and timestamp resolution. Initial real-capture samples and existing generated fixtures parse successfully (`src/preprocessing.py:62,154,173`).
- Supported link types are Ethernet, raw IP, Linux SLL and SLL2; unsupported classic headers raise an error. Ethernet VLAN handling exists. UDP is skipped, so the scope excludes HTTP/3/QUIC DoH. IPv6 extension-header coverage is limited.
- File rules are explicit, ordered, case-insensitive substring matches. Unknown files are skipped, `exclude` rules are recorded, and empty label rules fail loudly (`src/preprocessing.py:464,473`). The default rules are intentionally empty; the successful saved runs used explicit benign/exfiltration/malicious mappings. A broad keyword earlier in the mapping can override a later exclusion: retain precise naming and rule tests.
- Bidirectional keys use sorted endpoints; direction is relative to the first observed sender, not guaranteed client identity. Idle gaps over 120 seconds or a new SYN after observed closure split flows. Capture-start/midstream and tuple-reuse semantics need to be acknowledged.
- RF feature order is explicit, and feature construction slices arrays at `t`. Sequence features also slice at `t`. Absolute epoch timestamps, full-flow size and filenames are not model features. The two separate sequence-building implementations currently agree in their formulas; a shared parity regression test would reduce future drift.
- Prefix eligibility uses available stored arrays, not invented packets. Splitting precedes learning. Flow metadata permits full-capture byte denominators without giving them to the model.

### Confirmed edge-case defects in the current parser

These findings were reproduced with tiny in-memory capture fixtures against the current code, not inferred from a previous report. Existing source hashes remained unchanged afterward. They are **not evidence that the provided Ethernet captures actually contain these formats or ordering patterns**.

| Case | Observed result | Precise cause / smallest correction |
|---|---|---|
| Single big-endian PCAPNG section | Passed: one packet at expected timestamp | Existing initial-byte-order support works. |
| Little-endian section followed by big-endian section | Failed: only first packet emitted; later section treated as truncated | Block length is decoded using old byte order before reading the new section's byte-order magic (`preprocessing.py:183–205`). Decode section headers specially before length validation. |
| PCAPNG with Ethernet and raw-IP interfaces | Failed: reader emits two records, flow builder skips second valid packet as non-IP | Per-interface link type is recorded but parsing uses global `reader.linktype` (`preprocessing.py:210,225,391`). Carry interface link type with each packet. |
| PCAPNG interface `if_tsoffset=10`, packet timestamp 1 | Failed: observed 1 rather than 11 seconds | Timestamp offset option is ignored. Apply it per interface. |
| PCAPNG Simple Packet Block | Missing-time handling failed for latency use: emitted timestamp is fabricated `0.0` | SPB has no timestamp (`preprocessing.py:254`). Reject/flag timing-dependent use instead of treating zero as measured time. |
| Distinct TCP payload ranges arrive sequence 200–249 then 100–149 | Failed: lower range is dropped as a retransmission | High-water sequence check equates “below highest end” with “already seen” (`preprocessing.py:421–427`). Track covered ranges or conservatively retain uncertain overlaps. |

Retransmission filtering happens **before** final stable timestamp sorting (`preprocessing.py:455`). Sorting cannot restore a segment already dropped, or earlier packets arriving after the 200-packet storage cap. Partial overlaps and payload-free packets are not byte-level deduplicated. Therefore the documentation's broad “sort before aggregation” and “retransmissions removed” wording overstates the guarantee.

Malformed/truncated fixture tests pass for existing cases, and preprocessing reports counters. However, `inspect_pcap` samples the beginning and does not establish whole-file validity or fully expose a later truncated stream. A “valid” sample report should not be treated as a checksum or full structural validation.

### Does the selected traffic represent DoH tunneling?

The saved configuration selects **all TCP/443 traffic**, with no resolver allow-list and no payload/protocol verification. Capture-level labels represent the collection scenario. TCP/443 alone does not distinguish DoH from other HTTPS traffic. This threatens the semantic interpretation of “DoH detection,” even if scenario classification is accurate.

Observed diagnostic: all 138,098 malicious flows start with an IP-length-60 packet; benign first-packet lengths have median 52 and range 40–273. The first two directions are forward/reverse for all malicious flows and 977/978 benign flows. Benign sample timestamps lie in June 2025 and malicious ones in May 2026. These are **confirmed distribution differences**; learning a handshake/tool/host/session fingerprint is a **hypothesis**, not a demonstrated explanation. Packet-flag/payload-phase inspection, source-isolated tests and ablations are needed before interpreting two-packet accuracy as detection of tunneling behavior.

## 6. Member 2: model training, calibration, and saved artifacts

### Random Forest

Feature order (`src/feature_extraction.py`, consumed by `src/train_baseline.py:31,42`):

```text
n_packets, bytes_total, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
fwd_packet_frac, len_mean, len_std, len_min, len_max, fwd_len_mean,
bwd_len_mean, iat_mean, iat_std, iat_min, iat_max, duration
```

Labels are 0 benign and 1 malicious; inference locates class 1 through `model.classes_`, avoiding an assumed probability column. There is no fitted RF scaler to leak across partitions. The saved configuration uses 200 trees, `min_samples_leaf=3`, `balanced_subsample`, seeded training and inverse-prefix-count flow weights. Each training flow has equal total row weight before class weighting.

For flow grouping, calibration uses training-only, flow-grouped OOF predictions (`src/calibration.py:39`), followed by a final RF fit on outer training. For capture grouping, the pipeline deliberately uses a capture-disjoint fitting/calibration holdout because the OOF API groups only by flow (`src/pipeline.py:388`). Monotonic Platt mappings and fallback behavior are tested. This is reasonable boundary handling, but the completed runs share capture context and calibration prevalence is highly atypical.

**Persistence defect/limitation:** neither completed run contains the fitted RF estimator. RF calibrator coefficients are described in the run record, but the trees are absent. Saved scores and metrics can be audited; saved scores cannot be regenerated from the historical fitted forest. Exact historical RF reload consistency and TreeSHAP are therefore blocked. A new forest would not be an explanation of the reported model. No surrogate/replacement forest was trained in this audit.

### CNN-GRU architecture and numerical behavior

Input is `(B,T,8)` plus integer lengths; output is one scalar logit per sample. Feature order:

```text
pkt_len, direction, iat, elapsed, cum_pkts_fwd, cum_pkts_bwd,
cum_bytes_fwd, cum_bytes_bwd
```

Two left-padded convolution blocks, custom masked batch normalization, packed GRU sequences and the classification head are implemented in `src/cnn_gru_model.py:92`. Padding positions are excluded from normalization statistics and sequence lengths control packing. Existing mixed-length/padding tests pass. During training, masked batch normalization aggregates valid positions across training prefixes, so intermediate training activations are not pointwise causal across time. At inference, `eval()` uses frozen statistics; a prediction at `t` uses only its prefix. Keep this distinction explicit when claiming causal inference.

`SequenceNormalizer` is fitted only on train-fit prefix sequences. Its saved mean/std were independently recomputed from the recorded train-fit flows and matched within `1e-6` for both completed runs. Repeated positions across prefixes affect weighting of these training statistics but do not introduce test data.

Training uses float binary targets, `BCEWithLogitsLoss`, `pos_weight = n_negative/n_positive`, Adam, and gradient clipping at 5. Batch losses are accumulated with batch-size weighting, then divided by sample count (`cnn_gru_model.py:444–486`). The class weight being below one is expected when malicious is the majority. Unlike RF, CNN training does not compensate each flow for its number of available prefixes; comparison is not a controlled architecture-only experiment.

Actual saved models each have **64,193 parameters**, all finite. Bounded diagnostics produced finite logits and input gradients at all tested lengths. Batched-versus-single inference differed by at most `3.8147e-6` in logits (zero for the sampled 10/20 model). Before the updated master restriction, a single optimizer step on a disposable model copy produced finite loss `7.81526e-5`, finite gradients and changed head weights. That diagnostic used two sampled inputs, including held-out inputs; it was not experimental fitting, was discarded, and never changed a saved model or score. Historical per-step gradients were not logged, so the audit cannot retrospectively certify every training update.

### Early stopping and the apparent zero loss

`src/pipeline.py:154` prints loss with `.5f`, as do training debug messages. Saved values are positive; `0.00000` is rounding, not evidence of a zero or incorrectly reduced loss.

| Run | Actual history length | Saved best epoch/loss | Lower recorded validation loss |
|---|---:|---|---|
| `(10,20)` | 12 epochs | Epoch 5; `9.991545884e-7` | Epoch 12; `1.441829455e-7` |
| `(2,4,8,10,15)` | 17 epochs | Epoch 10; `2.977775760e-6` | Epoch 14; `2.495553653e-6` |

Confirmed cause: `val_loss < best_val_loss - 1e-6` gates both checkpoint replacement and patience (`cnn_gru_model.py:493`). Once the best loss is below `1e-6`, no nonnegative loss can satisfy that improvement condition. A minimum improvement for patience is defensible, but calling this the lowest-loss checkpoint is inaccurate. Separate “save any new minimum” from “reset patience after material improvement,” or document the intended rule. `epochs_trained` and the training log incorrectly use `best_epoch` instead of the history length (`pipeline.py:154,656`). No architecture/optimizer change is warranted merely to make the curve look different.

### Calibration boundaries and artifact reproducibility

CNN calibration is fitted on `train_cal`, excluded from train-fit and early-stop roles; outer validation selects thresholds only (`pipeline.py:118–153`; `cnn_gru_calibration.py:29`). This is flow-disjoint in the actual runs, not source-disjoint.

Both actual `cnn_gru_best.pt` files contain `epoch`, `model_state`, `val_loss`, and `normalizer`, but omit `model_kwargs`. Public `load_model()` reads that missing key at `cnn_gru_model.py:582`: **both actual checkpoints raise `KeyError: 'model_kwargs'`**. The existing round-trip test exercises `save_model()` instead, which does include kwargs, so it misses the real training-save path (`cnn_gru_model.py:496` versus `561`). Manual in-memory reconstruction from run-record architecture settings succeeds and matches the saved epoch/normalizer.

CNN calibration parameters are not serialized with the checkpoint or recorded fully in the run record. Before the master instruction, a one-off reconstruction of the 10/20 calibrator from its saved train-cal assignment completed in **3.64 seconds**, using 24,035 calibration prefix rows. On 32 validation and 32 test flows sampled across classes, reconstructed calibrated predictions matched saved values to less than `1e-18`. This supports that checkpoint's connection to the reported experiment, but **does not repair the missing calibrator artifact**. It was not repeated. The second run's calibrated-model replay was not attempted.

Minimum persistence correction: save one versioned model bundle containing architecture, weights/forest, feature order, normalizer, calibration maps, checkpoint list and frozen thresholds; test the *actual training output* through the public loader. Record input hashes, code/dirty-state hashes, dependency lock and actual epoch count. Optimizer state is additionally needed only if exact training resumption is a deliverable.

## 7. Member 3: adaptive policy and evaluation logic

### What is implemented correctly

- `run_policy` uses the **first** `p >= tau` crossing (`adaptive_detector.py:68`). Later checkpoints do not change an earlier decision.
- Validation FPR counts a benign flow once if **any** selected checkpoint alerts. Threshold sweeping uses the maximum score over the entire policy, not independent per-checkpoint 1% constraints (`adaptive_detector.py:106`).
- The lowest feasible grid threshold is selected; the grid includes an explicit never-alert sentinel above one. Thresholds are persisted before test metrics, and existing test-label-mutation checks show test labels cannot change them.
- Adaptive and every fixed policy use identical terminal-eligible flows within a run. A fixed window selects one column of the same model's scores, with its own validation-selected threshold. These are policy comparisons using shared models, not separately optimized classifiers trained only at each window.
- Confusion counts, recall/coverage, precision, F1, early/terminal alerts, observation costs, latency and DEBD recompute correctly from the saved normal score tables in both completed runs.

### Definitions and limits

`src/evaluation.py:57` defines coverage as recall over **eligible** malicious flows. An unresolved eligible malicious flow is an FN; an unresolved benign flow is a TN. Excluded short flows are outside those denominators, not silently counted as benign.

Observation cost is observed packets divided by the run's maximum checkpoint. Unresolved adaptive flows reach cost 1. A fixed-`t` policy stops observing at `t`, even without an alert, so its cost is `t/Tmax`. The README states this correctly; the blanket unresolved-cost wording in the evaluation module docstring is too broad. Cost excludes repeated feature computation and inference, so packet savings do not establish CPU/runtime savings.

Latency is first-packet-to-alert-packet elapsed capture time, averaged only over detected malicious flows. It is not end-to-end detector wall-clock latency. Misses have undefined latency, not zero; latency comparisons need detected counts and coverage. Early/terminal labels refer to the common run horizon even for shorter fixed windows.

DEBD is the proportion of counted **IP bytes in both directions**, including headers and the alert-triggering packet, observed before the decision. The denominator is all counted captured flow bytes, including beyond the stored prefix cap. Unresolved malicious flows receive 100% in all-flow variants. It is an exposure proxy, not measured exfiltrated application bytes. Truncation and retransmission handling affect it; detected-only and all-flow versions must not be interchanged.

### Evasion experiments

`src/evasion.py:35,42` applies timestamp dilation or `iat * exp(N(0,sigma²)) + U(0,a)` to malicious test flows only, with per-flow seeded randomness. RF aggregates and CNN IAT/elapsed features are rebuilt, then predictions are recomputed with frozen normalization, calibration and thresholds. Benign scores remain unchanged by design, so unchanged FPR is not evidence of resilience to benign timing shifts.

The jitter is not mean-preserving: expected IAT becomes `iat * exp(sigma²/2) + a/2`. These are offline timing simulations, not active attacks or a TCP network simulation. Size/count changes, congestion feedback, packet loss and retransmissions are not regenerated. If transformed gaps cross the 120-second flow timeout, the already-built flows are not resegmented; this is an additional scope limit.

Saved results show meaningful model differences. In the 10/20 run, strong jitter (`sigma=1`, additive maximum 0.2 seconds) reduces CNN detections to zero, while RF still detects all 4,132 eligible malicious flows. Moderate jitter gives CNN adaptive TP 2,247 and fixed-20 TP 1,431. In the 2–15 run, strongest jitter gives CNN adaptive TP **13,687** versus fixed-15 **11,121**, out of 27,620; RF remains at 27,620. These support a bounded sensitivity finding, not live-attack robustness.

## 8. Statistical reliability of the 1% FPR claim

The code implements an **empirical validation constraint**, not a population-FPR guarantee. With 132 or 124 eligible benign validation flows, one false positive is allowed by the 1% constraint. The resolution is approximately 0.758% or 0.806%; threshold estimates can change substantially with one example.

For adaptive test policies with zero false positives, `src/evaluation.py:42` implements exact Clopper–Pearson intervals. Recalculated values:

| Run | Observed FP / benign test flows | Observed FPR | Nominal two-sided 95% interval | Nominal one-sided 95% upper bound |
|---|---:|---:|---|---:|
| `(10,20)` | 0 / 131 | 0% | [0%, 2.7767%] | 2.2609% |
| `(2,4,8,10,15)` | 0 / 127 | 0% | [0%, 2.8629%] | 2.3312% |

For zero events, the one-sided upper bound is `1 - 0.05^(1/n)`. It takes at least **299 independent benign trials with zero false positives** to put this bound at or below 1%; the upper endpoint of a two-sided 95% interval requires at least **368**. These numbers are not a prescription to generate more correlated flows from the same two captures.

**Independence qualification:** the reported denominators are flow counts, not verified independent trials. Test flows come from the same two benign captures represented in training. There are **zero unseen benign capture groups** in the completed runs. A defensible exact population interval based on genuinely independent benign examples cannot be obtained from these artifacts alone. Cluster-aware/capture-level uncertainty requires more sessions and an explicit sampling target; two shared benign captures cannot establish a deployment FPR guarantee.

Correct reporting now: “Observed adaptive FPR was 0/131 (or 0/127) on a flow-held-out, terminal-eligible subset from shared captures; nominal independent-flow intervals exceed 1%, and unseen-capture generalization is untested.”

Also, some fixed policies in the 2–15 run violate even the observed test target: RF fixed-10 and fixed-15 each have **2/127 = 1.5748% FPR**, despite validation feasibility. Do not describe every policy as satisfying test FPR below 1%.

## 9. Model-specific results, artifacts, documentation, and plots

### Available experiments

| Directory under `data/results/` | Evidence/status |
|---|---|
| `exp_ufu_real_cp10_20/` | Complete real-data run; both models, 20 CSV tables, 21 figures, scores, histories and CNN checkpoint. |
| `exp_ufu_real_cp2_4_8_10_15/` | Complete real-data run; both models, 26 CSV tables, 21 figures, scores, histories and CNN checkpoint. |
| `exp_ufu_real_20261018/` | Partial: split assignment/manifest and dataset stats; not a completed experiment. No original failure log was found to establish the historical failure cause. |
| `member3_smoke_20261017/` | Existing synthetic smoke-test reference; not real-data evidence. |

No local `paper_results_v2` experiment was found among the accessible project result assets. Its capture-isolation and metrics remain unverified reference claims; they are not imported into this verdict.

### Why identical normal metrics are plausible

For `(10,20)`, RF and CNN calibrated test probabilities differ on every one of the 4,263 eligible rows at both checkpoints. Maximum absolute differences are about **0.0950 at 10** and **0.0576 at 20**. RF thresholds are adaptive/fixed-10 **0.130**, fixed-20 **0.021**; CNN thresholds are **0.001** for all three. Both models nevertheless produce TP 4,132, FN 0, FP 0, TN 131 for these normal policies.

| 10/20 normal policy, either model | Malicious observation cost | Mean detected latency | Mean all-malicious DEBD |
|---|---:|---:|---:|
| Adaptive | 0.5 | about 0.005381 s | 63.0140% |
| Fixed 10 | 0.5 | about 0.005381 s | 63.0140% |
| Fixed 20 | 1.0 | about 0.006244 s | 97.5985% |

**This run demonstrates no incremental adaptive benefit over fixed-10 on normal traffic.** A comparison only with fixed-20 can support earlier observation than that horizon, but cannot establish that adaptation itself is necessary.

In `(2,4,8,10,15)`, probabilities differ between models across all 27,747 eligible rows at all five checkpoints. Adaptive thresholds are RF **0.186** and CNN **0.011**. Both detect all 27,620 eligible malicious flows without a benign alert. RF alerts every malicious flow at two packets; CNN has one later detection. Costs are respectively **0.133333** and **0.133348**, with DEBD **3.34947%** and **3.35152%**. CNN fixed-2/fixed-4 miss one malicious flow; RF fixed-2 has one FP and fixed-4 has none. These very small differences require independent captures and uncertainty before supporting a superiority claim.

All normal validation/test summary metrics were independently reconstructed from saved scores plus flow timestamps/byte totals; no difference exceeded `1e-8`. Saved threshold sweeps' benign false-positive counts and lowest feasible thresholds were also reconstructed. RF historical-model identity remains limited by the missing forest; CNN checkpoint provenance is stronger because of the normalizer/history checks and the earlier sampled calibration replay.

### Documentation and final-deliverable discrepancies

| Evidence | Discrepancy / required reporting correction |
|---|---|
| `README.md:23,181` | Says no local real-capture execution, despite two completed real runs and local raw captures. Update status with run IDs and limitations. |
| `DATASET_AND_PREPARATION.md:6` | “Validated, fully tested” with 113 tests is stale and overbroad. Current suite is 128 tests plus two subtests; confirmed edge cases remain. |
| `DATASET_AND_PREPARATION.md:193–224` | Describes unconditional capture-group isolation. Actual default and completed runs use flow grouping. State the grouping explicitly. |
| `DATASET_AND_PREPARATION.md:112` | Sorting-before-aggregation claim conflicts with retransmission filtering and capping before final sorting. |
| `DATASET_AND_PREPARATION.md:34,42`; `README.md:34` | Publication/tool provenance requires correction or a primary citation; upstream description does not substantiate all current claims. |
| `README.md:79–94`; default config | 40-packet examples do not describe either successful real run and cannot retain malicious examples here. Separate generic defaults from measured protocols. |
| `main.py:26–85` | CLI exposes neither checkpoint override nor label-rule selection. Reproducing successful run-record configurations currently requires explicit programmatic `Config` construction or source configuration changes. Do not give a default CLI command as an exact reproduction. |
| `requirements.txt` | Lower bounds only; no lock for historical experiments. Run records omit dataset hashes and a complete dirty-source fingerprint. |

### Plot and frontend review

Inspected plotting sources and two actual saved figures: the 10/20 RF-versus-CNN adaptive comparison and 2–15 CNN robustness panel. Values and labels agree with the saved tables. The robustness panel explicitly labels simulated malicious timing changes and unchanged benign inputs. The zero-FPR comparison panel lacks uncertainty and automatically spans negative tick values; publication formatting should show a meaningful nonnegative range and interval/sample-count context.

Generic figures and RF-prefixed figures are intentional compatibility duplicates in `src/pipeline.py:569–608`, not independent measurements. Avoid presenting both as separate corroborating evidence. Current `_save` writes 150-dpi PNG (`src/plots.py:23`); publication export to PDF/SVG, readable uncertainty, cohort counts and consistent captions remains to be implemented. No plots were regenerated in this audit.

The frontend is a **standalone synthetic preview**, as stated in `frontend/README.md` and visible in `frontend/src/App.tsx:125,132–135,144,152`. Flow rows, scores and run cards are hard-coded illustrations; there is no research API/inference connection. Its “evasion lab” changes UI state rather than executing the detector. This is appropriately disclosed and is not a research result. No frontend build/browser test was run, no dependencies were changed, and no UI file was modified. Connecting actual run artifacts is optional product work after research interfaces stabilize.

## 10. Advanced test coverage and failures

**Status vocabulary:** Passed means an executed test or explicit artifact check passed within the stated scope. Failed means an observed defect or failed research-validity condition. Blocked means missing data/artifacts/dependencies. Not implemented means no automated check or pipeline capability was found. Not applicable includes justification. Passing synthetic tests does not promote real-data conclusions to validated status.

| Check | Status | Evidence / scope |
|---|---|---|
| Existing full Python suite | **Passed** | 128 passed, 2 subtests; 135.94 s. Run earlier in this audit, not repeated after master task. |
| Standard PCAP/PCAPNG generated fixtures | **Passed** | `tests/test_prefixes.py`, `tests/test_pcap_end_to_end.py`. |
| Actual local capture initial compatibility | **Passed** | 2,000-packet inspection per unique capture; bounded new-file parse. Not a full validity proof. |
| Basic malformed/truncated/empty/unsupported cases | **Passed** | Existing parser/integration tests cover these cases. |
| Mixed-endian PCAPNG sections | **Failed** | Adversarial in-memory fixture emits one of two packets. |
| Multiple link types within PCAPNG | **Failed** | Second interface's packet parsed using first interface type. |
| PCAPNG timestamp offset | **Failed** | Offset ignored. |
| Missing timestamps in Simple Packet Blocks | **Failed** | Fabricated zero timestamp is unsuitable for latency features. |
| Existing ordinary retransmission/timestamp-sort tests | **Passed** | Current synthetic suite. |
| Out-of-order nonoverlapping TCP sequence ranges | **Failed** | High-water filter drops unseen lower range. |
| Explicit labels, unknown/excluded rule handling | **Passed** | Source and existing tests; naming/scenario labels only. |
| Packet-level confirmation that every selected TCP/443 flow is DoH | **Blocked** | No per-flow protocol ground truth/resolver validation in artifacts. |
| Byte-identical capture detection | **Passed** | Full SHA-256 comparison; five mirrors identified, additions distinct. |
| Statistical independence of captures | **Blocked** | Session/host/campaign provenance insufficient. Hash uniqueness is not independence. |
| Class counts and checkpoint eligibility | **Passed** | Recalculated from 139,076 saved flows and reconciled with CSV/manifests. |
| Flow-prefix partition disjointness | **Passed** | Existing tests and saved assignments; no repeated flow IDs across roles. |
| Capture-group splitter behavior on sufficient synthetic groups | **Passed** | `tests/test_splitting.py` and pipeline role tests. |
| Capture isolation of current real results | **Failed** | Every source appears across outer partitions. This is the configured protocol, not an unreported splitter bug. |
| Capture-isolated five-role run on available captures | **Blocked** | Only 4 benign / 3 malicious captures including additions. |
| Prefix feature causality | **Passed** | Existing tail-mutation tests; additional real CNN prefix tail mutation unchanged. |
| RF versus CNN sequence-builder parity regression | **Not implemented** | Two equivalent formula implementations; no dedicated cross-module parity test located. |
| CNN shapes, variable-length packing/padding | **Passed** | `tests/test_cnn_gru.py`; bounded actual-model inference. |
| Finite actual parameters/logits/input gradients | **Passed** | Both saved models and sampled checkpoint lengths; historical gradients unavailable. |
| Optimizer changes parameters | **Passed** | Disposable one-step numerical check completed before master task; no experiment modification. |
| RF class weighting and equal-flow prefix weights | **Passed** | `tests/test_pipeline.py:182`; source/config inspected. Does not validate deployment prevalence. |
| CNN class imbalance behavior | **Passed** | BCE class ratio implementation and bounded finite-loss checks. No independent-prevalence performance test. |
| Training-fit-only CNN normalization | **Passed** | Recomputed mean/std match both saved normalizers. |
| Calibration and early-stop flow separation | **Passed** | `tests/test_pipeline.py:192`; saved inner assignments. Capture independence remains failed. |
| Calibration outputs/monotonicity/fallbacks | **Passed** | Existing calibration tests and saved bounded probabilities. |
| RF historical model reload | **Blocked** | Fitted estimator absent. |
| Dedicated CNN `save_model`/`load_model` round-trip | **Passed** | Existing helper test. |
| Actual CNN training-checkpoint public reload | **Failed** | `model_kwargs` missing in both real checkpoints. |
| Persisted CNN calibrator reload | **Not implemented** | Calibrator not saved. Earlier reconstruction is not a reload. |
| 10/20 CNN sampled calibrated-score replay | **Passed** | Earlier diagnostic, 32 flows per val/test split, error < `1e-18`; not repeated. |
| Second run full calibrated-model replay | **Blocked** | Missing saved calibrator; no refit under master task. |
| Best-checkpoint/epoch-count reporting | **Failed** | Minimum-delta selection differs from minimum history loss; best epoch misreported as epochs trained. |
| Seeded synthetic pipeline reproducibility | **Passed** | Existing pipeline test. No expensive historical real-run reproduction attempted. |
| Exact historical environment/data/code reproduction | **Blocked** | No full lock or dataset/dirty-code identity bundle; RF artifact absent. |
| RF early stopping | **Not applicable** | RF has no early-stopping procedure. |
| First crossing, sequential FPR, threshold freezing | **Passed** | Existing stopping/pipeline tests and saved sweep checks. |
| Test-label independence of threshold selection | **Passed** | `tests/test_pipeline.py:86` and inspected ordering. |
| Equal fixed/adaptive evaluation population | **Passed** | Saved IDs/denominators match within each run. |
| FPR interval arithmetic | **Passed** | Exact binomial calculation and existing metrics tests. |
| Population FPR below 1% established | **Failed** | Nominal upper bounds exceed 1%; independence also unsupported. |
| Short-flow exclusion and unresolved arithmetic | **Passed** | Existing tests and reconstructed summaries; conditional population explicitly limited. |
| Operational short-flow/end-of-flow policy | **Not implemented** | Terminal-ineligible flows are excluded from policy evaluation. |
| Observation cost/latency/DEBD arithmetic | **Passed** | All normal val/test table summaries reproduced to `1e-8`. |
| Timing perturbation determinism and feature rebuild | **Passed** | `tests/test_evasion.py`, pipeline tests, separate model robustness tables. |
| Live attacker/network robustness | **Not applicable** | Current evasion scope is offline timestamp simulation. |
| RF TreeSHAP shape/finiteness on reported forest | **Blocked** | SHAP and historical fitted forest absent. |
| CNN Deep/Gradient/Kernel SHAP execution | **Blocked** | SHAP absent; architecture-specific risks remain. |
| Captum implementation test | **Blocked** | Captum absent; native PyTorch IG is distinct evidence. |
| Native CNN IG shape/finiteness/repeatability | **Passed** | Actual saved model, `(2,10,8)` and `(2,20,8)`; repeated t=10 values identical. |
| IG convergence suitable for publication | **Not implemented** | Two quadrature budgets checked; further baseline/convergence analysis required. |
| Automated attribution prefix/mask/additivity regression | **Not implemented** | Ad-hoc prefix/shape checks only; no repository XAI suite. |
| Saved metrics versus inspected figures | **Passed** | Source/table tracing and visual inspection of two figures; no universal pixel/CSV regression test. |
| CI workflow | **Not implemented** | No tracked CI configuration or `.github/workflows` found. Tests are local. |
| Frontend research-model integration | **Not implemented** | Intentionally synthetic preview; frontend runtime/build not tested in this audit. |

## 11. XAI/SHAP compatibility with the actual models

No SHAP/Captum package was installed, and no new forest or CNN was trained for explanations. “Feasible” below is not equivalent to an executed package compatibility test.

| Approach | Actual evidence and compatibility judgment | Minimal adapter; recommended scope |
|---|---|---|
| RF TreeExplainer | Supported estimator family in SHAP, but **blocked locally** by absent SHAP and absent saved fitted RF. Actual SHAP values/shapes have not been tested. | Persist historical/future fitted forest; use exact 18-column order, class-1 output, per-checkpoint training background. No core architecture change. |
| CNN DeepExplainer | **Not recommended as first choice.** Not installed/tested. SHAP's PyTorch handlers do not provide dedicated GRU or this custom masked-normalization support; unsupported-module paths require strict validation. | Fixed-prefix single-output wrapper; require additivity and gradient tests before accepting values. Do not treat successful construction as correctness. |
| CNN GradientExplainer | **Plausible, not verified.** Native input gradients through saved GRU work, but this does not prove SHAP's interface/operations work. | Fixed-`t` tensor-only wrapper returning `(B,1)`, fixed lengths, eval mode; small training backgrounds; seeded samples. |
| CNN KernelExplainer | **Plausible tiny fallback, not executed.** Model-agnostic inference avoids backward-operation support issues, but cost and unrealistic feature coalitions are substantial. | Flatten `T×8`, reshape inside batched prediction function, keep `t` fixed; only 1–2 explanations initially. |
| CNN Integrated Gradients | **Native PyTorch POC passed on actual saved model.** Captum not installed/tested. Strongest direct local evidence of feasibility. | Direct differentiable forward with fixed lengths; raw-logit target; train-derived baseline; convergence/repeatability checks. No model architecture edit. |

TreeExplainer supports tree ensembles and probability explanations with an interventional background. Select the malicious class using `model.classes_`; do not blindly assume older binary-output list conventions. Verify actual returned dimensions, expected value plus attribution sum, and agreement with `predict_proba`. [TreeExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

For RF, explain the forest's **uncalibrated class-1 probability** and label it accordingly. TreeSHAP for a forest is not automatically SHAP for the nonlinear Platt-calibrated composite. An explanation of a calibrated probability needs a separately validated composite approach; displaying calibrated score alongside raw-forest attributions must not imply matching additive units.

SHAP's current PyTorch DeepExplainer source lacks dedicated handlers for this GRU/custom normalization combination; generic backward behavior or warnings are not an assurance of correct DeepLIFT rules. GradientExplainer instead uses expected gradients and requires a compatible differentiable output. [DeepExplainer PyTorch implementation](https://github.com/shap/shap/blob/main/shap/explainers/_deep/deep_pytorch.py), [GradientExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.GradientExplainer.html).

### Actual CNN Integrated Gradients diagnostic

Used the 10/20 saved checkpoint, manually reconstructed from saved kwargs/configuration, its saved normalizer and `eval()` mode. Target was the **raw malicious logit**, not calibrated probability. Baseline was the mean normalized prefix of eight benign train-fit flows; the first eight happened to come from File01, so this is a small, biased engineering baseline, not a justified global reference population. Explained two terminal-eligible held-out flows, one per class. Neither explanations nor labels were used to tune thresholds.

Gauss–Legendre integration used 64 and 128 points:

| Prefix | Attribution shape | Finite | 128-point completeness residuals, benign / malicious | Maximum coordinate change, 64→128 | Measured combined 64+128 diagnostic time |
|---|---|---|---|---:|---:|
| 10 | `(2,10,8)` | Yes | `-2.80e-5` / `+0.013895` | 0.01829 | 1.075 s |
| 20 | `(2,20,8)` | Yes | `-3.37e-6` / `-0.052202` | 0.18143 | 1.831 s |

Malicious baseline-to-input logit differences were about 19.37 and 19.59 respectively. The nonzero residuals and attribution changes warrant a convergence study before publication. A warmed, two-thread repeated 128-point t=10 calculation took 0.119 seconds for both repetitions and produced **maximum absolute attribution difference 0.0**. These timings are different measurement contexts, not universal runtime guarantees.

Changing packets after t=10 left the real flow's t=10 feature array exactly unchanged. This confirms the diagnostic prefix input construction; a permanent per-checkpoint attribution regression test remains missing.

The public scoring helper is decorated with `@torch.no_grad()` (`cnn_gru_model.py:520`), so gradient explainers must call differentiable model forward directly. Use fixed equal-length prefixes without padding initially; if mixed lengths are introduced, test that masked positions contribute zero and that changing padded values cannot change scores or attributions. Do not interpolate integer lengths or use later packets in an earlier explanation.

Integrated Gradients completeness checks and same-shaped attributions are supported by Captum's design, but the executed diagnostic was a native implementation and does not establish Captum installation compatibility. [Captum Integrated Gradients documentation](https://captum.ai/docs/extension/integrated_gradients).

### Practical sample and resource budgets

- **RF first check:** 16–32 training background rows per checkpoint, 8–16 explanation rows balanced for diagnostic coverage; assert `(n,18)` after class selection, finite values and additivity. Expand to 100–500 explained flows only after profiling. The attribution tensor itself is tiny (500×18 float64 values ≈72 KB for one class); tree/background computation dominates. Runtime is unmeasured without the saved forest/package, so use a time-bounded pilot rather than a promised duration.
- **CNN IG:** start with 2–8 explained flows per checkpoint and 64/128 integration points, then increase until convergence is stable. A 128-point, 8-flow, t=20 float32 input tensor is about 0.625 MiB; convolution/GRU activations and backward state dominate memory. Batch integration points to bound peak memory. Current CPU POC timings support a seconds-scale small check, not a large-study estimate.
- **CNN GradientSHAP:** start with 8–16 training backgrounds, 2 explained flows, fixed seed and roughly 100–200 samples. Runtime/memory must be measured after optional installation; native gradients suggest feasibility but do not bound SHAP overhead.
- **KernelSHAP:** at t=20, there are 160 flattened inputs. Four backgrounds ×128 coalition masks is approximately 512 model rows per explained flow (about 0.625 MiB of float64 feature values before other working arrays). Defaults can use roughly `2*d+2048 = 2368` masks: with 20 backgrounds this is about 47,360 model rows per explanation and around 61 MB for just the expanded feature array. Budget two tiny explanations and a hard runtime cap, not a global heatmap. These are cost estimates, not executed benchmarks. [KernelExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html).

All sequence approaches face correlated/deterministic features: elapsed time is cumulative IAT, and cumulative counts/bytes derive from direction/length. Independent masking/interpolation can create physically inconsistent packets. Explain model-input sensitivity, disclose the background and output units, and consider grouping correlated channels or packet positions. Attribution does not establish causal effects, malicious intent, or robustness.

## 12. Recommended XAI figures and minimal implementation approach

| Figure | Required source data | Correct interpretation / prerequisites |
|---|---|---|
| RF global SHAP bar | Persisted RF; per-checkpoint feature matrices; class-1 SHAP values | Mean absolute contribution to specified model output, conditional on background/sample. No ranking is currently available. |
| RF beeswarm | Same values plus actual feature values | Direction/magnitude of contributions; separate checkpoints or explicitly control checkpoint composition. |
| RF individual waterfall/decision plot | One selected flow prefix, expected value, signed contributions, raw probability | Additive decomposition in matching output units; annotate calibrated probability separately. |
| CNN packet × feature heatmap | Converged IG (or validated GradientSHAP), fixed prefix, raw-logit target | Signed sensitivity by the eight channels and observed packet positions only. |
| CNN per-feature aggregate | Absolute and signed attributions aggregated across valid positions, same checkpoint | Describe aggregation and sequence-length dependence; never treat padded positions as observed packets. |
| Per-checkpoint panels | Separate explanations/backgrounds restricted to each t | Descriptive evolution of evidence as packets arrive; not an explanation using future packets. |
| Benign versus malicious comparison | Prespecified held-out sample; identical explainers/output scale/background | Descriptive model behavior by label. Report sample sizes and capture composition; do not tune model/threshold on the comparison. |
| Early versus later alert explanation | Saved frozen stopping decision and attribution at each flow's actual stop | Different stop lengths create selection effects. Compare matched checkpoints where possible. Current normal results offer very few later malicious alerts. |

Minimal implementation, **after approval**:

1. Fix model bundle persistence and actual-training-output reload tests first. Keep existing results immutable; never substitute a newly trained forest as if it were the old model.
2. Add a separate explanation module/command reading a frozen bundle and an explicit train-derived background-ID list. Do not put XAI fitting inside threshold selection or core training.
3. Implement RF TreeExplainer and fixed-prefix CNN IG first. A tensor-only CNN wrapper supplies constant lengths and returns the desired raw logit; use saved normalization with no refitting. Support calibrated-output explanations only after the calibrator is persisted and the output semantics are validated.
4. Add shape, finiteness, additivity/completeness, seed-repeatability, future-tail invariance, padding invariance and background-partition checks. Test at every reported checkpoint. Save raw attribution arrays, sample IDs, target output, baseline IDs, versions and numerical error alongside plots.
5. Add SVG/PDF export and carefully captioned cohort counts. SHAP/IG figures remain unavailable until those computations actually run; no ranking or plot has been fabricated here.

Optional dependency command, **proposed, not executed**, using only the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install shap captum
```

Check Python 3.14 compatibility and resolve/pin versions before any installation. A separate explanation environment may be preferable to disturbing the current research environment. Native PyTorch IG already works without either package; do not install both merely to demonstrate a capability already verified.

## 13. Resource and runtime risks

Observed machine: approximately **15.69 GiB RAM**, with **2.42 GiB available at one audit sampling point**; an Intel CPU and no CUDA GPU. Free disk was about **677 GiB**. The memory query used Windows' native memory API after a CIM access-denied response. Availability is a point-in-time measurement, not a fixed resource guarantee.

The saved flow pickle is 69,083,270 bytes; its NumPy array buffers occupy about **39.90 MiB**. Python object/dictionary/list overhead is additional. Current subset processing is plausible on this machine, but preprocessing retains active and completed flow objects, prefix generation materializes DataFrames, CNN training materializes multiple prefix sequences, and RF uses all CPU cores. The implementation is packet-streaming, not constant-memory end-to-end processing.

Adding roughly 555 MB of benign capture bytes does not imply a known number of new flows; new class counts and peak-memory needs remain unknown until a controlled build. The 50,000-packet new-file samples took about 0.2 seconds each to parse, but this is not a full-run runtime estimate. Packet limiting changes the population, truncates active flows and biases byte denominators; limited runs must remain explicitly diagnostic.

The upstream full-dataset workflow recommends much larger compute/memory resources than this workstation. That recommendation applies to its full conversion workload, not proof that this selected 1.7 GB capture subset is infeasible. Do not extract/process the complete archive to resolve the current split problem. [Upstream PCAP processing guidance](https://raw.githubusercontent.com/CborgesUFU/DNS-360-Dataset/main/data/PCAP/README.md).

Before a later approved run, use a fresh output directory, record peak RSS and elapsed time, cap worker/thread counts deliberately, and estimate stored prefix expansion. No full training time or full new-data preprocessing time was measured in this audit. Existing histories and the completed local runs demonstrate execution, not a guaranteed repeat runtime.

## 14. Prioritized findings with precise evidence

P0 means a blocker to trustworthy core conclusions, including a methodological limitation rather than necessarily a software exception. P1 must be addressed or explicitly bounded before final reporting. P2 is an enhancement.

| ID | Priority/type | Evidence and impact | Smallest appropriate response |
|---|---|---|---|
| F01 | **P0 — confirmed design limitation** | Both real `split_manifest.json` / `split_assignment.csv` files use flow grouping; all captures span train/val/test. No unseen-session claim is supported. | Reframe current results as within-capture; obtain more sessions or approve an explicitly exploratory grouped protocol. |
| F02 | **P0 — confirmed statistical limitation** | Adaptive 0/131 and 0/127, nominal one-sided bounds 2.26%/2.33%; zero held-out capture units. | Remove population ≤1% claims; design independent benign evaluation with uncertainty. |
| F03 | **P0 — confirmed configuration/data incompatibility** | `config.py:72` default terminal 40; recalculated malicious eligibility at 30/40 is zero; `pipeline.py` rejects single-class terminal curves. | Prespecify a data-supported horizon and publish excluded-flow accounting before another real run. |
| F04 | **P0 if claimed — unsupported central conclusion** | 10/20 adaptive equals fixed-10 on all normal decisions/costs. | Do not claim adaptation outperforms fixed windows generally; compare every prespecified fixed policy under the same protocol. |
| F05 | **P1 — confirmed persistence defect** | Training save at `cnn_gru_model.py:496` omits kwargs required by loader at 582. Both actual artifacts fail. | Unify training/model-bundle save path; regression-test actual checkpoint loading. |
| F06 | **P1 — confirmed missing artifacts** | No saved forest; CNN calibration not persisted. | Persist complete bundles and prediction round-trip checks; historical RF replay remains unavailable. |
| F07 | **P1 — confirmed selection limitation** | Terminal 20 excludes 117,376 malicious flows; terminal 15 still excludes 320 benign flows. | Label conditional results and implement/report operational short-flow handling separately. |
| F08 | **P1 — observed confound; mechanism unproven** | TCP/443 without protocol validation; distinctive early packet lengths; class-associated capture dates. | Verify provenance/protocol scope; test held-out hosts/sessions and handshake-sensitive ablations. |
| F09 | **P1 — confirmed parser defects** | Mixed-endian/multi-interface/timestamp-offset fixtures fail; SPB fabricated timestamps; out-of-order unseen sequence dropped. | Fix targeted parser contracts and add these exact fixtures. Establish whether current captures are affected before revising their results. |
| F10 | **P1 — confirmed reporting/selection defect** | `cnn_gru_model.py:493`; `pipeline.py:154,656`; history minima differ from saved checkpoint and epoch count. | Separate patience delta from minimum-save tracking; report scientific notation and actual epochs. |
| F11 | **P1 — confirmed documentation/provenance mismatch** | README real-run status stale; dataset doc unconditionally claims capture isolation; unsupported tool/publication details. | Update documentation from actual manifests, cite verified provenance and disclose uncertainty. |
| F12 | **P1 — reproducibility limitation** | Unlocked lower-bound dependencies, changed NumPy version, no capture/dirty-source hashes in run records; incomplete CLI config surface. | Export complete immutable run identity and exact invocation/configuration. |
| F13 | **P1 — reporting qualification** | Simulated timing only; no resegmentation/TCP feedback; benign unchanged; observed packet latency excludes processing. | Keep simulation/latency scope explicit and avoid operational resilience claims. |
| F14 | **P2 — explainability enhancement** | Native IG works; SHAP/Captum missing; no XAI artifacts/tests. | Add small independent XAI layer after persistence, with convergence and causal-prefix checks. |
| F15 | **P2 — plots/automation** | 150-dpi PNG only, no CI, no plot-data regression; some compatibility duplicate figures. | Vector export, uncertainty/sample counts, CI, and selected metric-to-figure checks. |
| F16 | **P2 — maintainability** | Sequence formulas duplicated across `feature_extraction.py` and `cnn_gru_model.py`; frontend is intentionally mock. | Add formula-parity test; connect UI to real artifacts only as a separately approved deliverable. |

## 15. Commands actually executed and measured outcomes

The audit used targeted `rg`, PowerShell reads, graph traversal and inline Python, without saving helper scripts. The entries below distinguish exact shell commands from Python entry points invoked inside diagnostic here-strings. No unexecuted proposed full run is listed as completed.

### State, inventory and inspection

```powershell
git status --short
git branch --show-current
git log -1 --oneline
git ls-files
Get-ChildItem DoH_dataset -File | Select-Object Name,Length
Get-ChildItem data/results -Directory | Select-Object Name
Get-Content requirements.txt
Test-Path .github/workflows
```

Outcomes: branch/HEAD and dirty state recorded in section 2; seven unique capture names; four result directories; no CI workflow. `rg`/Git emitted access-denied warnings for two existing pytest temporary directories. No cleanup was performed.

Targeted navigation commands included:

```powershell
Get-Content graphify-out/GRAPH_REPORT.md -TotalCount 80
rg -n 'frontend|README|DATASET|feature_extraction|prefix_generation' graphify-out/manifest.json
rg -n 'best_val_loss|best_epoch|epochs_trained|torch.save|model_kwargs' src/cnn_gru_model.py src/pipeline.py
rg -n 'endian|tsresol|interfaces|linktype|retrans|argsort|truncated' src/preprocessing.py
rg --files src tests
```

Inline Python loaded graph JSON into NetworkX, traversed relevant neighborhoods, compared all 38 manifest hashes and checked edge endpoints. Capture SHA-256 calculations streamed 1 MiB chunks; binary captures were not dumped into tool context.

### Existing tests, completed before the master instruction

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.\.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider
```

Outcome: **128 passed, 2 subtests passed in 135.94 s**. The run used the permissions needed for Windows temporary-directory/worker execution. The suite includes bounded synthetic training/calibration fixtures; it did not launch a new real-data experiment. It was not repeated after the master instruction.

### Targeted diagnostic entry points already executed

| Python/API invocation or operation | Outcome / bound |
|---|---|
| `inspect_pcap(path, max_packets=2000)` for each unique capture | Seven initial samples parsed. |
| `build_flows_from_pcap(..., Config(max_packets_per_file=50000), Counter())` on File02 and File03 | In-memory only; 21 and 8 sample flows; no saved processed outputs. Completed before master task. |
| `load_flows('data/processed')`, per-source/label/length aggregation | 139,076 flows; counts in section 4. No full PCAP preprocessing. |
| `split_flows(meta, 'source_file', seed=42)` and addition group-count feasibility check | Insufficient group counts; original and augmented source inventories cannot pass current requirement. |
| `load_model(path)` on each actual `cnn_gru_best.pt` | Both fail with missing `model_kwargs`. |
| `torch.load(..., map_location='cpu', weights_only=False)` on local trusted experiment checkpoints; manual `CnnGruClassifier` reconstruction | Finite actual model parameters, normalizer/history checks and bounded inference/gradient diagnostics. |
| `FlowCurves` + `policy_outcomes` + `summarize`, rebuilt from saved scores and flow metadata | All normal validation/test summaries match within `1e-8`; validation sweep counts and selected thresholds also agree. |
| `clopper_pearson(0,131)` and `(0,127)`; zero-event sample-size formula | Bounds and 299/368 independent-trial requirements in section 8. |
| `fit_calibrator_cnn` once on recorded 10/20 train-cal flows, before master task | 3.64 s, 24,035 prefix rows; sampled val/test calibrated scores match. No model/calibrator saved; not repeated. |
| Native `torch.autograd.grad` with Gauss–Legendre IG quadrature | Finite `(2,t,8)` at t=10/20; residuals/timings in section 11. |
| Repeated t=10, 128-point IG and actual-flow future-tail mutation | Maximum attribution repeat difference 0.0; prefix input unchanged by later-packet mutation. |
| `BytesIO`/mocked file-open PCAPNG and TCP adversarial fixtures | One big-endian case passes; five limitations/failures detailed in section 5. No fixture files retained. |
| Read-only image viewing of two saved PNGs | Table-consistent values, simulation labels, and publication-format limitations noted. |

Primary upstream documentation was consulted for dataset provenance and XAI interfaces; links are next to the relevant claims. No package was installed. No full preprocessing, new real-data training, figure regeneration, model overwrite, commit or push was performed.

## 16. Exact remaining tasks by responsibility area and files changed

### Required next steps, in priority order

1. **Shared research design:** decide the target claim and independent sampling unit. Treat existing results as within-capture exploration. Obtain additional benign and malicious sessions with host/tool/time metadata, or approve a clearly limited grouped exploratory protocol. Prespecify checkpoint horizon, short-flow handling and fixed baselines before a new test.
2. **Member 1:** register the seven unique captures with hashes and verified label/provenance rules; avoid duplicate directories. Fix/reject the identified PCAPNG timing/interface cases and TCP sequence-range issue with fixtures. Establish whether they affect existing captures. After approval, process the two additions into a new versioned dataset, measure full class/eligibility counts and peak memory, and produce source-disjoint manifests for the approved protocol.
3. **Member 2:** repair training-checkpoint/public-loader compatibility; persist RF, CNN normalizer and calibration together with architecture/feature schema. Add actual-bundle score round-trip tests. Correct epoch/loss reporting and the best-checkpoint rule. Document different RF/CNN sample and flow-weighting regimes; do not alter architecture merely because loss rounds to zero.
4. **Member 3:** retain complete-policy validation thresholding and frozen test evaluation. Report capture grouping, cohort exclusions and FPR uncertainty prominently. Compare every prespecified fixed window; distinguish observation latency from inference runtime. Preserve offline simulation terminology and expose unresolved/short-flow population handling.
5. **Shared reproducibility/testing:** add CI, immutable run identity, dependency versions/lock and exact config invocation; include parser and actual-persistence regressions. Freeze new test captures and stop adapting methods to previously examined test outcomes.
6. **Shared XAI with Member 2/3:** implement persisted-RF TreeSHAP and fixed-prefix CNN IG first. Verify shape, additivity/completeness, mask/future-tail invariance and baseline sensitivity; save raw explanation data and captions. GradientSHAP is optional after a small compatibility check. No broad DeepSHAP/KernelSHAP job is justified yet.
7. **All members, final deliverable:** update README and dataset methodology from verified facts; replace unconditional validation/generalization statements; export vector figures with counts and uncertainty. Keep frontend mock visuals explicitly separate from experimental evidence.

**Another real-data run is necessary for stronger research claims**, but rerunning the same flow split with two additional benign captures would not resolve the central validity issues. Complete design and persistence corrections first; then obtain explicit approval for processing/training work.

### Files changed by this audit

Only new persistent deliverable:

```text
reports/full_project_audit.md
```

No application, test, configuration, dependency, frontend, Graphify or experiment artifact was edited. No existing file was overwritten. A before/after comparison of **216 protected source/test/configuration/processed/result/graph files** found **zero changes in size, modification timestamp or SHA-256** before report creation. Existing user modifications remain in place. Temporary test fixtures were managed by the test suite; pytest cache writing and Python bytecode writing were disabled for the audit test command.

The audit stops here. The changes above are recommendations for authorization, not implemented fixes.
