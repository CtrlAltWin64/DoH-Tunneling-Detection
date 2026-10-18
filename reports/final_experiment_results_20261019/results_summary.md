# Research results — preliminary within-capture evaluation



## 1. Experimental setup

Fresh real-capture RF and CNN-GRU experiments, seed 42, packet checkpoints 10 and 20, validation target FPR 1%. Existing model architecture and first-crossing policy are retained. These are exploratory flow-isolated results, not an unseen-capture confirmation.

## 2. Dataset characteristics and class distribution

Full parsing of seven unique captures produced 140,193 TCP/443 flows: 2,095 benign and 138,098 malicious. Labels are filename/scenario labels (0 benign; 1 malicious), not independently verified application labels. Four benign and three malicious captures were hashed and processed without a packet limit.

| checkpoint | label | total | eligible | excluded |
|---|---|---|---|---|
| 10 | 0 | 2095 | 1380 | 715 |
| 10 | 1 | 138098 | 138098 | 0 |
| 20 | 0 | 2095 | 1372 | 723 |
| 20 | 1 | 138098 | 20722 | 117376 |
| 30 | 0 | 2095 | 946 | 1149 |
| 30 | 1 | 138098 | 0 | 138098 |
| 40 | 0 | 2095 | 876 | 1219 |
| 40 | 1 | 138098 | 0 | 138098 |


Only 15.0053% of malicious flows reach 20 packets. No malicious flow reaches 30 or 40; those checkpoints cannot support a two-class detector comparison. The 10/20 schedule is already documented in the earlier audit and is a supported subset of the default checkpoints. It was selected for data support before examining new model scores.

## 3. Train/validation/test splitting procedure

The existing five-fold stratified group splitter rejects capture isolation with four benign and three malicious files. We explicitly use preliminary flow grouping: fold 0 test, fold 1 validation, remaining folds train. All prefixes stay with their flow. All seven captures occur across partitions; this is a major dependence/confounding limitation.

| partition | label | flows |
|---|---|---|
| test | 0 | 276 |
| test | 1 | 27620 |
| train | 0 | 828 |
| train | 1 | 82858 |
| val | 0 | 276 |
| val | 1 | 27620 |


The outer test has 27,896 flows reaching at least 10 packets. The common policy cohort has 4,412 flows reaching 20, with 23,484 excluded. Flows below 10 are excluded before splitting and documented in the dataset table. Native-prefix results in main_performance.csv separately retain all eligible 10-packet test flows.

## 4. Models and stopping policies evaluated

RF: 200 trees, minimum leaf size 3, balanced_subsample, equal total training weight per flow, five-fold flow-grouped out-of-fold Platt calibration. CNN-GRU: existing 64-channel two-convolution / two-layer 64-hidden GRU, up to 30 epochs with patience 7. Outer training is divided into fit, calibration, and early-stopping roles; threshold validation and final test remain separate. CNN weights prefixes equally; RF weights flows equally. The existing CNN minimum-improvement tolerance is 1e-6, so the saved epoch need not be the numerical minimum of every later validation loss.

Adaptive alerts at the first probability ≥ its validation-selected threshold; otherwise it observes 20 packets and records no alert. Fixed 10 and fixed 20 have their own validation-selected thresholds. Each policy uses the lowest threshold on the existing 0.001 grid satisfying empirical validation FPR ≤1%. Validation threshold tuning uses the common ≥20 cohort; native-prefix test rows reuse those thresholds.

## 5. Main performance comparison

| model | policy | TP | FP | TN | FN | precision | recall | f1 | fpr | fpr_ci95_high | eligible | excluded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rf | adaptive | 4137 | 3 | 272 | 0 | 0.999275 | 1 | 0.999638 | 0.0109091 | 0.0315482 | 4412 | 23484 |
| rf | fixed_10 | 4137 | 3 | 272 | 0 | 0.999275 | 1 | 0.999638 | 0.0109091 | 0.0315482 | 4412 | 23484 |
| rf | fixed_20 | 4137 | 3 | 272 | 0 | 0.999275 | 1 | 0.999638 | 0.0109091 | 0.0315482 | 4412 | 23484 |
| cnn | adaptive | 4137 | 0 | 275 | 0 | 1 | 1 | 1 | 0 | 0.0133245 | 4412 | 23484 |
| cnn | fixed_10 | 4137 | 0 | 275 | 0 | 1 | 1 | 1 | 0 | 0.0133245 | 4412 | 23484 |
| cnn | fixed_20 | 4137 | 0 | 275 | 0 | 1 | 1 | 1 | 0 | 0.0133245 | 4412 | 23484 |


Full tables include accuracy, precision, recall, F1, FPR, FNR, exclusion counts, exact confidence bounds, and explicit event numerators and denominators. Missing predictions are not imputed as benign.

## 6. Adaptive stopping and observation-cost findings

| model | policy | mean_packets | median_packets | p95_packets | reduction_pct | reaching_20_pct | detected_malicious | mean_latency_s |
|---|---|---|---|---|---|---|---|---|
| rf | adaptive | 10.6165 | 10 | 20 | 46.9175 | 6.165 | 4137 | 0.0053965 |
| rf | fixed_10 | 10 | 10 | 10 | 50 | 0 | 4137 | 0.0053965 |
| rf | fixed_20 | 20 | 20 | 20 | 0 | 100 | 4137 | 0.00624662 |
| cnn | adaptive | 10.6233 | 10 | 20 | 46.8835 | 6.233 | 4137 | 0.0053965 |
| cnn | fixed_10 | 10 | 10 | 10 | 50 | 0 | 4137 | 0.0053965 |
| cnn | fixed_20 | 20 | 20 | 20 | 0 | 100 | 4137 | 0.00624662 |


| model | n | extra_false_positives_vs_tuned_fixed20 | extra_false_negatives_vs_tuned_fixed20 | recovered_true_positives_vs_tuned_fixed20 | early_false_positives_vs_same_threshold_terminal | recovered_true_positives_vs_same_threshold_terminal |
|---|---|---|---|---|---|---|
| rf | 4412 | 3 | 0 | 0 | 3 | 0 |
| cnn | 4412 | 0 | 0 | 0 | 0 | 0 |


| model | matched_true_positives | adaptive_mean_latency_s | fixed20_mean_latency_s | mean_paired_saving_s |
|---|---|---|---|---|
| rf | 4137 | 0.0053965 | 0.00624662 | 0.000850118 |
| cnn | 4137 | 0.0053965 | 0.00624662 | 0.000850118 |


Observation reductions are calculated from actual stopping traces relative to fixed 20. Fixed 10 already costs exactly 10 packets, so reducing cost relative to fixed 20 does not by itself establish an advantage over the earliest fixed baseline. Paired latency uses only attacks detected by both adaptive and fixed 20. Ordinary latency summaries condition on detection and can be selection-biased when recall differs. RF adaptive and fixed 20 have equal aggregate error counts but flag different benign flows: adaptive adds three false positives relative to fixed 20 while avoiding three of fixed 20’s false positives. Its three additional early false positives also disappear when scoring only the terminal checkpoint at the adaptive threshold. CNN has no such decision disagreements.

## 7. Statistical uncertainty and FPR assessment

95% two-sided Clopper–Pearson exact binomial intervals and one-sided 95% FPR upper bounds are reported. These intervals assume independent flows and are optimistic under shared captures; they do not establish unseen-session error bounds. No FPR <1% claim follows merely from zero observed false positives.

| model | FP | n_benign | fpr | fpr_ci95_low | fpr_ci95_high | fpr_upper95_one_sided |
|---|---|---|---|---|---|---|
| rf | 3 | 275 | 0.0109091 | 0.00225539 | 0.0315482 | 0.0279523 |
| cnn | 0 | 275 | 0 | 0 | 0.0133245 | 0.0108345 |


At least 299 independent benign observations with zero false positives are required for a one-sided exact 95% upper bound below 1%; additional correlated flows are not a substitute for independent captures.

## 8. Evasive-traffic findings

No supplied per-flow rate, attack configuration, campaign or slow-tunneling labels identify genuine evasive groups. Prefix-length analysis is supported; an evasion-resistance conclusion is not. Existing timing-dilation code is an offline simulation, not measured slower tunneling. No synthetic timings are presented as real evasion findings. Needed evidence: labeled rate-controlled attack sessions, matched benign sessions, attack parameters, packet timestamps and capture-isolated partitions.

## 9. Interpretation of the results

RF: adaptive recall 100.0000%, FPR 1.0909% (3/275), mean observation 10.6165 packets (46.92% reduction versus 20). Fixed 10 recall is 100.0000% and FPR 1.0909%. These comparisons are conditional on the same long-flow cohort.

CNN: adaptive recall 100.0000%, FPR 0.0000% (0/275), mean observation 10.6233 packets (46.88% reduction versus 20). Fixed 10 recall is 100.0000% and FPR 0.0000%. These comparisons are conditional on the same long-flow cohort.

For both models, adaptive and fixed 10 produce the same classification counts and the same mean attack-detection latency (5.3965 ms). Adaptive observes about 10.62 packets across all flows versus exactly 10 for fixed 10; it therefore shows no measured benefit over fixed 10 here. Against fixed 20, mean detected-attack latency drops from 6.2466 ms to 5.3965 ms, a paired saving of 0.8501 ms. CNN zero false positives among 275 benign flows still yields a one-sided 95% FPR upper bound of 1.0834% (two-sided upper endpoint 1.3325%). RF empirical FPR is 1.0909%, exceeding the target.

## 10. Threats to validity and limitations

Shared capture sessions, extreme class imbalance, filename-derived labels, selection against short attacks, and possible handshake/host/tool fingerprints prevent generalization claims. TCP/443 filtering alone does not authenticate DoH. One seed and one flow split do not estimate split variability. Confidence intervals do not account for capture correlation. Parser retransmission filtering uses a sequence high-water mark and can drop unseen reordered payload; timestamps are sorted only after extraction and only the first 200 retained packets are stored. See preprocessing counters. Complete structural inspection found no mixed-endian sections, non-Ethernet interfaces, simple packet blocks or nonzero timestamp offsets in these PCAPNG files; this narrows but does not eliminate parser limitations.

The earlier audit describes two real-data runs from a different workspace. Their score files, processed data and model artifacts are absent here; their numerical findings were not copied into this study. Saved synthetic scores were independently checked only as software evidence. The existing 113-test suite passed. An initial RF launch preceded completion of split preparation and failed without producing model results; it was rerun successfully after preparation. Logs preserve both attempts.

Additional dependence diagnostic: 629 distinct first-20 length/direction signatures occur in both training and test. Across the full common-horizon population, 20,375 flows belong to repeated signature groups. These signatures omit timing and do not prove duplicate capture bytes, but reinforce the need for capture-isolated evaluation.

## 11. Overall conclusion

These results measure the adaptive policy’s cost/error trade-off on a common long-flow subset of real captures. Adaptive stopping reduces observation versus fixed 20 but provides no measured benefit over fixed 10. They do not establish that unseen-capture FPR is approximately 1% or that the detector resists evasive traffic. The common-cohort measurements and their exclusions are suitable for an explicitly preliminary report; a confirmatory generalization claim requires additional independent capture sessions and a prespecified short-flow protocol.
