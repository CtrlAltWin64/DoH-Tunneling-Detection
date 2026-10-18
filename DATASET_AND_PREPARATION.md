# Dataset and preparation

This document describes the actual 19 October 2026 experiment, superseding the
older preparation notes. Earlier assertions about independence, attack tools,
publication dates and licensing have not been independently authenticated.

## Inputs

The seven captures were directly inside `data/`, not `data/raw/` or the nested
checkout. See the [file list](data/README.md) and [SHA-256 manifest](reports/final_experiment_results_20261019/metrics/capture_manifest.csv).
Total size: 1,703,660,902 bytes (approximately 1.59 GiB). All are ignored by Git.

The project identifies these as UFU-DoH-EXF / DNS-360 scenario captures. Hashes
establish byte identity, not origin, independence or application protocol. Confirm
upstream provenance and redistribution terms before externally sharing datasets.
Labels are explicit: Benign filenames = 0; Exfiltration filenames = 1 (positive).

## Extraction and eligibility

Full-file extraction uses TCP/443, no resolver allow-list, a 120-second idle timeout,
and SYN after closure as a new-flow boundary. It includes zero-payload packets,
stores IP total length and direction relative to the first observed endpoint, and
retains up to 200 packet entries while counting full-flow packet/byte totals.
The existing sequence high-water retransmission heuristic is retained. Retained
out-of-order timestamps are sorted after extraction.

| Required packets | Eligible benign | Eligible malicious |
|---:|---:|---:|
| 10 | 1,380 | 138,098 |
| 20 | 1,372 | 20,722 |
| 30 | 946 | 0 |
| 40 | 876 | 0 |

Total extracted: 140,193 flows (2,095 benign / 138,098 malicious). Per-capture
counts and exclusions are in [checkpoint eligibility](reports/final_experiment_results_20261019/tables/checkpoint_eligibility.md).
Missing prefixes are never fabricated. Only about 15% of attacks reach 20 packets;
30 and 40 cannot support a two-class performance comparison.

## Splitting and model roles

Four benign and three malicious captures fail the existing five-fold capture
splitter's minimum five groups per class. This study explicitly uses a **preliminary
flow split**, seed 42: fold 0 test, fold 1 validation, folds 2–4 training. Only flows
reaching 10 packets enter the split. All prefixes of one flow stay together.

| Partition | Benign | Malicious |
|---|---:|---:|
| Train | 828 | 82,858 |
| Validation | 276 | 27,620 |
| Test | 276 | 27,620 |

**All seven captures occur in every outer partition.** There are no unseen test
captures. Repeated length/direction signatures also span training and test.
RF calibration uses training-only flow-grouped out-of-fold scores. The study's CNN
uses separate fitting, calibration and early-stopping roles inside outer training,
with seeds 42 and 43. Outer validation selects thresholds; test uses frozen models
and thresholds. The generic CLI differs in its early-stopping arrangement.

The common comparison requires at least 20 packets: 4,137 malicious and 275 benign
test flows, excluding 23,484 outer test flows. Native 10-packet results are separate
and reuse thresholds selected on common-cohort validation data.

## Features and timing

RF uses 18 prefix aggregate features. CNN-GRU uses eight channels: length, direction,
inter-arrival time, elapsed time, cumulative forward/reverse packet counts and
cumulative forward/reverse bytes. Features only read the requested prefix; source
filenames and full-flow totals are not direct classifier inputs.

Latency is alert-packet timestamp minus first-retained-packet timestamp, in seconds.
It is not inference runtime or time since attack onset. No-alert flows incur the
terminal observation cost. Latency summaries condition on detected attacks, with
paired comparisons separately reported for shared true positives.

## Data quality and limits

Complete PCAPNG inspection found Ethernet interfaces, no mixed-endian sections,
no timestamp-less Simple Packet Blocks and no nonzero timestamp offsets. Extracted
timestamps were finite and nondecreasing after processing. Full counters are in
[preprocessing.json](reports/final_experiment_results_20261019/metrics/preprocessing.json).

The parser's retransmission heuristic can drop unseen reordered payload; sorting
afterward cannot restore it. The storage cap limits recovery of late-arriving earlier
packets. TCP/443 is not DoH authentication. Labels do not identify genuine evasive
rates or tools. Session fingerprints and capture correlation limit conclusions.

One additional benign and two additional malicious captures would merely pass the
outer splitter's count guard; more independent sessions may be required for every
model/calibration/validation/test role and meaningful capture-level uncertainty.

## Reproduction and storage

Use [reproduction.md](reports/final_experiment_results_20261019/reproduction.md).
Scripts, aggregate outputs, figures, source snapshots and configuration are Git-visible.
Captures, processed data, per-flow scores, model bundles and test temporaries remain
local. See [GIT_READINESS.md](GIT_READINESS.md). A clean clone does not include the
large artifacts needed for score-based replay; restore them or perform a fresh run.
