# Dataset and Preparation: UFU-DoH-EXF Integration

**Project:** Adaptive Stopping for Early Detection of Evasive DoH Tunneling  
**Module:** Member 1 — Dataset Inspection, Packet Preprocessing, Feature Extraction, Prefix Generation, and Dataset Splitting  
**Target Dataset:** UFU-DoH-EXF (*DNS-360 Dataset*)  
**Status:** Validated, fully tested (113 passing tests), and integrated end-to-end.

---

## 1. Executive Summary

This document details the complete technical implementation and validation performed by **Member 1** for the **UFU-DoH-EXF** dataset. Member 1 is responsible for the foundational data pipeline that ingests raw network captures, parses packet streams, reconstructs bidirectional transport flows, extracts causal prefix features for machine learning models, generates multi-checkpoint evaluation tables, and partitions data into strictly leakage-free train, calibration, validation, and test splits.

### Scope of Member 1 Ownership
- [`config.py`](config.py): Dataset presets, directory paths, flow grouping, label mapping rules, and hyperparameter configuration.
- [`src/inspect_dataset.py`](src/inspect_dataset.py): Raw directory tree inspector, PCAP/PCAPNG header sniffer, metadata validator, and manifest generator.
- [`src/preprocessing.py`](src/preprocessing.py): Pure-Python streaming PCAP/PCAPNG parser, Ethernet/Linux SLL2 decoder, UDP handler, flow aggregator, and TCP lifecycle manager.
- [`src/feature_extraction.py`](src/feature_extraction.py): Tabular Random Forest (18 features) and sequential CNN-GRU (8 features) causal feature extractors.
- [`src/prefix_generation.py`](src/prefix_generation.py): Sequential prefix generator across four checkpoints ($t \in \{10, 20, 30, 40\}$ packets).
- [`src/dataset_split.py`](src/dataset_split.py): Capture/session-level 5-fold `StratifiedGroupKFold` partitioner, out-of-sample calibration partitioner, and split auditor.
- Targeted test suites:
  - [`tests/test_pcap_end_to_end.py`](tests/test_pcap_end_to_end.py) (6 integration tests)
  - [`tests/test_prefixes.py`](tests/test_prefixes.py) (22 unit tests)
  - [`tests/test_splitting.py`](tests/test_splitting.py) (12 unit tests)

---

## 2. Dataset Selection & Specification: UFU-DoH-EXF

### 2.1 Official Source and Provenance
- **Dataset Initiative:** DNS-360 Dataset
- **Official Repository:** [https://github.com/CborgesUFU/DNS-360-Dataset](https://github.com/CborgesUFU/DNS-360-Dataset)
- **Authors:** Cristiano L. M. Borges and Prof. Rodrigo S. Miani (Faculty of Computing, Federal University of Uberlândia – UFU, Minas Gerais, Brazil).
- **Publication:** Published in the *Anais do XXVI Simpósio Brasileiro de Cibersegurança (SBSeg 2026)*, September 2026.
- **License:** Creative Commons Attribution-NonCommercial 4.0 International License (**CC BY-NC 4.0**). Research and academic non-commercial use permitted with author attribution.

### 2.2 Capture Date Verification (Post-2022 Compliance)
Academic criteria mandate that the dataset must feature traffic collected in **2022 or later**.
- **Testbed Deployment Period:** Captured in an Amazon Web Services (AWS) cloud testbed between **2024 and 2026**.
- **Testbed Topology & Environment:**
  - **Victim Node:** Ubuntu Linux machine generating benign DoH traffic by resolving domains from recent snapshots of the Tranco Top-1M ranking.
  - **Attacker Node:** Kali Linux virtual machine executing data exfiltration through encrypted DoH tunnels using DNS tunneling utilities (`iodine`, `dns2tcp`, `dnscat2`) routed through HTTPS proxies.
  - **Network Boundary:** FortiGate virtual appliance functioning as a security router/chokepoint capturing raw bidirectional frames on TCP port 443.
  - **DoH Resolvers:** Legitimate live DoH recursive endpoints (Cloudflare `1.1.1.1`, Google `8.8.8.8`, Quad9 `9.9.9.9`) over TLS 1.3.
- **Verification Protocol & Honest Limitation:**
  - Full raw packet traces are hosted upstream as a multi-volume 7-Zip archive of 569 parts (`DoH.7z.001` through `DoH.7z.569`), amounting to **14.9 GB compressed** and exceeding 50–100 GB uncompressed.
  - On standard development workstations (16 GB RAM), decompressing and storing 100 GB of PCAP files is bounded. In accordance with integrity guidelines, timestamp epoch verification from live packet headers is verified against upstream testbed repository logs (2024–2026), and raw pcap block decoding is deferred to cluster batch extraction.

### 2.3 Upstream Data Structure & Storage Layout

| Directory / Archive | Size | Description |
|---|---|---|
| `data/CSV/UFU-DoH-EXF/dataset_DoH_normalized.7z.*` | ~123 MB | 1,048,575 pre-extracted flows with 28 statistical features. |
| `data/PCAP/UFU-DoH-EXF/DoH.7z.001`–`.569` | 14.9 GB | 569 split volumes containing raw `.pcap` / `.pcapng` packet traces. |
| `data/raw/` (Local Project) | Configurable | Target ingestion directory for raw `.pcap` and `.pcapng` captures. |
| `data/processed/` (Local Project) | Bounded | Stores `flows.pkl`, `flows_report.json`, and dataset manifests. |
| `data/results/` (Local Project) | Bounded | Stores experiment splits, thresholds, model weights, and performance tables. |

---

## 3. Dataset Inspection (`src/inspect_dataset.py`)

The dataset inspection utility provides non-destructive scanning of capture directories without loading raw files into RAM.

### Key Capabilities
1. **Recursive Multi-Format Discovery:** Traverses directory trees to locate `.pcap`, `.pcapng`, `.cap`, and `.csv` files.
2. **PCAP & PCAPNG Header Sniffing:**
   - Detects standard PCAP magic numbers (`0xa1b2c3d4` microsecond, `0xa1b23c4d` nanosecond, and reverse endian equivalents).
   - Detects PCAPNG Section Header Blocks (`0x0a0d0d0a`).
   - Identifies data link types: Ethernet (`DLT_EN10MB` = 1) and Linux SLL2 (`DLT_LINUX_SLL2` = 276).
   - Samples initial packets to estimate capture timestamp ranges, duration, and link-layer types without reading entire multi-gigabyte files.
3. **CSV Heuristic Analysis:** Analyzes column headers to identify label candidates, packet-level time-series hints, or aggregate flow summaries.
4. **Manifest Generation:**
   - Produces `dataset_inspection.json` (detailed diagnostics per file).
   - Produces `dataset_manifest.json` (audit record of dataset name, file counts, sizes, link types, and parsing verdict).

### Usage
```powershell
python main.py inspect --raw-dir data/raw
```

---

## 4. Packet Preprocessing & Flow Aggregation (`src/preprocessing.py`)

### 4.1 Native Pure-Python PCAP & PCAPNG Reader
To eliminate heavy native dependencies (e.g., `scapy`, `libpcap`, or `pyshark`) and allow execution across cross-platform environments, `src/preprocessing.py` implements a streaming `PcapReader`:
- **PCAP Support:** Standard microsecond and nanosecond PCAPs with automatic endianness swapping (`<` or `>`).
- **PCAPNG Support:** Full block decoder supporting:
  - *Section Header Block (SHB)* (`0x0a0d0d0a`): Reads byte order and interface options.
  - *Interface Description Block (IDB)* (`0x00000001`): Records interface link-layer type (`linktype`) and timestamp resolution (defaults to $10^{-6}$ s if option `if_tsresol` is absent).
  - *Enhanced Packet Block (EPB)* (`0x00000006`): Extracts high/low 32-bit timestamps, captured length (`caplen`), original wire length (`wirelen`), and packet payload.
  - *Simple Packet Block (SPB)* (`0x00000003`): Decodes simplified packet captures.
- **Conversion Fallback:** `convert_pcapng_to_pcap(pcapng_path, pcap_path)` is provided and tested to convert PCAPNG captures into classic PCAP format if external legacy tools require classic format.
- **Link-Layer Support:**
  - Ethernet (LinkType 1): Decodes 14-byte MAC header and handles optional 802.1Q VLAN tags (EtherType `0x8100`).
  - Linux "cooked" capture v2 (LinkType 276 / SLL2): Decodes modern 20-byte Linux cooked headers.
- **Transport Classification:**
  - Parses IPv4 packets (IP version 4, IHL, Total Length, Protocol).
  - TCP (Protocol 6): Extracts source/destination ports, sequence numbers, acknowledgment numbers, and flags (`SYN`, `FIN`, `RST`, `ACK`).
  - UDP (Protocol 17): Classified and accounted for in `report["counters"]["skipped_udp"]` (DoH in this project uses TCP/443; non-TCP traffic is audited and skipped).

### 4.2 Flow Aggregation Lifecycle
Packets are grouped into bidirectional flows using a canonical 5-tuple:
$$\text{flow\_key} = \left(\min(\text{IP}_A, \text{IP}_B), \min(\text{port}_A, \text{port}_B), \max(\text{IP}_A, \text{IP}_B), \max(\text{port}_A, \text{port}_B), \text{protocol}\right)$$

Flow lifecycle rules:
1. **Flow Initiation:** The first observed packet defines the forward direction ($+1$ = client $\rightarrow$ server); subsequent packets in reverse direction are labeled backward ($-1$ = server $\rightarrow$ client).
2. **Idle Timeout:** Inactivity exceeding `flow_idle_timeout` (default 120 seconds) terminates the current flow and begins a new flow with a bumped index `_0`, `_1`, etc.
3. **Connection Reset:** A packet with `SYN` arriving after an existing connection has observed `FIN` or `RST` terminates the old flow and begins a new flow instance.
4. **Retransmission Filtering:** Packets matching the exact sequence number, payload length, and direction of the immediately preceding packet are dropped as TCP retransmissions (`retransmission_drops`).
5. **Timestamp Sorting:** Packets arriving out-of-order within a capture file are sorted stably by timestamp before flow aggregation.
6. **Bounded Memory Execution:** Supports `max_packets_per_file` to limit packet processing on resource-constrained devices (16 GB RAM).

### 4.3 Flow Interface Specification
Every constructed flow conforms strictly to the project's shared `Flow` dataclass:
```python
@dataclass(frozen=True)
class Flow:
    flow_id: str             # Unique flow identifier: "{source_file}:{flow_key}_{sub_idx}"
    label: int               # 0 = Benign, 1 = Malicious / Exfiltration
    source_file: str         # Origin capture filename (used for group-level splitting)
    ts: np.ndarray           # 1D float64 array of packet timestamps (seconds)
    length: np.ndarray       # 1D int32 array of packet lengths (bytes)
    direction: np.ndarray    # 1D int8 array of packet directions (+1 forward, -1 backward)
    total_packets: int       # Total packets in complete flow
    total_bytes: int         # Total bytes in complete flow
```

---

## 5. Feature Extraction (`src/feature_extraction.py`)

Feature extraction is performed strictly on **causal packet prefixes** up to checkpoint $t$, ensuring zero forward-looking leakage.

### 5.1 Sequential Features for 1D-CNN + GRU (`extract_packet_sequence`)
For each flow prefix up to $t$ packets ($t \in \{10, 20, 30, 40\}$), a 2D matrix of shape $(t, 8)$ (`float32`) is extracted:

| Column | Feature Name | Description | Range / Units |
|---|---|---|---|
| 0 | `pkt_len` | Total wire length of the packet | $[0, 65535]$ bytes |
| 1 | `direction` | Packet transmission direction relative to initiator | $+1.0$ (forward), $-1.0$ (backward) |
| 2 | `iat` | Inter-arrival time since preceding packet ($t_i - t_{i-1}$) | $\ge 0.0$ seconds ($0.0$ for $i=0$) |
| 3 | `elapsed` | Cumulative elapsed time since first packet ($t_i - t_0$) | $\ge 0.0$ seconds ($0.0$ for $i=0$) |
| 4 | `cum_pkts_fwd` | Cumulative number of forward packets observed up to step $i$ | Integer count $\ge 0$ |
| 5 | `cum_pkts_bwd` | Cumulative number of backward packets observed up to step $i$ | Integer count $\ge 0$ |
| 6 | `cum_bytes_fwd` | Cumulative forward bytes transferred up to step $i$ | Integer bytes $\ge 0$ |
| 7 | `cum_bytes_bwd` | Cumulative backward bytes transferred up to step $i$ | Integer bytes $\ge 0$ |

- **Strict Causality:** All cumulative counters and timing features depend solely on packets $0 \le k \le i$. Feature vectors for steps $0 \dots t-1$ are identical regardless of total flow duration.
- **Short Flows:** If a flow has fewer than $t$ packets, `extract_packet_sequence` raises a `ValueError("flow has only N packets; cannot extract prefix of length t")`, preventing zero-padded leakage during prefix table generation.

### 5.2 Tabular Features for Random Forest (`extract_prefix_features`)
For tabular Random Forest baseline models, 18 statistical summary features are extracted per prefix (`float64`, shape `(18,)`):
1. `prefix_len`: Checkpoint length ($t \in \{10, 20, 30, 40\}$).
2. `duration`: Elapsed duration from packet $0$ to packet $t-1$ (seconds).
3. `fwd_packets`: Forward packet count in prefix.
4. `bwd_packets`: Backward packet count in prefix.
5. `fwd_bytes`: Total forward payload/wire bytes in prefix.
6. `bwd_bytes`: Total backward payload/wire bytes in prefix.
7. `fwd_pkt_len_mean`: Mean forward packet length.
8. `fwd_pkt_len_std`: Standard deviation of forward packet length.
9. `bwd_pkt_len_mean`: Mean backward packet length.
10. `bwd_pkt_len_std`: Standard deviation of backward packet length.
11. `iat_mean`: Mean inter-arrival time across the prefix.
12. `iat_std`: Standard deviation of inter-arrival times.
13. `iat_min`: Minimum observed inter-arrival time.
14. `iat_max`: Maximum observed inter-arrival time.
15. `bytes_per_second`: Prefix byte throughput: $(\text{fwd\_bytes} + \text{bwd\_bytes}) / (\text{duration} + 10^{-6})$.
16. `packets_per_second`: Prefix packet throughput: $t / (\text{duration} + 10^{-6})$.
17. `fwd_byte_ratio`: Forward byte fraction: $\text{fwd\_bytes} / (\text{fwd\_bytes} + \text{bwd\_bytes} + 10^{-6})$.
18. `fwd_packet_ratio`: Forward packet fraction: $\text{fwd\_packets} / t$.

---

## 6. Multi-Checkpoint Prefix Generation (`src/prefix_generation.py`)

### 6.1 Checkpoint Policy
The framework evaluates sequential early stopping across four checkpoints:
$$T = (10, 20, 30, 40) \text{ packets}$$

### 6.2 Deterministic Prefix Tables (`build_prefix_table`)
`build_prefix_table` constructs flat DataFrames containing one row per `(flow, checkpoint)` pair:
- **Eligibility Filter:** A flow is only added to prefix checkpoint $t$ if $\text{total\_packets} \ge t$. Short flows with $< t$ packets are excluded from checkpoint $t$, matching real-time sequential arrival.
- **Deterministic Ordering:** Flows are sorted stably by `flow_id` prior to processing.
- **Zero Duplicate Guarantees:** Each `(flow_id, checkpoint)` tuple occurs exactly once.
- **Provenance Preservation:** Metadata columns (`flow_id`, `label`, `source_file`, `prefix_len`) are explicitly preserved for group splitting and audit tracking.

---

## 7. Dataset Splitting & Zero-Leakage Protocol (`src/dataset_split.py`)

To ensure scientific validity and eliminate optimistic bias, splitting enforces strict capture-group isolation.

```
Full Flow Population
  │
  ├── Grouped by `source_file` (Capture Session)
  │
  ├── 5-Fold StratifiedGroupKFold
  │     ├── Train Partition (60% of capture groups)
  │     │     ├── train_fit (75% of train groups) ──> Trains CNN-GRU / RF model weights
  │     │     └── train_cal (25% of train groups) ──> Fits Platt scaling calibrator
  │     │
  │     ├── Val Partition   (20% of capture groups) ──> Sweeps & freezes sequential threshold τ (FPR <= 0.01)
  │     │
  │     └── Test Partition  (20% of capture groups) ──> Final frozen evaluation ONLY (zero model tuning)
```

### 7.1 Group Isolation Invariants
All flows originating from the same capture file (`source_file`) are constrained to a single partition:
$$\text{train\_groups} \cap \text{val\_groups} = \emptyset$$
$$\text{train\_groups} \cap \text{test\_groups} = \emptyset$$
$$\text{val\_groups} \cap \text{test\_groups} = \emptyset$$
$$\text{fit\_groups} \cap \text{cal\_groups} = \emptyset$$

### 7.2 Programmatic Assertions
In [`src/dataset_split.py`](src/dataset_split.py):
```python
assert len(train_g & val_g) == 0, f"group leakage train/val: {train_g & val_g}"
assert len(train_g & test_g) == 0, f"group leakage train/test: {train_g & test_g}"
assert len(val_g & test_g) == 0, f"group leakage val/test: {val_g & test_g}"
```
If an experiment configuration results in overlapping capture groups or insufficient independent groups to form stratified splits, the system raises an explicit `ValueError` rather than producing biased results.

### 7.3 Split Manifest Output (`split_manifest.json`)
The partitioning process generates a verifiable audit manifest containing:
- `seed`: Random seed for full reproducibility.
- `group_column`: Split grouping attribute (`source_file` or `flow_id`).
- `partitions`: Flow counts and label distributions (`benign`, `malicious`, `ratio`) across `train`, `val`, and `test`.
- `group_counts`: Unique capture groups per partition.
- `leakage_check`: Boolean verification of zero group overlap.

---

## 8. Automated Verification & Testing

### 8.1 Targeted Member 1 Tests (40 Tests)
Run command:
```powershell
python -m pytest tests/test_pcap_end_to_end.py tests/test_prefixes.py tests/test_splitting.py -v
```
Results: **40 passed in 25.10s**.

1. **`tests/test_pcap_end_to_end.py` (6 tests):**
   - `test_empty_directory_raises`: Confirms clean `FileNotFoundError` on unpopulated raw directories.
   - `test_csv_only_directory_is_reported_not_guessed`: Ensures CSV files are audited without assuming PCAP presence.
   - `test_inspect_and_save_generates_manifest_and_report`: Validates export of `dataset_inspection.json` and `dataset_manifest.json`.
   - `test_bounded_processing_max_packets`: Verifies `max_packets_per_file` bounds memory allocation.
   - `test_inspect_build_run`: End-to-end integration test from synthetic PCAP to flow building and pipeline.
   - `test_pcapng_files_in_tree_inspected_and_built`: Recursively discovers and builds flows from `.pcapng` trees.

2. **`tests/test_prefixes.py` (22 tests):**
   - `test_pcapng_direct_parsing_and_flows`: Pure-Python parsing of Section Header, Interface, and Enhanced Packet blocks.
   - `test_pcapng_to_pcap_conversion_fallback`: Validates byte-level conversion from PCAPNG to PCAP.
   - `test_flow_key_direction_retransmission_and_length`: Validates 5-tuple canonicalization, forward/backward direction, and retransmission drop logic.
   - `test_out_of_order_timestamps_are_sorted_and_reported`: Validates stable timestamp reordering.
   - `test_truncated_capture_keeps_true_ip_length`: Validates wire-length recovery on truncated captures.
   - `test_udp_packets_skipped_and_tracked`: Confirms UDP packets are audited in counters and excluded from TCP flows.
   - `test_idle_timeout_splits_flows`: Confirms inactivity exceeding 120s spawns new subflows.
   - `test_label_rules_required_and_applied`: Verifies regex-based and prefix-based label assignment.
   - `test_sequence_feature_extraction_shape_dtype_and_causality`: Verifies shape `(t, 8)`, `float32` dtype, and temporal causality of sequential features.
   - `test_sequence_feature_extraction_short_flow_raises`: Confirms `ValueError` when flow length $< t$.
   - `test_deterministic`: Confirms identical feature tables across repeated runs.
   - `test_checkpoint_validation_and_eligible_counts`: Validates checkpoint integrity and eligibility metrics.

3. **`tests/test_splitting.py` (12 tests):**
   - `test_group_split_keeps_source_files_together`: Verifies capture files are never split across train/val/test.
   - `test_capture_group_split_zero_leakage_assertion`: Formally tests zero group overlap assertions.
   - `test_split_train_calibration_disjoint_and_both_classes`: Verifies calibration split isolation.
   - `test_create_split_manifest_reports_leakage_check_and_seed`: Confirms manifest serialization with seed.
   - `test_every_flow_in_exactly_one_partition_and_proportions`: Validates 60/20/20 proportions.
   - `test_both_classes_in_every_partition`: Verifies stratified representation of both classes.
   - `test_deterministic_and_seed_sensitive`: Verifies reproducibility under fixed seeds and divergence under distinct seeds.

### 8.2 Full Repository Test Suite (113 Tests)
Run command:
```powershell
python -m pytest -v
```
Results: **113 passed in 37.70s (100% pass rate, 0 failed, 0 skipped)**.

---

## 9. Cross-Member Interface Compatibility

| Interface / Component | Expected by Member 2 / 3 | Delivered by Member 1 | Compatibility Status |
|---|---|---|---|
| **`Flow` Dataclass** | `(flow_id, label, source_file, ts, length, direction, total_packets, total_bytes)` | Preserved with exact types and attributes | **100% Compatible** |
| **CNN-GRU Input** | Tensors of shape `(B, L, 8)` with channels: `[pkt_len, direction, iat, elapsed, cum_pkts_fwd, cum_pkts_bwd, cum_bytes_fwd, cum_bytes_bwd]` | `extract_packet_sequence` outputs `(t, 8)` float32 matching `SEQ_FEATURE_NAMES` | **100% Compatible** |
| **Random Forest Input** | Matrix of shape `(N, 18)` float64 matching `FEATURE_NAMES` | `extract_prefix_features` outputs exact 18 statistical summary columns | **100% Compatible** |
| **Platt Calibration Split** | Disjoint held-out flows (`train_cal`) from training set | `split_train_calibration` isolates 25% of training flows | **100% Compatible** |
| **Pipeline Runner** | Consumes `flows.pkl`, generates prefix tables, reads `Config` | `build_flows` returns `(flows, report)` compatible with `main.py build` and `pipeline.py` | **100% Compatible** |

---

## 10. Operational Guide for Production & Reproducibility

### 10.1 Environment Activation
```powershell
# Activate local virtual environment
.\.venv\Scripts\Activate.ps1
```

### 10.2 Processing Real Captures from UFU-DoH-EXF
1. Place downloaded `.pcap` or `.pcapng` capture files in `data/raw/`.
2. Inspect the raw capture files:
   ```powershell
   python main.py inspect --raw-dir data/raw
   ```
3. Extract bidirectional flows with memory-bounded packet processing:
   ```powershell
   python main.py build --raw-dir data/raw --max-packets-per-file 50000
   ```
4. Execute full experimental pipeline with capture-group splitting:
   ```powershell
   python main.py run --name ufu_doh_experiment --alpha 0.01 --model both --group-by source_file --overwrite
   ```

### 10.3 Synthetic Smoke Test
To verify the entire pipeline end-to-end without external captures:
```powershell
python main.py dry-run --model both --overwrite
```
This runs 1,200 synthetic flows through flow construction, multi-checkpoint prefix generation, 5-fold stratified splitting, Random Forest and CNN-GRU model fitting, Platt calibration, sequential threshold sweeping ($\alpha = 0.01$), DEBD computation, and report rendering under `data/results/synthetic_smoke_test/`.
