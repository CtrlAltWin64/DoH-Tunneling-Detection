# Local dataset inventory

Verified in this checkout on 19 October 2026:

| Present capture | Bytes |
|---|---:|
| `data/raw/Benign/File01_DoH_Benign.pcapng` | 364,378,452 |
| `data/raw/Benign/File04_DoH_Benign.pcapng` | 209,847,788 |
| `data/raw/Malicious/File06_DoH_Exfiltration.pcap` | 40,634,319 |
| `data/raw/Malicious/File07_DoH_Exfiltration.pcap` | 66,230,439 |
| `data/raw/Malicious/File10_DoH_Exfiltration.pcap` | 467,388,156 |

Total: 1,148,479,154 bytes. Current hashes are recorded in
[cleanup saved evidence](../reports/cleanup_verification_20261019/saved_evidence.json).
All five match the corresponding entries in the archived
[capture manifest](../reports/final_experiment_results_20261019/metrics/capture_manifest.csv).

`data/processed/flows.pkl` contains 139,076 flows: 978 benign and 138,098 malicious.
`flows_report.json`, `dataset_inspection.json` and `dataset_manifest.json`
describe that local dataset. Existing completed experiments are under
`data/results/exp_ufu_real_cp10_20/` and
`data/results/exp_ufu_real_cp2_4_8_10_15/`; see the
[results guide](../RESULTS_GUIDE.md) for authority and limitations.

The separate seven-capture archived report also used
`File02_DoH_Benign.pcapng` (287,519,176 bytes) and
`File03_DoH_Benign.pcapng` (267,662,572 bytes). **Both are absent here**.
That study's harness expects all seven directly under its workspace's `data/`.
Its processed data and models are also absent; the five-capture data cannot replace them.

The generic CLI recursively discovers captures under `data/raw/`, using ordered
label rules. Inspect first and verify filename-to-label rules. Labels are
scenario-derived: benign=0, exfiltration/malicious=1. TCP/443 filtering does not
authenticate DoH. Flow grouping does not hold out capture sessions; two benign
capture groups are insufficient for the current capture split.

Captures, processed data and local results are ignored by Git. Do not force-add
them or recursively ingest duplicate copies from multiple layouts. Use new
processed/result directories when running experiments. A clean clone will not
contain the local files listed above.
