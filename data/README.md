# Local capture files

The recorded real-data experiment used these files directly in this directory:

```text
data/File01_DoH_Benign.pcapng
data/File02_DoH_Benign.pcapng
data/File03_DoH_Benign.pcapng
data/File04_DoH_Benign.pcapng
data/File06_DoH_Exfiltration.pcap
data/File07_DoH_Exfiltration.pcap
data/File10_DoH_Exfiltration.pcap
```

All `.pcap`, `.pcapng` and `.cap` files are ignored by Git, including these seven
inputs (1,703,660,902 bytes total, approximately 1.59 GiB). They remain on disk.
Do not force-add them.

Verify file identity against
[`capture_manifest.csv`](../reports/final_experiment_results_20261019/metrics/capture_manifest.csv).
Capture labels are scenario-derived: benign=0, exfiltration=1.

The study script discovers `data/*.pcap*` nonrecursively. `raw/`, `processed/` and
`results/` are generic CLI locations; their contents are local except `.gitkeep`.
Do not ingest duplicate copies of captures from multiple directories.
