"""Inspect whatever is in the raw-data folder BEFORE any dataset-specific assumption is made.

Reports: file types and sizes, pcap headers, CSV columns, label-like columns with value
counts, and a heuristic verdict on whether the CSVs hold per-packet records or only
completed-flow statistics. The verdict is a heuristic; read the printed columns yourself.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .preprocessing import PcapFormatError, PcapReader

PACKET_HINTS = ("packet", "pkt", "timestamp", "iat", "inter")
FLOW_STAT_HINTS = ("mean", "median", "std", "skew", "mode", "coefficient", "duration", "flowbytes", "flowsent", "flowreceived")
LABEL_HINTS = ("label", "class", "category", "type")


def inspect_csv(path: Path, nrows_preview: int = 5) -> dict:
    head = pd.read_csv(path, nrows=nrows_preview, low_memory=False)
    cols = list(head.columns)
    info = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2), "n_columns": len(cols), "columns": cols}
    low = [c.lower() for c in cols]
    n_flow_stat = sum(any(h in c for h in FLOW_STAT_HINTS) for c in low)
    info["flow_statistic_like_columns"] = n_flow_stat
    info["packet_hint_columns"] = [c for c, l in zip(cols, low) if any(h in l for h in PACKET_HINTS)]
    labels = {}
    for c in [c for c, l in zip(cols, low) if any(h in l for h in LABEL_HINTS)]:
        cnt = Counter()
        for chunk in pd.read_csv(path, usecols=[c], chunksize=200_000, low_memory=False):
            cnt.update(chunk[c].astype(str).value_counts().to_dict())
        if len(cnt) <= 50:
            labels[c] = dict(cnt)
    info["label_columns"] = labels
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=[cols[0]], chunksize=500_000, low_memory=False):
        n_rows += len(chunk)
    info["n_rows"] = n_rows
    info["rows_have_flow_endpoints"] = any("ip" in c or "port" in c for c in low)
    return info


def inspect_pcap(path: Path, max_packets: int = 2000) -> dict:
    info = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2)}
    try:
        r = PcapReader(path)
    except PcapFormatError as e:
        info["error"] = str(e)
        return info
    n, first, last = 0, None, None
    for ts, *_ in r:
        n += 1
        first = ts if first is None else first
        last = ts
        if n >= max_packets:
            break
    info.update(linktype=r.linktype, nanosecond=r.nano, packets_sampled=n,
                sampled_span_s=None if first is None else round(last - first, 3))
    return info


def inspect_dir(raw_dir: str | Path) -> dict:
    raw = Path(raw_dir)
    if not raw.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw}")
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"{raw} is empty. Place the dataset files there (see README section D).")
    ext = Counter(p.suffix.lower() or "<none>" for p in files)
    report = {"raw_dir": str(raw), "n_files": len(files), "extensions": dict(ext),
              "directories": sorted({p.parent.relative_to(raw).as_posix() for p in files}),
              "csv": [], "pcap": [], "other_examples": []}
    for p in files:
        s = p.suffix.lower()
        if s == ".csv":
            try:
                report["csv"].append(inspect_csv(p))
            except Exception as e:                                  # report, never crash the inspection
                report["csv"].append({"file": p.name, "error": repr(e)})
        elif s in (".pcap", ".pcapng", ".cap"):
            report["pcap"].append({"path": p.relative_to(raw).as_posix(), **inspect_pcap(p)})
        elif len(report["other_examples"]) < 10:
            report["other_examples"].append(p.relative_to(raw).as_posix())
    if report["pcap"]:
        verdict = ("PCAP files found: packet-level prefixes CAN be built. Set config.pcap_label_rules from the "
                   "directories above, then run `python main.py build`.")
    elif report["csv"]:
        verdict = ("Only CSV files found. If their columns are completed-flow statistics (typical for flow "
                   "meters), genuine progressive prefixes CANNOT be built from them: full-flow statistics must "
                   "not be used as partial-traffic features. Obtain the raw PCAPs (see README).")
    else:
        verdict = "No CSV or PCAP files found."
    report["verdict"] = verdict
    return report


def print_report(report: dict) -> None:
    print(f"Raw directory : {report['raw_dir']}\nFiles         : {report['n_files']}  {report['extensions']}")
    print("Directories   :", report["directories"][:20])
    for c in report["csv"]:
        print(f"\n[CSV] {c['file']}  rows={c.get('n_rows')}  cols={c.get('n_columns')}  size={c.get('size_mb')} MB")
        if "error" in c:
            print("  ERROR:", c["error"]); continue
        print("  columns:", c["columns"])
        print("  label-like columns:", c["label_columns"])
        print("  packet-hint columns:", c["packet_hint_columns"], "| flow-stat-like columns:", c["flow_statistic_like_columns"])
    for p in report["pcap"][:30]:
        print("\n[PCAP]", p)
    if len(report["pcap"]) > 30:
        print(f"... {len(report['pcap']) - 30} more PCAP files in the JSON report")
    print("\nVERDICT:", report["verdict"])


def save_report(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "dataset_inspection.json"
    p.write_text(json.dumps(report, indent=2))
    return p
