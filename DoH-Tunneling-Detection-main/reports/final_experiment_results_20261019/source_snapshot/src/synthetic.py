"""SYNTHETIC data for dry runs and unit tests ONLY.

Nothing here resembles measured CIRA-CIC-DoHBrw-2020 statistics. The generator
just produces two overlapping classes where some malicious flows reveal
themselves late (so that early vs. late decisions differ), which lets the
pipeline be exercised end to end. Results obtained on this data say nothing
about the real dataset or about the research question.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .preprocessing import Flow


def make_synthetic_flows(n_benign=400, n_malicious=400, seed=0, short_frac=0.1, n_files=10):
    rng = np.random.default_rng(seed)
    flows = []
    for label, n_flows in ((0, n_benign), (1, n_malicious)):
        for i in range(n_flows):
            n = int(rng.integers(5, 10)) if rng.random() < short_frac else int(rng.integers(45, 160))
            onset = int(rng.integers(0, 45))                   # packet index after which malicious traits appear
            strength = rng.uniform(0.3, 1.0)
            lens, iats, dirs = [], [], []
            for k in range(n):
                tell = label == 1 and k >= onset and rng.random() < strength
                if tell:
                    lens.append(rng.normal(180, 60)); iats.append(rng.lognormal(-4.0, 0.7))
                    dirs.append(1 if rng.random() < 0.65 else -1)
                else:
                    lens.append(rng.normal(300, 160)); iats.append(rng.lognormal(-3.0, 1.0))
                    dirs.append(1 if rng.random() < 0.5 else -1)
            length = np.clip(np.round(lens), 40, 1500).astype(np.int64)
            ts = 1000.0 + np.concatenate([[0.0], np.cumsum(iats[1:])])
            flows.append(Flow(
                flow_id=f"syn_{label}_{i}", label=label,
                source_file=f"syn_file_{label}_{i % n_files}.pcap",
                ts=ts, length=length, direction=np.array(dirs, dtype=np.int8),
                total_packets=n, total_bytes=int(length.sum())))
    return flows


# ------------------------------------------------------------------ tiny pcap writer (tests)
def _tcp_frame(src, sport, dst, dport, seq, flags, payload_len, ack=0):
    """Ethernet+IPv4+TCP frame with zeroed checksums; payload is zero bytes (metadata only)."""
    ip_total = 20 + 20 + payload_len
    eth = b"\x00" * 12 + b"\x08\x00"
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, ip_total, 0, 0x4000, 64, 6, 0,
                     bytes(map(int, src.split("."))), bytes(map(int, dst.split("."))))
    tcp = struct.pack("!HHIIBBHHH", sport, dport, seq & 0xFFFFFFFF, ack, 5 << 4, flags, 65535, 0, 0)
    return eth + ip + tcp + b"\x00" * payload_len


def write_pcap(path: Path, packets, snaplen_cut: int | None = None):
    """packets: iterable of (ts, src, sport, dst, dport, seq, flags, payload_len). Classic pcap, little endian."""
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for ts, src, sport, dst, dport, seq, flags, pl in packets:
            fr = _tcp_frame(src, sport, dst, dport, seq, flags, pl)
            cap = fr if snaplen_cut is None else fr[:snaplen_cut]
            sec = int(ts)
            f.write(struct.pack("<IIII", sec, int(round((ts - sec) * 1e6)), len(cap), len(fr)))
            f.write(cap)
