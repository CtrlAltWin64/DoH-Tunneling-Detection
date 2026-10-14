"""PCAP -> labelled bidirectional flows with per-packet metadata.

Only metadata is kept: timestamp, IP-level packet length, direction.
No TCP payload bytes are ever read or stored.

Design decisions (all configurable in config.py):
  * Flow key  : TCP 5-tuple with the two endpoints sorted, so both directions
                map to the same flow. A flow is split into a NEW flow when
                (a) the idle gap exceeds ``flow_idle_timeout_s`` or
                (b) a pure SYN arrives after the flow already saw FIN/RST.
  * Direction : +1 = same direction as the first packet we saw for the flow
                (normally client->server), -1 = reverse. If a capture starts
                mid-connection the "initiator" is simply the first sender seen.
  * Length    : IP total length (IP header + TCP header + payload; no Ethernet
                header). Taken from the IP header, so snaplen truncation does
                not shrink it.
  * Retransmissions: a TCP segment with payload whose end sequence number is
                <= the highest end sequence already seen in that direction is
                dropped (counted in the report). Limitation: a genuinely
                out-of-order (not retransmitted) segment is also dropped.
  * Timestamps: if packets of a flow are not chronological they are stably
                sorted; the number of such flows is reported.
  * Only IPv4/IPv6-over-TCP is parsed (classic pcap, Ethernet / Linux-SLL / raw
                IP). pcapng is rejected with instructions.
"""
from __future__ import annotations

import json
import pickle
import socket
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABEL_MAP = {"benign": 0, "malicious": 1}


class PcapFormatError(ValueError):
    pass


@dataclass
class Flow:
    flow_id: str
    label: int                 # 0 = benign DoH, 1 = malicious DoH tunnelling
    source_file: str
    ts: np.ndarray             # float64 seconds, chronological
    length: np.ndarray         # int64 IP-level bytes
    direction: np.ndarray      # int8, +1 / -1
    total_packets: int         # all counted packets of the flow (may exceed len(ts))
    total_bytes: int           # all counted bytes of the flow (DEBD denominator)


# --------------------------------------------------------------------------- pcap reading
class PcapReader:
    """Minimal classic-pcap reader (no third-party dependency)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.truncated_file = False
        with open(self.path, "rb") as f:
            hdr = f.read(24)
        if len(hdr) < 24:
            raise PcapFormatError(f"{self.path}: shorter than a pcap global header")
        magic = hdr[:4]
        if magic == b"\x0a\x0d\x0d\x0a":
            raise PcapFormatError(
                f"{self.path}: this is pcapng. Convert first, e.g. "
                f"`editcap -F pcap in.pcapng out.pcap` (Wireshark) and re-run.")
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            self.endian = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            self.endian = ">"
        else:
            raise PcapFormatError(f"{self.path}: not a classic pcap file (magic={magic!r})")
        self.nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
        self.linktype = struct.unpack(self.endian + "I", hdr[20:24])[0]
        if self.linktype not in (1, 101, 113):
            raise PcapFormatError(
                f"{self.path}: unsupported link type {self.linktype} (supported: 1=Ethernet, 101=raw IP, 113=Linux SLL)")

    def __iter__(self):
        rec = struct.Struct(self.endian + "IIII")
        div = 1e9 if self.nano else 1e6
        with open(self.path, "rb") as f:
            f.seek(24)
            while True:
                h = f.read(16)
                if not h:
                    return
                if len(h) < 16:
                    self.truncated_file = True
                    return
                sec, frac, caplen, origlen = rec.unpack(h)
                data = f.read(caplen)
                if len(data) < caplen:
                    self.truncated_file = True
                    return
                yield sec + frac / div, caplen, origlen, data


def parse_tcp_packet(data: bytes, linktype: int):
    """Return ((src_ip, sport, dst_ip, dport, ip_len, flags, seq, payload_len), None)
    or (None, reason). IPs are raw bytes."""
    if linktype == 1:
        if len(data) < 14:
            return None, "malformed"
        et = struct.unpack("!H", data[12:14])[0]
        off = 14
        while et in (0x8100, 0x88A8):          # VLAN tags
            if len(data) < off + 4:
                return None, "malformed"
            et = struct.unpack("!H", data[off + 2:off + 4])[0]
            off += 4
    elif linktype == 113:
        if len(data) < 16:
            return None, "malformed"
        et = struct.unpack("!H", data[14:16])[0]
        off = 16
    else:  # raw IP
        if not data:
            return None, "malformed"
        et = 0x0800 if data[0] >> 4 == 4 else 0x86DD if data[0] >> 4 == 6 else 0
        off = 0

    if et == 0x0800:
        if len(data) < off + 20:
            return None, "malformed"
        ihl = (data[off] & 0x0F) * 4
        ip_len = struct.unpack("!H", data[off + 2:off + 4])[0]
        if ihl < 20 or ip_len < ihl:
            return None, "malformed"
        if struct.unpack("!H", data[off + 6:off + 8])[0] & 0x1FFF:
            return None, "fragment"
        if data[off + 9] != 6:
            return None, "non_tcp"
        src, dst = data[off + 12:off + 16], data[off + 16:off + 20]
        t = off + ihl
    elif et == 0x86DD:
        if len(data) < off + 40:
            return None, "malformed"
        if data[off + 6] != 6:
            return None, "non_tcp"      # includes IPv6 extension headers (not parsed)
        ip_len = 40 + struct.unpack("!H", data[off + 4:off + 6])[0]
        src, dst = data[off + 8:off + 24], data[off + 24:off + 40]
        t = off + 40
        ihl = 40
    else:
        return None, "non_ip"

    if len(data) < t + 20:
        return None, "malformed"
    sport, dport, seq = struct.unpack("!HHI", data[t:t + 8])
    data_off = (data[t + 12] >> 4) * 4
    flags = data[t + 13]
    payload = ip_len - ihl - data_off
    if data_off < 20 or payload < 0:
        return None, "malformed"
    return (src, sport, dst, dport, ip_len, flags, seq, payload), None


# --------------------------------------------------------------------------- flow building
class _State:
    __slots__ = ("init", "peer", "ts", "length", "dir", "n", "bytes", "high", "closed",
                 "last_ts", "unordered")

    def __init__(self, init, peer):
        self.init, self.peer = init, peer
        self.ts, self.length, self.dir = [], [], []
        self.n = self.bytes = 0
        self.high = [None, None]
        self.closed = False
        self.last_ts = None
        self.unordered = False


def _ip_str(b: bytes) -> str:
    return socket.inet_ntop(socket.AF_INET if len(b) == 4 else socket.AF_INET6, b)


def build_flows_from_pcap(path: Path, label: int, source_name: str, cfg, stats: Counter) -> list[Flow]:
    import ipaddress
    allowed_ips = {ipaddress.ip_address(x).packed for x in cfg.doh_server_ips}
    reader = PcapReader(path)
    active: dict = {}
    done: list = []   # (first_ts, creation_idx, state_key_str, state)
    counter = 0

    def close(key, st):
        done.append((st.ts[0] if st.ts else 0.0, len(done), key, st))

    for i, (ts, caplen, origlen, data) in enumerate(reader):
        if cfg.max_packets_per_file is not None and i >= cfg.max_packets_per_file:
            stats["pcap_files_limited_by_max_packets"] += 1
            break
        stats["packets_read"] += 1
        if caplen < origlen:
            stats["truncated_packets"] += 1
        parsed, reason = parse_tcp_packet(data, reader.linktype)
        if parsed is None:
            stats["skipped_" + reason] += 1
            continue
        src, sport, dst, dport, ip_len, flags, seq, payload = parsed
        if cfg.doh_port is not None and cfg.doh_port not in (sport, dport):
            stats["skipped_not_doh_port"] += 1
            continue
        if allowed_ips and src not in allowed_ips and dst not in allowed_ips:
            stats["skipped_not_doh_server_ip"] += 1
            continue
        if cfg.drop_zero_payload_packets and payload == 0:
            stats["skipped_zero_payload"] += 1
            continue

        a, b = (src, sport), (dst, dport)
        key = (min(a, b), max(a, b))
        st = active.get(key)
        pure_syn = bool(flags & 0x02) and not flags & 0x10
        if st is not None and (
            (st.last_ts is not None and ts - st.last_ts > cfg.flow_idle_timeout_s)
            or (st.closed and pure_syn)
        ):
            close(key, st)
            st = None
        if st is None:
            st = _State(a, b)
            active[key] = st
        d = 1 if a == st.init else -1

        if payload > 0:                                   # retransmission / duplicate filter
            di = 0 if d == 1 else 1
            end = (seq + payload) & 0xFFFFFFFF
            hi = st.high[di]
            if hi is not None and ((hi - end) & 0xFFFFFFFF) < 0x80000000:
                stats["dropped_retransmissions"] += 1
                continue
            st.high[di] = end

        if st.last_ts is not None and ts < st.last_ts:
            st.unordered = True
        st.last_ts = ts if st.last_ts is None else max(st.last_ts, ts)
        st.n += 1
        st.bytes += ip_len
        if len(st.ts) < cfg.max_stored_packets:
            st.ts.append(ts)
            st.length.append(ip_len)
            st.dir.append(d)
        if flags & 0x05:                                   # FIN or RST
            st.closed = True

    if reader.truncated_file:
        stats["pcap_files_truncated_at_end"] += 1
    for key, st in active.items():
        close(key, st)

    done.sort(key=lambda x: (x[0], x[1]))
    flows = []
    for idx, (_, _, key, st) in enumerate(done):
        ts = np.asarray(st.ts, dtype=np.float64)
        ln = np.asarray(st.length, dtype=np.int64)
        dr = np.asarray(st.dir, dtype=np.int8)
        if st.unordered:
            stats["flows_with_out_of_order_timestamps_sorted"] += 1
            order = np.argsort(ts, kind="stable")
            ts, ln, dr = ts[order], ln[order], dr[order]
        fid = (f"{source_name}#{idx}:{_ip_str(st.init[0])}:{st.init[1]}>"
               f"{_ip_str(st.peer[0])}:{st.peer[1]}")
        flows.append(Flow(fid, label, source_name, ts, ln, dr, st.n, st.bytes))
    stats["flows_built"] += len(flows)
    return flows


def match_label_rule(rel_path: str, rules: dict):
    low = rel_path.lower()
    for kw, lab in rules.items():
        if kw.lower() in low:
            return lab
    return None


def build_flows(cfg, raw_dir: str | Path | None = None):
    """Walk raw_dir, label PCAPs via cfg.pcap_label_rules, return (flows, report)."""
    cfg.validate()
    raw = Path(raw_dir or cfg.raw_dir)
    if not raw.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw}")
    if not cfg.pcap_label_rules:
        raise ValueError(
            "config.pcap_label_rules is empty. Run `python main.py inspect` to see your folder layout, "
            "then set e.g. {'benign': 'benign', 'malicious': 'malicious'} in config.py. "
            "Labels are never guessed.")
    exts = tuple(e.lower() for e in cfg.pcap_extensions)
    files = sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(
            f"no files with extensions {exts} under {raw}. If you only have the CSV files, "
            "genuine packet prefixes cannot be built - see README 'CSV-only situation'.")
    stats: Counter = Counter()
    flows: list[Flow] = []
    used, skipped, excluded = [], [], []
    for p in files:
        rel = p.relative_to(raw).as_posix()
        rule = match_label_rule(rel, cfg.pcap_label_rules)
        if rule is None:
            skipped.append(rel)
            continue
        if rule == "exclude":
            excluded.append(rel)
            continue
        flows.extend(build_flows_from_pcap(p, LABEL_MAP[rule], rel, cfg, stats))
        used.append({"file": rel, "label": rule})
    if not used:
        raise ValueError(f"no PCAP matched pcap_label_rules {cfg.pcap_label_rules}; found: {[f.relative_to(raw).as_posix() for f in files][:10]}")
    report = {
        "raw_dir": str(raw), "files_used": used, "files_unlabelled_skipped": skipped,
        "files_excluded_by_rule": excluded, "counters": dict(stats),
        "flows_total": len(flows),
        "note": "length = IP total length; retransmissions dropped; direction relative to first packet seen",
    }
    return flows, report


def save_flows(flows, report, processed_dir: str | Path, name: str = "flows") -> Path:
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(flows, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out / f"{name}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return out / f"{name}.pkl"


def load_flows(processed_dir: str | Path, name: str = "flows"):
    path = Path(processed_dir) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python main.py build` first")
    with open(path, "rb") as f:     # pickle is only ever read from files this project wrote
        return pickle.load(f)
