import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import Config
from src.feature_extraction import FEATURE_NAMES, META_COLUMNS, prefix_feature_vector
from src.prefix_generation import build_prefix_table
from src.preprocessing import PcapFormatError, PcapReader, build_flows, build_flows_from_pcap
from src.synthetic import write_pcap
from tests._helpers import make_flow

CPS = (10, 20, 30, 40)


class PrefixConstruction(unittest.TestCase):
    def test_rows_exist_only_where_flow_is_long_enough(self):
        flows = [make_flow(45, fid="a"), make_flow(25, fid="b", seed=1), make_flow(9, fid="c", seed=2)]
        df = build_prefix_table(flows, CPS)
        got = {fid: sorted(g["checkpoint"]) for fid, g in df.groupby("flow_id")}
        self.assertEqual(got["a"], [10, 20, 30, 40])
        self.assertEqual(got["b"], [10, 20])
        self.assertNotIn("c", got)                       # shorter than 10 packets: no padding, no row

    def test_prefix_length_feature_equals_checkpoint(self):
        df = build_prefix_table([make_flow(60)], CPS)
        self.assertTrue((df["n_packets"] == df["checkpoint"]).all())

    def test_cannot_build_prefix_longer_than_flow(self):
        f = make_flow(15)
        with self.assertRaises(ValueError):
            prefix_feature_vector(f.ts, f.length, f.direction, 20)

    def test_hand_computed_features(self):
        ts = np.array([0.0, 1.0, 3.0, 6.0, 100.0])
        ln = np.array([100, 200, 300, 400, 999])
        dr = np.array([1, -1, 1, -1, 1])
        v = dict(zip(FEATURE_NAMES, prefix_feature_vector(ts, ln, dr, 4)))
        self.assertEqual(v["n_packets"], 4)
        self.assertEqual(v["bytes_total"], 1000)
        self.assertEqual((v["fwd_packets"], v["bwd_packets"]), (2, 2))
        self.assertEqual((v["fwd_bytes"], v["bwd_bytes"]), (400, 600))
        self.assertAlmostEqual(v["len_mean"], 250.0)
        self.assertAlmostEqual(v["len_std"], np.std([100, 200, 300, 400]))
        self.assertEqual((v["len_min"], v["len_max"]), (100, 400))
        self.assertAlmostEqual(v["iat_mean"], 2.0)                 # gaps 1,2,3
        self.assertEqual((v["iat_min"], v["iat_max"]), (1.0, 3.0))
        self.assertAlmostEqual(v["duration"], 6.0)                 # prefix duration, NOT flow duration

    def test_causality_future_packets_do_not_change_features(self):
        """Leakage detector: scrambling every packet after t must not change the t-packet features."""
        f = make_flow(80, seed=3)
        for t in CPS:
            base = prefix_feature_vector(f.ts, f.length, f.direction, t)
            ts2, ln2, dr2 = f.ts.copy(), f.length.copy(), f.direction.copy()
            ts2[t:] += 1e6 * np.arange(len(ts2) - t)
            ln2[t:] = 1
            dr2[t:] = -dr2[t:]
            np.testing.assert_array_equal(base, prefix_feature_vector(ts2, ln2, dr2, t))

    def test_prefix_table_row_independent_of_flow_tail_and_totals(self):
        f = make_flow(80, seed=4)
        g = replace(f, ts=f.ts.copy(), length=f.length.copy(), direction=f.direction.copy(),
                    total_bytes=f.total_bytes * 7)
        g.length[40:] = 1500
        a = build_prefix_table([f], CPS)
        b = build_prefix_table([g], CPS)
        np.testing.assert_array_equal(a[FEATURE_NAMES].to_numpy(), b[FEATURE_NAMES].to_numpy())

    def test_flow_level_totals_are_not_model_features(self):
        self.assertTrue(set(FEATURE_NAMES).isdisjoint(META_COLUMNS))
        self.assertFalse(any(n.startswith("flow_total") for n in FEATURE_NAMES))

    def test_deterministic(self):
        flows = [make_flow(70, fid=f"x{i}", seed=i) for i in range(5)]
        a, b = build_prefix_table(flows, CPS), build_prefix_table(flows, CPS)
        self.assertTrue(a.equals(b))


class PcapPreprocessing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = Config(doh_port=443)

    def tearDown(self):
        self.tmp.cleanup()

    def _pcap(self, name, packets, **kw):
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        write_pcap(p, packets, **kw)
        return p

    def test_flow_key_direction_retransmission_and_length(self):
        C, S = "10.0.0.1", "1.1.1.1"
        pk = [
            (1.0, C, 40000, S, 443, 1000, 0x18, 100),   # fwd, payload 100
            (1.1, S, 443, C, 40000, 5000, 0x18, 300),   # bwd
            (1.2, C, 40000, S, 443, 1000, 0x18, 100),   # exact retransmission -> dropped
            (1.3, C, 40000, S, 443, 1100, 0x18, 50),    # fwd new data
            (1.4, "10.0.0.2", 41000, S, 443, 1, 0x18, 70),  # unrelated connection
            (1.5, C, 40000, "8.8.8.8", 53, 1, 0x18, 70),    # wrong port -> filtered
        ]
        p = self._pcap("t.pcap", pk)
        from collections import Counter as C_
        st = C_()
        flows = build_flows_from_pcap(p, 0, "t.pcap", self.cfg, st)
        self.assertEqual(len(flows), 2)                          # both directions merged; other host separate
        f = flows[0]
        self.assertEqual(list(f.direction), [1, -1, 1])
        self.assertEqual(list(f.length), [140, 340, 90])         # IP total length = 20 + 20 + payload
        self.assertEqual(f.total_bytes, 570)
        self.assertEqual(st["dropped_retransmissions"], 1)
        self.assertEqual(st["skipped_not_doh_port"], 1)

    def test_out_of_order_timestamps_are_sorted_and_reported(self):
        C, S = "10.0.0.1", "1.1.1.1"
        pk = [(2.0, C, 40000, S, 443, 1, 0x10, 10), (1.0, S, 443, C, 40000, 1, 0x10, 10),
              (3.0, C, 40000, S, 443, 11, 0x10, 10)]
        from collections import Counter as C_
        st = C_()
        f = build_flows_from_pcap(self._pcap("o.pcap", pk), 1, "o.pcap", self.cfg, st)[0]
        self.assertTrue(np.all(np.diff(f.ts) >= 0))
        self.assertEqual(st["flows_with_out_of_order_timestamps_sorted"], 1)

    def test_idle_timeout_splits_flows(self):
        C, S = "10.0.0.1", "1.1.1.1"
        pk = [(1.0, C, 40000, S, 443, 1, 0x10, 10), (500.0, C, 40000, S, 443, 11, 0x10, 10)]
        from collections import Counter as C_
        flows = build_flows_from_pcap(self._pcap("i.pcap", pk), 0, "i.pcap", self.cfg, C_())
        self.assertEqual(len(flows), 2)

    def test_truncated_capture_keeps_true_ip_length(self):
        C, S = "10.0.0.1", "1.1.1.1"
        pk = [(1.0, C, 40000, S, 443, 1, 0x10, 500)]
        from collections import Counter as C_
        st = C_()
        f = build_flows_from_pcap(self._pcap("s.pcap", pk, snaplen_cut=80), 0, "s.pcap", self.cfg, st)[0]
        self.assertEqual(f.length[0], 540)
        self.assertEqual(st["truncated_packets"], 1)

    def test_label_rules_required_and_applied(self):
        C, S = "10.0.0.1", "1.1.1.1"
        pk = [(1.0, C, 40000, S, 443, 1, 0x10, 10)]
        self._pcap("Benign/a.pcap", pk)
        self._pcap("Malicious/b.pcap", pk)
        self._pcap("other/c.pcap", pk)
        with self.assertRaises(ValueError):
            build_flows(self.cfg, self.dir)                       # empty rules -> refuse to guess
        cfg = replace(self.cfg, pcap_label_rules={"benign": "benign", "malicious": "malicious"})
        flows, rep = build_flows(cfg, self.dir)
        self.assertEqual(Counter(f.label for f in flows), Counter({0: 1, 1: 1}))
        self.assertEqual(rep["files_unlabelled_skipped"], ["other/c.pcap"])

    def test_pcapng_and_garbage_rejected(self):
        p = self.dir / "x.pcap"
        p.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 40)
        with self.assertRaises(PcapFormatError):
            PcapReader(p)
        p.write_bytes(b"not a pcap at all, definitely" * 3)
        with self.assertRaises(PcapFormatError):
            PcapReader(p)


if __name__ == "__main__":
    unittest.main()
