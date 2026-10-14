"""PCAP -> inspect -> build -> full pipeline, on tiny SYNTHETIC pcap files written by the test itself."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import Config
from src.inspect_dataset import inspect_dir
from src.pipeline import run_experiment
from src.preprocessing import build_flows
from src.synthetic import write_pcap


def make_pcap_tree(root: Path, seed=0, files_per_class=6, flows_per_file=12, pkts=50):
    rng = np.random.default_rng(seed)
    for cls, folder in ((0, "Benign"), (1, "Malicious")):
        for fi in range(files_per_class):
            packets, t0 = [], 1000.0
            for k in range(flows_per_file):
                client, sport = f"10.{cls}.{fi}.{k + 1}", 40000 + k
                seq = {1: 1000, -1: 5000}
                t = t0 + k * 0.01
                for i in range(pkts):
                    d = 1 if rng.random() < (0.65 if cls else 0.5) else -1
                    pl = int(abs(rng.normal(150 if cls else 300, 80))) + 1
                    t += rng.lognormal(-4 if cls else -3, 0.8)
                    src, sp, dst, dp = (client, sport, "1.1.1.1", 443) if d == 1 else ("1.1.1.1", 443, client, sport)
                    packets.append((t, src, sp, dst, dp, seq[d], 0x18, pl))
                    seq[d] += pl
            packets.sort(key=lambda p: p[0])
            (root / folder).mkdir(parents=True, exist_ok=True)
            write_pcap(root / folder / f"cap{fi}.pcap", packets)


class PcapEndToEnd(unittest.TestCase):
    def test_inspect_build_run(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            make_pcap_tree(raw)
            rep = inspect_dir(raw)
            self.assertEqual(len(rep["pcap"]), 12)
            self.assertIn("packet-level prefixes CAN be built", rep["verdict"])

            cfg = Config(raw_dir=str(raw), processed_dir=td, results_dir=td, rf_n_estimators=20, seed=3,
                         pcap_label_rules={"benign": "benign", "malicious": "malicious"},
                         jitter_settings=((0.5, 0.05),), low_and_slow_factors=(5.0,))
            flows, report = build_flows(cfg)
            self.assertEqual(len(flows), 2 * 6 * 12)
            self.assertEqual({f.label for f in flows}, {0, 1})
            self.assertTrue(all(len(f.ts) == 50 for f in flows))      # flows kept separate, none merged
            res = run_experiment(flows, cfg, Path(td) / "out", log=lambda *a: None)
            self.assertIn("adaptive", res["thresholds"])
            self.assertTrue((Path(td) / "out" / "summary.md").exists())

            cfg_g = replace(cfg, group_by="source_file")              # file-grouped split also works
            run_experiment(flows, cfg_g, Path(td) / "out_g", log=lambda *a: None)

    def test_csv_only_directory_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "l2.csv").write_text("Duration,FlowBytesSent,PacketLengthMean,Label\n1,2,3,Benign\n4,5,6,Malicious\n")
            rep = inspect_dir(td)
            self.assertEqual(rep["csv"][0]["label_columns"]["Label"], {"Benign": 1, "Malicious": 1})
            self.assertIn("CANNOT be built", rep["verdict"])
            cfg = Config(raw_dir=td, pcap_label_rules={"x": "benign"})
            with self.assertRaises(FileNotFoundError):
                build_flows(cfg)

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(FileNotFoundError):
            inspect_dir(td)


if __name__ == "__main__":
    unittest.main()
