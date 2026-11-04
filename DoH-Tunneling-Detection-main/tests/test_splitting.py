import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dataset_split import (
    create_split_manifest,
    save_split_manifest,
    split_flows,
    split_train_calibration,
)
from src.prefix_generation import build_prefix_table
from src.synthetic import make_synthetic_flows

CPS = (10, 20, 30, 40)


def meta_of(flows):
    return pd.DataFrame({
        "flow_id": [f.flow_id for f in flows],
        "label": [f.label for f in flows],
        "source_file": [f.source_file for f in flows],
    })


class FlowLevelSplit(unittest.TestCase):
    def setUp(self):
        self.flows = make_synthetic_flows(150, 150, seed=1, short_frac=0.0, n_files=10)
        self.meta = meta_of(self.flows)

    def test_every_flow_in_exactly_one_partition_and_proportions(self):
        a = split_flows(self.meta, "flow", 7)
        self.assertEqual(set(a), set(self.meta["flow_id"]))
        counts = pd.Series(a).value_counts()
        self.assertEqual(counts.sum(), len(self.flows))
        self.assertAlmostEqual(counts["train"] / len(self.flows), 0.6, delta=0.05)

    def test_all_prefixes_of_a_flow_share_the_partition(self):
        a = split_flows(self.meta, "flow", 7)
        df = build_prefix_table(self.flows, CPS)
        df["partition"] = df["flow_id"].map(a)
        self.assertFalse(df["partition"].isna().any())
        self.assertTrue((df.groupby("flow_id")["partition"].nunique() == 1).all())
        self.assertGreater((df.groupby("flow_id").size() > 1).sum(), 0)  # flows really have several prefix rows

    def test_no_flow_overlap_between_partitions(self):
        a = split_flows(self.meta, "flow", 7)
        parts = {p: {k for k, v in a.items() if v == p} for p in ("train", "val", "test")}
        self.assertFalse(parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"])

    def test_both_classes_in_every_partition(self):
        a = split_flows(self.meta, "flow", 7)
        m = self.meta.assign(p=self.meta["flow_id"].map(a))
        self.assertTrue((m.groupby("p")["label"].nunique() == 2).all())

    def test_deterministic_and_seed_sensitive(self):
        self.assertEqual(split_flows(self.meta, "flow", 7), split_flows(self.meta, "flow", 7))
        self.assertNotEqual(split_flows(self.meta, "flow", 7), split_flows(self.meta, "flow", 8))

    def test_group_split_keeps_source_files_together(self):
        a = split_flows(self.meta, "source_file", 3)
        m = self.meta.assign(p=self.meta["flow_id"].map(a))
        self.assertTrue((m.groupby("source_file")["p"].nunique() == 1).all())

    def test_too_few_groups_raises(self):
        m = self.meta.copy()
        m["source_file"] = m["label"].map({0: "b.pcap", 1: "m.pcap"})
        with self.assertRaises(ValueError):
            split_flows(m, "source_file", 0)

    def test_duplicate_flow_ids_rejected(self):
        m = pd.concat([self.meta, self.meta.iloc[:1]])
        with self.assertRaises(ValueError):
            split_flows(m, "flow", 0)

    def test_missing_column_raises(self):
        m = self.meta.drop(columns=["source_file"])
        with self.assertRaises(ValueError):
            split_flows(m, "flow", 0)

    def test_split_train_calibration_disjoint_and_both_classes(self):
        assign = split_train_calibration(self.meta, "flow", 42, cal_ratio=0.25)
        self.assertEqual(set(assign), set(self.meta["flow_id"]))
        parts = pd.Series(assign)
        self.assertIn("train_fit", parts.values)
        self.assertIn("train_cal", parts.values)
        m = self.meta.assign(sub=self.meta["flow_id"].map(assign))
        for sp in ("train_fit", "train_cal"):
            self.assertEqual(m[m["sub"] == sp]["label"].nunique(), 2)

    def test_create_split_manifest_reports_leakage_check_and_seed(self):
        a = split_flows(self.meta, "flow", 7)
        manifest = create_split_manifest(self.meta, a, "flow", seed=7)
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(manifest["group_by"], "flow")
        self.assertEqual(manifest["total_flows"], len(self.meta))
        self.assertTrue(manifest["leakage_check"]["verified_disjoint"])
        self.assertEqual(manifest["leakage_check"]["train_val_overlap"], 0)
        self.assertEqual(manifest["leakage_check"]["train_test_overlap"], 0)
        self.assertEqual(manifest["leakage_check"]["val_test_overlap"], 0)
        self.assertIn("train", manifest["partitions"])
        self.assertIn("val", manifest["partitions"])
        self.assertIn("test", manifest["partitions"])

        # Test saving manifest
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "split_manifest.json"
            saved = save_split_manifest(manifest, out_p)
            self.assertTrue(saved.exists())
            import json
            loaded = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(loaded["seed"], 7)
            self.assertTrue(loaded["leakage_check"]["verified_disjoint"])

    def test_capture_group_split_zero_leakage_assertion(self):
        """Strict capture-level test: verify no source capture file appears in multiple partitions."""
        a = split_flows(self.meta, "source_file", 42)
        m = self.meta.assign(part=self.meta["flow_id"].map(a))

        train_files = set(m[m["part"] == "train"]["source_file"])
        val_files = set(m[m["part"] == "val"]["source_file"])
        test_files = set(m[m["part"] == "test"]["source_file"])

        self.assertEqual(len(train_files & val_files), 0, "Train and Val share capture files!")
        self.assertEqual(len(train_files & test_files), 0, "Train and Test share capture files!")
        self.assertEqual(len(val_files & test_files), 0, "Val and Test share capture files!")


if __name__ == "__main__":
    unittest.main()
