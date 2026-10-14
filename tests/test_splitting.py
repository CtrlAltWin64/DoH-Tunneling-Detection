import unittest

import pandas as pd

from src.dataset_split import split_flows
from src.prefix_generation import build_prefix_table
from src.synthetic import make_synthetic_flows

CPS = (10, 20, 30, 40)


def meta_of(flows):
    return pd.DataFrame({"flow_id": [f.flow_id for f in flows], "label": [f.label for f in flows],
                         "source_file": [f.source_file for f in flows]})


class FlowLevelSplit(unittest.TestCase):
    def setUp(self):
        self.flows = make_synthetic_flows(150, 150, seed=1, short_frac=0.0)
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
        self.assertGreater((df.groupby("flow_id").size() > 1).sum(), 0)   # flows really have several prefix rows

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


if __name__ == "__main__":
    unittest.main()
