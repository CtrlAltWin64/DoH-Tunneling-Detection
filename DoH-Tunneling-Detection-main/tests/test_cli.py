import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import main


class Cli(unittest.TestCase):
    def test_all_requires_name_before_building(self):
        with patch.object(main, "build_flows") as build, self.assertRaises(SystemExit):
            main.main(["all"])
        build.assert_not_called()

    def test_build_preserves_existing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flows.pkl"
            path.write_bytes(b"valuable dataset")
            with patch.object(main, "build_flows") as build, self.assertRaises(FileExistsError):
                main.main(["build", "--processed-dir", tmp])
            build.assert_not_called()
            self.assertEqual(path.read_bytes(), b"valuable dataset")

    def test_all_checks_run_destination_before_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "existing"
            out.mkdir()
            (out / "summary.md").write_text("preserve")
            with patch.object(main, "build_flows") as build, self.assertRaises(FileExistsError):
                main.main(["all", "--name", "existing", "--results-dir", tmp])
            build.assert_not_called()

    def test_dry_run_honors_paths_checkpoints_epochs_and_overwrite(self):
        with patch.object(main, "run_experiment") as run, patch.object(main, "make_synthetic_flows", return_value=[]):
            main.main(["dry-run", "--name", "new_run", "--processed-dir", "custom_processed",
                       "--results-dir", "custom_results", "--checkpoints", "10", "20",
                       "--cnn-epochs", "2"])
            cfg = run.call_args.args[1]
            self.assertEqual(cfg.checkpoints, (10, 20))
            self.assertEqual(cfg.cnn_epochs, 2)
            self.assertEqual(cfg.processed_dir, "custom_processed")
            self.assertEqual(str(run.call_args.args[2]), str(main.Path("custom_results/new_run")))
            self.assertFalse(run.call_args.kwargs["overwrite"])

    def test_invalid_checkpoints_fail_before_execution(self):
        with patch.object(main, "run_experiment") as run, self.assertRaises(ValueError):
            main.main(["dry-run", "--checkpoints", "20", "10"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
