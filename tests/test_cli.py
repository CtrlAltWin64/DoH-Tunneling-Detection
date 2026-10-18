import unittest
from unittest.mock import patch

import main


class Cli(unittest.TestCase):
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
