import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rubik_rl.checkpoint import CheckpointManager
from rubik_rl.evaluate_checkpoint import _aggregate_metrics, build_parser, run_evaluation
from rubik_rl.policy import LinearSoftmaxPolicy


class TestEvaluateCheckpoint(unittest.TestCase):
    def test_parser_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.episodes_per_scramble, 100000)
        self.assertEqual(args.scramble_min, 1)
        self.assertEqual(args.scramble_max, 20)
        self.assertEqual(args.max_episode_steps, 100)
        self.assertEqual(args.eval_batch_size, 4096)
        self.assertEqual(args.progress, "on")

    def test_metrics_aggregation(self):
        solved = np.array([True, False, True, False, True], dtype=bool)
        steps = np.array([3, 100, 5, 100, 7], dtype=np.int64)
        m = _aggregate_metrics(scramble_depth=4, solved=solved, steps=steps, eval_time_sec=2.0)
        self.assertEqual(m.scramble_depth, 4)
        self.assertEqual(m.episodes, 5)
        self.assertEqual(m.solved_count, 3)
        self.assertEqual(m.unsolved_count, 2)
        self.assertAlmostEqual(m.success_rate, 0.6, places=6)
        self.assertAlmostEqual(m.steps_solved_min, 3.0, places=6)
        self.assertAlmostEqual(m.steps_solved_mean, 5.0, places=6)
        self.assertAlmostEqual(m.steps_solved_max, 7.0, places=6)
        self.assertAlmostEqual(m.steps_all_min, 3.0, places=6)
        self.assertAlmostEqual(m.steps_all_mean, 43.0, places=6)
        self.assertAlmostEqual(m.steps_all_max, 100.0, places=6)
        self.assertAlmostEqual(m.episodes_per_sec, 2.5, places=6)

    def test_smoke_evaluation_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "ckpt"
            out_dir = Path(td) / "out"
            mgr = CheckpointManager(str(ckpt_dir))
            policy = LinearSoftmaxPolicy(seed=1)
            mgr.save(policy, episode=42)

            args = argparse.Namespace(
                checkpoint_dir=str(ckpt_dir),
                checkpoint_path=None,
                device="cpu",
                episodes_per_scramble=64,
                scramble_min=1,
                scramble_max=2,
                max_episode_steps=10,
                eval_batch_size=16,
                seed=123,
                output_dir=str(out_dir),
                output_prefix="smoke",
                progress="off",
            )
            out = run_evaluation(args)
            self.assertTrue(Path(out["sr_plot"]).exists())
            self.assertTrue(Path(out["steps_plot"]).exists())
            self.assertTrue(Path(out["csv"]).exists())
            self.assertTrue(Path(out["json"]).exists())
            self.assertEqual(len(out["metrics"]), 2)

    def test_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                checkpoint_dir=td,
                checkpoint_path=None,
                device="cpu",
                episodes_per_scramble=8,
                scramble_min=1,
                scramble_max=1,
                max_episode_steps=5,
                eval_batch_size=4,
                seed=1,
                output_dir=td,
                output_prefix="x",
                progress="off",
            )
            with self.assertRaises(RuntimeError):
                run_evaluation(args)


if __name__ == "__main__":
    unittest.main()
