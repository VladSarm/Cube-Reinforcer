import tempfile
import unittest
from pathlib import Path

import torch

from rubik_rl.checkpoint import CheckpointManager
from rubik_rl.policy import LinearSoftmaxPolicy


class TestCheckpoint(unittest.TestCase):
    def test_save_load_roundtrip_and_latest(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(td)
            p = LinearSoftmaxPolicy(seed=42)

            with torch.no_grad():
                for param in p.parameters():
                    param.add_(0.123)
            path1 = mgr.save(p, episode=1000, metadata={"foo": 1})

            with torch.no_grad():
                for param in p.parameters():
                    param.add_(0.2)
            path2 = mgr.save(p, episode=2000, metadata={"foo": 2})

            self.assertEqual(Path(path2), mgr.latest_path())

            loaded, episode = mgr.load_latest()
            self.assertIsNotNone(loaded)
            self.assertEqual(episode, 2000)
            for p_ref, p_loaded in zip(p.parameters(), loaded.parameters()):
                self.assertTrue(torch.allclose(p_ref, p_loaded))
            self.assertTrue(path1.exists())
            self.assertTrue(path2.exists())

    def test_latest_path_supports_large_episode_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(td)
            p = LinearSoftmaxPolicy(seed=1)
            big_ep = 44_000_000
            path = mgr.save(p, episode=big_ep)
            self.assertEqual(mgr.latest_path(), path)
            loaded, episode = mgr.load_latest()
            self.assertIsNotNone(loaded)
            self.assertEqual(episode, big_ep)

    def test_incompatible_checkpoint_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(td)
            bad_path = Path(td) / "policy_ep0001234.pt"
            torch.save(
                {
                    "episode": 1234,
                    "model_state_dict": {
                        "linear1.weight": torch.randn(10, 10),
                        "linear1.bias": torch.randn(10),
                    },
                },
                bad_path,
            )
            loaded, episode = mgr.load_latest()
            self.assertIsNone(loaded)
            self.assertEqual(episode, 0)


if __name__ == "__main__":
    unittest.main()
