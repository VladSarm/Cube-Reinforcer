import tempfile
import unittest
from pathlib import Path

import numpy as np

from rubik_rl.checkpoint import CheckpointManager
from rubik_rl.policy import LinearSoftmaxPolicy


class TestCheckpoint(unittest.TestCase):
    def test_save_load_roundtrip_and_latest(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = CheckpointManager(td)
            p = LinearSoftmaxPolicy(seed=42)

            p.W1 += 0.123
            p.b1 -= 0.456
            p.W2 += 0.234
            p.b2 -= 0.567
            path1 = mgr.save(p, episode=1000, metadata={"foo": 1})

            p.W1 += 0.2
            p.b1 += 0.2
            p.W2 += 0.2
            p.b2 += 0.2
            path2 = mgr.save(p, episode=2000, metadata={"foo": 2})

            self.assertEqual(Path(path2), mgr.latest_path())

            loaded, episode = mgr.load_latest()
            self.assertIsNotNone(loaded)
            self.assertEqual(episode, 2000)
            self.assertTrue(np.allclose(loaded.W1, p.W1))
            self.assertTrue(np.allclose(loaded.b1, p.b1))
            self.assertTrue(np.allclose(loaded.W2, p.W2))
            self.assertTrue(np.allclose(loaded.b2, p.b2))
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


if __name__ == "__main__":
    unittest.main()
