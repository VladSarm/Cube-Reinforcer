import argparse
import tempfile
import threading
import time
import unittest

from rubik_rl.checkpoint import CheckpointManager
from rubik_rl.infer import run_inference
from rubik_rl.trainer import ReinforceTrainer
from rubik_sim.engine import RubikEngine
from rubik_sim.server import RubikHTTPServer


class TestRLIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = RubikEngine()
        self.server = RubikHTTPServer(engine=self.engine, host="127.0.0.1", port=0, mode="headless")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.05)

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=1.0)

    def test_step_animated_headless_fallback(self):
        from rubik_rl.client import RubikAPIClient

        client = RubikAPIClient(host=self.server.host, port=self.server.port)
        out = client.step_animated(0, duration_ms=300)
        self.assertIn("state", out)
        self.assertEqual(out["action"], 0)

    def test_train_and_infer_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            train_args = argparse.Namespace(
                host=self.server.host,
                port=self.server.port,
                external_server=True,
                num_envs=1,
                episodes=2,
                max_episode_steps=8,
                scramble_steps=2,
                gamma=1.0,
                lr=0.01,
                save_every=1,
                checkpoint_dir=td,
                seed=123,
                log_interval=1,
                stats_window=10,
            )
            trainer = ReinforceTrainer(train_args)
            trainer.run()

            ckpt = CheckpointManager(td)
            self.assertIsNotNone(ckpt.latest_path())

            infer_args = argparse.Namespace(
                host=self.server.host,
                port=self.server.port,
                scramble_steps=2,
                max_steps=10,
                step_duration_ms=10,
                checkpoint_dir=td,
                seed=123,
            )
            solved = run_inference(infer_args)
            self.assertIsInstance(solved, bool)


if __name__ == "__main__":
    unittest.main()
