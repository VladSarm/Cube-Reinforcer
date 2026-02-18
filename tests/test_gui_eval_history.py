import os
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rubik_sim.engine import RubikEngine
from rubik_sim.gui import RubikGUI


class _PolicyStub:
    def __init__(self):
        self.last_hist_arg = None

    def history_one_hot(self, action_history):
        hist = np.zeros((48,), dtype=np.float64)
        for i, a in enumerate(action_history[-4:]):
            hist[(4 - len(action_history[-4:]) + i) * 12 + int(a)] = 1.0
        return hist

    def sample_action(self, state_one_hot, hist_oh):
        self.last_hist_arg = np.asarray(hist_oh, dtype=np.float64).copy()
        probs = np.full((12,), 1.0 / 12.0, dtype=np.float64)
        return 0, probs


class TestGUIEvalHistory(unittest.TestCase):
    def test_eval_tick_passes_history_to_policy(self):
        gui = RubikGUI(engine=RubikEngine(), host="127.0.0.1", port=0, scramble_steps=1)
        try:
            stub = _PolicyStub()
            gui.eval_enabled = True
            gui.eval_policy = stub
            gui.eval_action_history = [3, 7, 1]
            gui.animating = False
            gui.engine.scramble(steps=1, seed=123)

            gui._eval_tick()

            self.assertIsNotNone(stub.last_hist_arg)
            expected = stub.history_one_hot([3, 7, 1])
            self.assertTrue(np.array_equal(stub.last_hist_arg, expected))
            self.assertEqual(gui.eval_action_history[-1], 0)
        finally:
            gui.server.httpd.server_close()
            import pygame

            pygame.quit()


if __name__ == "__main__":
    unittest.main()
