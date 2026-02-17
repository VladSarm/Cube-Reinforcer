import unittest

import numpy as np
import torch

from rubik_rl.reward import INVERSE_ACTION_PENALTY, REPEAT_FOUR_PENALTY, STEP_REWARD
from rubik_rl.torch_env import TorchRubikBatchEnv
from rubik_sim.engine import RubikEngine


class TestTorchEnv(unittest.TestCase):
    def test_step_matches_engine_sequence(self):
        # This guarantees training backend transitions match simulator/GUI engine transitions.
        device = torch.device("cpu")
        env = TorchRubikBatchEnv(batch_size=4, device=device)
        env.reset()

        # Build deterministic per-env action sequences.
        action_seq = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [6, 6, 6, 7, 8],
                [10, 9, 8, 7, 6],
                [4, 4, 4, 4, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        refs = [RubikEngine() for _ in range(4)]
        ref_done = [False] * 4
        for r in refs:
            r.reset()

        for t in range(action_seq.shape[1]):
            a = action_seq[:, t]
            env.step(a)
            for i, r in enumerate(refs):
                if ref_done[i]:
                    continue
                r.step(int(a[i].item()))
                ref_done[i] = r.is_solved()
                self.assertTrue(
                    np.array_equal(env.state[i].cpu().numpy(), r.get_state()),
                    msg=f"Mismatch at step={t}, env_idx={i}",
                )

        for i, r in enumerate(refs):
            self.assertTrue(np.array_equal(env.state[i].cpu().numpy(), r.get_state()))

    def test_scramble_depths_fixed(self):
        device = torch.device("cpu")
        env = TorchRubikBatchEnv(batch_size=256, device=device)
        g = torch.Generator(device=device)
        g.manual_seed(123)
        depths = env.scramble(scramble_steps=7, generator=g)
        self.assertTrue(torch.all(depths == 7).item())

    def test_reward_components_match_rules(self):
        device = torch.device("cpu")
        env = TorchRubikBatchEnv(batch_size=2, device=device)
        env.reset()

        # Force history so first env does inverse, second env does repeat-4.
        env.action_hist[0] = torch.tensor([-1, -1, -1, 2], device=device)  # inverse for action 3
        env.action_hist[1] = torch.tensor([5, 5, 5, 5], device=device)  # repeat if action 5

        out = env.step(torch.tensor([3, 5], dtype=torch.long, device=device))
        r = out["reward_total"].cpu().numpy()

        self.assertAlmostEqual(float(r[0]), STEP_REWARD - INVERSE_ACTION_PENALTY, places=6)
        self.assertAlmostEqual(float(r[1]), STEP_REWARD - REPEAT_FOUR_PENALTY, places=6)


if __name__ == "__main__":
    unittest.main()
