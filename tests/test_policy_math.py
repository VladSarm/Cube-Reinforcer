import unittest

import numpy as np
import torch

from rubik_rl.policy import LinearSoftmaxPolicy


class TestPolicyMath(unittest.TestCase):
    def test_softmax_probabilities_sum_to_one(self):
        p = LinearSoftmaxPolicy(seed=1)
        state = np.zeros((24, 6), dtype=np.float64)
        state[np.arange(24), np.arange(24) % 6] = 1.0
        hist = np.zeros((48,), dtype=np.float64)
        probs = p.action_probs(state, hist)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
        self.assertTrue(np.all(probs >= 0.0))

    def test_autograd_gradient_exists_and_finite(self):
        policy = LinearSoftmaxPolicy(seed=0)
        state = np.zeros((24, 6), dtype=np.float64)
        state[np.arange(24), (np.arange(24) * 2) % 6] = 1.0
        hist = np.zeros((48,), dtype=np.float64)
        hist[3 * 12 + 4] = 1.0
        action, _, log_prob = policy.sample_action(state, hist, return_log_prob=True)
        self.assertTrue(0 <= action < 12)

        loss = -log_prob
        policy.zero_grad()
        loss.backward()

        for p in policy.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())

    def test_state_flatten_shape(self):
        state = np.zeros((24, 6), dtype=np.float64)
        state[np.arange(24), np.arange(24) % 6] = 1.0
        hist = np.zeros((48,), dtype=np.float64)
        x = LinearSoftmaxPolicy.build_observation(state, hist)
        self.assertEqual(x.shape, (192,))

    def test_batched_sampling_from_logits(self):
        policy = LinearSoftmaxPolicy(seed=0)
        x = torch.randn(16, policy.INPUT_DIM)
        logits = policy.forward_logits(x)
        actions, log_probs = policy.sample_actions_from_logits(logits)
        self.assertEqual(tuple(logits.shape), (16, policy.ACTION_DIM))
        self.assertEqual(tuple(actions.shape), (16,))
        self.assertEqual(tuple(log_probs.shape), (16,))
        probs = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
