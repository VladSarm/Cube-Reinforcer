import unittest

import numpy as np

from rubik_rl.policy import LinearSoftmaxPolicy


class TestPolicyMath(unittest.TestCase):
    def test_softmax_probabilities_sum_to_one(self):
        p = LinearSoftmaxPolicy(seed=1)
        state = np.zeros((24, 6), dtype=np.float64)
        state[np.arange(24), np.arange(24) % 6] = 1.0
        hist = np.zeros((48,), dtype=np.float64)
        probs = p.action_probs(state, hist)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=8)
        self.assertTrue(np.all(probs >= 0.0))

    def test_manual_gradient_matches_finite_difference(self):
        policy = LinearSoftmaxPolicy(seed=0)
        state = np.zeros((24, 6), dtype=np.float64)
        state[np.arange(24), (np.arange(24) * 2) % 6] = 1.0
        hist = np.zeros((48,), dtype=np.float64)
        hist[3 * 12 + 4] = 1.0
        action = 3

        analytic_dW1, analytic_db1, analytic_dW2, analytic_db2 = policy.log_policy_gradients(state, hist, action)

        eps = 1e-6

        def logp() -> float:
            probs_local = policy.action_probs(state, hist)
            return float(np.log(probs_local[action] + 1e-15))

        check_pairs_w1 = [(0, 0), (5, 4), (23, 11), (120, 2)]
        for i, j in check_pairs_w1:
            old = policy.W1[i, j]
            policy.W1[i, j] = old + eps
            f_plus = logp()
            policy.W1[i, j] = old - eps
            f_minus = logp()
            policy.W1[i, j] = old
            numeric = (f_plus - f_minus) / (2.0 * eps)
            self.assertAlmostEqual(numeric, analytic_dW1[i, j], places=5)

        for j in [0, 3, 8, 11, 64, 127]:
            old = policy.b1[j]
            policy.b1[j] = old + eps
            f_plus = logp()
            policy.b1[j] = old - eps
            f_minus = logp()
            policy.b1[j] = old
            numeric = (f_plus - f_minus) / (2.0 * eps)
            self.assertAlmostEqual(numeric, analytic_db1[j], places=5)

        check_pairs_w2 = [(0, 0), (5, 4), (23, 11), (64, 2), (127, 9)]
        for i, j in check_pairs_w2:
            old = policy.W2[i, j]
            policy.W2[i, j] = old + eps
            f_plus = logp()
            policy.W2[i, j] = old - eps
            f_minus = logp()
            policy.W2[i, j] = old
            numeric = (f_plus - f_minus) / (2.0 * eps)
            self.assertAlmostEqual(numeric, analytic_dW2[i, j], places=5)

        for j in [0, 3, 8, 11]:
            old = policy.b2[j]
            policy.b2[j] = old + eps
            f_plus = logp()
            policy.b2[j] = old - eps
            f_minus = logp()
            policy.b2[j] = old
            numeric = (f_plus - f_minus) / (2.0 * eps)
            self.assertAlmostEqual(numeric, analytic_db2[j], places=5)

    def test_state_flatten_shape(self):
        state = np.zeros((24, 6), dtype=np.float64)
        state[np.arange(24), np.arange(24) % 6] = 1.0
        hist = np.zeros((48,), dtype=np.float64)
        x = LinearSoftmaxPolicy.build_observation(state, hist)
        self.assertEqual(x.shape, (192,))


if __name__ == "__main__":
    unittest.main()
