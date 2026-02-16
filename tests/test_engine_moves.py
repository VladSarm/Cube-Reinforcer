import unittest

import numpy as np

from rubik_sim.engine import RubikEngine


class TestEngineMoves(unittest.TestCase):
    def test_inverse_actions_restore_state(self):
        inverse_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        for a_pos, a_neg in inverse_pairs:
            engine = RubikEngine()
            initial = engine.get_state()
            engine.step(a_pos)
            engine.step(a_neg)
            self.assertTrue(np.array_equal(initial, engine.get_state()), msg=f"Failed for actions {a_pos}/{a_neg}")

    def test_four_quarter_turns_restore_state(self):
        clockwise_actions = [0, 2, 4, 6, 8, 10]
        for action in clockwise_actions:
            engine = RubikEngine()
            initial = engine.get_state()
            for _ in range(4):
                engine.step(action)
            self.assertTrue(np.array_equal(initial, engine.get_state()), msg=f"Failed for action {action}")

    def test_scramble_is_deterministic_for_fixed_seed(self):
        e1 = RubikEngine()
        e2 = RubikEngine()

        s1, a1 = e1.scramble(steps=30, seed=123)
        s2, a2 = e2.scramble(steps=30, seed=123)

        self.assertEqual(a1, a2)
        self.assertTrue(np.array_equal(s1, s2))

    def test_scramble_has_no_immediate_inverse_move(self):
        e = RubikEngine()
        _, actions = e.scramble(steps=200, seed=99)
        for prev_a, next_a in zip(actions[:-1], actions[1:]):
            self.assertNotEqual(next_a, (prev_a ^ 1))


if __name__ == "__main__":
    unittest.main()
