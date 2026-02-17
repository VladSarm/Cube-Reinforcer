import unittest

import numpy as np

from rubik_sim.actions import ACTION_TABLE, FACE_INDEX, MOVE_PERMUTATIONS, STICKERS_PER_FACE
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

    def test_turn_moves_face_and_adjacent_strips(self):
        """Regression: face turns must move side strips too (not face-only rotation)."""
        base = np.arange(STICKERS_PER_FACE * 6, dtype=np.int32)
        for action in range(12):
            face, _ = ACTION_TABLE[action]
            perm = MOVE_PERMUTATIONS[action]
            moved = base[perm]
            changed = moved != base

            face_start = FACE_INDEX[face] * STICKERS_PER_FACE
            face_end = face_start + STICKERS_PER_FACE
            changed_on_face = int(changed[face_start:face_end].sum())
            changed_total = int(changed.sum())
            changed_off_face = changed_total - changed_on_face

            self.assertEqual(changed_on_face, 8, msg=f"action={action} {face}: expected 8 moved on turning face")
            self.assertEqual(changed_off_face, 12, msg=f"action={action} {face}: expected 12 moved on adjacent strips")
            self.assertEqual(changed_total, 20, msg=f"action={action} {face}: expected total moved stickers=20")


if __name__ == "__main__":
    unittest.main()
