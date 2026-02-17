import unittest

from rubik_sim.actions import ORIENTATION_PERMUTATIONS, solved_state
from rubik_sim.solved_check import is_solved_orientation_invariant


class TestSolvedCheck(unittest.TestCase):
    def test_solved_state_is_true(self):
        state = solved_state()
        self.assertTrue(is_solved_orientation_invariant(state))

    def test_global_rotation_of_solved_is_true(self):
        state = solved_state()
        rotated = None
        for perm in ORIENTATION_PERMUTATIONS:
            candidate = state[perm]
            if not (candidate == state).all():
                rotated = candidate
                break

        self.assertIsNotNone(rotated)
        self.assertTrue(is_solved_orientation_invariant(rotated))

    def test_corrupted_state_is_false(self):
        state = solved_state().copy()
        state[0], state[9] = state[9], state[0]
        self.assertFalse(is_solved_orientation_invariant(state))


if __name__ == "__main__":
    unittest.main()
