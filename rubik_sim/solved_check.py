"""Solved-state checks for the Rubik simulator."""

from __future__ import annotations

import numpy as np

from .actions import ORIENTATION_PERMUTATIONS, solved_state
from .state_codec import StateValidationError, validate_state

_CANONICAL_SOLVED = solved_state()


def is_solved_orientation_invariant(state: list[int] | np.ndarray) -> bool:
    arr = validate_state(state)
    for perm in ORIENTATION_PERMUTATIONS:
        oriented = arr[perm]
        if np.array_equal(oriented, _CANONICAL_SOLVED):
            return True
    return False


def assert_valid_and_solved(state: list[int] | np.ndarray) -> None:
    arr = validate_state(state)
    if not is_solved_orientation_invariant(arr):
        raise StateValidationError("State is valid but not solved")
