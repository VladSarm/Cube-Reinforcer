"""Shared reward shaping for trainer and MC baseline rollouts."""

from __future__ import annotations

INVERSE_ACTION_PENALTY = 20.0
REPEAT_FOUR_PENALTY = 100.0
TIMEOUT_PENALTY = 100.0
STEP_REWARD = -1.0


def is_inverse_action(prev_action: int | None, action: int) -> bool:
    if prev_action is None:
        return False
    return action == (prev_action ^ 1)


def compute_step_reward(action_history: list[int], action: int) -> float:
    reward = STEP_REWARD
    prev_action = action_history[-1] if action_history else None
    if is_inverse_action(prev_action, action):
        reward -= INVERSE_ACTION_PENALTY
    new_hist = (action_history + [action])[-4:]
    if len(new_hist) == 4 and len(set(new_hist)) == 1:
        reward -= REPEAT_FOUR_PENALTY
    return reward
