"""Shared reward shaping for trainer and MC baseline rollouts."""

from __future__ import annotations

INVERSE_ACTION_PENALTY = 200.0
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


def reward_components(action_history: list[int], action: int) -> dict[str, float]:
    """Return per-term reward decomposition for logging."""
    step = STEP_REWARD
    inverse = -INVERSE_ACTION_PENALTY if is_inverse_action(action_history[-1] if action_history else None, action) else 0.0
    new_hist = (action_history + [action])[-4:]
    repeat = -REPEAT_FOUR_PENALTY if len(new_hist) == 4 and len(set(new_hist)) == 1 else 0.0
    return {
        "step": float(step),
        "inverse_penalty": float(inverse),
        "repeat_penalty": float(repeat),
        "timeout_penalty": 0.0,
        "total": float(step + inverse + repeat),
    }
