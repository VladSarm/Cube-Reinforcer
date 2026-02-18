# rubik_rl/reward.py

from __future__ import annotations

import numpy as np

from rubik_sim.actions import ORIENTATION_PERMUTATIONS, solved_state  # uses your simulator perms :contentReference[oaicite:4]{index=4}

INVERSE_ACTION_PENALTY = 20.0
REPEAT_FOUR_PENALTY = 100.0
TIMEOUT_PENALTY = 100.0
STEP_REWARD = -1.0

# new knobs
ALPHA_STATE_PROGRESS = 0.25     # reward per +1 sticker correct (tune)
SOLVE_BONUS = 200.0             # bonus when solved (tune)


def is_inverse_action(prev_action: int | None, action: int) -> bool:
    if prev_action is None:
        return False
    return action == (prev_action ^ 1)


def _one_hot_to_colors(state_one_hot: np.ndarray) -> np.ndarray:
    """state_one_hot: (24,6) float/int -> colors: (24,) int8"""
    arr = np.asarray(state_one_hot)
    if arr.shape != (24, 6):
        raise ValueError(f"state_one_hot must have shape (24,6), got {arr.shape}")
    return np.argmax(arr, axis=1).astype(np.int8) #(26, 1) sticker <-> color


def state_match_score_orientation_invariant(state_one_hot: np.ndarray) -> int:
    """
    Score = max over 24 global orientations of
            number of stickers that match canonical solved_state().
    """
    colors = _one_hot_to_colors(state_one_hot)
    target = solved_state() 
    best = 0
    for perm in ORIENTATION_PERMUTATIONS:  # 24 perms :contentReference[oaicite:6]{index=6}
        oriented = colors[perm]
        m = int(np.sum(oriented == target))
        if m > best:
            best = m
    return best


def compute_step_reward(
    action_history: list[int],
    action: int,
    state_before_one_hot: np.ndarray, # encoded state before action
    state_after_one_hot: np.ndarray,  # encoded state after action
    solved_after: bool,
) -> float:
    # --- action-based part 
    reward = STEP_REWARD
    prev_action = action_history[-1] if action_history else None
    if is_inverse_action(prev_action, action):
        reward -= INVERSE_ACTION_PENALTY
    new_hist = (action_history + [action])[-4:] # alwas keep only 4 previuos actions
    if len(new_hist) == 4 and len(set(new_hist)) == 1:
        reward -= REPEAT_FOUR_PENALTY

    #--- state-based shaping
    sb = state_match_score_orientation_invariant(state_before_one_hot) # how many stickers already match the solved cube
    sa = state_match_score_orientation_invariant(state_after_one_hot)
    reward += ALPHA_STATE_PROGRESS * float(sa - sb)

    if solved_after:
        reward += SOLVE_BONUS

    return float(reward) / 100.0


# """Shared reward shaping for trainer and MC baseline rollouts."""

# from __future__ import annotations

# INVERSE_ACTION_PENALTY = 20.0
# REPEAT_FOUR_PENALTY = 100.0
# TIMEOUT_PENALTY = 100.0
# STEP_REWARD = -1.0


# def is_inverse_action(prev_action: int | None, action: int) -> bool:
#     if prev_action is None:
#         return False
#     return action == (prev_action ^ 1)


# def compute_step_reward(action_history: list[int], action: int) -> float:
#     reward = STEP_REWARD
#     prev_action = action_history[-1] if action_history else None
#     if is_inverse_action(prev_action, action):
#         reward -= INVERSE_ACTION_PENALTY
#     new_hist = (action_history + [action])[-4:]
#     if len(new_hist) == 4 and len(set(new_hist)) == 1:
#         reward -= REPEAT_FOUR_PENALTY
#     return reward
