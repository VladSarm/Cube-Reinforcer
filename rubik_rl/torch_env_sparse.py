"""Batched torch environment for sparse Rubik 2x2 training (7 cells, 6 actions, H fixed)."""

from __future__ import annotations

import numpy as np
import torch

from rubik_sim.actions import ACTION_6_TO_12, MOVE_PERMUTATIONS

from .sparse_state import KEY_SLOTS, CURRICULUM_ORDER, TARGET_PIECE, REWARD_SHAPE_MATRIX

# Scalar reward constants
SOLVE_REWARD = 100.0       # +100 for reaching curriculum goal
TIMEOUT_PENALTY = 100.0    # -100 for timeout / not reaching goal (applied in trainer at episode end)
STEP_PENALTY = -1.0        # -1 per action by default
# Shaping: corner in target cell +1, corner on one of its 3 faces +0.33 (see REWARD_SHAPE_MATRIX)
# Penalty for wasteful patterns:
#   repeat-3: a,a,a  (3rd same action = 270° = wasteful, 2 reps = 180° is OK)
#   oscillate: a,b,a,b where b == a^1 (infinite back-and-forth)
REPEAT_PENALTY = 1.0

# Number of observable corners (a..g); H is fixed
NUM_KEY_SLOTS = 7


class TorchSparseBatchEnv:
    """Batched environment using permutation-based state, 6 actions only (H corner fixed).

    State: perm [B, 24] — identity means solved.
    Observation: [B, 31] = [B, 7] sparse binary state + [B, 24] action history one-hot.
    Actions: 0-5 (mapped to 12-action space via ACTION_6_TO_12).
    Reward: -1 per step; +1 per corner in target cell, +0.33 per corner on one of its faces (shaping);
    +100 if curriculum goal reached, -100 if timeout (applied in trainer).

    Curriculum: set target_n via set_target_n(n). Episode is "solved" when the first
    target_n corners in curriculum order (d, f, b, a, c, e, g) are at home: stage 1 = D→d
    (corner above static H), stage 2 = +F, stage 3 = +B, stages 4–7 = +a,+c,+e,+g.
    """

    ACTION_DIM = 6
    STATE_SIZE = 24
    HISTORY_LEN = 4
    OBS_DIM = 7 + HISTORY_LEN * ACTION_DIM  # 31

    def __init__(self, batch_size: int, device: torch.device):
        self.batch_size = int(batch_size)
        self.device = device

        # 6 permutations selected from MOVE_PERMUTATIONS via ACTION_6_TO_12
        sparse_perms = np.array([MOVE_PERMUTATIONS[ACTION_6_TO_12[a]] for a in range(6)], dtype=np.int64)
        self.move_perms = torch.as_tensor(sparse_perms, dtype=torch.long, device=device)  # [6, 24]

        # Identity = solved
        self.identity = torch.arange(self.STATE_SIZE, dtype=torch.long, device=device)  # [24]

        # KEY_SLOTS tensor for building sparse observation (all 7 corners a..g)
        self.key_slots = torch.as_tensor(KEY_SLOTS, dtype=torch.long, device=device)  # [7]

        # Curriculum: how many of the first target_n corners (in CURRICULUM_ORDER) must be at home to "solve"
        self.target_n: int = NUM_KEY_SLOTS  # default: all 7 (full solve)
        self._curriculum_slots = torch.as_tensor(
            KEY_SLOTS[CURRICULUM_ORDER], dtype=torch.long, device=device
        )  # [7] slots to check per curriculum level (d, f, b, a, c, e, g)
        self._reward_shape = torch.as_tensor(
            REWARD_SHAPE_MATRIX, dtype=torch.float32, device=device
        )  # [7, 24]: reward per (cell, slot) — 1.0 target, 0.33 on face, 0 else
        self._target_piece = torch.as_tensor(TARGET_PIECE, dtype=torch.long, device=device)  # [7]

        self.perm = self.identity.unsqueeze(0).expand(self.batch_size, -1).clone()
        self.action_hist = torch.full((self.batch_size, self.HISTORY_LEN), -1, dtype=torch.long, device=device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        self.steps = torch.zeros(self.batch_size, dtype=torch.long, device=device)

    def set_target_n(self, n: int) -> None:
        """Set curriculum level: episode solved when first n corners in order (d, f, b, a, c, e, g) are at home."""
        if not (1 <= n <= NUM_KEY_SLOTS):
            raise ValueError(f"target_n must be in [1, {NUM_KEY_SLOTS}], got {n}")
        self.target_n = int(n)

    def reset(self, batch_size: int | None = None) -> None:
        if batch_size is not None and int(batch_size) != self.batch_size:
            self.batch_size = int(batch_size)
            self.perm = self.identity.unsqueeze(0).expand(self.batch_size, -1).clone()
            self.action_hist = torch.full((self.batch_size, self.HISTORY_LEN), -1, dtype=torch.long, device=self.device)
            self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            self.steps = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        else:
            self.perm = self.identity.unsqueeze(0).expand(self.batch_size, -1).clone()
            self.action_hist[:] = -1
            self.done[:] = False
            self.steps[:] = 0

    def _apply_actions(self, actions_6: torch.Tensor, active: torch.Tensor) -> None:
        """Apply 6-action indices to permutations for active envs."""
        perms = self.move_perms[actions_6]  # [B, 24]
        new_perm = torch.gather(self.perm, dim=1, index=perms)
        self.perm = torch.where(active.unsqueeze(1), new_perm, self.perm)
        self.steps = self.steps + active.long()

    def _is_solved(self) -> torch.Tensor:
        """True if the first target_n corners in curriculum order (d, f, b, a, c, e, g) are at home. Returns [B] bool."""
        slots = self._curriculum_slots[:self.target_n]  # [target_n]
        key_pieces = torch.gather(
            self.perm, dim=1,
            index=slots.unsqueeze(0).expand(self.batch_size, -1),
        )  # [B, target_n]
        return (key_pieces == slots.unsqueeze(0)).all(dim=1)  # [B]

    def build_observation(self) -> torch.Tensor:
        """Returns [B, 31]: 7-bit sparse state + 24-dim action history one-hot."""
        # Sparse state: perm[key_slots[i]] == key_slots[i] (target == slot in identity)
        key_pieces = torch.gather(
            self.perm, dim=1,
            index=self.key_slots.unsqueeze(0).expand(self.batch_size, -1)
        )  # [B, 7]
        sparse_state = (key_pieces == self.key_slots.unsqueeze(0)).to(torch.float32)  # [B, 7]

        # Action history one-hot: [B, HISTORY_LEN, ACTION_DIM] -> [B, 24]
        hist_oh = torch.zeros(
            (self.batch_size, self.HISTORY_LEN, self.ACTION_DIM),
            dtype=torch.float32, device=self.device
        )
        valid = self.action_hist >= 0
        if valid.any():
            b_idx, slot_idx = valid.nonzero(as_tuple=True)
            act_idx = self.action_hist[b_idx, slot_idx]
            hist_oh[b_idx, slot_idx, act_idx] = 1.0
        hist_flat = hist_oh.reshape(self.batch_size, -1)  # [B, 24]

        return torch.cat([sparse_state, hist_flat], dim=1)  # [B, 31]

    def scramble(self, scramble_steps: int, generator: torch.Generator) -> None:
        """Scramble all envs with scramble_steps random 6-actions (no inverse of prev)."""
        steps = int(scramble_steps)
        if steps < 1:
            raise ValueError("scramble_steps must be >= 1")
        active = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        prev = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        for _ in range(steps):
            actions = torch.randint(
                low=0, high=self.ACTION_DIM, size=(self.batch_size,),
                generator=generator, device=self.device, dtype=torch.long,
            )
            # Avoid undoing previous action (inverse of a6 is a6 ^ 1 for pairs 0↔1, 2↔3, 4↔5)
            need_resample = (prev >= 0) & (actions == (prev ^ 1))
            while need_resample.any():
                actions[need_resample] = torch.randint(
                    low=0, high=self.ACTION_DIM,
                    size=(int(need_resample.sum().item()),),
                    generator=generator, device=self.device, dtype=torch.long,
                )
                need_resample = (prev >= 0) & (actions == (prev ^ 1))
            self._apply_actions(actions, active)
            prev = actions

        # Scramble does not count as episode steps
        self.steps.zero_()
        self.action_hist[:] = -1
        self.done[:] = False

    def step(self, actions_6: torch.Tensor) -> dict[str, torch.Tensor]:
        """Step all envs. actions_6: [B] tensor with values 0-5."""
        actions_6 = actions_6.to(self.device, dtype=torch.long).view(-1)

        active = ~self.done
        new_hist = torch.cat([self.action_hist[:, 1:], actions_6.unsqueeze(1)], dim=1)
        # new_hist columns: [t-3, t-2, t-1, t]

        # Repeat-3: a,a,a — 3rd consecutive identical action (270° = wasteful; 2 reps = 180° OK)
        repeat3 = (
            active
            & (new_hist[:, 1] >= 0)
            & (new_hist[:, 1] == new_hist[:, 2])
            & (new_hist[:, 2] == new_hist[:, 3])
        )
        # Oscillate: a,b,a,b where b == a^1 (e.g. U+, U-, U+, U-)
        oscillate = (
            active
            & (new_hist[:, 0] >= 0)
            & (new_hist[:, 0] == new_hist[:, 2])          # t-3 == t-1
            & (new_hist[:, 1] == new_hist[:, 3])          # t-2 == t
            & (new_hist[:, 1] == (new_hist[:, 0] ^ 1))   # t-2 is inverse of t-3
        )
        bad_pattern = repeat3 | oscillate

        self._apply_actions(actions_6, active)
        self.action_hist = torch.where(active.unsqueeze(1), new_hist, self.action_hist)

        solved = self._is_solved()
        self.done = self.done | solved

        # Base: -1 per active step, 0 if already done
        reward_base = torch.where(
            active,
            torch.full((self.batch_size,), STEP_PENALTY, dtype=torch.float32, device=self.device),
            torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
        )
        # Shaping: for each corner, +1 if piece in target cell, +0.33 if on one of its faces
        inv_perm = torch.zeros_like(self.perm, device=self.device)
        inv_perm.scatter_(1, self.perm, torch.arange(self.STATE_SIZE, dtype=torch.long, device=self.device).unsqueeze(0).expand(self.batch_size, -1))
        reward_shaping = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        for i in range(7):
            slots_i = inv_perm[:, self._target_piece[i]]  # [B] slot where piece for cell i is
            reward_shaping += self._reward_shape[i, slots_i]
        # Solve bonus: +100 when curriculum goal reached this step
        reward_solve = torch.where(
            active & solved,
            torch.full((self.batch_size,), SOLVE_REWARD, dtype=torch.float32, device=self.device),
            torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
        )
        reward_step = reward_base + reward_shaping + reward_solve
        # Penalty for wasteful patterns (repeat-3 or oscillate)
        reward_repeat = torch.where(
            bad_pattern,
            torch.full((self.batch_size,), -REPEAT_PENALTY, dtype=torch.float32, device=self.device),
            torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
        )

        return {
            "reward": reward_step + reward_repeat,
            "reward_step": reward_step,
            "reward_repeat": reward_repeat,
            "done": self.done.clone(),
            "active": active,
        }
