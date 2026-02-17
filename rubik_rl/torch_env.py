"""Batched torch environment for fast Rubik 2x2 training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rubik_rl.reward import INVERSE_ACTION_PENALTY, REPEAT_FOUR_PENALTY, STEP_REWARD
from rubik_sim.actions import MOVE_PERMUTATIONS, ORIENTATION_PERMUTATIONS, solved_state


class TorchRubikBatchEnv:
    ACTION_DIM = 12
    STATE_SIZE = 24

    def __init__(self, batch_size: int, device: torch.device):
        self.batch_size = int(batch_size)
        self.device = device

        self.move_perms = torch.as_tensor(MOVE_PERMUTATIONS, dtype=torch.long, device=device)  # [12,24]
        self.orientation_perms = torch.as_tensor(ORIENTATION_PERMUTATIONS, dtype=torch.long, device=device)  # [24,24]
        self.solved = torch.as_tensor(solved_state(), dtype=torch.long, device=device)  # [24]

        self.state = torch.empty((self.batch_size, self.STATE_SIZE), dtype=torch.long, device=device)
        self.action_hist = torch.full((self.batch_size, 4), -1, dtype=torch.long, device=device)
        self.done = torch.zeros((self.batch_size,), dtype=torch.bool, device=device)
        self.steps = torch.zeros((self.batch_size,), dtype=torch.long, device=device)
        self.reset(self.batch_size)

    def reset(self, batch_size: int | None = None) -> None:
        if batch_size is not None and int(batch_size) != self.batch_size:
            self.batch_size = int(batch_size)
            self.state = torch.empty((self.batch_size, self.STATE_SIZE), dtype=torch.long, device=self.device)
            self.action_hist = torch.full((self.batch_size, 4), -1, dtype=torch.long, device=self.device)
            self.done = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
            self.steps = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
        self.state[:] = self.solved.unsqueeze(0).expand(self.batch_size, -1)
        self.action_hist[:] = -1
        self.done[:] = False
        self.steps[:] = 0

    def _apply_actions(self, actions: torch.Tensor, active: torch.Tensor) -> None:
        perms = self.move_perms[actions]  # [B,24]
        moved = torch.gather(self.state, dim=1, index=perms)
        self.state = torch.where(active.unsqueeze(1), moved, self.state)
        self.steps = self.steps + active.long()

    def _is_solved_orientation_invariant(self) -> torch.Tensor:
        # state[:, orientation_perms] => [B,24,24]
        oriented = self.state[:, self.orientation_perms]
        solved_mask = (oriented == self.solved.view(1, 1, -1)).all(dim=2).any(dim=1)
        return solved_mask

    def build_observation(self) -> torch.Tensor:
        state_oh = F.one_hot(self.state, num_classes=6).to(torch.float32).reshape(self.batch_size, -1)  # [B,144]
        hist_oh = torch.zeros((self.batch_size, 4, self.ACTION_DIM), dtype=torch.float32, device=self.device)
        valid = self.action_hist >= 0
        if valid.any():
            b_idx, slot_idx = valid.nonzero(as_tuple=True)
            act_idx = self.action_hist[b_idx, slot_idx]
            hist_oh[b_idx, slot_idx, act_idx] = 1.0
        hist_flat = hist_oh.reshape(self.batch_size, -1)  # [B,48]
        return torch.cat([state_oh, hist_flat], dim=1)  # [B,192]

    def scramble(self, scramble_steps: int, generator: torch.Generator) -> torch.Tensor:
        steps = int(scramble_steps)
        if steps < 1:
            raise ValueError("scramble_steps must be >= 1")
        depths = torch.full((self.batch_size,), steps, dtype=torch.long, device=self.device)
        prev = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        for _ in range(steps):
            active = torch.ones((self.batch_size,), dtype=torch.bool, device=self.device)
            if not active.any():
                break

            actions = torch.randint(
                low=0,
                high=self.ACTION_DIM,
                size=(self.batch_size,),
                generator=generator,
                device=self.device,
                dtype=torch.long,
            )
            need_resample = active & (prev >= 0) & (actions == (prev ^ 1))
            while need_resample.any():
                actions[need_resample] = torch.randint(
                    low=0,
                    high=self.ACTION_DIM,
                    size=(int(need_resample.sum().item()),),
                    generator=generator,
                    device=self.device,
                    dtype=torch.long,
                )
                need_resample = active & (prev >= 0) & (actions == (prev ^ 1))

            self._apply_actions(actions, active)
            prev = torch.where(active, actions, prev)

        # Scramble does not count as episode steps/rewards.
        self.steps.zero_()
        self.action_hist[:] = -1
        self.done[:] = False
        return depths

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        actions = actions.to(self.device, dtype=torch.long).view(-1)
        if actions.shape[0] != self.batch_size:
            raise ValueError(f"actions must have shape ({self.batch_size},), got {tuple(actions.shape)}")

        active = ~self.done
        prev = self.action_hist[:, 3]
        new_hist = torch.cat([self.action_hist[:, 1:], actions.unsqueeze(1)], dim=1)

        inverse = active & (prev >= 0) & (actions == (prev ^ 1))
        repeat = active & (new_hist[:, 0] >= 0) & (new_hist[:, 0] == new_hist[:, 1]) & (new_hist[:, 1] == new_hist[:, 2]) & (
            new_hist[:, 2] == new_hist[:, 3]
        )

        reward_step = torch.where(active, torch.full((self.batch_size,), STEP_REWARD, dtype=torch.float32, device=self.device), torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device))
        reward_inverse = torch.where(
            inverse,
            torch.full((self.batch_size,), -float(INVERSE_ACTION_PENALTY), dtype=torch.float32, device=self.device),
            torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device),
        )
        reward_repeat = torch.where(
            repeat,
            torch.full((self.batch_size,), -float(REPEAT_FOUR_PENALTY), dtype=torch.float32, device=self.device),
            torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device),
        )
        reward_total = reward_step + reward_inverse + reward_repeat

        self._apply_actions(actions, active)
        self.action_hist = torch.where(active.unsqueeze(1), new_hist, self.action_hist)

        solved = self._is_solved_orientation_invariant()
        self.done = self.done | solved

        return {
            "reward_total": reward_total,
            "reward_step": reward_step,
            "reward_inverse": reward_inverse,
            "reward_repeat": reward_repeat,
            "done": self.done.clone(),
            "active": active,
        }
