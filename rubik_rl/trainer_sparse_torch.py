"""PyTorch REINFORCE trainer for sparse Rubik 2x2 (7 cells, 6 actions, H fixed). GPU-first."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .policy_sparse_torch import SparsePolicyTorch
from .reward import TIMEOUT_PENALTY
from .torch_env_sparse import TorchSparseBatchEnv


def _load_sparse_torch_latest(
    checkpoint_dir: str,
) -> tuple[SparsePolicyTorch | None, torch.optim.Optimizer | None, int]:
    """Load latest SparsePolicyTorch checkpoint (.pt) from directory. Returns (policy, optimizer_state, episode)."""
    import re
    pattern = re.compile(r"policy_ep(\d+)\.pt$")
    dir_path = Path(checkpoint_dir)
    if not dir_path.is_dir():
        return None, None, 0
    best_ep, best_path = -1, None
    for p in dir_path.glob("policy_ep*.pt"):
        m = pattern.search(p.name)
        if m:
            ep = int(m.group(1))
            if ep > best_ep:
                best_ep, best_path = ep, p
    if best_path is None:
        return None, None, 0
    data = torch.load(best_path, map_location="cpu")
    policy = SparsePolicyTorch.from_state_dict(data["model_state_dict"])
    opt_state = data.get("optimizer_state_dict")
    episode = int(data.get("episode", 0))
    return policy, opt_state, episode


class ReinforceSparseTrainerTorch:
    CURRICULUM_THRESHOLD = 0.80
    CURRICULUM_MAX_SCRAMBLE = 10

    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.scramble_steps < 1:
            raise ValueError("--scramble-steps must be >= 1")
        if args.num_envs < 1:
            raise ValueError("--num-envs must be >= 1")

        if args.seed is not None:
            torch.manual_seed(args.seed)

        self.device = self._resolve_device(args.device)
        self.policy, opt_state, self.start_episode = _load_sparse_torch_latest(args.checkpoint_dir)
        if self.policy is None:
            hidden_dim = getattr(args, "hidden_dim", None) or SparsePolicyTorch.HIDDEN_DIM
            self.policy = SparsePolicyTorch(hidden_dim=hidden_dim, seed=args.seed)
            self.start_episode = 0
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

        self.ckpt_dir = Path(args.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = getattr(args, "exp_name", None)
        if exp_name:
            tb_logdir = str(Path(args.tensorboard_logdir) / exp_name / f"run_{run_ts}")
        else:
            tb_logdir = str(Path(args.tensorboard_logdir) / f"run_{run_ts}")
        self.tb_writer = SummaryWriter(log_dir=tb_logdir)
        self.tb_logdir = tb_logdir

        self._rng = torch.Generator(device=self.device)
        if args.seed is not None:
            self._rng.manual_seed(args.seed)

        self.env = TorchSparseBatchEnv(batch_size=args.num_envs, device=self.device)
        self.current_scramble_steps = min(int(args.scramble_steps), self.CURRICULUM_MAX_SCRAMBLE)
        self._episode_bar: tqdm | None = None

        self._log(
            "trainer_init sparse_torch "
            f"episodes={args.episodes} num_envs={args.num_envs} "
            f"scramble_steps_start={self.current_scramble_steps} scramble_steps_max={self.CURRICULUM_MAX_SCRAMBLE} "
            f"curriculum_threshold={self.CURRICULUM_THRESHOLD:.2f} max_episode_steps={args.max_episode_steps} "
            f"gamma={args.gamma} lr={args.lr} optimizer=adam "
            f"device={self.device.type} tensorboard_logdir={tb_logdir} "
            f"checkpoint_dir={args.checkpoint_dir}"
        )
        if self.start_episode == 0:
            self._log("checkpoint_status no checkpoint found, initialized random policy")
        else:
            self._log(f"checkpoint_status resumed from episode={self.start_episode}")

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        name = device_name.lower()
        if name == "cpu":
            return torch.device("cpu")
        if name == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Requested --device cuda, but CUDA is not available")
            return torch.device("cuda")
        if name == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise ValueError("Requested --device mps, but MPS is not available")
            return torch.device("mps")
        raise ValueError("--device must be one of: cpu, cuda, mps")

    def _log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {message}"
        if self._episode_bar is not None:
            self._episode_bar.write(text)
        else:
            print(text, flush=True)

    def _rollout_batch(self) -> dict[str, torch.Tensor]:
        B = self.args.num_envs
        T = self.args.max_episode_steps
        gamma = float(self.args.gamma)

        self.env.reset(B)
        scramble_used = int(self.current_scramble_steps)
        self.env.scramble(scramble_used, generator=self._rng)

        rewards = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        rewards_repeat = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        active_mask = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        log_probs = torch.zeros((T, B), dtype=torch.float32, device=self.device)

        for t in range(T):
            obs = self.env.build_observation()  # [B, 31]
            logits = self.policy.forward_logits(obs)  # [B, 6]
            actions, lp = SparsePolicyTorch.sample_actions_from_logits(logits)

            step_out = self.env.step(actions)
            rewards[t] = step_out["reward"]
            rewards_repeat[t] = step_out["reward_repeat"]
            active_mask[t] = step_out["active"].to(torch.float32)
            log_probs[t] = lp

        # Timeout penalty: add to last active step for unsolved episodes
        timeout_penalty_per_env = torch.zeros(B, dtype=torch.float32, device=self.device)
        unsolved = ~self.env.done
        has_steps = self.env.steps > 0
        apply_to = unsolved & has_steps
        if apply_to.any():
            idx = torch.nonzero(apply_to, as_tuple=False).squeeze(1)
            t_idx = (self.env.steps[idx] - 1).clamp(0, T - 1)
            rewards[t_idx, idx] += -float(TIMEOUT_PENALTY)
            timeout_penalty_per_env[idx] = -float(TIMEOUT_PENALTY)

        # Discounted returns
        returns = torch.zeros_like(rewards)
        running = torch.zeros(B, dtype=torch.float32, device=self.device)
        for t in range(T - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running

        valid_count = torch.clamp(active_mask.sum(), min=1.0)
        loss = -((log_probs * returns * active_mask).sum() / valid_count)

        episode_return = rewards.sum(dim=0)  # [B]
        episode_steps = self.env.steps.to(torch.float32)
        solved = self.env.done.to(torch.float32)
        solved_steps = episode_steps[self.env.done]

        return {
            "loss": loss,
            "sr": solved.mean(),
            "steps_mean": episode_steps.mean(),
            "steps_to_solve_mean": solved_steps.mean() if solved_steps.numel() else torch.tensor(0.0, device=self.device),
            "return_mean": episode_return.mean(),
            "return_repeat_mean": rewards_repeat.sum(dim=0).mean(),
            "return_timeout_mean": timeout_penalty_per_env.mean(),
            "last_steps": self.env.steps[-1].item(),
            "last_solved": bool(self.env.done[-1].item()),
            "last_return": float(episode_return[-1].item()),
            "scramble_steps": scramble_used,
        }

    def _save_checkpoint(self, episode: int) -> None:
        path = self.ckpt_dir / f"policy_ep{episode:07d}.pt"
        metadata: dict[str, Any] = {
            "episode": episode,
            "lr": self.args.lr,
            "gamma": self.args.gamma,
            "num_envs": self.args.num_envs,
            "scramble_steps": self.current_scramble_steps,
            "curriculum_threshold": self.CURRICULUM_THRESHOLD,
            "curriculum_max_scramble": self.CURRICULUM_MAX_SCRAMBLE,
            "device": self.device.type,
            "mode": "sparse_torch",
        }
        torch.save(
            {
                "episode": episode,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": metadata,
            },
            path,
        )
        self._log(f"checkpoint_saved episode={episode} path={path}")

    def run(self) -> None:
        total_to_run = int(self.args.episodes)
        episodes_done = 0
        global_episode = int(self.start_episode)
        global_batch = 0

        try:
            self._episode_bar = tqdm(
                total=total_to_run,
                desc="REINFORCE sparse torch episodes",
                unit="ep",
                mininterval=1.0,
                maxinterval=5.0,
            )

            while episodes_done < total_to_run:
                metrics = self._rollout_batch()
                batch_size = min(self.args.num_envs, total_to_run - episodes_done)
                loss = metrics["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                global_batch += 1
                global_episode += batch_size
                episodes_done += batch_size

                sr = float(metrics["sr"].item())
                steps_mean = float(metrics["steps_mean"].item())
                steps_to_solve_mean = float(metrics["steps_to_solve_mean"].item())
                return_mean = float(metrics["return_mean"].item())
                ret_timeout = float(metrics["return_timeout_mean"].item())
                ret_repeat = float(metrics["return_repeat_mean"].item())
                loss_value = float(loss.detach().item())

                self.tb_writer.add_scalar("train/sr_batch", sr, global_batch)
                self.tb_writer.add_scalar("train/steps_mean_batch", steps_mean, global_batch)
                self.tb_writer.add_scalar("train/steps_to_solve_mean_batch", steps_to_solve_mean, global_batch)
                self.tb_writer.add_scalar("train/return_total_mean_batch", return_mean, global_batch)
                self.tb_writer.add_scalar("train/return_timeout_penalty_mean_batch", ret_timeout, global_batch)
                self.tb_writer.add_scalar("train/return_repeat_penalty_mean_batch", ret_repeat, global_batch)
                self.tb_writer.add_scalar("train/loss_batch", loss_value, global_batch)
                self.tb_writer.add_scalar("train/lr", float(self.args.lr), global_batch)
                self.tb_writer.add_scalar("train/scramble_steps_current", int(metrics["scramble_steps"]), global_batch)

                if self.args.log_interval > 0 and global_episode % self.args.log_interval == 0:
                    self._log(
                        "batch_stats "
                        f"episode={global_episode} batch={global_batch} "
                        f"scramble_steps={int(metrics['scramble_steps'])} "
                        f"sr={sr:.3f} steps_mean={steps_mean:.2f} "
                        f"steps_to_solve_mean={steps_to_solve_mean:.2f} "
                        f"return_mean={return_mean:.2f} ret_timeout={ret_timeout:.2f} "
                        f"ret_repeat={ret_repeat:.2f} loss={loss_value:.4f}"
                    )

                if sr > self.CURRICULUM_THRESHOLD and self.current_scramble_steps < self.CURRICULUM_MAX_SCRAMBLE:
                    prev = self.current_scramble_steps
                    self.current_scramble_steps += 1
                    self._log(
                        f"curriculum_update episode={global_episode} batch={global_batch} "
                        f"sr={sr:.3f} scramble_steps {prev}->{self.current_scramble_steps}"
                    )

                self._episode_bar.update(batch_size)
                self._episode_bar.set_postfix(
                    {
                        "scr": int(metrics["scramble_steps"]),
                        "steps": int(metrics["last_steps"]),
                        "solved": bool(metrics["last_solved"]),
                        "ret": f"{metrics['last_return']:.1f}",
                        "sr": f"{sr:.2f}",
                        "loss": f"{loss_value:.3f}",
                    }
                )

                if global_episode % self.args.save_every == 0:
                    self._save_checkpoint(global_episode)

        finally:
            self.tb_writer.flush()
            self.tb_writer.close()
            if self._episode_bar is not None:
                self._episode_bar.close()
                self._episode_bar = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train REINFORCE sparse policy for Rubik 2x2 (PyTorch, 7 cells, 6 actions, CUDA)"
    )
    p.add_argument("--num-envs", type=int, default=16, help="Batch size of parallel environments")
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument(
        "--scramble-steps", type=int, required=True,
        help="Initial scramble depth. Curriculum increases by +1 when SR > 0.8, up to 10.",
    )
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--tensorboard-logdir", default="runs/cube_reinforcer_sparse")
    p.add_argument("--exp-name", type=str, default=None, help="Optional experiment name for TensorBoard grouping")
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--checkpoint-dir", default="checkpoints_sparse_torch")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--hidden-dim", type=int, default=None, help="First hidden layer size (default 512); second layer is fixed at 128")
    return p


def main() -> None:
    args = build_parser().parse_args()
    trainer = ReinforceSparseTrainerTorch(args)
    trainer.run()


if __name__ == "__main__":
    main()
