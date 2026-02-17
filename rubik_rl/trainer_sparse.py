"""REINFORCE training for sparse environment: 7 cells a..g, target A..G, H fixed, 6 actions (n/b, u/j, k/l)."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
from tqdm import tqdm

from rubik_sim.actions import ACTION_6_TO_12
from rubik_sim.engine import RubikEngine

from .checkpoint import CheckpointManager, load_sparse_latest
from .policy_sparse import SparseLinearSoftmaxPolicy
from .reward import TIMEOUT_PENALTY
from .sparse_state import is_solved, piece_permutation, scalar_reward, sparse_state_from_perm
from .types import Transition


def load_config(path: str | Path) -> dict:
    """Load YAML config. Returns dict with 'policy' and 'training' keys."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _sample_scramble_steps(rng: np.random.Generator, scramble_steps_max: int) -> int:
    return int(rng.integers(1, scramble_steps_max + 1))


def _scramble_6_only(env: RubikEngine, steps: int, rng: np.random.Generator) -> None:
    """Scramble using only 6 actions (U/L/B) so corner H stays fixed."""
    prev_a6: int | None = None
    for _ in range(steps):
        candidates = [a for a in range(6) if prev_a6 is None or a != (prev_a6 ^ 1)]
        a6 = int(rng.choice(candidates))
        env.step(ACTION_6_TO_12[a6])
        prev_a6 = a6


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return returns


def _worker_episode_gradients(task: dict) -> dict:
    """Run one episode in sparse env (6 actions, perm-based state) and return gradients."""
    W1 = task["W1"]
    b1 = task["b1"]
    W2 = task["W2"]
    b2 = task["b2"]
    seed = int(task["seed"])
    scramble_steps_max = int(task["scramble_steps_max"])
    max_episode_steps = int(task["max_episode_steps"])
    gamma = float(task["gamma"])

    policy = SparseLinearSoftmaxPolicy(W1=W1, b1=b1, W2=W2, b2=b2, seed=seed)
    rng = np.random.default_rng(seed)

    env = RubikEngine(cube_size=2)
    env.reset()
    scramble_steps = _sample_scramble_steps(rng, scramble_steps_max)
    _scramble_6_only(env, scramble_steps, rng)

    traj: list[Transition] = []
    solved = False
    history_6: list[int] = []

    for _ in range(max_episode_steps):
        perm = piece_permutation(env.history)
        sparse_state = sparse_state_from_perm(perm).astype(np.float64)
        hist_oh = policy.history_one_hot(history_6)
        probs = policy.action_probs(sparse_state, hist_oh)
        action_6 = int(rng.choice(6, p=probs))
        env.step(ACTION_6_TO_12[action_6])
        perm_new = piece_permutation(env.history)
        solved = is_solved(perm_new)
        reward = scalar_reward(perm_new)

        traj.append(
            Transition(
                state_one_hot=sparse_state,
                action_history_one_hot=hist_oh.copy(),
                action=action_6,
                reward=reward,
            )
        )
        history_6 = (history_6 + [action_6])[-4:]
        if solved:
            break

    if (not solved) and len(traj) > 0:
        traj[-1].reward -= TIMEOUT_PENALTY

    rewards = np.asarray([t.reward for t in traj], dtype=np.float64)
    returns = _discounted_returns(rewards, gamma)
    advantages = returns

    dW1_acc = np.zeros_like(policy.W1)
    db1_acc = np.zeros_like(policy.b1)
    dW2_acc = np.zeros_like(policy.W2)
    db2_acc = np.zeros_like(policy.b2)
    adv_values: list[float] = []
    v_values: list[float] = []

    for t, advantage in zip(traj, advantages):
        advantage = float(advantage)
        adv_values.append(advantage)
        v_values.append(0.0)
        dW1, db1, dW2, db2 = policy.log_policy_gradients(
            sparse_state=t.state_one_hot,
            action_history_one_hot=t.action_history_one_hot,
            action=t.action,
        )
        dW1_acc += advantage * dW1
        db1_acc += advantage * db1
        dW2_acc += advantage * dW2
        db2_acc += advantage * db2

    return {
        "dW1": dW1_acc,
        "db1": db1_acc,
        "dW2": dW2_acc,
        "db2": db2_acc,
        "solved": bool(solved),
        "steps": int(len(traj)),
        "return": float(np.sum(rewards)),
        "adv_mean": float(np.mean(adv_values)) if adv_values else 0.0,
        "v_mean": float(np.mean(v_values)) if v_values else 0.0,
    }


class ReinforceSparseTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._episode_bar: tqdm | None = None
        if args.scramble_steps < 1:
            raise ValueError("--scramble-steps must be >= 1")

        self.ckpt = CheckpointManager(args.checkpoint_dir)
        loaded_policy, start_ep = load_sparse_latest(args.checkpoint_dir)
        if loaded_policy is None:
            policy_kw = {}
            if getattr(args, "hidden_dim", None) is not None:
                policy_kw["hidden_dim"] = args.hidden_dim
            if getattr(args, "init_scale", None) is not None:
                policy_kw["init_scale"] = args.init_scale
            self.policy = SparseLinearSoftmaxPolicy(seed=args.seed, **policy_kw)
            self.start_episode = 0
        else:
            self.policy = loaded_policy
            self.start_episode = start_ep

        self.env_rngs = [
            np.random.default_rng(None if args.seed is None else args.seed + i + 1)
            for i in range(args.num_envs)
        ]

        self._recent_solved = deque(maxlen=args.stats_window)
        self._recent_steps = deque(maxlen=args.stats_window)
        self._recent_returns = deque(maxlen=args.stats_window)
        self._solved_total = 0
        self._episodes_done = 0
        self._last_adv_mean = 0.0
        self._last_v_mean = 0.0
        self._log_solved_sum = 0
        self._log_steps_sum = 0.0
        self._log_return_sum = 0.0
        self._log_adv_sum = 0.0
        self._log_v_sum = 0.0
        self._log_count = 0

        self._log(
            "trainer_init sparse "
            f"episodes={args.episodes} num_envs={args.num_envs} "
            f"scramble_steps_max={args.scramble_steps} max_episode_steps={args.max_episode_steps} "
            f"gamma={args.gamma} lr={args.lr} checkpoint_dir={args.checkpoint_dir}"
        )
        if loaded_policy is None:
            self._log("checkpoint_status no checkpoint found, initialized random sparse policy")
        else:
            self._log(f"checkpoint_status resumed from episode={self.start_episode}")

    def _log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {message}"
        if self._episode_bar is not None:
            self._episode_bar.write(text)
        else:
            print(text, flush=True)

    def _record_episode_metrics(
        self,
        solved: bool,
        steps: int,
        total_return: float,
        adv_mean: float,
        v_mean: float,
    ) -> None:
        self._episodes_done += 1
        self._solved_total += int(solved)
        self._recent_solved.append(1 if solved else 0)
        self._recent_steps.append(steps)
        self._recent_returns.append(total_return)
        self._log_count += 1
        self._log_solved_sum += int(solved)
        self._log_steps_sum += float(steps)
        self._log_return_sum += float(total_return)
        self._log_adv_sum += float(adv_mean)
        self._log_v_sum += float(v_mean)

    def _maybe_log_interval_stats(self, global_episode: int) -> None:
        if self.args.log_interval <= 0:
            return
        if global_episode % self.args.log_interval != 0:
            return
        if self._log_count == 0:
            return
        recent_n = len(self._recent_solved)
        solve_rate_recent = (sum(self._recent_solved) / recent_n) if recent_n else 0.0
        avg_steps_recent = (sum(self._recent_steps) / recent_n) if recent_n else 0.0
        avg_return_recent = (sum(self._recent_returns) / recent_n) if recent_n else 0.0
        solve_rate_total = self._solved_total / self._episodes_done if self._episodes_done else 0.0
        solve_rate_interval = self._log_solved_sum / self._log_count
        avg_steps_interval = self._log_steps_sum / self._log_count
        avg_return_interval = self._log_return_sum / self._log_count
        avg_adv_interval = self._log_adv_sum / self._log_count
        avg_v_interval = self._log_v_sum / self._log_count
        self._log(
            "episode_stats "
            f"episode={global_episode} interval_episodes={self._log_count} "
            f"solve_rate_interval={solve_rate_interval:.3f} "
            f"avg_steps_interval={avg_steps_interval:.2f} "
            f"avg_return_interval={avg_return_interval:.2f} "
            f"adv_mean_interval={avg_adv_interval:.2f} "
            f"v_mean_interval={avg_v_interval:.2f} "
            f"solve_rate_recent={solve_rate_recent:.3f} "
            f"avg_steps_recent={avg_steps_recent:.2f} "
            f"avg_return_recent={avg_return_recent:.2f} "
            f"solve_rate_total={solve_rate_total:.3f}"
        )
        self._log_solved_sum = 0
        self._log_steps_sum = 0.0
        self._log_return_sum = 0.0
        self._log_adv_sum = 0.0
        self._log_v_sum = 0.0
        self._log_count = 0

    def _should_update_progress_postfix(self, global_episode: int, total_to_run: int, episodes_done_local: int) -> bool:
        if episodes_done_local >= total_to_run:
            return True
        if self.args.log_interval <= 0:
            return False
        return (global_episode % self.args.log_interval) == 0

    def _episode_gradients(self, traj: list[Transition]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rewards = np.asarray([t.reward for t in traj], dtype=np.float64)
        returns = _discounted_returns(rewards, self.args.gamma)
        advantages = returns
        dW1_acc = np.zeros_like(self.policy.W1)
        db1_acc = np.zeros_like(self.policy.b1)
        dW2_acc = np.zeros_like(self.policy.W2)
        db2_acc = np.zeros_like(self.policy.b2)
        adv_values: list[float] = []
        v_values: list[float] = []
        for t, advantage in zip(traj, advantages):
            advantage = float(advantage)
            adv_values.append(advantage)
            v_values.append(0.0)
            dW1, db1, dW2, db2 = self.policy.log_policy_gradients(
                sparse_state=t.state_one_hot,
                action_history_one_hot=t.action_history_one_hot,
                action=t.action,
            )
            dW1_acc += advantage * dW1
            db1_acc += advantage * db1
            dW2_acc += advantage * dW2
            db2_acc += advantage * db2
        self._last_adv_mean = float(np.mean(adv_values)) if adv_values else 0.0
        self._last_v_mean = float(np.mean(v_values)) if v_values else 0.0
        return dW1_acc, db1_acc, dW2_acc, db2_acc

    def run(self) -> None:
        total_to_run = self.args.episodes
        episodes_done_local = 0
        global_episode = self.start_episode

        try:
            self._episode_bar = tqdm(
                total=total_to_run,
                desc="REINFORCE sparse episodes",
                unit="ep",
                mininterval=1.0,
                maxinterval=5.0,
            )

            with ProcessPoolExecutor(max_workers=self.args.num_envs) as pool:
                while episodes_done_local < total_to_run:
                    batch_size = min(self.args.num_envs, total_to_run - episodes_done_local)
                    tasks = []
                    for _ in range(batch_size):
                        seed = int(self.env_rngs[0].integers(0, 2**31 - 1))
                        tasks.append(
                            {
                                "W1": self.policy.W1,
                                "b1": self.policy.b1,
                                "W2": self.policy.W2,
                                "b2": self.policy.b2,
                                "seed": seed,
                                "scramble_steps_max": self.args.scramble_steps,
                                "max_episode_steps": self.args.max_episode_steps,
                                "gamma": self.args.gamma,
                            }
                        )

                    batch_results = list(pool.map(_worker_episode_gradients, tasks))

                    batch_dW1 = np.zeros_like(self.policy.W1)
                    batch_db1 = np.zeros_like(self.policy.b1)
                    batch_dW2 = np.zeros_like(self.policy.W2)
                    batch_db2 = np.zeros_like(self.policy.b2)
                    save_marks: list[int] = []

                    for res in batch_results:
                        batch_dW1 += res["dW1"]
                        batch_db1 += res["db1"]
                        batch_dW2 += res["dW2"]
                        batch_db2 += res["db2"]
                        solved = bool(res["solved"])
                        steps = int(res["steps"])
                        total_return = float(res["return"])
                        adv_mean = float(res.get("adv_mean", 0.0))
                        v_mean = float(res.get("v_mean", 0.0))
                        global_episode += 1
                        episodes_done_local += 1
                        self._record_episode_metrics(
                            solved=solved,
                            steps=steps,
                            total_return=total_return,
                            adv_mean=adv_mean,
                            v_mean=v_mean,
                        )
                        self._maybe_log_interval_stats(global_episode)
                        if global_episode % self.args.save_every == 0:
                            save_marks.append(global_episode)

                    batch_dW1 /= batch_size
                    batch_db1 /= batch_size
                    batch_dW2 /= batch_size
                    batch_db2 /= batch_size
                    self.policy.apply_gradients(batch_dW1, batch_db1, batch_dW2, batch_db2, lr=self.args.lr)

                    self._episode_bar.update(batch_size)
                    if batch_results and self._should_update_progress_postfix(
                        global_episode, total_to_run, episodes_done_local
                    ):
                        solved_last = bool(batch_results[-1]["solved"])
                        steps_last = int(batch_results[-1]["steps"])
                        ret_last = float(batch_results[-1]["return"])
                        self._episode_bar.set_postfix(
                            {"steps": steps_last, "solved": solved_last, "ret": f"{ret_last:.1f}"}
                        )

                    for ep in save_marks:
                        metadata = {
                            "episode": ep,
                            "lr": self.args.lr,
                            "gamma": self.args.gamma,
                            "num_envs": self.args.num_envs,
                            "scramble_steps": self.args.scramble_steps,
                            "baseline": "discounted_returns",
                            "mode": "sparse",
                        }
                        path = self.ckpt.save_sparse(self.policy, episode=ep, metadata=metadata)
                        self._log(f"checkpoint_saved episode={ep} path={path}")

        finally:
            if self._episode_bar is not None:
                self._episode_bar.close()
                self._episode_bar = None


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    d = defaults or {}
    train = d.get("training", {})
    pol = d.get("policy", {})
    p = argparse.ArgumentParser(description="Train REINFORCE policy for sparse Rubik 2x2 (7 cells, 6 actions)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (training + policy params)")
    p.add_argument("--num-envs", type=int, default=train.get("num_envs", 16), help="Number of parallel simulators")
    p.add_argument("--episodes", type=int, default=train.get("episodes"), help="Total episodes (required if no --config)")
    p.add_argument("--max-episode-steps", type=int, default=train.get("max_episode_steps", 30))
    p.add_argument("--scramble-steps", type=int, default=train.get("scramble_steps"), help="Max scramble steps (required if no --config)")
    p.add_argument("--gamma", type=float, default=train.get("gamma", 1.0))
    p.add_argument("--lr", type=float, default=train.get("lr", 0.0001))
    p.add_argument("--save-every", type=int, default=train.get("save_every", 100000))
    p.add_argument("--checkpoint-dir", type=str, default=train.get("checkpoint_dir", "checkpoints_sparse"))
    p.add_argument("--seed", type=int, default=train.get("seed"))
    p.add_argument("--log-interval", type=int, default=train.get("log_interval", 10000))
    p.add_argument("--stats-window", type=int, default=train.get("stats_window", 1000))
    # Policy (for docs / override from config only; not in CLI by default)
    p.add_argument("--hidden-dim", type=int, default=pol.get("hidden_dim"), dest="hidden_dim", help="Policy hidden layer size")
    p.add_argument("--init-scale", type=float, default=pol.get("init_scale"), dest="init_scale", help="Policy weight init scale")
    return p


def main() -> None:
    # Pre-parse to get --config
    pre = argparse.ArgumentParser()
    pre.add_argument("--config", type=str, default=None)
    pre_args, rest = pre.parse_known_args()

    defaults = {}
    if pre_args.config:
        defaults = load_config(pre_args.config)

    parser = build_parser(defaults)
    args = parser.parse_args()

    if args.episodes is None:
        parser.error("--episodes required (or set in --config)")
    if args.scramble_steps is None:
        parser.error("--scramble-steps required (or set in --config)")

    trainer = ReinforceSparseTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
