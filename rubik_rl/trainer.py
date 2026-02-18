"""REINFORCE training entrypoint."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat

import numpy as np
from tqdm import tqdm

from .checkpoint import CheckpointManager
from .client import RubikAPIClient
from .policy import LinearSoftmaxPolicy
from .reward import TIMEOUT_PENALTY, compute_step_reward
from .types import Transition
from rubik_sim.engine import RubikEngine
from rubik_sim.state_codec import encode_one_hot


def _sample_scramble_steps(rng: np.random.Generator, scramble_steps_max: int) -> int:
    return int(rng.integers(1, scramble_steps_max + 1))


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return returns

def _compute_returns_for_traj(task: dict) -> dict:
    W1 = task["W1"]
    b1 = task["b1"]
    W2 = task["W2"]
    b2 = task["b2"]
    seed = int(task["seed"])
    scramble_steps_max = int(task["scramble_steps_max"])
    max_episode_steps = int(task["max_episode_steps"])
    gamma = float(task["gamma"])
    policy = LinearSoftmaxPolicy(W1=W1, b1=b1, W2=W2, b2=b2, seed=seed)
    rng = np.random.default_rng(seed)

    env = RubikEngine(cube_size=2)
    env.reset()
    scramble_steps = _sample_scramble_steps(rng, scramble_steps_max)
    env.scramble(steps=scramble_steps, seed=seed + 17)

    traj: list[Transition] = []
    solved = False
    action_history: list[int] = []

    for _ in range(max_episode_steps):
        state_oh = encode_one_hot(env.get_state()).astype(np.float64)
        hist_oh = policy.history_one_hot(action_history) # (48,) (flattened 12*4)
        
        probs = policy.action_probs(state_oh, hist_oh)
        action = int(rng.choice(12, p=probs))
        env.step(action)

        solved = env.is_solved()
        next_state_oh = encode_one_hot(env.get_state()).astype(np.float64)
        
        reward = compute_step_reward(
                            action_history=action_history,
                            action=action,
                            state_before_one_hot=state_oh,
                            state_after_one_hot=next_state_oh,
                            solved_after=solved,
                        ) # compute_step_reward(action_history, action) # не зависит от state?

        traj.append(
            Transition(
                state_one_hot=state_oh,
                action_history_one_hot=hist_oh.copy(),
                action=action,
                reward=reward,
            )
        )
        action_history = (action_history + [action])[-4:]
        if solved:
            break

    if (not solved) and len(traj) > 0:
        traj[-1].reward -= TIMEOUT_PENALTY

    rewards = np.asarray([t.reward for t in traj], dtype=np.float64)
    returns = _discounted_returns(rewards, gamma)

    return returns

def _worker_episode_gradients(task: dict, baseline=None) -> dict:
    """
    samples ONE full trajectory (one episode)
    computes the REINFORCE gradient for that trajectory
    returns the accumulated gradient
    """
    W1 = task["W1"]
    b1 = task["b1"]
    W2 = task["W2"]
    b2 = task["b2"]
    seed = int(task["seed"])
    scramble_steps_max = int(task["scramble_steps_max"])
    max_episode_steps = int(task["max_episode_steps"])
    gamma = float(task["gamma"])
    policy = LinearSoftmaxPolicy(W1=W1, b1=b1, W2=W2, b2=b2, seed=seed)
    rng = np.random.default_rng(seed)

    env = RubikEngine(cube_size=2)
    env.reset()
    scramble_steps = _sample_scramble_steps(rng, scramble_steps_max)
    env.scramble(steps=scramble_steps, seed=seed + 17)

    traj: list[Transition] = []
    solved = False
    action_history: list[int] = []

    for _ in range(max_episode_steps):
        state_oh = encode_one_hot(env.get_state()).astype(np.float64)
        hist_oh = policy.history_one_hot(action_history) # (48,) (flattened 12*4)
        
        probs = policy.action_probs(state_oh, hist_oh)
        action = int(rng.choice(12, p=probs))
        env.step(action)

        solved = env.is_solved()
        next_state_oh = encode_one_hot(env.get_state()).astype(np.float64)
        
        reward = compute_step_reward(
                            action_history=action_history,
                            action=action,
                            state_before_one_hot=state_oh,
                            state_after_one_hot=next_state_oh,
                            solved_after=solved,
                        ) # compute_step_reward(action_history, action) # не зависит от state?

        traj.append(
            Transition(
                state_one_hot=state_oh,
                action_history_one_hot=hist_oh.copy(),
                action=action,
                reward=reward,
            )
        )
        action_history = (action_history + [action])[-4:]
        if solved:
            break

    if (not solved) and len(traj) > 0:
        traj[-1].reward -= TIMEOUT_PENALTY

    rewards = np.asarray([t.reward for t in traj], dtype=np.float64)
    returns = _discounted_returns(rewards, gamma)
    if baseline is not None:
        T = len(returns)
        advantages = returns - baseline[:T]
    else:
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
            state_one_hot=t.state_one_hot,
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


class ReinforceTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._episode_bar: tqdm | None = None
        if args.scramble_steps < 1:
            raise ValueError("--scramble-steps must be >= 1")

        self.external_client: RubikAPIClient | None = None
        if args.external_server:
            if args.num_envs != 1:
                raise ValueError("--external-server supports only --num-envs 1")
            self.external_client = RubikAPIClient(host=args.host, port=args.port)
            self._targets = [f"{args.host}:{args.port}"]
        else:
            self._targets = [f"local_proc_env_{i}" for i in range(args.num_envs)]

        self.ckpt = CheckpointManager(args.checkpoint_dir)

        loaded_policy, start_ep = self.ckpt.load_latest()
        if loaded_policy is None:
            self.policy = LinearSoftmaxPolicy(seed=args.seed)
            self.start_episode = 0
        else:
            self.policy = loaded_policy
            self.start_episode = start_ep

        self.env_rngs = [np.random.default_rng(None if args.seed is None else args.seed + i + 1) for i in range(args.num_envs)]

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

        self.baseline = args.baseline

        self._log(
            "trainer_init "
            f"targets={self._targets} episodes={args.episodes} num_envs={args.num_envs} "
            f"scramble_steps_max={args.scramble_steps} max_episode_steps={args.max_episode_steps} "
            f"gamma={args.gamma} lr={args.lr} baseline=discounted_returns "
            f"checkpoint_dir={args.checkpoint_dir}"
        )
        self._log(
            f"simulator_mode external_server={args.external_server} "
            f"backend={'process_pool' if not args.external_server else 'external_http'}"
        )
        if loaded_policy is None:
            self._log("checkpoint_status no checkpoint found, initialized random policy")
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

    def _collect_episode_external(self, client: RubikAPIClient, rng: np.random.Generator) -> tuple[list[Transition], bool, int, float]:
        client.reset() # puts the cube into solved state
        scramble_steps = _sample_scramble_steps(rng, self.args.scramble_steps)
        client.scramble(steps=scramble_steps) # applies random moves so the episode starts from a scrambled cube

        traj: list[Transition] = []
        solved = False
        action_history: list[int] = []

        for _ in range(self.args.max_episode_steps):
            state_oh = client.get_state()
            hist_oh = self.policy.history_one_hot(action_history)
            probs = self.policy.action_probs(state_oh, hist_oh)
            action = int(rng.choice(12, p=probs))

            out = client.step(action)
            solved = bool(out["solved"])
            next_state_oh = np.asarray(out["state"], dtype=np.float64)

            reward = compute_step_reward(
                action_history=action_history,
                action=action,
                state_before_one_hot=state_oh,
                state_after_one_hot=next_state_oh,
                solved_after=solved,
            )
            traj.append(
                Transition(
                    state_one_hot=state_oh,
                    action_history_one_hot=hist_oh.copy(),
                    action=action,
                    reward=reward,
                )
            )
            action_history = (action_history + [action])[-4:]
            if solved:
                break

        if (not solved) and len(traj) > 0:
            traj[-1].reward -= TIMEOUT_PENALTY

        total_return = float(sum(t.reward for t in traj))
        return traj, solved, len(traj), total_return

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

        for t, advantage in zip(traj, advantages):  # Monte-Carlo episode loop
            advantage = float(advantage)
            adv_values.append(advantage)
            v_values.append(0.0)
            dW1, db1, dW2, db2 = self.policy.log_policy_gradients(
                state_one_hot=t.state_one_hot,
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
                desc="REINFORCE episodes",
                unit="ep",
                mininterval=1.0,
                maxinterval=5.0,
            )

            if self.args.external_server:
                assert self.external_client is not None
                while episodes_done_local < total_to_run:
                    traj, solved, steps, total_return = self._collect_episode_external(
                        self.external_client,
                        self.env_rngs[0],
                    )
                    dW1, db1, dW2, db2 = self._episode_gradients(traj)
                    self.policy.apply_gradients(dW1, db1, dW2, db2, lr=self.args.lr)

                    global_episode += 1
                    episodes_done_local += 1
                    self._record_episode_metrics(
                        solved=solved,
                        steps=steps,
                        total_return=total_return,
                        adv_mean=self._last_adv_mean,
                        v_mean=self._last_v_mean,
                    )

                    self._episode_bar.update(1)
                    if self._should_update_progress_postfix(global_episode, total_to_run, episodes_done_local):
                        self._episode_bar.set_postfix({"steps": steps, "solved": solved, "ret": f"{total_return:.1f}"})
                    self._maybe_log_interval_stats(global_episode)

                    if global_episode % self.args.save_every == 0:
                        metadata = {
                            "episode": global_episode,
                            "lr": self.args.lr,
                            "gamma": self.args.gamma,
                            "num_envs": self.args.num_envs,
                            "scramble_steps": self.args.scramble_steps,
                            "baseline": "discounted_returns",
                        }
                        path = self.ckpt.save(self.policy, episode=global_episode, metadata=metadata)
                        self._log(f"checkpoint_saved episode={global_episode} path={path}")

            else:
                with ProcessPoolExecutor(max_workers=self.args.num_envs) as pool:
                    while episodes_done_local < total_to_run:
                        #batch size = number of sampled episodes (trajectories) used to estimate the Monte-Carlo gradient
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
                            ) # len(tasks)=number of sampled trajectories

                        # Run _worker_episode_gradients(task) for every task in tasks in parallel processes collect all returned results into a list
                        if self.baseline:
                            returns = list(pool.map(_compute_returns_for_traj, tasks)) 
                            T_max = max(len(R) for R in returns)
                            R_pad = np.zeros((batch_size, T_max), dtype=np.float64)
                            for i, R in enumerate(returns):
                                T = len(R)
                                R_pad[i, :T] = R
                            baseline = np.mean(R_pad, axis=0)
                        else:
                            baseline = None

                        batch_results = list(pool.map(_worker_episode_gradients, tasks, repeat(baseline))) 

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
                            global_episode,
                            total_to_run,
                            episodes_done_local,
                        ):
                            solved_last = bool(batch_results[-1]["solved"])
                            steps_last = int(batch_results[-1]["steps"])
                            ret_last = float(batch_results[-1]["return"])
                            self._episode_bar.set_postfix({"steps": steps_last, "solved": solved_last, "ret": f"{ret_last:.1f}"})

                        for ep in save_marks:
                            metadata = {
                                "episode": ep,
                                "lr": self.args.lr,
                                "gamma": self.args.gamma,
                                "num_envs": self.args.num_envs,
                                "scramble_steps": self.args.scramble_steps,
                                "baseline": "discounted_returns",
                            }
                            path = self.ckpt.save(self.policy, episode=ep, metadata=metadata)
                            self._log(f"checkpoint_saved episode={ep} path={path}")

        finally:
            if self._episode_bar is not None:
                self._episode_bar.close()
                self._episode_bar = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train REINFORCE policy for Rubik 2x2")
    p.add_argument("--host", default="127.0.0.1", help="Used only with --external-server")
    p.add_argument("--port", type=int, default=8000, help="Used only with --external-server")
    p.add_argument("--external-server", action="store_true", help="Use already running simulator server")
    p.add_argument("--num-envs", type=int, default=1, help="Number of parallel simulators for batch episodes")
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--scramble-steps", type=int, required=True)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--stats-window", type=int, default=100)
    p.add_argument("--baseline", action="store_true", help="Use time-step batch baseline")
    return p


def main() -> None:
    args = build_parser().parse_args()
    print(args.external_server)
    if args.baseline:
        print('Baseline approach is used')
    trainer = ReinforceTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
