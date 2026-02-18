# plot_objective_compare.py
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_objective(log_path: Path):
    """
    Extract (episode, avg_return_interval) from lines like:
    ... episode=595000 ... avg_return_interval=-13.76 ... solve_rate_total=0.732
    """
    episodes = []
    obj = []

    # avg_return_interval can be negative, so include '-' in the regex
    pattern = re.compile(r"episode=(\d+).*avg_return_interval=([-0-9.]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "episode_stats" not in line:
                continue
            m = pattern.search(line)
            if m:
                episodes.append(int(m.group(1)))
                obj.append(float(m.group(2)))

    return np.array(episodes, dtype=np.int64), np.array(obj, dtype=np.float64)


def smooth_xy(x, y, window: int):
    if window is None or window <= 1 or len(y) < window:
        return x, y
    y_s = np.convolve(y, np.ones(window) / window, mode="valid")
    x_s = x[window - 1 :]
    return x_s, y_s


def main():
    parser = argparse.ArgumentParser(description="Plot objective (avg_return_interval) from two logs")
    parser.add_argument("--baseline-log", required=True, help="Log file for run WITH baseline")
    parser.add_argument("--nobaseline-log", required=True, help="Log file for run WITHOUT baseline")
    parser.add_argument("--output", default="objective_compare.png", help="Output image path")
    parser.add_argument("--smooth", type=int, default=0, help="Moving average window (0/1 = off)")
    parser.add_argument("--label-baseline", default="With baseline", help="Legend label for baseline curve")
    parser.add_argument("--label-nobaseline", default="Without baseline", help="Legend label for no-baseline curve")
    args = parser.parse_args()

    base_path = Path(args.baseline_log)
    nobase_path = Path(args.nobaseline_log)
    if not base_path.exists():
        raise FileNotFoundError(f"Baseline log not found: {base_path}")
    if not nobase_path.exists():
        raise FileNotFoundError(f"No-baseline log not found: {nobase_path}")

    x1, y1 = parse_objective(base_path)
    x2, y2 = parse_objective(nobase_path)

    if len(x1) == 0:
        raise RuntimeError(f"No episode_stats/avg_return_interval found in baseline log: {base_path}")
    if len(x2) == 0:
        raise RuntimeError(f"No episode_stats/avg_return_interval found in no-baseline log: {nobase_path}")

    x1, y1 = smooth_xy(x1, y1, args.smooth)
    x2, y2 = smooth_xy(x2, y2, args.smooth)

    plt.figure(figsize=(9, 5))
    plt.plot(x1, y1, linewidth=2, label=args.label_baseline)
    plt.plot(x2, y2, linewidth=2, label=args.label_nobaseline)

    plt.xlabel("Episode")
    plt.ylabel("Objective estimate (avg_return_interval)")
    plt.title("Objective comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
