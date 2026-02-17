
import os
import glob
import math
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TITLES = {
    "train/sr_batch": "Success Rate",
    "train/steps_mean_batch": "Mean Steps",
    "train/steps_to_solve_mean_batch": "Steps to Solve",
    "train/return_total_mean_batch": "Total Return",
    "train/return_step_mean_batch": "Step Reward",
    "train/return_inverse_penalty_mean_batch": "Inverse Penalty",
    "train/return_repeat_penalty_mean_batch": "Repeat Penalty",
    "train/return_timeout_penalty_mean_batch": "Timeout Penalty",
    "train/loss_batch": "Loss",
    "train/lr": "Learning Rate",
    "train/scramble_steps_current": "Scramble Depth",
}

SIZE_GUIDANCE = {
    'compressedHistograms': 0, 'images': 0, 'audio': 0, 'scalars': 0, 'histograms': 0,
}

def get_sorted_run_dirs(base_dir):
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return []
    return sorted([entry.path for entry in os.scandir(base_dir) if entry.is_dir() and entry.name.startswith("run_")])

def extract_tags(run_dir):
    if not glob.glob(os.path.join(run_dir, "events.out.tfevents.*")): return []
    ea = EventAccumulator(run_dir, size_guidance=SIZE_GUIDANCE)
    ea.Reload()
    return ea.Tags().get('scalars', [])

def extract_data(run_dir, tags):
    ea = EventAccumulator(run_dir, size_guidance=SIZE_GUIDANCE)
    ea.Reload()
    available = set(ea.Tags().get('scalars', []))
    data = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            data[tag] = ([e.step for e in events], [e.value for e in events])
        else:
            data[tag] = ([], [])
    return data

def concat_and_plot(run_dirs):
    all_tags = sorted({t for d in run_dirs for t in extract_tags(d)})
    if not all_tags:
        print("No scalar tags found.")
        return

    print(f"Processing {len(run_dirs)} runs with {len(all_tags)} tags...")
    
    combined = {tag: {"steps": [], "values": []} for tag in all_tags}
    total_offset = 0

    for run_dir in run_dirs:
        print(f"  Reading {os.path.basename(run_dir)}...")
        run_data = extract_data(run_dir, all_tags)
        max_step = 0
        has_data = False

        for tag in all_tags:
            steps, values = run_data[tag]
            if not steps: continue
            
            has_data = True
            max_step = max(max_step, max(steps))
            # Filter first 1000 steps of each run to prevent jumps, except for the oldest run
            is_oldest = os.path.basename(run_dir) == "run_20260217_205610"
            cutoff = min(steps) + (0 if is_oldest else 1000)
            indices = [i for i, s in enumerate(steps) if s >= cutoff]

            if indices:
                combined[tag]["steps"].extend([steps[i] + total_offset for i in indices])
                combined[tag]["values"].extend([values[i] for i in indices])

        if has_data:
            total_offset += max_step

    os.makedirs("media", exist_ok=True)
    chunk_size = 4
    num_parts = math.ceil(len(all_tags) / chunk_size)

    for i in range(num_parts):
        tags_subset = all_tags[i*chunk_size : (i+1)*chunk_size]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Metrics - Part {i + 1}", fontsize=16)
        axes = axes.flatten()

        for j, tag in enumerate(tags_subset):
            ax = axes[j]
            steps, values = combined[tag]["steps"], combined[tag]["values"]
            if steps:
                ax.plot(steps, values)
                ax.set_title(TITLES.get(tag, tag))
                ax.set_xlabel("Global Steps")
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(TITLES.get(tag, tag))

        for j in range(len(tags_subset), 4): fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = f"media/combined_plot_part_{i+1}.png"
        plt.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")

if __name__ == "__main__":
    base_dirs = [
        "runs/cube_reinforcer",
        "../runs/cube_reinforcer",
        "/home/phuc/working/rl_sk/Cube-Reinforcer/runs/cube_reinforcer"
    ]
    for d in base_dirs:
        if os.path.exists(d):
            concat_and_plot(get_sorted_run_dirs(d))
            break
    else:
        print("Log directory not found.")
