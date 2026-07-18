import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(base_dir):
    matches = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name.startswith("events.out.tfevents"):
                matches.append(os.path.join(root, name))
    return matches


def load_tags(event_file):
    accumulator = EventAccumulator(event_file, size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator.Tags().get("scalars", [])


def choose_tag(tags_by_file):
    if not tags_by_file:
        return None
    sets = [set(tags) for tags in tags_by_file.values() if tags]
    if not sets:
        return None
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    candidates = sorted(common) if common else sorted(set.union(*sets))
    preferences = [
        "episode_reward",
        "ep_reward",
        "episode_return",
        "return",
        "reward",
        "eval/episode_reward",
        "test/episode_reward",
    ]
    for pref in preferences:
        for tag in candidates:
            if pref in tag:
                return tag
    return candidates[0] if candidates else None


def load_scalar_series(event_file, tag):
    accumulator = EventAccumulator(event_file, size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    events = accumulator.Scalars(tag)
    if not events:
        return None
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return pd.DataFrame({"step": steps, "value": values})


def algo_from_path(event_file, base_dir):
    rel = os.path.relpath(event_file, base_dir)
    parts = rel.split(os.sep)
    return parts[0] if parts else "run"


def build_algo_series(event_files, base_dir, tag):
    per_algo = defaultdict(list)
    for event_file in event_files:
        df = load_scalar_series(event_file, tag)
        if df is None or df.empty:
            continue
        algo = algo_from_path(event_file, base_dir)
        df = df.drop_duplicates(subset=["step"]).sort_values("step")
        per_algo[algo].append(df)
    algo_series = {}
    for algo, dfs in per_algo.items():
        merged = pd.concat(dfs, axis=0, ignore_index=True)
        merged = merged.groupby("step", as_index=False)["value"].mean()
        algo_series[algo] = merged.sort_values("step")
    return algo_series


def apply_smoothing(df, window):
    if window <= 1:
        return df
    smoothed = df.copy()
    smoothed["value"] = smoothed["value"].rolling(window=window, min_periods=1).mean()
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/hard_data/user/majunjie/dcd/logs/baseline_20k")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smooth", type=int, default=1)
    args = parser.parse_args()

    event_files = find_event_files(args.log_dir)
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files found under {args.log_dir}")

    tags_by_file = {f: load_tags(f) for f in event_files}
    tag = args.tag or choose_tag(tags_by_file)
    if not tag:
        raise RuntimeError("No scalar tags found in event files.")

    algo_series = build_algo_series(event_files, args.log_dir, tag)
    if not algo_series:
        raise RuntimeError(f"No scalar data found for tag: {tag}")

    plt.figure(figsize=(10, 6))
    for algo, df in sorted(algo_series.items(), key=lambda x: x[0]):
        df = apply_smoothing(df, args.smooth)
        plt.plot(df["step"].values, df["value"].values, label=algo)

    plt.xlabel("step")
    plt.ylabel(tag)
    plt.title(tag)
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path = args.out or os.path.join(args.log_dir, "tb_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"tag: {tag}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
