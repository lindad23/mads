#!/usr/bin/env python
"""Evaluate Maze baselines at timestep-aligned checkpoints.

This script is intentionally offline: it loads saved checkpoints, evaluates all
methods on the same sequence of MultiGrid seeds, and writes one CSV row per
method/checkpoint target.  It does not start training.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval import Evaluator
from envs.wrappers import ParallelAdversarialVecEnv
from util import DotDict, is_discrete_actions, make_agent


DEFAULT_RUNS = {
    "ppo": "baseline-domain_randomization-MultiGrid_Task2_v0-seed7001-20260707_1130_clean",
    "procurl_target": "baseline-procurl_target-MultiGrid_Task2_v0-seed7101-20260707_1130_clean",
    "cp_drl": "baseline-cp_drl-MultiGrid_Task2_v0-seed7201-30m_20260713",
}


@dataclass(frozen=True)
class CheckpointCandidate:
    path: str
    name: str
    steps: int
    update: Optional[int]
    is_archive: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Maze baselines at timestep-aligned checkpoints.")
    parser.add_argument(
        "--base_path",
        type=str,
        default="logs_tmp/baselines_clean_20260707_1130",
        help="Directory containing run subdirectories.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=XPID",
        help="Run to evaluate. Can be repeated. Defaults to current Maze baselines.")
    parser.add_argument(
        "--timesteps",
        type=str,
        default="5000000,10000000,20000000,30000000",
        help="Comma-separated target environment steps.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MultiGrid-Task2-v0",
        help="Evaluation environment.")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=500,
        help="Number of fixed-seed evaluation episodes per checkpoint.")
    parser.add_argument(
        "--seed",
        type=int,
        default=100000,
        help="Base seed. Episode i uses seed + i.")
    parser.add_argument(
        "--selection",
        choices=["le", "nearest"],
        default="le",
        help="'le' selects the closest checkpoint not after the target timestep.")
    parser.add_argument(
        "--max_step_gap",
        type=int,
        default=0,
        help="If >0, skip selected checkpoints whose absolute step gap exceeds this value.")
    parser.add_argument(
        "--model_name",
        choices=["agent", "adversary_agent"],
        default="agent",
        help="Which agent state to evaluate.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy actions for evaluation.")
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results/maze_aligned_checkpoints.csv",
        help="Output CSV path.")
    parser.add_argument(
        "--list_only",
        action="store_true",
        help="Only write the selected checkpoint manifest; do not evaluate.")
    return parser.parse_args()


def parse_runs(run_args: Sequence[str]) -> Dict[str, str]:
    if not run_args:
        return dict(DEFAULT_RUNS)

    runs: Dict[str, str] = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"--run must be LABEL=XPID, got {item!r}")
        label, xpid = item.split("=", 1)
        label = label.strip()
        xpid = xpid.strip()
        if not label or not xpid:
            raise ValueError(f"--run must be LABEL=XPID, got {item!r}")
        runs[label] = xpid
    return runs


def parse_timesteps(raw: str) -> List[int]:
    timesteps = []
    for item in raw.split(","):
        item = item.strip().replace("_", "")
        if item:
            timesteps.append(int(item))
    if not timesteps:
        raise ValueError("At least one timestep is required.")
    return timesteps


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def read_last_log_row(log_path: str) -> Optional[dict]:
    if not os.path.exists(log_path):
        return None
    last = None
    header = None
    data_lines = []
    with open(log_path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                header = line[1:].strip()
            elif line.strip():
                data_lines.append(line)
    if header is None or not data_lines:
        return None
    reader = csv.DictReader(data_lines, fieldnames=next(csv.reader([header])))
    for row in reader:
        last = row
    return last


def int_from_row(row: Optional[dict], key: str) -> Optional[int]:
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return int(float(value))


def steps_per_update(meta_args: dict) -> int:
    return int(meta_args["num_steps"]) * int(meta_args["num_processes"])


def discover_checkpoints(run_dir: str) -> List[CheckpointCandidate]:
    meta_path = os.path.join(run_dir, "meta.json")
    logs_path = os.path.join(run_dir, "logs.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json in {run_dir}")

    meta_args = load_json(meta_path)["args"]
    step_unit = steps_per_update(meta_args)
    last_row = read_last_log_row(logs_path)
    latest_steps = int_from_row(last_row, "steps")
    latest_update = int_from_row(last_row, "total_student_grad_updates")

    candidates: List[CheckpointCandidate] = []
    archive_re = re.compile(r"^model_(\d+)\.tar$")
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        match = archive_re.match(name)
        if match:
            update = int(match.group(1))
            candidates.append(CheckpointCandidate(
                path=path,
                name=name,
                steps=update * step_unit,
                update=update,
                is_archive=True))
        elif name == "model.tar" and latest_steps is not None:
            candidates.append(CheckpointCandidate(
                path=path,
                name=name,
                steps=latest_steps,
                update=latest_update,
                is_archive=False))

    candidates.sort(key=lambda item: (item.steps, item.name))
    return candidates


def select_checkpoint(
    candidates: Sequence[CheckpointCandidate],
    target_steps: int,
    selection: str,
    max_step_gap: int,
) -> Optional[CheckpointCandidate]:
    if not candidates:
        return None
    if selection == "le":
        valid = [item for item in candidates if item.steps <= target_steps]
        if not valid:
            return None
        selected = max(valid, key=lambda item: item.steps)
    else:
        selected = min(candidates, key=lambda item: abs(item.steps - target_steps))

    if max_step_gap > 0 and abs(selected.steps - target_steps) > max_step_gap:
        return None
    return selected


def make_eval_venv(env_name: str, xpid_flags: DotDict, device: str):
    make_fn = [lambda: Evaluator.make_env(
        env_name,
        record_video=False,
        use_global_policy=xpid_flags.get("use_global_policy", False))]
    venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
    return Evaluator.wrap_venv(venv, env_name=env_name, device=device)


def load_agent(run_dir: str, checkpoint_path: str, model_name: str, env_name: str, device: str):
    meta = load_json(os.path.join(run_dir, "meta.json"))
    xpid_flags = DotDict(meta["args"])
    venv = make_eval_venv(env_name, xpid_flags, device)
    agent = make_agent(name="agent", env=venv, args=xpid_flags, device=device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "runner_state_dict" in checkpoint:
        state_dict = checkpoint["runner_state_dict"]["agent_state_dict"][model_name]
    else:
        state_dict = checkpoint
    agent.algo.actor_critic.load_state_dict(state_dict)
    agent.eval()
    return agent, venv


def to_scalar(value, default=float("nan")) -> float:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def evaluate_one_checkpoint(
    run_dir: str,
    checkpoint: CheckpointCandidate,
    env_name: str,
    num_episodes: int,
    seed: int,
    model_name: str,
    deterministic: bool,
    device: str = "cpu",
) -> dict:
    agent, venv = load_agent(run_dir, checkpoint.path, model_name, env_name, device)
    discrete_actions = is_discrete_actions(venv)

    returns: List[float] = []
    lengths: List[float] = []
    spls: List[float] = []
    successes: List[float] = []
    passable_values: List[float] = []
    shortest_paths: List[float] = []

    try:
        for episode_idx in range(num_episodes):
            episode_seed = seed + episode_idx
            venv.set_seed([episode_seed])
            obs = venv.reset()

            shortest_path = to_scalar(venv.get_shortest_path_length())
            passable = to_scalar(venv.get_passable())
            shortest_paths.append(shortest_path)
            passable_values.append(passable)

            recurrent_hidden_states = torch.zeros(
                1,
                agent.algo.actor_critic.recurrent_hidden_state_size,
                device=device)
            if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == "lstm":
                recurrent_hidden_states = (
                    recurrent_hidden_states,
                    torch.zeros_like(recurrent_hidden_states))
            masks = torch.ones(1, 1, device=device)

            while True:
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = agent.act(
                        obs,
                        recurrent_hidden_states,
                        masks,
                        deterministic=deterministic)

                action_np = action.cpu().numpy()
                if not discrete_actions:
                    action_np = agent.process_action(action_np)
                obs, _, done, infos = venv.step(action_np)
                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=device)

                if done[0] and "episode" in infos[0]:
                    epinfo = infos[0]["episode"]
                    episode_return = float(epinfo["r"])
                    episode_length = float(epinfo["l"])
                    success = 1.0 if episode_return > 0 else 0.0

                    returns.append(episode_return)
                    lengths.append(episode_length)
                    successes.append(success)
                    if success and math.isfinite(shortest_path) and shortest_path > 0:
                        spls.append(shortest_path / max(shortest_path, episode_length))
                    else:
                        spls.append(0.0)
                    break
    finally:
        venv.close()

    return {
        "episodes": len(returns),
        "solved_rate": float(np.mean(successes)) if successes else float("nan"),
        "eval_return_mean": float(np.mean(returns)) if returns else float("nan"),
        "eval_return_std": float(np.std(returns)) if returns else float("nan"),
        "episode_length_mean": float(np.mean(lengths)) if lengths else float("nan"),
        "spl_mean": float(np.mean(spls)) if spls else float("nan"),
        "passable_rate": float(np.mean(passable_values)) if passable_values else float("nan"),
        "shortest_path_mean": float(np.mean(shortest_paths)) if shortest_paths else float("nan"),
    }


def write_rows(output_path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "label",
        "xpid",
        "target_steps",
        "checkpoint_name",
        "checkpoint_steps",
        "checkpoint_gap",
        "checkpoint_update",
        "checkpoint_is_archive",
        "episodes",
        "solved_rate",
        "eval_return_mean",
        "eval_return_std",
        "episode_length_mean",
        "spl_mean",
        "passable_rate",
        "shortest_path_mean",
        "status",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    base_path = os.path.expanduser(os.path.expandvars(args.base_path))
    runs = parse_runs(args.run)
    timesteps = parse_timesteps(args.timesteps)

    rows: List[dict] = []
    for label, xpid in runs.items():
        run_dir = os.path.join(base_path, xpid)
        candidates = discover_checkpoints(run_dir)

        for target_steps in timesteps:
            selected = select_checkpoint(
                candidates,
                target_steps=target_steps,
                selection=args.selection,
                max_step_gap=args.max_step_gap)
            if selected is None:
                rows.append({
                    "label": label,
                    "xpid": xpid,
                    "target_steps": target_steps,
                    "checkpoint_name": "",
                    "checkpoint_steps": "",
                    "checkpoint_gap": "",
                    "checkpoint_update": "",
                    "checkpoint_is_archive": "",
                    "episodes": 0,
                    "status": "missing_checkpoint",
                })
                continue

            row = {
                "label": label,
                "xpid": xpid,
                "target_steps": target_steps,
                "checkpoint_name": selected.name,
                "checkpoint_steps": selected.steps,
                "checkpoint_gap": selected.steps - target_steps,
                "checkpoint_update": selected.update if selected.update is not None else "",
                "checkpoint_is_archive": int(selected.is_archive),
                "status": "selected" if args.list_only else "ok",
            }

            if not args.list_only:
                metrics = evaluate_one_checkpoint(
                    run_dir=run_dir,
                    checkpoint=selected,
                    env_name=args.env_name,
                    num_episodes=args.num_episodes,
                    seed=args.seed,
                    model_name=args.model_name,
                    deterministic=args.deterministic,
                    device="cpu")
                row.update(metrics)
            rows.append(row)

    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
