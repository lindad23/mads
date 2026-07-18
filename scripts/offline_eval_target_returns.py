#!/usr/bin/env python3
"""Offline target-environment evaluation for saved training checkpoints.

This script evaluates saved checkpoints from baseline or MADS runs on a shared
target environment and writes aligned scalar curves. It is intended for fair
comparison when each algorithm trained on a different curriculum distribution.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import Evaluator
from util import DotDict, make_agent, is_discrete_actions
from envs.wrappers import ParallelAdversarialVecEnv


CHECKPOINT_RE = re.compile(r"^model(?:_(?P<index>\d+))?\.tar$")


@dataclass
class CheckpointJob:
    run_dir: str
    run_name: str
    model_file: str
    model_name: str
    checkpoint_index: Optional[int]
    eval_step: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints on a shared target environment."
    )

    def add_bool_arg(name: str, default: bool, help_text: str = "") -> None:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
        group.add_argument(
            f"--no_{name}",
            dest=name,
            action="store_false",
            help=f"Disable {name.replace('_', ' ')}.",
        )
        parser.set_defaults(**{name: default})

    parser.add_argument(
        "--run_dir",
        action="append",
        default=[],
        help="Run directory containing meta.json and model*.tar. Can be repeated.",
    )
    parser.add_argument(
        "--run_glob",
        action="append",
        default=[],
        help="Glob for run directories. Can be repeated.",
    )
    parser.add_argument(
        "--target_env",
        required=True,
        help="Target evaluation env, e.g. MultiGrid-Task2-v0.",
    )
    parser.add_argument(
        "--output_dir",
        default="logs_tmp/offline_target_eval",
        help="Directory for CSV and TensorBoard output.",
    )
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_processes", type=int, default=10)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=100000,
        help="Base seed for fixed target-level evaluation distributions.",
    )
    parser.add_argument(
        "--multigrid_eval_mode",
        default="reset_random",
        choices=["reset_random"],
        help="How to instantiate playable target levels for MultiGrid envs.",
    )
    parser.add_argument("--model_name", default="agent", choices=["agent", "adversary_agent"])
    add_bool_arg(
        "include_final",
        True,
        "Evaluate model.tar in addition to archived model_N.tar files.",
    )
    parser.add_argument(
        "--max_checkpoints_per_run",
        type=int,
        default=None,
        help="Optional cap after sorting checkpoints by eval step.",
    )
    parser.add_argument(
        "--checkpoint_stride",
        type=int,
        default=1,
        help="Evaluate every Nth checkpoint after sorting.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda:0 if available, otherwise cpu.",
    )
    add_bool_arg("deterministic", False)
    add_bool_arg("write_tensorboard", True)
    add_bool_arg("overwrite", False, "Overwrite existing result CSV.")
    return parser.parse_args()


def _read_meta(run_dir: str) -> DotDict:
    with open(os.path.join(run_dir, "meta.json")) as f:
        return DotDict(json.load(f)["args"])


def _read_logs_csv(run_dir: str) -> List[Dict[str, str]]:
    logs_csv = os.path.join(run_dir, "logs.csv")
    if not os.path.exists(logs_csv):
        return []

    fields = None
    rows: List[Dict[str, str]] = []
    with open(logs_csv, newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# "):
                fields = next(csv.reader([line[2:]]))
                continue
            if fields is None:
                continue
            values = next(csv.reader([line]))
            if len(values) == len(fields):
                rows.append(dict(zip(fields, values)))
    return rows


def _to_int(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _basis_field(meta_args: DotDict) -> str:
    basis = getattr(meta_args, "checkpoint_basis", "total_student_grad_updates")
    if basis == "student_grad_updates":
        return "total_student_grad_updates"
    if basis in {"num_updates", "total_num_edits"}:
        return basis
    return basis


def _step_for_checkpoint(
    rows: List[Dict[str, str]],
    meta_args: DotDict,
    checkpoint_index: Optional[int],
) -> int:
    if not rows:
        return 0

    if checkpoint_index is None:
        return _to_int(rows[-1].get("steps")) or 0

    basis = _basis_field(meta_args)
    best = None
    for row in rows:
        basis_value = _to_int(row.get(basis))
        if basis_value is None:
            continue
        if basis_value >= checkpoint_index:
            best = row
            break
        best = row

    if best is None:
        best = rows[-1]
    return _to_int(best.get("steps")) or checkpoint_index


def _discover_run_dirs(args: argparse.Namespace) -> List[str]:
    run_dirs = list(args.run_dir)
    for pattern in args.run_glob:
        run_dirs.extend(glob.glob(pattern))

    deduped = []
    seen = set()
    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        if run_dir in seen:
            continue
        seen.add(run_dir)
        if os.path.isdir(run_dir) and os.path.exists(os.path.join(run_dir, "meta.json")):
            deduped.append(run_dir)
        else:
            print(f"Skipping non-run directory: {run_dir}", flush=True)
    return sorted(deduped)


def _discover_checkpoints(
    run_dir: str,
    meta_args: DotDict,
    rows: List[Dict[str, str]],
    model_name: str,
    include_final: bool,
) -> List[CheckpointJob]:
    run_name = os.path.basename(run_dir)
    jobs = []
    for path in glob.glob(os.path.join(run_dir, "model*.tar")):
        name = os.path.basename(path)
        match = CHECKPOINT_RE.match(name)
        if not match:
            continue
        raw_index = match.group("index")
        if raw_index is None and not include_final:
            continue
        checkpoint_index = int(raw_index) if raw_index is not None else None
        jobs.append(
            CheckpointJob(
                run_dir=run_dir,
                run_name=run_name,
                model_file=path,
                model_name=model_name,
                checkpoint_index=checkpoint_index,
                eval_step=_step_for_checkpoint(rows, meta_args, checkpoint_index),
            )
        )

    jobs.sort(key=lambda job: (job.eval_step, job.checkpoint_index is None, job.model_file))
    return jobs


def _make_eval_venv(
    env_name: str,
    num_processes: int,
    meta_args: DotDict,
    device: torch.device,
):
    make_fn = [
        lambda: Evaluator.make_env(
            env_name,
            frame_stack=meta_args.frame_stack,
            grayscale=meta_args.grayscale,
            num_action_repeat=getattr(meta_args, "num_action_repeat", 1),
            use_global_critic=meta_args.use_global_critic,
            use_global_policy=meta_args.use_global_policy,
        )
    ] * num_processes
    venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
    return Evaluator.wrap_venv(venv, env_name=env_name, device=device)


def _make_eval_agent(meta_args: DotDict, env_name: str, device: torch.device):
    dummy_venv = _make_eval_venv(env_name, 1, meta_args, device)
    agent = make_agent(name="agent", env=dummy_venv, args=meta_args, device=device)
    return agent, dummy_venv


def _load_checkpoint(agent, job: CheckpointJob) -> None:
    checkpoint = torch.load(job.model_file, map_location="cpu")
    if "runner_state_dict" in checkpoint:
        state_dict = checkpoint["runner_state_dict"]["agent_state_dict"][job.model_name]
    else:
        state_dict = checkpoint
    agent.algo.actor_critic.load_state_dict(state_dict)
    agent.eval()


def _safe_run_name(run_name: str) -> str:
    return run_name.replace(os.sep, "_")


def _seed_eval_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_recurrent_states(agent, num_processes: int, device: torch.device):
    hidden_size = agent.algo.actor_critic.recurrent_hidden_state_size
    recurrent_hidden_states = torch.zeros(num_processes, hidden_size, device=device)
    if (
        agent.algo.actor_critic.is_recurrent
        and getattr(agent.algo.actor_critic.rnn, "arch", None) == "lstm"
    ):
        recurrent_hidden_states = (
            recurrent_hidden_states,
            torch.zeros_like(recurrent_hidden_states),
        )
    return recurrent_hidden_states


def _zero_recurrent_state(agent, recurrent_hidden_states, index: int) -> None:
    if not agent.is_recurrent:
        return
    if isinstance(recurrent_hidden_states, tuple):
        recurrent_hidden_states[0][index].zero_()
        recurrent_hidden_states[1][index].zero_()
    else:
        recurrent_hidden_states[index].zero_()


def _evaluate_multigrid_target(
    agent,
    meta_args: DotDict,
    target_env: str,
    num_episodes: int,
    num_processes: int,
    device: torch.device,
    deterministic: bool,
    eval_seed: int,
) -> Dict[str, float]:
    """Evaluate adversarial MultiGrid policies on playable fixed-seed levels."""
    returns: List[float] = []
    solved_episodes = 0
    max_processes = max(1, min(num_processes, num_episodes))
    is_discrete = None

    while len(returns) < num_episodes:
        batch_size = min(max_processes, num_episodes - len(returns))
        venv = _make_eval_venv(target_env, batch_size, meta_args, device)
        try:
            if is_discrete is None:
                is_discrete = is_discrete_actions(venv)

            seeds = [eval_seed + len(returns) + i for i in range(batch_size)]
            venv.set_seed(seeds)
            obs = venv.reset_random()

            recurrent_hidden_states = _init_recurrent_states(agent, batch_size, device)
            masks = torch.ones(batch_size, 1, device=device)
            active = np.ones(batch_size, dtype=bool)

            while active.any():
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = agent.act(
                        obs,
                        recurrent_hidden_states,
                        masks,
                        deterministic=deterministic,
                    )

                action = action.cpu().numpy()
                if not is_discrete:
                    action = agent.process_action(action)

                obs, _, done, infos = venv.step_env(action)
                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=device,
                )

                for i, info in enumerate(infos):
                    if not active[i] or "episode" not in info:
                        continue
                    episode_return = float(info["episode"]["r"])
                    returns.append(episode_return)
                    if episode_return > 0:
                        solved_episodes += 1
                    active[i] = False
                    _zero_recurrent_state(agent, recurrent_hidden_states, i)
        finally:
            venv.close()

    return {
        f"test_returns:{target_env}": float(np.mean(returns)),
        f"solved_rate:{target_env}": solved_episodes / float(num_episodes),
    }


def _evaluate_job(
    job: CheckpointJob,
    meta_args: DotDict,
    target_env: str,
    num_episodes: int,
    num_processes: int,
    device: torch.device,
    deterministic: bool,
    eval_seed: int,
) -> Dict[str, object]:
    _seed_eval_rng(eval_seed)
    agent, dummy_venv = _make_eval_agent(meta_args, target_env, device)
    try:
        _load_checkpoint(agent, job)
        if target_env.startswith("MultiGrid"):
            stats = _evaluate_multigrid_target(
                agent,
                meta_args,
                target_env,
                num_episodes,
                num_processes,
                device,
                deterministic,
                eval_seed,
            )
        else:
            evaluator = Evaluator(
                [target_env],
                num_processes=min(num_processes, num_episodes),
                num_episodes=num_episodes,
                frame_stack=meta_args.frame_stack,
                grayscale=meta_args.grayscale,
                num_action_repeat=getattr(meta_args, "num_action_repeat", 1),
                use_global_critic=meta_args.use_global_critic,
                use_global_policy=meta_args.use_global_policy,
                device=device,
            )
            try:
                stats = evaluator.evaluate(agent, deterministic=deterministic)
            finally:
                evaluator.close()
    finally:
        dummy_venv.close()

    return {
        "run_name": job.run_name,
        "run_dir": job.run_dir,
        "model_file": os.path.basename(job.model_file),
        "model_name": job.model_name,
        "checkpoint_index": "" if job.checkpoint_index is None else job.checkpoint_index,
        "eval_step": job.eval_step,
        "target_env": target_env,
        "target_agent_return": stats[f"test_returns:{target_env}"],
        "target_solved_rate": stats[f"solved_rate:{target_env}"],
    }


def _write_result_header(csv_path: str, overwrite: bool) -> None:
    if os.path.exists(csv_path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "run_dir",
                "model_file",
                "model_name",
                "checkpoint_index",
                "eval_step",
                "target_env",
                "target_agent_return",
                "target_solved_rate",
            ],
        )
        writer.writeheader()


def _append_result(csv_path: str, row: Dict[str, object]) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    csv_path = os.path.join(args.output_dir, f"target_eval_{args.target_env}.csv")
    _write_result_header(csv_path, args.overwrite)

    run_dirs = _discover_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run directories found.")

    writers: Dict[str, SummaryWriter] = {}
    try:
        for run_dir in run_dirs:
            meta_args = _read_meta(run_dir)
            rows = _read_logs_csv(run_dir)
            jobs = _discover_checkpoints(
                run_dir,
                meta_args,
                rows,
                model_name=args.model_name,
                include_final=args.include_final,
            )
            jobs = jobs[:: max(args.checkpoint_stride, 1)]
            if args.max_checkpoints_per_run is not None:
                jobs = jobs[: args.max_checkpoints_per_run]
            if not jobs:
                print(f"No checkpoints found for {run_dir}", flush=True)
                continue

            for job in jobs:
                print(
                    f"Evaluating {job.run_name}/{os.path.basename(job.model_file)} "
                    f"at step {job.eval_step} on {args.target_env}",
                    flush=True,
                )
                row = _evaluate_job(
                    job,
                    meta_args,
                    args.target_env,
                    args.num_episodes,
                    args.num_processes,
                    device,
                    args.deterministic,
                    args.eval_seed,
                )
                _append_result(csv_path, row)

                if args.write_tensorboard:
                    writer = writers.get(job.run_name)
                    if writer is None:
                        writer = SummaryWriter(
                            os.path.join(args.output_dir, "tb", _safe_run_name(job.run_name))
                        )
                        writers[job.run_name] = writer
                    step = int(row["eval_step"])
                    writer.add_scalar("target_agent_return", float(row["target_agent_return"]), step)
                    writer.add_scalar("target_solved_rate", float(row["target_solved_rate"]), step)
                    writer.flush()
    finally:
        for writer in writers.values():
            writer.close()

    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
