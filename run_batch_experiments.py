import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExperimentSpec:
    tag: str
    env_name: str
    test_env_name: str


def _parse_int_list(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def _nvidia_smi_query() -> List[Tuple[int, int, int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if not out:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, used, total, util = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        rows.append((idx, used, total, util))
    return rows


def _pick_idle_gpu(
    allowed_gpus: Optional[List[int]],
    min_free_mem_mb: int,
    max_gpu_util: int,
    running_counts: Dict[int, int],
    max_jobs_per_gpu: int,
    mem_per_job_mb: int,
) -> Optional[int]:
    rows = _nvidia_smi_query()
    if not rows:
        return None
    candidates: List[Tuple[int, int]] = []
    for idx, used, total, util in rows:
        if allowed_gpus is not None and idx not in allowed_gpus:
            continue
        if util > max_gpu_util:
            continue
        free = total - used
        current = running_counts.get(idx, 0)
        if current >= max_jobs_per_gpu:
            continue
        if free >= (min_free_mem_mb + mem_per_job_mb * (current + 1)):
            candidates.append((idx, free))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _prune_finished(running: Dict[int, List[subprocess.Popen]]) -> None:
    finished_keys: List[int] = []
    for key, procs in running.items():
        alive: List[subprocess.Popen] = []
        for p in procs:
            if p.poll() is None:
                alive.append(p)
            else:
                _ = p.wait()
        if alive:
            running[key] = alive
        else:
            finished_keys.append(key)
    for key in finished_keys:
        running.pop(key, None)


def _build_train_cmd(
    base_xpid: str,
    spec: ExperimentSpec,
    seed: int,
    log_dir: str,
) -> List[str]:
    xpid = f"{base_xpid}-{spec.tag}-seed{seed}"
    cmd = [
        "python",
        "-m",
        "train_mads",
        f"--xpid={xpid}",
        f"--env_name={spec.env_name}",
        "--use_gae=True",
        "--gamma=0.99",
        "--gae_lambda=0.9",
        f"--seed={seed}",
        "--num_control_points=12",
        "--recurrent_arch=lstm",
        "--recurrent_agent=False",
        "--recurrent_adversary_env=False",
        "--recurrent_hidden_size=1",
        "--use_global_critic=False",
        "--lr=0.0003",
        "--num_steps=2048",
        "--num_processes=16",
        "--num_env_steps=100000000",
        "--ppo_epoch=5",
        "--num_mini_batch=32",
        "--entropy_coef=0.001",
        "--value_loss_coef=0.5",
        "--clip_param=0.2",
        "--clip_value_loss=False",
        "--adv_entropy_coef=0.01",
        "--max_grad_norm=0.5",
        "--algo=ppo",
        "--ued_algo=paired",
        "--use_plr=False",
        "--level_replay_prob=0.0",
        "--level_replay_rho=0.5",
        "--level_replay_seed_buffer_size=1000",
        "--level_replay_score_transform=rank",
        "--level_replay_temperature=0.1",
        "--staleness_coef=0.5",
        "--no_exploratory_grad_updates=False",
        "--use_editor=False",
        "--level_editor_prob=0",
        "--level_editor_method=random",
        "--num_edits=0",
        "--base_levels=batch",
        "--use_accel_paired=False",
        "--accel_paired_score_function=paired",
        "--use_lstm=False",
        "--use_behavioural_cloning=False",
        "--kl_loss_coef=0.0",
        "--kl_update_step=1",
        "--use_kl_only_agent=False",
        "--log_interval=10",
        "--screenshot_interval=200",
        "--log_grad_norm=True",
        "--normalize_returns=True",
        "--checkpoint_basis=student_grad_updates",
        "--archive_interval=5000",
        "--reward_shaping=True",
        "--use_categorical_adv=True",
        "--use_skip=False",
        "--choose_start_pos=False",
        "--sparse_rewards=False",
        "--handle_timelimits=True",
        "--adv_max_grad_norm=0.5",
        "--adv_ppo_epoch=8",
        "--adv_num_mini_batch=4",
        "--adv_normalize_returns=True",
        "--adv_use_popart=False",
        "--level_replay_strategy=positive_value_loss",
        f"--test_env_names={spec.test_env_name}",
        f"--log_dir={log_dir}",
        "--test_interval=10",
        "--test_num_episodes=10",
        "--test_num_processes=2",
        "--log_plr_buffer_stats=True",
        "--log_replay_complexity=True",
        "--checkpoint=True",
        "--log_action_complexity=False",
    ]
    return cmd


def _launch_process(cmd: List[str], gpu_id: Optional[int], log_dir: str) -> subprocess.Popen:
    os.makedirs(log_dir, exist_ok=True)
    env = os.environ.copy()
    if gpu_id is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = cmd + ["--no_cuda=True"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out_path = os.path.join(log_dir, "launcher.out")
    out_f = open(out_path, "a", buffering=1)
    return subprocess.Popen(cmd, stdout=out_f, stderr=out_f, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=str, default="/hard_data/user/majunjie/dcd/logs/ppo")
    parser.add_argument("--seeds", nargs="+", default=["88", "89", "90"])
    parser.add_argument("--gpus", nargs="*", default=None)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--min_free_mem_mb", type=int, default=20000)
    parser.add_argument("--max_gpu_util", type=int, default=10)
    parser.add_argument("--max_jobs_per_gpu", type=int, default=1)
    parser.add_argument("--mem_per_job_mb", type=int, default=20000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    allowed_gpus = _parse_int_list(args.gpus) if args.gpus is not None and len(args.gpus) > 0 else None

    base_xpid = "ued-BipedalWalker-MADS-v0-paired-lr0.0003-epoch5-mb32-v0.5-gc0.5-henv0.01-ha0.001-tl_0"
    experiments = [
        ExperimentSpec(tag="task1", env_name="BipedalWalker-MADS-Task1-v0", test_env_name="BipedalWalker-MADS-Task1-Eval-v0"),
        ExperimentSpec(tag="task0", env_name="BipedalWalker-MADS-Task0-v0", test_env_name="BipedalWalker-MADS-Task0-Eval-v0"),
        ExperimentSpec(tag="medium", env_name="BipedalWalker-MADS-Medium-v0", test_env_name="BipedalWalker-MADS-Medium-Eval-v0"),
        ExperimentSpec(tag="hard", env_name="BipedalWalker-MADS-Hard-v0", test_env_name="BipedalWalker-MADS-Hard-Eval-v0"),
    ]

    pending: List[Tuple[ExperimentSpec, int]] = [(spec, seed) for spec in experiments for seed in seeds]
    running: Dict[int, List[subprocess.Popen]] = {}

    if args.dry_run:
        for spec, seed in pending:
            log_dir = os.path.join(args.log_root, spec.tag, f"seed{seed}", "BipedalWalker")
            cmd = _build_train_cmd(base_xpid=base_xpid, spec=spec, seed=seed, log_dir=log_dir)
            print(" ".join(cmd))
        return

    while pending or running:
        _prune_finished(running)

        if pending:
            running_counts = {k: len(v) for k, v in running.items() if k >= 0}

            gpu_id = None
            gpu_rows: List[Tuple[int, int, int, int]] = []
            try:
                gpu_rows = _nvidia_smi_query()
                gpu_id = _pick_idle_gpu(
                    allowed_gpus=allowed_gpus,
                    min_free_mem_mb=args.min_free_mem_mb,
                    max_gpu_util=args.max_gpu_util,
                    running_counts=running_counts,
                    max_jobs_per_gpu=args.max_jobs_per_gpu,
                    mem_per_job_mb=args.mem_per_job_mb,
                )
            except Exception:
                gpu_id = None

            if gpu_id is None and gpu_rows:
                remaining = len(pending)
                active = sum(len(v) for v in running.values())
                allowed_str = ",".join(str(x) for x in allowed_gpus) if allowed_gpus is not None else "all"
                print(
                    f"No GPU available. pending={remaining} running={active} "
                    f"allowed={allowed_str} max_jobs_per_gpu={args.max_jobs_per_gpu} "
                    f"mem_per_job_mb={args.mem_per_job_mb} min_free_mem_mb={args.min_free_mem_mb} "
                    f"max_gpu_util={args.max_gpu_util}",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue

            spec, seed = pending.pop(0)
            log_dir = os.path.join(args.log_root, spec.tag, f"seed{seed}", "BipedalWalker")
            cmd = _build_train_cmd(base_xpid=base_xpid, spec=spec, seed=seed, log_dir=log_dir)

            remaining = len(pending)
            active = sum(len(v) for v in running.values())
            print(
                f"Launching: tag={spec.tag} seed={seed} gpu={gpu_id} "
                f"pending_after={remaining} running_before={active}",
                flush=True,
            )
            proc = _launch_process(cmd=cmd, gpu_id=gpu_id, log_dir=log_dir)
            key = gpu_id if gpu_id is not None else -1
            running.setdefault(key, []).append(proc)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
