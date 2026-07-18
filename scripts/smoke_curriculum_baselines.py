"""Smoke-test isolated curriculum baseline teachers.

This script does not launch RL training. It verifies that the reproduced
teacher interfaces can sample tasks and consume rollout/episode updates.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from curriculum_baselines import CPDRLTeacher, ProCuRLTargetTeacher, TaskSpace


def main():
    task_space = TaskSpace.from_bounds(
        [
            (0.0, 10.0),
            (0.0, 10.0),
            (0.0, 10.0),
            (0.0, 5.0),
            (0.0, 5.0),
        ]
    )

    procurl = ProCuRLTargetTeacher(
        task_space=task_space,
        seed=1,
        reward_bounds=(-200.0, 350.0),
        retrain_interval_episodes=4,
        buffer_size=64,
        num_target_samples=16,
    )
    for i in range(12):
        task = procurl.sample()
        procurl.update_episode(task, episode_return=float(np.sin(i) * 50.0))
    print("procurl_sample", np.round(procurl.sample(), 4).tolist())

    cp_drl = CPDRLTeacher(
        task_space=task_space,
        state_dim=24,
        action_dim=4,
        seed=2,
        reward_bounds=(-200.0, 350.0),
        retrain_interval_episodes=4,
        buffer_size=64,
        num_target_samples=16,
        ensemble_size=2,
        model_batch_size=2,
        transition_scale=1.0,
    )
    for i in range(8):
        task = cp_drl.sample()
        for _ in range(5):
            state = np.random.randn(24).astype(np.float32)
            action = np.random.uniform(-1, 1, size=4).astype(np.float32)
            next_state = state + 0.01 * np.random.randn(24).astype(np.float32)
            cp_drl.update_step(state, action, 0.1, next_state, False)
        cp_drl.update_episode(task, episode_return=float(i))
    print("cp_drl_sample", np.round(cp_drl.sample(), 4).tolist())
    print("cp_drl_last_novelty", round(cp_drl.last_novelty, 6))


if __name__ == "__main__":
    main()
