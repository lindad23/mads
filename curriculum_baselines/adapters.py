"""Adapters that connect isolated curriculum teachers to legacy runners."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .teachers import CPDRLTeacher, ProCuRLTargetTeacher, TaskSpace


def _task_space_from_minmax(mins: Sequence[float], maxs: Sequence[float]) -> TaskSpace:
    bounds = list(zip(mins, maxs))
    return TaskSpace.from_bounds(bounds)


class CurriculumTeacherAdapter:
    """Small legacy-compatible interface used by TeacherController."""

    def sample_task(self) -> np.ndarray:
        task = self.teacher.sample()
        return np.asarray(task, dtype=np.float32)

    def update(self, task: Iterable[float], reward: float) -> None:
        self.teacher.update_episode(task, reward)

    def update_step(self, state, action, reward, next_state, done) -> None:
        self.teacher.update_step(state, action, reward, next_state, done)

    def dump(self, dump_dict):
        dump_dict["baseline_teacher"] = self.teacher.__class__.__name__
        if hasattr(self.teacher, "target_samples"):
            target_samples = np.asarray(self.teacher.target_samples, dtype=np.float32)
            dump_dict["target_samples_count"] = int(target_samples.shape[0])
            dump_dict["target_samples_mean"] = target_samples.mean(axis=0)
        if hasattr(self.teacher, "last_novelty"):
            dump_dict["last_novelty"] = self.teacher.last_novelty
        return dump_dict


class ProCuRLTargetAdapter(CurriculumTeacherAdapter):
    def __init__(
        self,
        mins: Sequence[float],
        maxs: Sequence[float],
        reward_bounds: Optional[Tuple[float, float]] = None,
        seed: int = 0,
        params=None,
    ):
        params = params or {}
        self.teacher = ProCuRLTargetTeacher(
            task_space=_task_space_from_minmax(mins, maxs),
            seed=seed,
            reward_bounds=reward_bounds or (-200.0, 350.0),
            beta=params.get("beta", 110.0),
            buffer_size=params.get("buffer_size", 1000),
            target_samples=params.get("target_samples"),
            num_target_samples=params.get("num_target_samples", 250),
            retrain_interval_episodes=params.get("retrain_interval_episodes", 50),
            device=params.get("device", "cpu"),
        )


class CPDRLAdapter(CurriculumTeacherAdapter):
    def __init__(
        self,
        mins: Sequence[float],
        maxs: Sequence[float],
        reward_bounds: Optional[Tuple[float, float]] = None,
        seed: int = 0,
        params=None,
    ):
        params = params or {}
        self.teacher = CPDRLTeacher(
            task_space=_task_space_from_minmax(mins, maxs),
            state_dim=params["state_dim"],
            action_dim=params["action_dim"],
            seed=seed,
            reward_bounds=reward_bounds or (-200.0, 350.0),
            beta=params.get("beta", 50.0),
            transition_scale=params.get("transition_scale", 10.0),
            reward_scale=params.get("reward_scale", 0.0),
            state_scale=params.get("state_scale", 0.0),
            action_scale=params.get("action_scale", 0.0),
            aligned=params.get("aligned", False),
            ensemble_size=params.get("ensemble_size", 5),
            model_batch_size=params.get("model_batch_size", 512),
            buffer_size=params.get("buffer_size", 1000),
            target_samples=params.get("target_samples"),
            num_target_samples=params.get("num_target_samples", 250),
            retrain_interval_episodes=params.get("retrain_interval_episodes", 50),
            device=params.get("device", "cpu"),
        )
