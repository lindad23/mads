"""Teacher implementations for isolated curriculum baselines.

The classes here are intentionally framework-light. They expose a common
interface that can be driven by our PPO/MADS runners:

    task = teacher.sample()
    teacher.update_step(state, action, reward, next_state, done)
    teacher.update_episode(task, episode_return, success=False)

External source references:
- ProCuRL-Target: external_baselines/procurl_target/teachers_tma/proxcorl.py
- CP-DRL: external_baselines/cp_drl/bipedalwalker/TeachMyAgent/teachers/algos/cp_drl.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ArrayLike = Sequence[float]


@dataclass(frozen=True)
class TaskSpace:
    """Continuous task/context bounds used by curriculum teachers."""

    low: np.ndarray
    high: np.ndarray

    @classmethod
    def from_bounds(cls, bounds: Iterable[Tuple[float, float]]) -> "TaskSpace":
        low, high = zip(*bounds)
        return cls(np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))

    @property
    def dim(self) -> int:
        return int(self.low.shape[0])

    def sample_uniform(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n, self.dim)).astype(np.float32)

    def clip(self, tasks: np.ndarray) -> np.ndarray:
        return np.clip(tasks, self.low, self.high).astype(np.float32)

    def normalize(self, tasks: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.high - self.low, 1e-8)
        return (tasks - self.low) / denom


class ReturnRegressor(nn.Module):
    """Small MLP used to estimate returns over task parameters."""

    def __init__(self, task_dim: int, hidden_sizes: Sequence[int] = (256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = task_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        return self.net(task).squeeze(-1)


class ProCuRLTargetTeacher:
    """Generalized ProCuRL-Target task sampler.

    The IJCAI 2024 code is written for 2-D task spaces. This implementation keeps
    the same sampling idea while supporting arbitrary continuous task vectors:
    prefer tasks with predicted performance near the zone of proximal development
    and high similarity to samples from the target task distribution.
    """

    def __init__(
        self,
        task_space: TaskSpace,
        seed: int = 0,
        reward_bounds: Tuple[float, float] = (-200.0, 350.0),
        beta: float = 110.0,
        buffer_size: int = 1000,
        target_samples: Optional[np.ndarray] = None,
        num_target_samples: int = 250,
        retrain_interval_episodes: int = 50,
        device: str = "cpu",
    ):
        self.task_space = task_space
        self.rng = np.random.default_rng(seed)
        self.reward_min, self.reward_max = reward_bounds
        self.beta = beta
        self.buffer_size = buffer_size
        self.retrain_interval_episodes = retrain_interval_episodes
        self.device = torch.device(device)

        self.task_buffer = self.task_space.sample_uniform(self.rng, buffer_size)
        self.target_samples = (
            self.task_space.sample_uniform(self.rng, num_target_samples)
            if target_samples is None
            else self.task_space.clip(np.atleast_2d(np.asarray(target_samples, dtype=np.float32)))
        )
        self.history_tasks = []
        self.history_returns = []
        self.episode_counter = 0
        self.regressor = ReturnRegressor(task_space.dim).to(self.device)
        self.has_model = False

    def sample(self) -> np.ndarray:
        weights = self._sampling_weights()
        idx = self.rng.choice(len(self.task_buffer), p=weights)
        return self.task_buffer[idx].copy()

    def sample_batch(self, n: int) -> np.ndarray:
        return np.stack([self.sample() for _ in range(n)], axis=0)

    def update_step(self, *args, **kwargs) -> None:
        return None

    def update_episode(self, task: ArrayLike, episode_return: float, success: bool = False) -> None:
        self.history_tasks.append(np.asarray(task, dtype=np.float32))
        self.history_returns.append(float(episode_return))
        self.episode_counter += 1
        if self.episode_counter % self.retrain_interval_episodes == 0:
            self.fit_return_model()

    def fit_return_model(self) -> None:
        if len(self.history_tasks) < 8:
            return
        x = torch.as_tensor(np.asarray(self.history_tasks), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.asarray(self.history_returns), dtype=torch.float32, device=self.device)

        model = ReturnRegressor(self.task_space.dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        batch_size = min(64, len(x))
        for _ in range(80):
            order = torch.randperm(len(x), device=self.device)
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                pred = model(x[idx])
                loss = F.mse_loss(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.regressor = model
        self.has_model = True
        self.task_buffer = self.task_space.sample_uniform(self.rng, self.buffer_size)

    def _predict_normalized_return(self, tasks: np.ndarray) -> np.ndarray:
        if not self.has_model:
            return np.zeros(len(tasks), dtype=np.float32)
        with torch.no_grad():
            task_tensor = torch.as_tensor(tasks, dtype=torch.float32, device=self.device)
            pred = self.regressor(task_tensor).cpu().numpy()
        denom = max(self.reward_max - self.reward_min, 1e-8)
        return np.clip((pred - self.reward_min) / denom, 0.0, 1.0)

    def _target_similarity(self, tasks: np.ndarray) -> np.ndarray:
        norm_tasks = self.task_space.normalize(tasks)
        norm_targets = self.task_space.normalize(self.target_samples)
        dists = np.linalg.norm(norm_targets[:, None, :] - norm_tasks[None, :, :], axis=-1)
        return np.exp(-dists).max(axis=0)

    def _sampling_weights(self) -> np.ndarray:
        p = self._predict_normalized_return(self.task_buffer)
        zpd = p * (1.0 - p)
        target_sim = self._target_similarity(self.task_buffer)
        logits = self.beta * zpd * target_sim
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            return np.ones(len(self.task_buffer), dtype=np.float32) / len(self.task_buffer)
        return weights / weights.sum()


class _TransitionPrediction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(torch.cat([state, action], dim=-1))))


class _RewardPrediction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(torch.cat([state, action], dim=-1)))).squeeze(-1)


class _BetaVAE(nn.Module):
    def __init__(self, dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, dim)

    def forward(self, x: torch.Tensor):
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar


class CPDRLTeacher(ProCuRLTargetTeacher):
    """CP-DRL style teacher with causal novelty bonus.

    This reuses the ProCuRL-style task buffer for sampling, but augments episodic
    returns with ensemble disagreement terms from CP-DRL. It exposes the four
    scale knobs from the official PointMass runner (`t_scale`, `r_scale`,
    `s_scale`, `a_scale`) so experiments can match the paper settings.
    """

    def __init__(
        self,
        task_space: TaskSpace,
        state_dim: int,
        action_dim: int,
        seed: int = 0,
        reward_bounds: Tuple[float, float] = (-200.0, 350.0),
        beta: float = 50.0,
        transition_scale: float = 10.0,
        reward_scale: float = 0.0,
        state_scale: float = 0.0,
        action_scale: float = 0.0,
        aligned: bool = False,
        ensemble_size: int = 5,
        model_batch_size: int = 512,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            task_space=task_space,
            seed=seed,
            reward_bounds=reward_bounds,
            beta=beta,
            device=device,
            **kwargs,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transition_scale = transition_scale
        self.reward_scale = reward_scale
        self.state_scale = state_scale
        self.action_scale = action_scale
        self.aligned = aligned
        self.model_batch_size = max(int(model_batch_size), 1)

        self.transition_models = nn.ModuleList(
            [_TransitionPrediction(state_dim, action_dim, 32).to(self.device) for _ in range(ensemble_size)]
        )
        self.reward_models = nn.ModuleList(
            [_RewardPrediction(state_dim, action_dim, 256).to(self.device) for _ in range(ensemble_size)]
        )
        self.state_vaes = nn.ModuleList(
            [_BetaVAE(state_dim, 32).to(self.device) for _ in range(ensemble_size)]
        )
        self.action_vaes = nn.ModuleList(
            [_BetaVAE(action_dim, 16).to(self.device) for _ in range(ensemble_size)]
        )
        self.transition_optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in self.transition_models]
        self.reward_optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in self.reward_models]
        self.state_optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in self.state_vaes]
        self.action_optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in self.action_vaes]
        self.step_buffer = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}
        self.last_novelty = 0.0

    def update_step(self, state, action, reward, next_state, done) -> None:
        self.step_buffer["states"].append(np.asarray(state, dtype=np.float32))
        self.step_buffer["actions"].append(np.asarray(action, dtype=np.float32))
        self.step_buffer["rewards"].append(float(reward))
        self.step_buffer["next_states"].append(np.asarray(next_state, dtype=np.float32))
        self.step_buffer["dones"].append(bool(done))

    def update_episode(self, task: ArrayLike, episode_return: float, success: bool = False) -> None:
        novelty = self._consume_step_buffer()
        self.last_novelty = novelty
        super().update_episode(task, float(episode_return) + novelty, success=success)

    def _consume_step_buffer(self) -> float:
        if len(self.step_buffer["states"]) < 2:
            self._reset_step_buffer()
            return 0.0

        novelty = 0.0
        if self.state_scale != 0.0:
            novelty += self.state_scale * self._vae_disagreement(
                self.state_vaes, self.state_optimizers, self.step_buffer["states"]
            )
        if self.action_scale != 0.0:
            novelty += self.action_scale * self._vae_disagreement(
                self.action_vaes, self.action_optimizers, self.step_buffer["actions"]
            )
        if self.transition_scale != 0.0:
            novelty += self.transition_scale * self._prediction_disagreement(
                self.transition_models,
                self.transition_optimizers,
                self.step_buffer["states"][:-1],
                self.step_buffer["actions"][:-1],
                self.step_buffer["next_states"][:-1],
            )
        if self.reward_scale != 0.0:
            novelty += self.reward_scale * self._prediction_disagreement(
                self.reward_models,
                self.reward_optimizers,
                self.step_buffer["states"],
                self.step_buffer["actions"],
                self.step_buffer["rewards"],
            )
        self._reset_step_buffer()
        return float(novelty)

    def _reset_step_buffer(self) -> None:
        for values in self.step_buffer.values():
            values.clear()

    def _scale_disagreement(self, disagreement: torch.Tensor) -> torch.Tensor:
        if self.aligned:
            return 1.0 / torch.clamp(disagreement, min=1e-6)
        return disagreement

    def _batch_tensor(self, values, start: int, end: int) -> torch.Tensor:
        batch = values[start:end]
        return torch.as_tensor(np.asarray(batch), dtype=torch.float32, device=self.device)

    def _vae_disagreement(self, models: nn.ModuleList, optimizers: Sequence[torch.optim.Optimizer], x) -> float:
        total = 0.0
        total_count = 0
        for start in range(0, len(x), self.model_batch_size):
            end = min(start + self.model_batch_size, len(x))
            x_batch = self._batch_tensor(x, start, end)
            preds = []
            for model, optimizer in zip(models, optimizers):
                pred, mu, logvar = model(x_batch)
                preds.append(pred.detach())
                recon_loss = F.mse_loss(pred, x_batch)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                optimizer.zero_grad()
                (recon_loss + kl).backward()
                optimizer.step()
            disagreement = torch.stack(preds).std(dim=0).mean()
            count = end - start
            total += float(self._scale_disagreement(disagreement).detach().cpu()) * count
            total_count += count
        return total / max(total_count, 1)

    def _prediction_disagreement(
        self,
        models: nn.ModuleList,
        optimizers: Sequence[torch.optim.Optimizer],
        state,
        action,
        target,
    ) -> float:
        total = 0.0
        total_count = 0
        for start in range(0, len(state), self.model_batch_size):
            end = min(start + self.model_batch_size, len(state))
            state_batch = self._batch_tensor(state, start, end)
            action_batch = self._batch_tensor(action, start, end)
            target_batch = self._batch_tensor(target, start, end)
            preds = []
            for model, optimizer in zip(models, optimizers):
                pred = model(state_batch, action_batch)
                preds.append(pred.detach())
                loss = F.mse_loss(pred, target_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            disagreement = torch.stack(preds).std(dim=0).mean()
            count = end - start
            total += float(self._scale_disagreement(disagreement).detach().cpu()) * count
            total_count += count
        return total / max(total_count, 1)
