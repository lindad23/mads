# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from envs.registration import register as gym_register

from .racetracks import RaceTrack
from .racetracks import formula1
from .car_racing_bezier import CarRacingBezier, TRACK_WIDTH

import ast
import gym
import numpy as np


F1_MADS_PARAM_NAMES = (
    'x_scale',
    'y_scale',
    'track_width_scale',
    'road_friction',
)
F1_MADS_DEFAULT_PARAMS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
F1_MADS_PARAM_RANGES = {
    1: (0.85, 1.15),
    2: (0.85, 1.15),
    3: (0.75, 1.25),
    4: (0.65, 1.15),
}


def set_global(name, value):
    globals()[name] = value


racetracks = dict([(name, cls) for name, cls in formula1.__dict__.items() if isinstance(cls, RaceTrack)])


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


def _create_constructor(track):
	def constructor(self, **kwargs):
		return CarRacingBezier.__init__(self, 
			track_name=track.name,
			**kwargs)
	return constructor


class CarRacingF1MADS(CarRacingBezier):
    def __init__(
        self,
        track_name=None,
        random_z_dim=4,
        policy_feature_dim=96,
        adversary_step_magnitude=0.1,
        **kwargs):
        super().__init__(track_name=track_name, **kwargs)

        self.passable = True
        self.random_z_dim = random_z_dim
        self.policy_feature_dim = policy_feature_dim
        self.current_policy_feature = np.zeros(self.policy_feature_dim, dtype=np.float32)
        self.adversary_step_magnitude = adversary_step_magnitude
        self.adversary_max_steps = len(F1_MADS_DEFAULT_PARAMS)
        self.adversary_step_count = 0
        self.default_params = F1_MADS_DEFAULT_PARAMS.copy()
        self.level_params_vec = self.default_params.copy()
        self._base_playfield = self.playfield

        self.adversary_action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0, high=1, shape=(len(self.level_params_vec),), dtype=np.float32),
            'time_step': gym.spaces.Box(
                low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8'),
            'random_z': gym.spaces.Box(
                low=0, high=1.0, shape=(self.random_z_dim,), dtype=np.float32),
            'policy_feature': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.policy_feature_dim,), dtype=np.float32),
        })

        self._apply_level_params(self.level_params_vec)

    @property
    def processed_action_dim(self):
        return 1

    @property
    def level(self):
        return str(tuple(float(v) for v in self.level_params_vec))

    def _coerce_level_params(self, level):
        if isinstance(level, str):
            level = level.strip()
            if level.startswith('(') or level.startswith('['):
                level = ast.literal_eval(level)
            else:
                level = np.fromstring(level, sep=',')

        if isinstance(level, dict):
            level = [level[name] for name in F1_MADS_PARAM_NAMES]

        level_params = np.asarray(level, dtype=np.float32).reshape(-1)
        if level_params.shape[0] != self.adversary_max_steps:
            raise ValueError(
                f'Expected {self.adversary_max_steps} F1 level params, '
                f'got {level_params.shape[0]}.')

        clipped = []
        for idx, value in enumerate(level_params, start=1):
            p_min, p_max = F1_MADS_PARAM_RANGES[idx]
            clipped.append(float(np.clip(value, p_min, p_max)))
        return np.array(clipped, dtype=np.float32)

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def _normalize_params(self):
        norm_params = []
        for idx, value in enumerate(self.level_params_vec, start=1):
            p_min, p_max = F1_MADS_PARAM_RANGES[idx]
            norm_params.append((value - p_min) / (p_max - p_min))
        return np.array(norm_params, dtype=np.float32)

    def _apply_level_params(self, level_params_vec):
        x_scale, y_scale, width_scale, road_friction = level_params_vec
        self.track_x_scale = float(x_scale)
        self.track_y_scale = float(y_scale)
        self.track_width = float(TRACK_WIDTH * width_scale)
        self.road_friction = float(road_friction)
        self.playfield = float(self._base_playfield * max(self.track_x_scale, self.track_y_scale))

    def _adversary_obs(self):
        return {
            'image': self._normalize_params(),
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z(),
            'policy_feature': self.current_policy_feature,
        }

    def reset(self, policy_feature=None):
        self.adversary_step_count = 0
        self.level_params_vec = self.default_params.copy()
        self._apply_level_params(self.level_params_vec)
        if policy_feature is not None:
            self.current_policy_feature = np.asarray(policy_feature, dtype=np.float32)
        return self._adversary_obs()

    def reset_agent(self):
        return super().reset()

    def reset_to_level(self, level):
        self.adversary_step_count = self.adversary_max_steps
        self.level_params_vec = self._coerce_level_params(level)
        self._apply_level_params(self.level_params_vec)
        return self.reset_agent()

    def reset_random(self):
        self.adversary_step_count = self.adversary_max_steps
        self.level_params_vec = np.array([
            np.random.uniform(*F1_MADS_PARAM_RANGES[idx])
            for idx in range(1, self.adversary_max_steps + 1)
        ], dtype=np.float32)
        self._apply_level_params(self.level_params_vec)
        return self.reset_agent()

    def reset_alp_gmm(self, level):
        level = np.asarray(level, dtype=np.float32).reshape(-1)
        if level.shape[0] != self.adversary_max_steps:
            raise ValueError(
                f'Expected {self.adversary_max_steps} normalized F1 actions, '
                f'got {level.shape[0]}.')

        self.adversary_step_count = self.adversary_max_steps
        params = []
        for idx, action in enumerate(level, start=1):
            p_min, p_max = F1_MADS_PARAM_RANGES[idx]
            normalized = np.clip((float(action) - 0.0) / 2.0, 0.0, 1.0)
            params.append(normalized * (p_max - p_min) + p_min)

        self.level_params_vec = np.array(params, dtype=np.float32)
        self._apply_level_params(self.level_params_vec)
        return self._adversary_obs()

    def step_adversary(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)[0]
        action = float(np.clip(action, -1.0, 1.0))

        param_idx = self.adversary_step_count + 1
        p_min, p_max = F1_MADS_PARAM_RANGES[param_idx]
        p_range = p_max - p_min
        default_val = self.default_params[self.adversary_step_count]
        default_norm = (default_val - p_min) / p_range
        new_norm = np.clip(default_norm + action * self.adversary_step_magnitude, 0.0, 1.0)
        self.level_params_vec[self.adversary_step_count] = float(new_norm * p_range + p_min)

        self.adversary_step_count += 1
        done = self.adversary_step_count >= self.adversary_max_steps
        if done:
            self._apply_level_params(self.level_params_vec)

        return self._adversary_obs(), 0.0, done, {}


def _create_mads_constructor(track):
    def constructor(self, **kwargs):
        return CarRacingF1MADS.__init__(
            self,
            track_name=track.name,
            **kwargs)
    return constructor


for name, track in racetracks.items():
	class_name = f"CarRacingF1-{track.name}"
	env = type(class_name, (CarRacingBezier, ), {
	    "__init__": _create_constructor(track),
	})
	set_global(class_name, env)
	gym_register(
		id=f'CarRacingF1-{track.name}-v0', 
		entry_point=module_path + f':{class_name}',
	    max_episode_steps=track.max_episode_steps,
	    reward_threshold=900)

	mads_class_name = f"CarRacingF1-MADS-{track.name}"
	mads_env = type(mads_class_name, (CarRacingF1MADS, ), {
	    "__init__": _create_mads_constructor(track),
	})
	set_global(mads_class_name, mads_env)
	gym_register(
		id=f'CarRacingF1-MADS-{track.name}-v0',
		entry_point=module_path + f':{mads_class_name}',
	    max_episode_steps=track.max_episode_steps,
	    reward_threshold=900)

	mads_eval_class_name = f"CarRacingF1-MADS-{track.name}-Eval"
	mads_eval_env = type(mads_eval_class_name, (CarRacingF1MADS, ), {
	    "__init__": _create_mads_constructor(track),
	})
	set_global(mads_eval_class_name, mads_eval_env)
	gym_register(
		id=f'CarRacingF1-MADS-{track.name}-Eval-v0',
		entry_point=module_path + f':{mads_eval_class_name}',
	    max_episode_steps=track.max_episode_steps,
	    reward_threshold=900)
