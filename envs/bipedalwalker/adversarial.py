# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import gym
import time
import numpy as np
import torch

from gym.envs.box2d import BipedalWalker, BipedalWalkerHardcore

from .walker_env import EnvConfig, BipedalWalkerCustom
from envs.registration import register as gym_register

"""
actions

1. ground_roughness
2,3. pit_gap
stump_width (fixed)
4,5. stump_height
stump_float (fixed)
6,7 stair_height
stair_width (fixed)
8 stair_steps

"""
PARAM_RANGES_DEBUG = {
    1: [0,0.01], # ground roughness
    2: [0,0], # pit gap 1
    3: [0.01,0.01], # pit gap 2
    4: [0,0], # stump height 1
    5: [0.01,0.01], # stump height 2
    6: [0,0], # stair height 1
    7: [0.01,0.01], # stair height 2
    8: [1,1], # stair steps
}

PARAM_RANGES_EASY = {
    1: [0,0.6], # ground roughness
    2: [0,0], # pit gap 1
    3: [0.8,0.8], # pit gap 2
    4: [0,0], # stump height 1
    5: [0.4,0.4], # stump height 2
    6: [0,0], # stair height 1
    7: [0.4,0.4], # stair height 2
    8: [1,1], # stair steps
}

PARAM_RANGES_FULL = {
    1: [0,10], # ground roughness
    2: [0,10], # pit gap 1
    3: [0,10], # pit gap 2
    4: [0,5], # stump height 1
    5: [0,5], # stump height 2
    6: [0,5], # stair height 1
    7: [0,5], # stair height 2
    8: [1,9], # stair steps
}

PARAM_MUTATIONS = {
    1: [0,0.6], # ground roughness
    2: [0.4], # pit gap 1
    3: [0.4], # pit gap 2
    4: [0.2], # stump height 1
    5: [0.2], # stump height 2
    6: [0.2], # stair height 1
    7: [0.2], # stair height 2
    8: [1], # stair steps
}

# DEFAULT_LEVEL_PARAMS_VEC = [0,0,10,0,5,0,5,9]
level_0_task = [1.0, 0, 0, 0, 0, 0, 0, 0]           # 0
level_1_task = [1.0, 0.8, 0.8, 0, 1.0, 0, 0, 0]     # 1
level_2_task = [4.0, 4.0, 6.0, 0.5, 2.0, 0.5, 2.0, 4]   # Medium
level_3_task = [8.0, 6.0, 10.0, 1.0, 4.0, 2.0, 5.0, 8]  # Hard
DEFAULT_LEVEL_PARAMS_VEC = level_1_task

POET_ROSE_RAW_PARAMS = {
    '1a': [5.6, 2.4, 2.82, 6.4, 4.48],
    '1b': [5.44, 1.8, 2.82, 6.72, 4.48],
    '2a': [7.2, 1.98, 2.82, 7.2, 5.6],
    '2b': [5.76, 2.16, 2.76, 7.2, 1.6],
    '3a': [5.28, 1.98, 2.76, 7.2, 4.8],
    '3b': [4.8, 2.4, 2.76, 4.48, 4.8],
}

POET_ROSE_LEVEL_PARAMS = {
    rose_id: [raw[0], raw[4], raw[3], raw[1], raw[2]]
    for rose_id, raw in POET_ROSE_RAW_PARAMS.items()
}

POET_PARAM_RANGES_FULL = {k: PARAM_RANGES_FULL[k] for k in range(1, 6)}

STUMP_WIDTH_RANGE = [1, 2]
STUMP_FLOAT_RANGE = [0, 1]
STAIR_WIDTH_RANGE = [4, 5]


def rand_int_seed():
    return int.from_bytes(os.urandom(4), byteorder="little")


class BipedalWalkerAdversarialEnv(BipedalWalkerCustom):
    def __init__(self, mode='full', poet=False, random_z_dim=10, seed=0, fixed_level_params_vec=None, fixed_level_seed=None, lock_params_on_reset=True):
        self.mode = mode
        self.level_seed = seed
        self.poet = poet # POET didn't use the stairs, not clear why
        self.fixed_level_seed = fixed_level_seed
        self.fixed_level_params_vec = fixed_level_params_vec
        self.lock_params_on_reset = lock_params_on_reset

        default_config = EnvConfig(
            name='default_conf',
            ground_roughness=0,
            pit_gap=[0,10],
            stump_width=[4,5],
            stump_height=[0,5],
            stump_float=[0,1],
            stair_height=[0,5],
            stair_width=[4,5],
            stair_steps=[1])

        super().__init__(default_config, seed=seed)

        if self.poet:
            self.adversary_max_steps = 5
        else:
            self.adversary_max_steps = 8
        self.random_z_dim = random_z_dim
        self.passable = True

        self.stump_width = STUMP_WIDTH_RANGE
        self.stump_float = STUMP_FLOAT_RANGE
        self.stair_width = STAIR_WIDTH_RANGE

        # Level vec is the *tunable* UED params
        self.level_params_vec = self.fixed_level_params_vec if self.fixed_level_params_vec is not None else DEFAULT_LEVEL_PARAMS_VEC
        if self.poet:
            self.level_params_vec = self.level_params_vec[:5]
        self._update_params(self.level_params_vec)
        self.set_env_config(self.get_config())

        if poet:
            self.mutations = {k:v for k,v in list(PARAM_MUTATIONS.items())[:5]}
        else:
            self.mutations = PARAM_MUTATIONS

        n_u_chars = max(12, len(str(rand_int_seed())))
        self.encoding_u_chars = np.dtype(('U', n_u_chars))

        # Create spaces for adversary agent's specs.
        self.adversary_action_dim = 1
        self.adversary_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.adversary_ts_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=self.adversary_max_steps, 
                shape=(1,), 
                dtype='uint8')
        self.adversary_randomz_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=1.0, 
                shape=(random_z_dim,), 
                dtype=np.float32)
        self.adversary_image_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=10.0, 
                shape=(len(self.level_params_vec),), 
                dtype=np.float32)
        self.adversary_observation_space = \
            gym.spaces.Dict({
                'image': self.adversary_image_obs_space, 
                'time_step': self.adversary_ts_obs_space, 
                'random_z': self.adversary_randomz_obs_space})

    def reset(self):
        self.step_count = 0
        self.adversary_step_count = 0

        # Reset to default parameters
        self.level_params_vec = self.fixed_level_params_vec if self.fixed_level_params_vec is not None else DEFAULT_LEVEL_PARAMS_VEC
        if self.poet:
            self.level_params_vec = self.level_params_vec[:5]

        self._update_params(self.level_params_vec)
        self.set_env_config(self.get_config())

        self.level_seed = self.fixed_level_seed if self.fixed_level_seed is not None else rand_int_seed()

        obs = {
            'image': self.get_obs(),
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs

    def get_obs(self):
        ## vector of *tunable* environment params
        obs = []
        obs += [self.ground_roughness]
        obs += self.pit_gap
        obs += self.stump_height
        if not self.poet:
            obs += self.stair_height
            obs += self.stair_steps

        return np.array(obs)

    def reset_agent(self):
        if self.lock_params_on_reset and self.fixed_level_params_vec is not None:
            self.level_params_vec = self.fixed_level_params_vec[:5] if self.poet else self.fixed_level_params_vec
            self._update_params(self.level_params_vec)
        self.set_env_config(self.get_config())
        if self.fixed_level_seed is not None:
            self.level_seed = self.fixed_level_seed
        super().seed(self.level_seed)
        obs = super()._reset_env()

        return obs

    def _update_params(self, level_params_vec):
        self.ground_roughness = level_params_vec[0]
        self.pit_gap = [level_params_vec[1],level_params_vec[2]]
        self.pit_gap.sort()
        self.stump_height = [level_params_vec[3],level_params_vec[4]]
        self.stump_height.sort()
        if self.poet:
            self.stair_height = []
            self.stair_steps = []
        else:
            self.stair_height = [level_params_vec[5],level_params_vec[6]]
            self.stair_height.sort()
            self.stair_steps = [int(round(level_params_vec[7]))]

    def get_complexity_info(self):
        complexity_info = {
            'ground_roughness': self.ground_roughness,
            'pit_gap_low': self.pit_gap[0],
            'pit_gap_high': self.pit_gap[1],
            'stump_height_low': self.stump_height[0],
            'stump_height_high': self.stump_height[1]
        }

        if not self.poet:
            complexity_info['stair_height_low'] = self.stair_height[0]
            complexity_info['stair_height_high'] = self.stair_height[1]
            complexity_info['stair_steps'] = self.stair_steps[0]

        return complexity_info

    def get_config(self):
        """
        Gets the config to use to create the level.
        If the range is zer or below a min threshold, we put blank entries.
        """
        if self.stump_height[1] < 0.2:
            stump_height = []
            stump_width = []
            stump_float = []
        else:
            stump_height = self.stump_height
            stump_width = self.stump_width
            stump_float = self.stump_float

        if self.pit_gap[1] < 0.8:
            pit_gap = []
        else:
            pit_gap = self.pit_gap

        if self.poet:
            stair_height = []
            stair_width = []
            stair_steps = []
        elif self.stair_height[1] < 0.2:
            stair_height = []
            stair_width = []
            stair_steps = []
        else:
            stair_height = self.stair_height
            stair_width = self.stair_width
            stair_steps = self.stair_steps

        # get the current config
        config = EnvConfig(
            name='config',
            ground_roughness=self.ground_roughness,
            pit_gap=pit_gap,
            stump_width=stump_width,
            stump_height=stump_height,
            stump_float=stump_float,
            stair_height=stair_height,
            stair_width=stair_width,
            stair_steps=stair_steps)

        return config

    def _reset_env_config(self):
        """
        Resets the environment based on current level encoding.
        """
        config = self.get_config()
        try:
            super().re_init(config, self.level_seed)
        except AssertionError:
            super().re_init(config, self.level_seed+1)

    def reset_to_level(self, level, editing=False):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == len(self.level_params_vec) + 1, \
            f'Level input is the wrong length.'

        self.level_params_vec = encoding[:-1]
        self._update_params(self.level_params_vec)
        self._reset_env_config()

        self.level_seed = int(level[-1])

        return self.reset_agent()

    @property
    def param_ranges(self):
        if self.mode == 'easy':
            param_ranges = PARAM_RANGES_EASY
        elif self.mode == 'full':
            param_ranges = PARAM_RANGES_FULL
        elif self.mode == 'debug':
            param_ranges = PARAM_RANGES_DEBUG
        else:
            raise ValueError("Mode must be 'easy' or 'full'")

        return param_ranges

    @property
    def encoding(self):
        enc = list(self.level_params_vec) + [self.level_seed]
        enc = [str(x) for x in enc]
        return np.array(enc, dtype=self.encoding_u_chars)

    @property
    def level(self):
        return self.encoding

    def reset_random(self):
        """
        Must reset randomly as step_adversary would otherwise do
        """
        if self.lock_params_on_reset and self.fixed_level_params_vec is not None:
            self.level_params_vec = self.fixed_level_params_vec[:5] if self.poet else self.fixed_level_params_vec
            self._update_params(self.level_params_vec)
            self.level_seed = self.fixed_level_seed if self.fixed_level_seed is not None else rand_int_seed()
            self._reset_env_config()
            return self.reset_agent()

        # action will be between [-1,1]
        # this maps to a range, depending on the index
        param_ranges = self.param_ranges

        rand_norm_params = np.random.rand(len(param_ranges))
        self.level_params_vec = \
            [rand_norm_params[i]*(param_range[1]-param_range[0]) + param_range[0] 
                for i,param_range in enumerate(param_ranges.values())]
        self._update_params(self.level_params_vec)

        self.level_seed = rand_int_seed()

        self._reset_env_config()

        return self.reset_agent()

    def reset_alp_gmm(self, level):
        self.reset()

        level = list(level)
        param_ranges = self.param_ranges
        for idx, action in enumerate(level):
            val_range = param_ranges[idx + 1]

            action -= 1
            value = ((action + 1)/2) * (val_range[1]-val_range[0]) + val_range[0]

            # update the level vec
            self.level_params_vec[idx] = value

        self.level_seed = rand_int_seed()
        self._update_params(self.level_params_vec)
        self._reset_env_config()

        obs = {
            'image': self.level_params_vec,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs

    @property
    def processed_action_dim(self):
        return 1

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def mutate_level(self, num_edits=1):
        if num_edits > 0:
            # Perform mutations on current level vector
            param_ranges = self.param_ranges
            edit_actions = np.random.randint(1, len(self.mutations) + 1, num_edits)
            edit_dirs = np.random.randint(0, 3, num_edits) - 1

            # Update level_params_vec
            for a,d in zip(edit_actions, edit_dirs):
                mutation_range = self.mutations[a]
                if len(mutation_range) == 1:
                    mutation = d*mutation_range[0]
                elif len(mutation_range) == 2:
                    mutation = d*np.random.uniform(*mutation_range)

                self.level_params_vec[a-1] = \
                    np.clip(self.level_params_vec[a-1]+mutation,
                            *PARAM_RANGES_FULL[a])

            self.level_seed = rand_int_seed()
            self._update_params(self.level_params_vec)
            self._reset_env_config()

        return self.reset_agent()

    def step_adversary(self, action):
        # action will be between [-1,1]
        # this maps to a range, depending on the index
        param_ranges = self.param_ranges
        val_range = param_ranges[self.adversary_step_count+1]

        if torch.is_tensor(action):
            action = action.item()

        # get unnormalized value from the action
        value = ((action + 1)/2) * (val_range[1]-val_range[0]) + val_range[0]

        # update the level vec
        self.level_params_vec[self.adversary_step_count] = value
        
        self.adversary_step_count += 1

        if self.adversary_step_count >= self.adversary_max_steps:
            self.level_seed = rand_int_seed()
            self._update_params(self.level_params_vec)
            self._reset_env_config()
            done=True
        else:
            done=False

        obs = {
            'image': self.level_params_vec,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs, 0, done, {}

class BipedalWalkerMADS(BipedalWalkerAdversarialEnv):
    def __init__(self, seed=0, policy_feature_dim=128, fixed_level_params_vec=None, fixed_level_seed=None, lock_params_on_reset=True,
                 mode='full', poet=False, mads_param_ranges=None, adversary_step_magnitude=0.1): # 默认维度设为你想要的
        # 调用父类初始化，保持 mode='full'
        super().__init__(mode=mode, poet=poet, seed=seed, fixed_level_params_vec=fixed_level_params_vec, fixed_level_seed=fixed_level_seed, lock_params_on_reset=lock_params_on_reset)
        
        self.policy_feature_dim = policy_feature_dim
        self.current_policy_feature = np.zeros(self.policy_feature_dim, dtype=np.float32)

        # 扩展观察空间：在父类基础上增加 policy_feature
        # 注意：我们需要先复制父类的 space，再添加，避免修改引用
        base_space = self.adversary_observation_space.spaces.copy()
        base_space['policy_feature'] = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.policy_feature_dim,), 
            dtype=np.float32
        )
        self.adversary_observation_space = gym.spaces.Dict(base_space)

        # MADS: 使用代码中已有的默认参数作为基准
        self.default_params = np.array(
            self.level_params_vec,
            dtype=np.float32,
        )

        self.level_params_vec = self.default_params.tolist()
        # 定义每一步的修改幅度 (Sensitivity/Magnitude)
        # 0.1 表示每次最多变化默认值±10%的范围
        self.adversary_step_magnitude = adversary_step_magnitude
        # self.adversary_step_magnitude = 0.15
        # 定义参数的物理边界 (用于归一化和截断)
        self.mads_param_ranges = mads_param_ranges if mads_param_ranges is not None else PARAM_RANGES_FULL

    def reset(self, policy_feature=None):
        # 1. 调用父类 reset (获取基础 obs)
        # 注意：父类 reset 不接受 policy_feature 参数，所以这里不能传给它
        obs = super().reset()
        self.level_params_vec = self.default_params.tolist()
        self._update_params(self.level_params_vec)
        self.set_env_config(self.get_config())
        obs['image'] = self.get_obs()
        
        # 2. 更新特征
        if policy_feature is not None:
            self.current_policy_feature = policy_feature
        
        # 3. 注入特征到 obs
        obs['policy_feature'] = self.current_policy_feature
        return obs

    def reset_agent(self):
        self.set_env_config(self.get_config())
        if self.fixed_level_seed is not None:
            self.level_seed = self.fixed_level_seed
        super().seed(self.level_seed)
        return super()._reset_env()

    def step_adversary(self, action):
        """
        Adversary 的单步决策函数。
        逻辑：
        1. 读取对应参数的默认值（Anchor）。
        2. 根据 Action ([-1, 1]) 和 Magnitude (例如 0.05) 计算偏离默认值的幅度。
        3. 归一化计算后还原为物理值。
        4. 放入 Observation 返回（Obs 中包含异步更新的 policy_feature）。
        """
        if torch.is_tensor(action):
            action = action.item()
        
        # 2. 确定当前正在设置哪个参数
        param_idx = self.adversary_step_count + 1
        
        # 获取该参数的物理边界 [Min, Max]
        p_min, p_max = self.mads_param_ranges[param_idx]
        p_range = p_max - p_min
        
        # 3. 获取该参数的默认值 (基准锚点)
        # self.default_params 应该在 __init__ 中初始化为 np.array(DEFAULT_LEVEL_PARAMS_VEC)
        default_physical_val = self.default_params[self.adversary_step_count]
        
        # 4. 将默认值归一化到 [0, 1] 空间
        # 这样神经网络只需要学习"相对于默认值改动多少百分比"
        if p_range > 1e-6:
            default_norm_val = (default_physical_val - p_min) / p_range
        else:
            default_norm_val = 0.0 # 范围为0，固定不可变
            
        # 5. 计算增量 (Delta)
        # action: [-1, 1]
        # self.adversary_step_magnitude: 超参数，默认 0.1 (10%)
        # 结果: 在默认值基础上左右浮动 10%
        delta = action * self.adversary_step_magnitude
        
        # 6. 应用增量并截断 (Clip)
        # 确保结果不会超出 [0, 1] 的物理定义边界
        new_norm_val = np.clip(default_norm_val + delta, 0.0, 1.0)
        
        # 7. 还原为物理值 (Denormalize)
        new_physical_val = float(new_norm_val * p_range + p_min)
        
        # 8. 更新环境参数向量
        self.level_params_vec[self.adversary_step_count] = new_physical_val
        
        # 9. 步数计数自增
        self.adversary_step_count += 1

        # 10. 判断是否结束 (所有参数都设置完毕)
        if self.adversary_step_count >= self.adversary_max_steps:
            # 应用生成的参数到物理引擎
            self.level_seed = rand_int_seed()
            self._update_params(self.level_params_vec)
            # self._update_params(self.default_params)
            self._reset_env_config()
            done = True
        else:
            done = False

        # 11. 构建 Observation
        # 重点：这里的 self.current_policy_feature 是通过 set_policy_feature 接口异步更新的
        obs = {
            'image': self.level_params_vec,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z(),
            'policy_feature': self.current_policy_feature 
        }

        return obs, 0, done, {}

class BipedalWalkerDev(BipedalWalker):
    def __init__(self, random_z_dim=5):
        super().__init__()
        self.adversary_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.adversary_max_steps = 5
        self.level_params_vec = [0]
        self.adversary_ts_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=self.adversary_max_steps, 
                shape=(1,), 
                dtype='uint8')
        self.adversary_randomz_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=1.0, 
                shape=(random_z_dim,), 
                dtype=np.float32)
        self.adversary_image_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=10.0, 
                shape=(len(self.level_params_vec),), 
                dtype=np.float32)
        self.adversary_observation_space = \
            gym.spaces.Dict({
                'image': self.adversary_image_obs_space, 
                'time_step': self.adversary_ts_obs_space, 
                'random_z': self.adversary_randomz_obs_space})

    def reset_random(self):
        seed = rand_int_seed()
        super().seed(seed)
        return super().reset()

    def reset_agent(self):
        return super().reset()

    def step_adversary(self):
        pass

    @property
    def processed_action_dim(self):
        return 1

    def get_complexity_info(self):

        complexity_info = {
            'ground_roughness': 0,
        }
        return complexity_info

class BipedalWalkerHC(BipedalWalkerHardcore):
    def __init__(self, random_z_dim=5, seed=0):
        super().__init__()
        self.adversary_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.adversary_max_steps = 5
        self.level_params_vec = [0]
        self.adversary_ts_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=self.adversary_max_steps, 
                shape=(1,), 
                dtype='uint8')
        self.adversary_randomz_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=1.0, 
                shape=(random_z_dim,), 
                dtype=np.float32)
        self.adversary_image_obs_space = \
            gym.spaces.Box(
                low=0, 
                high=10.0, 
                shape=(len(self.level_params_vec),), 
                dtype=np.float32)
        self.adversary_observation_space = \
            gym.spaces.Dict({
                'image': self.adversary_image_obs_space, 
                'time_step': self.adversary_ts_obs_space, 
                'random_z': self.adversary_randomz_obs_space})
        self.adversary_editor_action_space = gym.spaces.MultiDiscrete([3, 3])

    def reset_random(self):
        seed = rand_int_seed()
        super().seed(seed)
        return super().reset()

    def reset_agent(self):
        return super().reset()

    def step_adversary(self):
        pass

    @property
    def processed_action_dim(self):
        return 1

    def get_complexity_info(self):

        complexity_info = {
            'ground_roughness': 0,
        }
        return complexity_info


class BipedalWalkerFull(BipedalWalkerAdversarialEnv):
  def __init__(self, seed=0):
    super().__init__(mode='full', seed=seed)

class BipedalWalkerEasy(BipedalWalkerAdversarialEnv):
  def __init__(self, seed=0):
    super().__init__(mode='easy', seed=seed)

class BipedalWalkerDebug(BipedalWalkerDev):
  def __init__(self, seed=0):
    super().__init__()

class BipedalWalkerPOET(BipedalWalkerAdversarialEnv):
  def __init__(self, seed=0):
    super().__init__(mode='full', poet=True, seed=seed)

class BipedalWalkerEasyPOET(BipedalWalkerAdversarialEnv):
  def __init__(self, seed=0):
    super().__init__(mode='easy', poet=True, seed=seed)

class BipedalWalkerMADSPOETRose(BipedalWalkerMADS):
    def __init__(self, rose_id='1a', seed=0, policy_feature_dim=128, fixed_level_seed=None,
                 lock_params_on_reset=False, adversary_step_magnitude=0.1):
        if rose_id not in POET_ROSE_LEVEL_PARAMS:
            raise ValueError(f"Unknown POET Rose id: {rose_id}")
        self.rose_id = rose_id
        super().__init__(
            mode='full',
            poet=True,
            seed=seed,
            policy_feature_dim=policy_feature_dim,
            fixed_level_params_vec=list(POET_ROSE_LEVEL_PARAMS[rose_id]),
            fixed_level_seed=fixed_level_seed,
            lock_params_on_reset=lock_params_on_reset,
            mads_param_ranges=POET_PARAM_RANGES_FULL,
            adversary_step_magnitude=adversary_step_magnitude,
        )

class BipedalWalkerMADSEval(BipedalWalkerMADS):
    """
    专门用于 Evaluator 测试的包装类。
    继承自 BipedalWalkerMADS 以保持相同的默认参数 (Default Params)，
    但重写 reset() 使其返回 Agent 的观测 (obs) 而不是 Adversary 的观测 (dict)。
    """
    def reset(self):
        _ = super().reset()

        self._reset_env_config()

        return self.reset_agent()

class BipedalWalkerMADSPOETRoseEval(BipedalWalkerMADSPOETRose):
    def reset(self):
        _ = super().reset()
        self._reset_env_config()
        return self.reset_agent()

class BipedalWalkerAdversarialEval(BipedalWalkerFull):
    """
    用于 Baseline (Adversarial-v0) 的测试包装类。
    强制 reset() 返回 Agent 的观测 (24维向量)，而不是 Adversary 的字典。
    """
    def reset(self):
        # 1. 调用父类 reset 初始化环境参数 (随机生成难度)
        _ = super().reset()
        # 2. 强制刷新配置 (修复之前的隐形Bug)
        self._reset_env_config()
        # 3. 返回 Agent 视角的观测
        return self.reset_agent()

class BipedalWalkerAdversarialFixedEval(BipedalWalkerAdversarialEnv):
    def reset(self):
        _ = super().reset()
        self._reset_env_config()
        return self.reset_agent()


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

gym_register(id='BipedalWalker-Adversarial-v0',
             entry_point=module_path + ':BipedalWalkerFull',
             max_episode_steps=2000)

gym_register(id='BipedalWalker-Adversarial-Easy-v0',
             entry_point=module_path + ':BipedalWalkerEasy',
             max_episode_steps=2000)

gym_register(id='BipedalWalker-Vanilla-v0',
             entry_point=module_path + ':BipedalWalkerDebug',
             max_episode_steps=2000)

gym_register(id='BipedalWalker-HC-v0',
             entry_point=module_path + ':BipedalWalkerHC',
             max_episode_steps=2000)

gym_register(id='BipedalWalker-POET-v0',
             entry_point=module_path + ':BipedalWalkerPOET',
             max_episode_steps=2000)

gym_register(id='BipedalWalker-POET-Easy-v0',
             entry_point=module_path + ':BipedalWalkerEasyPOET',
             max_episode_steps=2000)

gym_register(id='BipedalWalker-MADS-v0', # [新增 ID]
             entry_point=module_path + ':BipedalWalkerMADS',
             max_episode_steps=2000)

gym_register(
    id='BipedalWalker-MADS-UED-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'lock_params_on_reset': False},
)

gym_register(
    id='BipedalWalker-MADS-Task0-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_0_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Task0-UED-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_0_task, 'fixed_level_seed': None, 'lock_params_on_reset': False},
)

gym_register(
    id='BipedalWalker-Adversarial-Task0-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialEnv',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_0_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Task1-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_1_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Task1-UED-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_1_task, 'fixed_level_seed': None, 'lock_params_on_reset': False},
)

gym_register(
    id='BipedalWalker-Adversarial-Task1-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialEnv',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_1_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Medium-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_2_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Medium-UED-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_2_task, 'fixed_level_seed': None, 'lock_params_on_reset': False},
)

gym_register(
    id='BipedalWalker-Adversarial-Medium-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialEnv',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_2_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Hard-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_3_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Hard-UED-v0',
    entry_point=module_path + ':BipedalWalkerMADS',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_3_task, 'fixed_level_seed': None, 'lock_params_on_reset': False},
)

gym_register(
    id='BipedalWalker-Adversarial-Hard-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialEnv',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_3_task, 'fixed_level_seed': None},
)

for rose_id in ['1a', '1b', '2a', '2b', '3a', '3b']:
    gym_register(
        id=f'BipedalWalker-MADS-POET-Rose-{rose_id}-v0',
        entry_point=module_path + ':BipedalWalkerMADSPOETRose',
        max_episode_steps=2000,
        kwargs={'rose_id': rose_id},
    )

gym_register(id='BipedalWalker-MADS-Eval-v0', # [新名字] 用这个名字来测试
             entry_point=module_path + ':BipedalWalkerMADSEval',
             max_episode_steps=2000)

gym_register(
    id='BipedalWalker-MADS-Task0-Eval-v0',
    entry_point=module_path + ':BipedalWalkerMADSEval',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_0_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-Adversarial-Task0-Eval-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialFixedEval',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_0_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Task1-Eval-v0',
    entry_point=module_path + ':BipedalWalkerMADSEval',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_1_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-Adversarial-Task1-Eval-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialFixedEval',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_1_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Medium-Eval-v0',
    entry_point=module_path + ':BipedalWalkerMADSEval',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_2_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-Adversarial-Medium-Eval-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialFixedEval',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_2_task, 'fixed_level_seed': None},
)

gym_register(
    id='BipedalWalker-MADS-Hard-Eval-v0',
    entry_point=module_path + ':BipedalWalkerMADSEval',
    max_episode_steps=2000,
    kwargs={'fixed_level_params_vec': level_3_task, 'fixed_level_seed': None},
)

for rose_id in ['1a', '1b', '2a', '2b', '3a', '3b']:
    gym_register(
        id=f'BipedalWalker-MADS-POET-Rose-{rose_id}-Eval-v0',
        entry_point=module_path + ':BipedalWalkerMADSPOETRoseEval',
        max_episode_steps=2000,
        kwargs={'rose_id': rose_id},
    )

gym_register(
    id='BipedalWalker-Adversarial-Hard-Eval-v0',
    entry_point=module_path + ':BipedalWalkerAdversarialFixedEval',
    max_episode_steps=2000,
    kwargs={'mode': 'full', 'fixed_level_params_vec': level_3_task, 'fixed_level_seed': None},
)

gym_register(id='BipedalWalker-Adversarial-Eval-v0', # [新 ID]
             entry_point=module_path + ':BipedalWalkerAdversarialEval',
              max_episode_steps=2000)
