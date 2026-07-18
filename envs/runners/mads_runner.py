# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import deque, defaultdict

import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd

from level_replay import LevelSampler, LevelStore
from util import \
    array_to_csv, \
    is_discrete_actions, \
    get_obs_at_index, \
    set_obs_at_index

from teachDeepRL.teachers.teacher_controller import TeacherController

import matplotlib as mpl
import matplotlib.pyplot as plt


class MADSRunner(object):
    """
    Performs rollouts of an adversarial environment, given 
    protagonist (agent), antogonist (adversary_agent), and
    environment adversary (advesary_env)
    """
    def __init__(
        self,
        args,
        venv,
        agent,
        ued_venv=None,
        adversary_agent=None,
        adversary_env=None,
        flexible_protagonist=False,
        train=False,
        plr_args=None,
        device='cpu'):
        """
        venv: Vectorized, adversarial gym env with agent-specific wrappers.
        agent: Protagonist trainer.
        ued_venv: Vectorized, adversarial gym env with adversary-env-specific wrappers.
        adversary_agent: Antogonist trainer.
        adversary_env: Environment adversary trainer.

        flexible_protagonist: Which agent plays the role of protagonist in
            calculating the regret depends on which has the lowest score.
        """
        self.args = args

        self.venv = venv
        if ued_venv is None:
            self.ued_venv = venv
        else:
            self.ued_venv = ued_venv # Since adv env can have different env wrappers

        self.is_discrete_actions = is_discrete_actions(self.venv)
        self.is_discrete_adversary_env_actions = is_discrete_actions(self.venv, adversary=True)

        self.agents = {
            'agent': agent,
            'adversary_agent': adversary_agent,
            'adversary_env': adversary_env,
        }

        self.agent_rollout_steps = args.num_steps
        self.adversary_env_rollout_steps = self.venv.adversary_observation_space['time_step'].high[0]

        self.is_dr = args.ued_algo == 'domain_randomization'
        self.is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
        self.is_paired = args.ued_algo in ['paired', 'flexible_paired']
        self.requires_batched_vloss = (args.use_editor and args.base_levels == 'easy' and args.use_accel_paired==False)

        self.is_alp_gmm = args.ued_algo == 'alp_gmm' 

        # Track running mean and std of env returns for return normalization
        if args.adv_normalize_returns:
            self.env_return_rms = RunningMeanStd(shape=())

        self.device = device

        if train:
            self.train()
        else:
            self.eval()

        self.reset()        
        self.use_accel_paired = args.use_accel_paired

        # Set up PLR
        self.level_store = None
        self.level_samplers = {}
        self.current_level_seeds = None
        self.weighted_num_edits = 0
        self.latest_env_stats = defaultdict(float)
        self.use_editor = False
        self.edit_prob = 0
        self.base_levels = None

        # === [新增] MADS RBF 初始化 ===
        # 1. 设置锚点数量 (N)。
        # 注意：最终的 feature_dim = num_rbf_anchors * action_dim
        # BipedalWalker 动作维度是 4，MultiGrid 动作空间是 7 (离散)，特征维度将是 32 * 7 = 224
        self.num_rbf_anchors = 32 
        self.rbf_sigma = 1.0  # 高斯核的带宽 (Sigma)，决定了对相似度的敏感性
        
        # 2. 初始化锚点状态
        self.anchor_states = self._init_anchor_states()
        
        # 3. 预计算核矩阵的逆 (K_inv)
        self.K_inv = self._precompute_rbf_kernel_inverse()

    # [mads_runner.py]
    def _init_anchor_states(self):
        env_name = self.args.env_name
        anchor_candidates = [
            f'anchors_{env_name}.npy',
            f"anchors_{env_name.replace('-UED-v0', '-v0')}.npy",
            f"anchors_{env_name.replace('-Eval-v0', '-v0')}.npy",
        ]
        if env_name.startswith('BipedalWalker-MADS-Hard'):
            anchor_candidates.append('anchors_BipedalWalker-MADS-Medium-v0.npy')
        if env_name.startswith('BipedalWalker'):
            anchor_candidates.extend([
                'anchors_BipedalWalker-v3.npy',
                'anchors_BipedalWalker-MADS-v0.npy',
            ])
        elif env_name.startswith('MultiGrid'):
            anchor_candidates.extend([
                'anchors_MultiGrid-MADS-Task0-v0.npy',
                'anchors_MultiGrid.npy',
            ])

        anchor_file = next(
            (candidate for candidate in anchor_candidates if os.path.exists(candidate)),
            None,
        )
        
        if anchor_file is not None:
            print(f"Loading anchor states from {anchor_file}...")
            loaded_data = np.load(anchor_file, allow_pickle=True)

            # 随机采样或重复采样以凑够数量
            if len(loaded_data) >= self.num_rbf_anchors:
                indices = np.random.choice(len(loaded_data), self.num_rbf_anchors, replace=False)
            else:
                indices = np.random.choice(len(loaded_data), self.num_rbf_anchors, replace=True)
            
            selected_anchors = loaded_data[indices]
            if isinstance(selected_anchors[0], dict):
                tensor_anchors = {
                    key: torch.from_numpy(
                        np.stack([np.asarray(anchor[key]) for anchor in selected_anchors])
                    ).float().to(self.device)
                    for key in selected_anchors[0].keys()
                }
                return tensor_anchors

            tensor_anchors = torch.from_numpy(selected_anchors).float().to(self.device)

            # === [关键修改点] ===
            # MultiGrid 的 Observation 通常是 (H, W, 3)，例如 (7, 7, 3)
            # PyTorch 需要 (N, C, H, W)，即 (N, 3, 7, 7)
            if len(tensor_anchors.shape) == 4 and tensor_anchors.shape[-1] == 3:
                tensor_anchors = tensor_anchors.permute(0, 3, 1, 2)
                print(f"Permuted anchors to: {tensor_anchors.shape}")
            
            return tensor_anchors
        else:
            print(
                f"Warning: No anchor file found for {env_name}. "
                "Using zero anchors matching the environment observation shape."
            )
            obs_space = self.venv.observation_space
            if hasattr(obs_space, 'spaces'):
                return {
                    key: torch.zeros(self.num_rbf_anchors, *space.shape).to(self.device)
                    for key, space in obs_space.spaces.items()
                }
            return torch.zeros(self.num_rbf_anchors, *obs_space.shape).to(self.device)

    def _precompute_rbf_kernel_inverse(self):
        s = self.anchor_states
        
        # === [关键修改点] ===
        # 如果 s 是图像 (N, C, H, W)，需要展平为 (N, Features) 才能计算距离
        if isinstance(s, dict):
            s_flat = torch.cat(
                [value.reshape(value.shape[0], -1).float() for value in s.values()],
                dim=-1,
            )
        elif len(s.shape) > 2:
            s_flat = s.reshape(s.shape[0], -1)
        else:
            s_flat = s

        # 计算成对距离矩阵
        dist_sq = torch.cdist(s_flat, s_flat, p=2) ** 2
        
        # 计算高斯核矩阵 K
        K = torch.exp(-dist_sq / (2 * self.rbf_sigma ** 2))
        
        # 增加数值稳定性 (Jitter)
        identity = torch.eye(self.num_rbf_anchors).to(self.device)
        K = K + identity * 1e-6 
        
        # 求逆
        K_inv = torch.linalg.inv(K)
        return K_inv

    def _anchor_batch_size(self):
        if isinstance(self.anchor_states, dict):
            return next(iter(self.anchor_states.values())).shape[0]
        return self.anchor_states.shape[0]

    def _make_policy_feature_rnn_hxs(self, actor_critic, batch_size):
        hidden_size = actor_critic.recurrent_hidden_state_size
        if hidden_size == 0:
            return None

        zeros = torch.zeros(batch_size, hidden_size, device=self.device)
        rnn = getattr(actor_critic, 'rnn', None)
        if getattr(rnn, 'is_lstm', False):
            return (zeros, zeros.clone())
        return zeros

    def _compute_policy_features(self, policy_agent_name='adversary_agent'):
        # MADS: the curriculum designer pi_delta should condition on the
        # dynamic learner pi_theta, which is `adversary_agent` in this runner.
        agent = self.agents.get(policy_agent_name)
        if agent is None:
            raise ValueError(
                f"Cannot compute curriculum policy features: "
                f"agent '{policy_agent_name}' is not available."
            )
        # 临时切换到 eval 模式
        training_mode = agent.algo.actor_critic.training
        agent.eval() 
        
        with torch.no_grad():
            anchor_batch_size = self._anchor_batch_size()
            rnn_hxs = self._make_policy_feature_rnn_hxs(
                agent.algo.actor_critic,
                anchor_batch_size,
            )
            masks = torch.zeros(anchor_batch_size, 1).to(self.device)
            
            # === [关键修改点 1] 获取动作分布 ===
            # 注意：这里我们不仅需要 action，还需要 distribution (action_log_dist)
            # 你的 Agent act 函数通常返回: value, action, action_log_prob, rnn_states
            # 请确保你的代码能访问到底层的 distribution，或者通过 evaluate_actions 获取
            
            # 假设 model.act 返回了 distribution 或者我们可以通过 model.get_dist 获取
            # 这里演示通过 standard act 获取 (需要确认你的 PPO 代码里 act 是否返回了 dist)
            value, action, action_log_dist, rnn_hxs = agent.act(self.anchor_states, rnn_hxs, masks)

            # === [关键修改点 2] 离散动作处理 ===
            if self.is_discrete_actions:
                # 离散动作：使用概率 (probs) 作为特征向量
                # action_log_dist 通常是 Categorical 分布
                if hasattr(action_log_dist, 'probs'):
                    action_vector = action_log_dist.probs # [N, 7]
                elif torch.is_tensor(action_log_dist):
                    action_vector = action_log_dist.softmax(dim=-1)
                else:
                    # 如果只有 logits，手动 softmax
                    action_vector = action_log_dist.logits.softmax(dim=-1)
            else:
                # 连续动作 (BipedalWalker)：直接用 action
                action_vector = action
            
            # 2. 矩阵乘法求解 Alpha: [N, N] @ [N, ActionDim]
            alpha = torch.matmul(self.K_inv, action_vector)
            
            # 3. 展平为一维向量
            alpha_flat = alpha.flatten()
            
        if training_mode:
            agent.train()
            
        return alpha_flat.cpu().numpy()

    def _set_env_policy_features(self, env, policy_feature):
        """
        将计算好的 Alpha 特征注入到并行环境中。
        兼容 SubprocVecEnv 和 DummyVecEnv。
        """
        print(f"DEBUG: Env Type: {type(env)}")
        print(f"DEBUG: Env Dir: {dir(env)}")
        # 如果是 DummyVecEnv (调试用)，可以直接访问 envs 列表
        if hasattr(env, 'envs'):
            for sub_env in env.envs:
                # 假设 sub_env 是被 Monitor 等包裹的，我们需要找到最底层的 BipedalWalkerMADS
                # 通常直接设置属性即可，只要你的 Wrapper 没把 __setattr__ 拦截
                if hasattr(sub_env, 'unwrapped'):
                    sub_env.unwrapped.current_policy_feature = policy_feature
                else:
                    sub_env.current_policy_feature = policy_feature
        
        # 如果是 SubprocVecEnv (多进程)，需要使用 env_method 或 set_attr
        elif hasattr(env, 'set_attr'):
            # set_attr 会把所有子进程环境的该属性都设为同一个值
            env.set_attr('current_policy_feature', policy_feature)
        
        # 如果是其他类型的 Wrapper，尝试直接调用
        else:
            try:
                env.set_attr('current_policy_feature', policy_feature)
            except AttributeError:
                print("Warning: Could not set policy feature for environment.")

    @property
    def use_byte_encoding(self):
        env_name = self.args.env_name
        if self.args.use_editor \
           or env_name.startswith('BipedalWalker') \
           or (env_name.startswith('MultiGrid') and self.args.use_reset_random_dr):
            return True
        else:
            return False

    def reset(self):
        self.num_updates = 0
        self.total_num_edits = 0
        self.total_episodes_collected = 0
        self.total_seeds_collected = 0
        self.student_grad_updates = 0
        self.sampled_level_info = None

        max_return_queue_size = 10
        self.agent_returns = deque(maxlen=max_return_queue_size)
        self.adversary_agent_returns = deque(maxlen=max_return_queue_size)

    def train(self):
        self.is_training = True
        [agent.train() if agent else agent for _,agent in self.agents.items()]

    def eval(self):
        self.is_training = False
        [agent.eval() if agent else agent for _,agent in self.agents.items()]

    def state_dict(self):
        agent_state_dict = {}
        optimizer_state_dict = {}
        for k, agent in self.agents.items():
            if agent:
                agent_state_dict[k] = agent.algo.actor_critic.state_dict()
                optimizer_state_dict[k] = agent.algo.optimizer.state_dict()

        return {
            'agent_state_dict': agent_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'agent_returns': self.agent_returns,
            'adversary_agent_returns': self.adversary_agent_returns,
            'num_updates': self.num_updates,
            'total_episodes_collected': self.total_episodes_collected,
            'total_seeds_collected': self.total_seeds_collected,
            'total_num_edits': self.total_num_edits,
            'student_grad_updates': self.student_grad_updates,
            'latest_env_stats': self.latest_env_stats,
            'level_store': self.level_store,
            'level_samplers': self.level_samplers,
        }

    def load_state_dict(self, state_dict):

        agent_state_dict = state_dict.get('agent_state_dict')

        for k,state in agent_state_dict.items():
            self.agents[k].algo.actor_critic.load_state_dict(state)

        optimizer_state_dict = state_dict.get('optimizer_state_dict')

        for k, state in optimizer_state_dict.items():
            self.agents[k].algo.optimizer.load_state_dict(state)

        self.agent_returns = state_dict.get('agent_returns')
        self.adversary_agent_returns = state_dict.get('adversary_agent_returns')
        self.num_updates = state_dict.get('num_updates')
        self.total_episodes_collected = state_dict.get('total_episodes_collected')
        self.total_seeds_collected = state_dict.get('total_seeds_collected')
        self.total_num_edits = state_dict.get('total_num_edits')
        self.student_grad_updates = state_dict.get('student_grad_updates')
        self.latest_env_stats = state_dict.get('latest_env_stats')

        self.level_store = state_dict.get('level_store')
        self.level_samplers = state_dict.get('level_samplers')

    def _get_batched_value_loss(self, agent, clipped=True, batched=True):
        batched_value_loss = agent.storage.get_batched_value_loss(
            signed=False, 
            positive_only=False, 
            clipped=clipped,
            batched=batched)

        return batched_value_loss

    def _get_rollout_return_stats(self, rollout_returns):
        mean_return = torch.zeros(self.args.num_processes, 1)
        max_return = torch.zeros(self.args.num_processes, 1)
        for b, returns in enumerate(rollout_returns):
            if len(returns) > 0:
                mean_return[b] = float(np.mean(returns))
                max_return[b] = float(np.max(returns))

        stats = {
            'mean_return': mean_return,
            'max_return': max_return,
            'returns': rollout_returns 
        }

        return stats
    
    def _calculate_paired_regret_scores(self, agent_rollout_info, adversary_agent_rollout_info, type="paired"):
        if type=="paired":
            external_scores = torch.max(adversary_agent_rollout_info['max_return'] - agent_rollout_info['mean_return'], \
                    torch.zeros_like(agent_rollout_info['mean_return']))
        else:
            raise NotImplementedError
            
        return external_scores

    def _get_env_stats_multigrid(self, agent_info, adversary_agent_info):
        num_blocks = np.mean(agent_info.get(
            'num_blocks', self.venv.get_num_blocks()))
        
        passable_ratio = np.mean(agent_info.get(
            'passable_ratio', self.venv.get_passable()))

        shortest_path_lengths = agent_info.get(
            'shortest_path_lengths', self.venv.get_shortest_path_length())
        shortest_path_length = np.mean(shortest_path_lengths)

        solved_idx =  agent_info.get('solved_idx', None)
        if solved_idx is None:
            if 'max_returns' in adversary_agent_info:
                solved_idx = \
                    (torch.max(agent_info['max_return'], \
                        adversary_agent_info['max_return']) > 0).numpy().squeeze()
            else:
                solved_idx = (agent_info['max_return'] > 0).numpy().squeeze()

        solved_path_lengths = np.array(shortest_path_lengths)[solved_idx]
        solved_path_length = np.mean(solved_path_lengths) if len(solved_path_lengths) > 0 else 0

        stats = {
            'num_blocks': num_blocks,
            'passable_ratio': passable_ratio,
            'shortest_path_length': shortest_path_length,
            'solved_path_length': solved_path_length
        }

        return stats

    def _get_env_stats_car_racing(self, agent_info, adversary_agent_info):
        infos = self.venv.get_complexity_info()
        num_envs = len(infos)

        sums = defaultdict(float)
        for info in infos:
            for k,v in info.items():
                sums[k] += v

        stats = {}
        for k,v in sums.items():
            stats['track_' + k] = sums[k]/num_envs

        return stats

    def _get_env_stats_bipedalwalker(self, agent_info, adversary_agent_info):
        def _summarize(infos_):
            num_envs_ = len(infos_)
            sums_ = defaultdict(float)
            for info_ in infos_:
                for k_, v_ in info_.items():
                    sums_[k_] += v_
            return num_envs_, sums_

        venv_infos = self.venv.get_complexity_info()
        venv_num_envs, venv_sums = _summarize(venv_infos)

        stats = {}
        for k, v in venv_sums.items():
            stats['track_' + k] = v / venv_num_envs
            stats['track_student_' + k] = v / venv_num_envs

        if getattr(self, 'ued_venv', None) is not None and self.ued_venv is not self.venv:
            ued_infos = self.ued_venv.get_complexity_info()
            ued_num_envs, ued_sums = _summarize(ued_infos)
            for k, v in ued_sums.items():
                stats['track_ued_' + k] = v / ued_num_envs

        return stats

    def _get_env_stats(self, agent_info, adversary_agent_info, log_replay_complexity=False):
        env_name = self.args.env_name
        if env_name.startswith('MultiGrid'):
            stats = self._get_env_stats_multigrid(agent_info, adversary_agent_info)
        elif env_name.startswith('CarRacing'):
            stats = self._get_env_stats_car_racing(agent_info, adversary_agent_info)
        elif env_name.startswith('BipedalWalker'):
            stats = self._get_env_stats_bipedalwalker(agent_info, adversary_agent_info)
        else:
            raise ValueError(f'Unsupported environment, {self.args.env_name}')

        stats_ = {}
        for k,v in stats.items():
            stats_['plr_' + k] = v if log_replay_complexity else None
            stats_[k] = v if not log_replay_complexity else None
            
        return stats_

    def _get_active_levels(self):
        assert self.args.use_plr, 'Only call _get_active_levels when using PLR.'

        env_name = self.args.env_name

        is_multigrid = env_name.startswith('MultiGrid')
        is_car_racing = env_name.startswith('CarRacing')
        is_bipedal_walker = env_name.startswith('BipedalWalker')

        if self.use_byte_encoding:
            return [x.tobytes() for x in self.ued_venv.get_encodings()]
        elif is_multigrid:
            return self.agents['adversary_env'].storage.get_action_traj(as_string=True)
        else:
            return self.ued_venv.get_level()

    def _get_level_sampler(self, name):
        other = 'adversary_agent'
        if name == 'adversary_agent':
            other = 'agent'

        level_sampler = self.level_samplers.get(name) or self.level_samplers.get(other)

        updateable = name in self.level_samplers

        return level_sampler, updateable

    @property
    def all_level_samplers(self):
        if len(self.level_samplers) == 0:
            return []

        return list(filter(lambda x: x is not None, [v for _, v in self.level_samplers.items()]))

    def _should_edit_level(self):
        if self.use_editor:
            return np.random.rand() < self.edit_prob
        else:
            return False

    def _update_plr_with_current_unseen_levels(self, parent_seeds=None):
        args = self.args
        levels = self._get_active_levels()
        self.current_level_seeds = \
            self.level_store.insert(levels, parent_seeds=parent_seeds)
        if args.log_plr_buffer_stats or args.reject_unsolvable_seeds:
            passable = self.venv.get_passable()
        else:
            passable = None
        self._update_level_samplers_with_external_unseen_sample(
            self.current_level_seeds, solvable=passable)

    def _update_level_samplers_with_external_unseen_sample(self, seeds, solvable=None):
        level_samplers = self.all_level_samplers

        if self.args.reject_unsolvable_seeds:
            solvable = np.array(solvable, dtype=np.bool)
            seeds = np.array(seeds, dtype=np.int)[solvable]
            solvable = solvable[solvable]

        for level_sampler in level_samplers:
            level_sampler.observe_external_unseen_sample(seeds, solvable)

    def agent_rollout(self, 
                      agent, 
                      num_steps, 
                      # MADS: 新增参数 specific_env，默认为 None
                      specific_env=None,
                      update=False, 
                      is_env=False, 
                      level_replay=False, 
                      level_sampler=None, 
                      update_level_sampler=False,
                      discard_grad=False, 
                      edit_level=False,
                      num_edits=0, 
                      fixed_seeds=None,
                      kl_dict=None,
                      update_agent_separately=False):
        args = self.args

        # MADS: 决定当前使用哪个环境实例 (active_env)
        if is_env:
            # 如果是环境生成阶段，强制使用 ued_venv (用于变异)
            active_env = self.ued_venv
        elif specific_env is not None:
            # 如果指定了环境，就用指定的 (用于把 Antagonist 指向 ued_venv)
            active_env = specific_env
        else:
            # 默认情况 self.venv (用于 Student 指向固定环境)
            active_env = self.venv
        

        if is_env:
            if edit_level: # Get mutated levels
                levels = [self.level_store.get_level(seed) for seed in fixed_seeds]
                active_env.reset_to_level_batch(levels)
                active_env.mutate_level(num_edits=num_edits)
                self._update_plr_with_current_unseen_levels(parent_seeds=fixed_seeds)
                return
            if level_replay: # Get replay levels
                self.current_level_seeds = [level_sampler.sample_replay_level() for _ in range(args.num_processes)]
                levels = [self.level_store.get_level(seed) for seed in self.current_level_seeds]
                active_env.reset_to_level_batch(levels)
                return self.current_level_seeds
            else:
                obs = active_env.reset() # Prepare for constructive rollout
                self.total_seeds_collected += args.num_processes
        else:
            obs = active_env.reset_agent()

        # Initialize first observation
        agent.storage.copy_obs_to_index(obs,0)
        
        rollout_info = {}
        rollout_returns = [[] for _ in range(args.num_processes)]
        
        if self.use_accel_paired:
            actor_seeds = {i: [] for i in range(args.num_processes)}

        if level_sampler and level_replay:
            rollout_info.update({
                'solved_idx': np.zeros(args.num_processes, dtype=np.bool)
            })
            
        for step in range(num_steps):
            if args.render:
                active_env.render_to_screen()
            # Sample actions
            with torch.no_grad():
                obs_id = agent.storage.get_obs(step)
                value, action, action_log_dist, recurrent_hidden_states = agent.act(
                    obs_id, agent.storage.get_recurrent_hidden_state(step), agent.storage.masks[step])
                if self.is_discrete_actions:
                    action_log_prob = action_log_dist.gather(-1, action)
                else:
                    action_log_prob = action_log_dist
            
            # 改了这里
            # Observe reward and next obs
            reset_random = self.is_dr and not args.use_plr

            # ----- MADS: adversary_env outputs delta; execute base + delta -----
            action_to_exec = action
            # ---------------------------------------------------------------

            _action = agent.process_action(action_to_exec.cpu())

            if is_env:
                obs, reward, done, infos = active_env.step_adversary(_action)
            else:
                obs, reward, done, infos = active_env.step_env(_action, reset_random=reset_random)
                if args.clip_reward:
                    reward = torch.clamp(reward, -args.clip_reward, args.clip_reward)


            if not is_env and step >= num_steps - 1:
                # Handle early termination due to cliffhanger rollout
                if agent.storage.use_proper_time_limits:
                    for i, done_ in enumerate(done):
                        if not done_:
                            infos[i]['cliffhanger'] = True
                            infos[i]['truncated'] = True
                            infos[i]['truncated_obs'] = get_obs_at_index(obs, i)

                done = np.ones_like(done, dtype=np.float)

            if level_sampler and level_replay:
                next_level_seeds = [s for s in self.current_level_seeds]
                
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    rollout_returns[i].append(info['episode']['r'])
                    
                    if self.use_accel_paired:
                        actor_seeds[i].append(self.current_level_seeds[i])

                    if reset_random:
                        self.total_seeds_collected += 1

                    if not is_env:
                        self.total_episodes_collected += 1

                        # Handle early termination
                        if agent.storage.use_proper_time_limits:
                            if 'truncated_obs' in info.keys():
                                truncated_obs = info['truncated_obs']
                                agent.storage.insert_truncated_obs(truncated_obs, index=i)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'truncated' in info.keys() else [1.0]
                 for info in infos])
            cliffhanger_masks = torch.FloatTensor(
                [[0.0] if 'cliffhanger' in info.keys() else [1.0]
                 for info in infos])

            # Need to store level seeds alongside non-env agent steps
            current_level_seeds = None
            if (not is_env) and level_sampler:
                current_level_seeds = torch.tensor(self.current_level_seeds, dtype=torch.int).view(-1, 1)

            agent.insert(
                obs, recurrent_hidden_states, 
                action, action_log_prob, action_log_dist, 
                value, reward, masks, bad_masks, 
                level_seeds=current_level_seeds,
                cliffhanger_masks=cliffhanger_masks)

            if level_sampler and level_replay:
                self.current_level_seeds = next_level_seeds


        rollout_info.update(self._get_rollout_return_stats(rollout_returns))
        if self.use_accel_paired:
            rollout_info['actor_seeds'] = actor_seeds

        # Update non-env agent if required
        if not is_env and update: 
            with torch.no_grad():
                obs_id = agent.storage.get_obs(-1)
                next_value = agent.get_value(
                    obs_id, agent.storage.get_recurrent_hidden_state(-1),
                    agent.storage.masks[-1]).detach()

            agent.storage.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda)

            # Compute batched value loss if using value_l1-maximizing adversary
            if self.requires_batched_vloss:
                # Don't clip value loss reward if env adversary normalizes returns
                clipped = not args.adv_use_popart and not args.adv_normalize_returns
                batched_value_loss = self._get_batched_value_loss(
                    agent, clipped=clipped, batched=True)
                rollout_info.update({'batched_value_loss': batched_value_loss})

            # Update level sampler and remove any ejected seeds from level store
            if not update_agent_separately:
                if level_sampler and update_level_sampler:
                    level_sampler.update_with_rollouts(agent.storage)

                value_loss, action_loss, dist_entropy, info = agent.update(discard_grad=discard_grad, kl_dict=kl_dict)

                if level_sampler and update_level_sampler:
                    level_sampler.after_update()
                
                if 'kl_loss' in info.keys():
                    kl_loss = info.pop('kl_loss')
                    rollout_info.update({'kl_loss': kl_loss})

                rollout_info.update({
                    'value_loss': value_loss,
                    'action_loss': action_loss,
                    'dist_entropy': dist_entropy,
                    'update_info': info,
                })

                # Compute LZ complexity of action trajectories
                if args.log_action_complexity:
                    rollout_info.update({'action_complexity': agent.storage.get_action_complexity()})

        return rollout_info
    
    def _update_agent_separately(self, 
                                 agent, 
                                 level_sampler=None, 
                                 update_level_sampler=False,
                                 discard_grad=False,
                                 kl_dict=None,
                                 external_scores=None):

        # Update level sampler and remove any ejected seeds level store
        if level_sampler and update_level_sampler:
            level_sampler.update_with_rollouts(agent.storage, external_scores=external_scores)

        value_loss, action_loss, dist_entropy, info = agent.update(discard_grad=discard_grad, kl_dict=kl_dict)

        if level_sampler and update_level_sampler:
            level_sampler.after_update()
        
        rollout_info = {
            'value_loss': value_loss,
            'action_loss': action_loss,
            'dist_entropy': dist_entropy,
            'update_info': info,
        }
        
        if 'kl_loss' in info.keys():
            kl_loss = info.pop('kl_loss')
            rollout_info.update({'kl_loss': kl_loss})

        # Compute LZ complexity of action trajectories
        if self.args.log_action_complexity:
            rollout_info.update({'action_complexity': agent.storage.get_action_complexity()})
        
        return rollout_info

    def _compute_env_return(self, agent_info, adversary_agent_info):
        # Use the dynamic learner pi_theta (`adversary_agent`) so the
        # curriculum reward is aligned with the curriculum feature source.
        antagonist_storage = self.agents['adversary_agent'].storage
        returns = antagonist_storage.returns[:-1]
        value_preds = antagonist_storage.value_preds[:-1]
        
        antagonist_advantage = (returns - value_preds).mean(dim=0)

        env_return = antagonist_advantage.clone().detach()
        
        if self.args.adv_normalize_returns:
            self.env_return_rms.update(env_return.flatten().cpu().numpy())
            env_return /= np.sqrt(self.env_return_rms.var + 1e-8)

        if self.args.adv_clip_reward is not None:
            env_return = env_return.clamp(-self.args.adv_clip_reward, self.args.adv_clip_reward)
        
        return env_return

    # MADS: 双策略同步 (DPS)
    def _dps_synchronize(self, lambda_skill=0.95, lambda_target=0.95):
        """
        参数:
        lambda_skill (float): 控制从 Antagonist 到 Student 的迁移速率。
                              值越小，Student 吸收 Antagonist 的技能越多。
                              (Paper建议动态计算，这里先用固定值)                      
        lambda_target (float): 控制从 Student 到 Antagonist 的约束速率。
                               值越小，Antagonist 被拉回 Target 的力度越大。
        """
        
        # 获取两个 Agent 的模型
        # agent (Student/Native Policy) = Phi (运行在固定环境)
        # adversary_agent (Antagonist/Policy Learner) = Theta (运行在变异环境)
        student_model = self.agents['agent'].algo.actor_critic
        antagonist_model = self.agents['adversary_agent'].algo.actor_critic
        
        phi_state = student_model.state_dict()
        theta_state = antagonist_model.state_dict()
        
        # 准备新的参数字典
        new_phi_state = {}   # Student 的新参数
        new_theta_state = {} # Antagonist 的新参数
        
        # 遍历所有参数键值 (假设两个模型结构完全一致)
        for key in phi_state.keys():
            param_phi = phi_state[key]
            param_theta = theta_state[key]
            
            # -----------------------------------------------------------
            # 方向 1: 技能迁移 (Skill Transfer: Theta -> Phi)
            # Formula: phi = lambda1 * phi + (1 - lambda1) * theta
            # -----------------------------------------------------------
            # 让 Student (Phi) 吸收一部分 Antagonist (Theta) 在变异环境中探索到的参数
            new_phi_state[key] = lambda_skill * param_phi + (1 - lambda_skill) * param_theta
            
            # -----------------------------------------------------------
            # 方向 2: 目标对齐 (Target Alignment: Phi -> Theta)
            # Formula: theta = lambda2 * theta + (1 - lambda2) * phi
            # -----------------------------------------------------------
            # 让 Antagonist (Theta) 被拉回一点，不要在变异环境中玩得太偏
            new_theta_state[key] = lambda_target * param_theta + (1 - lambda_target) * param_phi

        # 将计算好的新参数加载回模型
        student_model.load_state_dict(new_phi_state)
        antagonist_model.load_state_dict(new_theta_state)
        return {
            'dps_lambda_skill': lambda_skill,
            'dps_lambda_target': lambda_target,
        }

    def run(self):
        args = self.args

        adversary_env = self.agents['adversary_env']
        agent = self.agents['agent']
        adversary_agent = self.agents['adversary_agent']
        level_replay = False

        # Discard student gradients if not level replay (sampling new levels)
        student_discard_grad = False

        if self.is_training and not student_discard_grad:
            self.student_grad_updates += 1

        # =============[新增] MADS: 计算并广播 Alpha 特征=======================
        # 1. 计算当前动态学习者 pi_theta 的 Alpha
        current_alpha = self._compute_policy_features()
        # 2. 将 Alpha 广播给 Adversary 环境 (ued_venv)
        # 这样，Adversary 下一次 reset/step 时，就会把这个特征放入 Observation
        self.ued_venv.set_policy_feature(current_alpha)
        # =========================================================

        # Generate a batch of adversarial environments
        env_info = self.agent_rollout(
            agent=adversary_env, 
            num_steps=self.adversary_env_rollout_steps, 
            update=False,
            is_env=True,
            level_replay=level_replay,
            level_sampler=self._get_level_sampler('agent')[0],
            update_level_sampler=False)

        # Run agent episodes
        level_sampler, is_updateable = self._get_level_sampler('agent')
        kl_dict_agent = None
                
        agent_info = self.agent_rollout(
            agent=agent, 
            num_steps=self.agent_rollout_steps,
            update=self.is_training,
            specific_env=self.venv,  # MADS: 指定去固定环境跑
            level_replay=level_replay,
            level_sampler=level_sampler,
            update_level_sampler=is_updateable,
            discard_grad=student_discard_grad,
            kl_dict=kl_dict_agent,
            update_agent_separately=self.use_accel_paired)

        adversary_agent_info = defaultdict(float)

        # Run adversary agent episodes
        level_sampler, is_updateable = self._get_level_sampler('adversary_agent')
        
        kl_dict_adv_agent = None
        if not self.args.use_kl_only_agent:
            if self.is_training and self.args.use_behavioural_cloning:
                if ((self.student_grad_updates) % self.args.kl_update_step == 0):
                    kl_dict_adv_agent = {}
                    agent.eval()
                    kl_dict_adv_agent['antagonist_model'] = agent.algo.actor_critic
                    
        adversary_agent_info = self.agent_rollout(
            agent=adversary_agent, 
            num_steps=self.agent_rollout_steps, 
            update=self.is_training,
            specific_env=self.ued_venv,    # MADS: 指定去变异环境跑
            level_replay=level_replay,
            level_sampler=level_sampler,
            update_level_sampler=is_updateable,
            discard_grad=student_discard_grad,
            kl_dict=kl_dict_adv_agent)

        sampled_level_info = {
            'level_replay': False,
            'num_edits': [0 for _ in range(args.num_processes)]
        }

        # Update adversary agent final return
        env_return = self._compute_env_return(agent_info, adversary_agent_info)

        adversary_env_info = defaultdict(float)
        if self.is_training and self.is_training_env:
            with torch.no_grad():
                obs_id = adversary_env.storage.get_obs(-1)
                next_value = adversary_env.get_value(
                    obs_id, adversary_env.storage.get_recurrent_hidden_state(-1),
                    adversary_env.storage.masks[-1]).detach()
            adversary_env.storage.replace_final_return(env_return)
            adversary_env.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            env_value_loss, env_action_loss, env_dist_entropy, info = adversary_env.update()
            adversary_env_info.update({
                'action_loss': env_action_loss,
                'value_loss': env_value_loss,
                'dist_entropy': env_dist_entropy,
                'update_info': info
            })

        if self.is_training:
            self.num_updates += 1
            # MADS DPS: keep the dynamic learner pi_theta and target learner
            # pi_phi softly synchronized with fixed lambda weights.
            dps_info = self._dps_synchronize(
                lambda_skill=args.lambda1,
                lambda_target=args.lambda2,
            )
        else:
            dps_info = {}

        # === LOGGING ===
        # Only update env-related stats when run generates new envs (not level replay)
        log_replay_complexity = level_replay and args.log_replay_complexity
        if (not level_replay) or log_replay_complexity:
            stats = self._get_env_stats(agent_info, adversary_agent_info, 
                log_replay_complexity=log_replay_complexity)
            stats.update({
                'mean_env_return': env_return.mean().item(),
                'adversary_env_pg_loss': adversary_env_info['action_loss'],
                'adversary_env_value_loss': adversary_env_info['value_loss'],
                'adversary_env_dist_entropy': adversary_env_info['dist_entropy'],
            })
        else:
            stats = self.latest_env_stats.copy()

        [self.agent_returns.append(r) for b in agent_info['returns'] for r in reversed(b)]
        mean_agent_return = 0
        if len(self.agent_returns) > 0:
            mean_agent_return = np.mean(self.agent_returns)

        mean_adversary_agent_return = 0
        if self.is_paired or self.use_accel_paired:
            [self.adversary_agent_returns.append(r) for b in adversary_agent_info['returns'] for r in reversed(b)]
            if len(self.adversary_agent_returns) > 0:
                mean_adversary_agent_return = np.mean(self.adversary_agent_returns)

        self.sampled_level_info = sampled_level_info

        stats.update({
            'steps': (self.num_updates + self.total_num_edits) * args.num_processes * args.num_steps,
            'total_episodes': self.total_episodes_collected,
            'total_seeds': self.total_seeds_collected,
            'total_student_grad_updates': self.student_grad_updates,

            'mean_agent_return': mean_agent_return,
            'agent_value_loss': agent_info['value_loss'],
            'agent_pg_loss': agent_info['action_loss'],
            'agent_dist_entropy': agent_info['dist_entropy'],

            'mean_adversary_agent_return': mean_adversary_agent_return,
            'adversary_value_loss': adversary_agent_info['value_loss'],
            'adversary_pg_loss': adversary_agent_info['action_loss'],
            'adversary_dist_entropy': adversary_agent_info['dist_entropy'],
            
            'kl_loss_advagent_agent': agent_info.get('kl_loss', None),
            'kl_loss_agent_advagent': adversary_agent_info.get('kl_loss', None)
        })
        stats.update(dps_info)

        if args.log_grad_norm:
            agent_grad_norm = np.mean(agent_info['update_info']['grad_norms'])
            adversary_grad_norm = 0
            adversary_env_grad_norm = 0
            if self.is_paired:
                adversary_grad_norm = np.mean(adversary_agent_info['update_info']['grad_norms'])
            if self.is_training_env:
                adversary_env_grad_norm = np.mean(adversary_env_info['update_info']['grad_norms'])
            stats.update({
                'agent_grad_norm': agent_grad_norm,
                'adversary_grad_norm': adversary_grad_norm,
                'adversary_env_grad_norm': adversary_env_grad_norm
            })

        if args.log_action_complexity:
            stats.update({
                'agent_action_complexity': agent_info['action_complexity'],
                'adversary_action_complexity': adversary_agent_info['action_complexity']  
            }) 

        return stats
