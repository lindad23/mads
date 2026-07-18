import os
import argparse
import numpy as np
import torch
import gym
from gym.envs.registration import register

# ==============================================================================
# [区域 1: 环境注册] 
# 同时支持 BipedalWalker 和 MultiGrid 的手动注册
# ==============================================================================
try:
    # --- 1. BipedalWalker 注册 ---
    from envs.bipedalwalker.adversarial import BipedalWalkerMADS
    env_id_bw = 'BipedalWalker-MADS-Medium-v0'
    if env_id_bw not in gym.envs.registry.env_specs:
        register(
            id=env_id_bw,
            entry_point='envs.bipedalwalker.adversarial:BipedalWalkerMADS',
            max_episode_steps=2000,
        )
    
    # --- 2. MultiGrid 注册 (假设你的文件路径是 envs/multigrid/multigrid_envs.py) ---
    # 根据你之前的代码上下文推断
    import envs.multigrid.multigrid_envs # 触发注册逻辑
    # 如果你的 MultiGrid 注册逻辑写在 __init__.py 里，import envs.multigrid 即可

except ImportError as e:
    print(f"\n[WARNING] 自动导入环境失败，如果是标准Gym环境请忽略。错误: {e}")
    # 不中断，因为可能是跑标准环境

# ==============================================================================
# [区域 2: Policy 类映射]
# ==============================================================================
def get_policy_class(env_name):
    """
    根据环境名称返回对应的 Policy 类。
    """
    if 'BipedalWalker' in env_name:
        from models.walker_models import BipedalWalkerStudentPolicy
        return BipedalWalkerStudentPolicy
        
    elif 'MultiGrid' in env_name:
        # [修改点] 请确保这里指向你的 Grid Policy 类
        # 假设你的模型文件在 models/grid_models.py
        from models.grid_models import GridStudentPolicy 
        return GridStudentPolicy
        
    elif 'CarRacing' in env_name:
        raise NotImplementedError("CarRacing policy not yet linked.")
        
    else:
        raise ValueError(f"Unknown environment: {env_name}. Please register it in get_policy_class().")

# ==============================================================================
# [区域 3: 观测处理 (核心修改)]
# ==============================================================================
def process_obs(obs, device):
    """
    统一处理 Observation:
    1. Dict -> Array
    2. (H,W,C) -> (C,H,W) [MultiGrid 需要]
    3. Array -> Tensor
    """
    # 1. 提取原始数据
    if isinstance(obs, dict):
        if 'image' in obs:
            raw_data = obs['image']
        elif 'observation' in obs:
            raw_data = obs['observation']
        else:
            raw_data = list(obs.values())[0]
    else:
        raw_data = obs

    # 2. [新增] 维度转换 (针对 MultiGrid/图像环境)
    # raw_data shape 通常是 (H, W, C) -> PyTorch 需要 (C, H, W)
    if len(raw_data.shape) == 3:
        # 假设最后是 Channel (7,7,3)
        if raw_data.shape[-1] == 3: 
            raw_data = np.transpose(raw_data, (2, 0, 1)) # -> (3, 7, 7)
    
    # 3. 转为 Tensor 并增加 Batch 维度 (1, ...)
    return torch.from_numpy(raw_data).float().unsqueeze(0).to(device)

def get_raw_obs_data(obs):
    """
    保存锚点时，我们保存【原始】格式 (H,W,C)。
    这样 Runner 加载时可以灵活处理。
    """
    if isinstance(obs, dict):
        if 'image' in obs: return obs['image']
        return list(obs.values())[0]
    return obs

# ==============================================================================
# [主逻辑]
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generic Anchor Generator for MADS")
    
    parser.add_argument('--env_name', type=str, required=True, help='Gym environment ID')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pt file')
    
    parser.add_argument('--num_anchors', type=int, default=32, help='Number of anchors to collect')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save')
    parser.add_argument('--sample_interval', type=int, default=20, help='Steps between samples')
    parser.add_argument('--device', type=str, default='auto', help='cuda:0 or cpu')

    args = parser.parse_args()

    # 1. 设备设置
    if args.device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"[{args.env_name}] Starting Anchor Generation...")
    print(f"Model: {args.model_path}")

    # 2. 初始化环境
    try:
        env = gym.make(args.env_name)
        # 对于 TaskBased MultiGrid，可能需要指定 task_level，但这里默认跑 Task0 或 v0 即可
    except Exception as e:
        print(f"Error creating env {args.env_name}: {e}")
        return

    # 3. 初始化并加载模型
    try:
        PolicyClass = get_policy_class(args.env_name)
        
        # 处理 MultiGrid 特殊的 Observation Space (如果是 Dict)
        obs_shape = env.observation_space.shape
        if isinstance(env.observation_space, gym.spaces.Dict):
             obs_shape = env.observation_space['image'].shape
             # [注意] Policy 初始化时，如果是图像，通常需要把 (H,W,C) 转为 (C,H,W) 传入 shape 参数
             # 取决于你的 GridStudentPolicy 怎么写的，这里是一个常见坑
             if len(obs_shape) == 3:
                 obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        actor_critic = PolicyClass(obs_shape, env.action_space)
        actor_critic.to(device)
    except ImportError as e:
        print(f"Error importing Policy class: {e}")
        return

    if os.path.exists(args.model_path):
        print(f"Loading checkpoint...")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        state_dict = None
        # --- 智能加载逻辑 (Peeling) ---
        if 'runner_state_dict' in checkpoint:
            checkpoint = checkpoint['runner_state_dict']
        
        if 'agent_state_dict' in checkpoint:
            agents_dict = checkpoint['agent_state_dict']
            if 'agent' in agents_dict:
                state_dict = agents_dict['agent'] # Student
            else:
                state_dict = list(agents_dict.values())[0]
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        try:
            actor_critic.load_state_dict(state_dict, strict=False) # MultiGrid 建议 False，有时有冗余 key
            print("Model weights loaded.")
        except RuntimeError as e:
            print(f"[ERROR] Loading failed: {e}")
            return
    else:
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    actor_critic.eval()

    # 4. 采集循环
    collected_anchors = []
    
    # [修改点] 兼容不同的 Reset 方式
    obs = env.reset()
    if hasattr(env, 'reset_agent'):
        # 兼容你的 BipedalWalkerAdversarialEnv
        obs = env.reset_agent()
    
    rnn_hxs = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(1, 1).to(device)
    
    step_count = 0
    with torch.no_grad():
        while len(collected_anchors) < args.num_anchors:
            # 1. 处理观测 (转 Tensor, Transpose if needed)
            obs_tensor = process_obs(obs, device)
            
            # 2. 决策
            # 注意：MultiGrid 动作通常是离散的，act 输出可能是 index 或 vector
            # 这里我们假设 act 返回 (value, action, log_prob, rnn)
            _, action, _, rnn_hxs = actor_critic.act(
                obs_tensor, rnn_hxs, masks, deterministic=True
            )
            
            action_np = action.squeeze().cpu().numpy()
            
            # 3. 环境步进
            # MultiGrid 的 step 接受 int 或 scalar array
            if isinstance(env.action_space, gym.spaces.Discrete) and action_np.shape != ():
                 action_scalar = action_np.item() # tensor(2) -> 2
            else:
                 action_scalar = action_np

            next_obs, reward, done, info = env.step(action_scalar)
            
            # 4. 采集条件
            if step_count % args.sample_interval == 0 and not done:
                # [关键] 保存原始观测 (H,W,C)，不要保存 Tensor 转换后的
                raw_data = get_raw_obs_data(next_obs)
                collected_anchors.append(raw_data)
                print(f"\rCollecting: {len(collected_anchors)}/{args.num_anchors}", end="")

            # 5. Reset 处理
            if done:
                obs = env.reset()
                if hasattr(env, 'reset_agent'):
                    obs = env.reset_agent()
                rnn_hxs = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
            else:
                obs = next_obs
            
            step_count += 1

    print(f"\nCollection Complete.")

    # 5. 保存
    # 文件名自动适应
    env_clean_name = args.env_name.split('-')[0]
    if "MultiGrid" in args.env_name: env_clean_name = "MultiGrid" # 统称
    
    save_filename = f"anchors_{env_clean_name}.npy"
    # save_filename = f"anchors_{args.env_name}.npy" # 或者用全名
    
    save_path = os.path.join(args.output_dir, save_filename)
    
    anchors_np = np.array(collected_anchors)
    np.save(save_path, anchors_np)
    
    print(f"Saved to: {save_path}")
    print(f"Shape: {anchors_np.shape}") 
    # 对于 MultiGrid, 期望 Shape: (32, 7, 7, 3) 
    # Runner 读取时会 permute 成 (32, 3, 7, 7)

if __name__ == "__main__":
    main()