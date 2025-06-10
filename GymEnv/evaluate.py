import argparse
import os
import torch
import numpy as np
import gymnasium as gym

# 导入训练时用到的配置和 Runner/Env 接口
from train import get_hopper_v5_train_cfg
from on_policy import OnPolicyRunner
from GymEnv import GymVecEnv

def evaluate(model_path: str,
             env_name: str = "Hopper-v5",
             episodes: int = 1000,
             device: str = "cpu"):
    # 1. 准备配置
    cfg = get_hopper_v5_train_cfg()
    cfg["env_name"] = env_name
    cfg["num_envs"] = 1           # 评估时只用单环境
    cfg["device"] = device

    # 2. 创建环境和 Runner
    env = GymVecEnv(env_name, device=device, render_mode='human')  # 使用 Gym 接口创建环境
    runner = OnPolicyRunner(env=env, cfg=cfg, log_dir=None, device=device)

    # 3. 加载模型（仅权重，不加载优化器状态）
    runner.load(model_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    # 4. 多回合评估并渲染
    returns = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            # 模型输出动作
            with torch.no_grad():
                action = policy(obs.to(device))
            # 与训练时不同，这里我们用 Gym 接口来渲染画面
            obs, _, reward, terminated, info = env.step(action)
            # reward 可能是 Tensor 或 float
            r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
            ep_ret += r
            done = bool(terminated)
            env.render()   # 直接渲染到屏幕

        returns.append(ep_ret)
        print(f"Episode {ep:2d}  Return: {ep_ret:.2f}")

    avg_ret = np.mean(returns)
    print(f"\nAverage Return over {episodes} episodes: {avg_ret:.2f}")

    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser("SAC/TD3/PPO 模型评估与渲染")
    p.add_argument("--model-path", type=str, required=True,
                   help="训练保存的模型文件路径 (.pt)")
    p.add_argument("--env-name",   type=str, default="Hopper-v5")
    p.add_argument("--episodes",   type=int, default=5)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    evaluate(args.model_path,
             args.env_name,
             args.episodes,
             args.device)