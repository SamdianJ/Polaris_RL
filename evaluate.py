import gymnasium as gym
import time
import numpy as np
import DDPG_TD3
import argparse
import os
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",    default="TD3", help="Policy to use: TD3, DDPG or OurDDPG")
    parser.add_argument("--env",       default="BipedalWalker-v3", help="Gym env name")
    parser.add_argument("--seed",      type=int, default=0, help="Random seed")
    parser.add_argument("--model",     default="TD3", help="TD3")
    parser.add_argument("--max_steps", type=int, default=200000, help="演示最多步数")
    args = parser.parse_args()

    # 创建渲染环境
    env = gym.make(args.env, render_mode="human")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    # 环境维度
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    config = utils.Config()
    policy = DDPG_TD3.AgentTD3(config)

    # 加载参数
    #model_name = "TD3"
    #model_path = os.path.join(os.path.realpath(__file__), model_name)
    #print(f"Loading model from: {model_path}")
    #policy.load("policy_TD3_envBipedalWalker-v3/models/TD3")
    policy.load("TD3")

    # 演示运行
    obs, _ = env.reset()
    total_reward = 0.0
    for t in range(args.max_steps):
        # 通过 policy.select_action 获取动作
        action = policy.select_action(np.array(obs))

        # 交互并渲染
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if done:
            print(f"Episode done at step {t+1}, total reward {total_reward:.3f}")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()

if __name__ == "__main__":
    main()