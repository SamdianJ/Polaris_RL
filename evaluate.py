import gymnasium as gym
import time
import numpy as np
import DDPG_TD3
import argparse
import os
import utils

def main():
    print(gym.envs.registry.keys())
    policy_name = "TD3"
    env_name = "Humanoid-v5"
    seed = 0
    max_steps = 1000

    #creat evaluation env
    env = gym.make(env_name, render_mode="human")
    env.reset(seed = seed)
    env.action_space.seed(seed)

    config = utils.Config()
    if env_name == "Humanoid-v5":
        config.net_dims = [512, 512]
    config.env_name = env_name
    config.re_eval_config()
    policy = DDPG_TD3.AgentTD3(config)

    # 加载参数
    #model_name = "TD3"
    #model_path = os.path.join(os.path.realpath(__file__), model_name)
    #print(f"Loading model from: {model_path}")
    #policy.load("policy_TD3_envBipedalWalker-v3/models/TD3")

    policy.load("policy_{}_env_{}/models/seed-{}-".format(policy_name, env_name, seed))

    # 演示运行
    obs, _ = env.reset()
    total_reward = 0.0
    for t in range(max_steps):
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