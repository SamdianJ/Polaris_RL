import gymnasium as gym
import time
import numpy as np
import DDPG_TD3
import SAC
import argparse
import os
import utils

'''
All environments:
['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 
'MountainCarContinuous-v0', 'Pendulum-v1', 'Acrobot-v1', 
'phys2d/CartPole-v0', 'phys2d/CartPole-v1', 'phys2d/Pendulum-v0', 
'LunarLander-v3', 'LunarLanderContinuous-v3', 'BipedalWalker-v3', 
'BipedalWalkerHardcore-v3', 'CarRacing-v3', 'Blackjack-v1', 
'FrozenLake-v1', 'FrozenLake8x8-v1', 'CliffWalking-v0', 
'Taxi-v3', 'tabular/Blackjack-v0', 'tabular/CliffWalking-v0', 
'Reacher-v2', 'Reacher-v4', 'Reacher-v5', 
'Pusher-v2', 'Pusher-v4', 'Pusher-v5', 
'InvertedPendulum-v2', 'InvertedPendulum-v4', 'InvertedPendulum-v5', 
'InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v4', 
'InvertedDoublePendulum-v5', 'HalfCheetah-v2', 'HalfCheetah-v3', 
'HalfCheetah-v4', 'HalfCheetah-v5', 'Hopper-v2', 
'Hopper-v3', 'Hopper-v4', 'Hopper-v5', 
'Swimmer-v2', 'Swimmer-v3', 'Swimmer-v4', 
'Swimmer-v5', 'Walker2d-v2', 'Walker2d-v3', 
'Walker2d-v4', 'Walker2d-v5', 'Ant-v2', 
'Ant-v3', 'Ant-v4', 'Ant-v5', 'Humanoid-v2', 
'Humanoid-v3', 'Humanoid-v4', 'Humanoid-v5', 
'HumanoidStandup-v2', 'HumanoidStandup-v4', 'HumanoidStandup-v5', 
'GymV21Environment-v0', 'GymV26Environment-v0']
'''

def main():
    print(gym.envs.registry.keys())
    policy_name = "TD3"
    env_name = "Hopper-v5"
    seed = 0
    max_steps = 1000

    #creat evaluation env
    env = gym.make(env_name, render_mode="human")
    env.reset(seed = seed)
    env.action_space.seed(seed)

    config = utils.Config(policy_name)
    if env_name == "Humanoid-v5":
        config.net_dims = [512, 512]
    config.env_name = env_name
    config.re_eval_config()

    if policy_name == 'TD3':
        policy = DDPG_TD3.AgentTD3(config)
    elif policy_name == 'SAC':
        policy = SAC.AgentSAC(config)

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