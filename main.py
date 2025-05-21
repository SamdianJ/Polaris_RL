import gymnasium as gym
import time
import numpy as np
import DDPG_TD3
import SAC
import utils
import os
from datetime import datetime
from reward import BipedalWalkerHardcore_RewardShaping

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def explore_env(env_name, policy, seed,num_episodes=10):
    """
    Explore the environment using the given policy.
    """
    env = gym.make(env_name)
    env.reset(seed = seed)

    print("==========================")
    print(f"Exploring {env_name} with policy {policy.__class__.__name__}")
    print("==========================")

    avg_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = env.step(action)
            avg_reward += reward
            done = terminated or truncated
    avg_reward /= num_episodes
    print(f"Episode {num_episodes} - Average Reward: {avg_reward:.2f}")
    env.close()
    return avg_reward

if __name__ == "__main__":
    print(gym.envs.registry.keys())
    config = utils.Config('SAC')
    #config.from_xml("config.xml")
    config.env_name = "BipedalWalkerHardcore-v3"
   
    if config.env_name == "Humanoid-v5":
        config.max_timesteps = 2000000
        config.net_dims = [512,512]
        config.soft_update_tau = 0.002
        config.learning_rate = 1e-4

    if config.env_name == "BipedalWalkerHardcore-v3":
        config.max_timesteps = 1000000
        config.net_dims = [256,256]
        config.soft_update_tau = 0.005
        config.learning_rate = 1e-4
        reward_shaping = BipedalWalkerHardcore_RewardShaping(config.env_name, shaping_scale=0.1)
    config.re_eval_config()
    print(config)
    
    print("==========================")
    print(f"Policy: {config.policy_name}, Env: {config.env_name}, Seed: {config.random_seed}")
    print("==========================")

    env = gym.make(config.env_name)
    env.reset(seed=config.random_seed)
    env.action_space.seed(config.random_seed)
    
    config.init_before_training()

    replay_buffer = utils.ReplayBuffer(config.buffer_size, config.state_dim, config.action_dim)
    #policy = DDPG_TD3.AgentTD3(config)
    policy = SAC.AgentSAC(config)

    Evaluations = [explore_env(config.env_name, policy, config.random_seed)]

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start_time = time.time()  
    result_path = os.path.join(config.file_object.result_dir, "seed-{}-".format(config.random_seed))  
    rb_path = os.path.join(config.file_object.rb_dir, "seed-{}-".format(config.random_seed))

    for t in range(config.max_timesteps):

        episode_timesteps += 1

        if t < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, config.max_action * config.exploration_noise, size=config.action_dim)
            ).clip(-config.max_action, config.max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        if config.env_name == "BipedalWalkerHardcore-v3":
            #'''Trick1 for BipedalWalkerHardCore: done or dead
            if (reward <= -100):
                done_bool = done_bool
            else:
                done_bool = 0
            #'''

            #'''Trick2 for BipedalWalkerHardCore
            reward_shaping.apply(state, reward)
            #'''

        replay_buffer.add(state, action, reward, done_bool, next_state)

        state = next_state
        episode_reward += reward

        if t >= config.start_timesteps:
            policy.update(replay_buffer, config.batch_size)   

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, _ = env.reset()
            episode_num += 1
            episode_timesteps = 0
            episode_reward = 0

        if (t + 1) % config.eval_frequency == 0:
            print(f"Evaluating policy at timestep {t + 1}...")
            Evaluations.append(explore_env(config.env_name, policy, config.random_seed))
            np.save(result_path, Evaluations)
            print(f"Evaluations saved to {result_path}/Evaluations.npy")
            print(f"Total Timesteps: {t + 1} - Time: {time.time() - start_time:.2f}s")

    model_path = os.path.join(config.file_object.model_dir, "seed-{}-".format(config.random_seed))
    policy.save(model_path)
    print(f"Policy saved to {model_path}")

    env.close()






