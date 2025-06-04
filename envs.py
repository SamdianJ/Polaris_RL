import gymnasium as gym
import numpy as np
import os

def make_env(env_name, seed=0, capture_video=False, run_name=None):
    def thunk():
        """
        Create a gym environment with the specified name and seed.
        """
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(
            env, 
            lambda reward: np.clip(reward, -10, 10)
        )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        return env

    return thunk

