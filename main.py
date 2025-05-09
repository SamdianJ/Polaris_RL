import gymnasium as gym
import time
import numpy as np
import DDPG_TD3
import utils

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

if __name__ == "__main__":
    #print(gym.envs.registry.keys())
    config = utils.Config()
    print(config.env_name)


