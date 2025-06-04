import PPO
import gymnasium as gym
import time
import os
import numpy as np
import envs
from utils import Config

if __name__ == "__main__":
    print(gym.envs.registry.keys())
    config = Config('PPO')
    config.env_name = "Hopper-v5"
    config.max_timesteps = 10000000
    config.learning_rate = 2e-4
    config.entropy_coef = 0.001
    config.on_policy_minibatch_size = 256
    config.num_envs = 8
    config.num_epochs = 5
    config.re_eval_config()
    

    replay_buffer = PPO.ReplayBufferPPO(config)

    print("==========================")
    print(f"Policy: {config.policy_name}, Env: {config.env_name}, NumEnvs: {config.num_envs}, Seed: {config.random_seed}")
    print("==========================")

    config.init_before_training()
    policy = PPO.AgentPPO(config)

    envs = gym.vector.SyncVectorEnv(
        [envs.make_env(config.env_name, config.random_seed + i) for i in range(config.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    start_time = time.time()
    while (policy.global_step < config.max_timesteps):
        obs, actions, log_probs, advatanges, returns, values = policy.explore(envs, config.num_exploration_steps)
        replay_buffer.collect(obs, actions, log_probs, advatanges, returns, values)
        policy.update(replay_buffer, config.num_epochs)

    model_path = os.path.join(config.file_object.model_dir, "seed-{}-".format(config.random_seed))
    policy.save(model_path)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")