import gymnasium as gym
import torch
import numpy as np

# --- Minimal VecEnv for a single environment ---
class GymVecEnv:
    def __init__(self, env_name, device='cpu'):
        self.env = gym.make(env_name, render_mode=None) # Specify render_mode if needed by env
        self.num_envs = 1
        self.device = device

        obs_space = self.env.observation_space
        action_space = self.env.action_space

        self.num_obs = obs_space.shape[0]
        self.num_actions = action_space.shape[0]
        self.num_privileged_obs = None # Hopper-v5 typically doesn't have privileged obs

        self.max_episode_length = getattr(self.env.spec, 'max_episode_steps', 1000)
        
        # These buffers are expected by OnPolicyRunner or its components
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self._current_obs_tensor = None
        self._current_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._current_episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


    def reset(self, seed=None): # Matches runner's call: _, _ = self.env.reset()
        # gymnasium reset returns obs, info
        # The runner seems to discard the second return value from env.reset()
        obs, info = self.env.reset(seed=seed)
        self._current_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.episode_length_buf[0] = 0 # As per runner's expectation for its own buffer
        self._current_reward_sum[0] = 0
        self._current_episode_length[0] = 0
        return self._current_obs_tensor, info # Runner uses the first element

    def step(self, actions: torch.Tensor):
        # actions: (num_envs, num_actions)
        action_np = actions[0].cpu().numpy() # Action for the single env
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        done = terminated or truncated

        self._current_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        rewards_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor([done], dtype=torch.bool, device=self.device)

        self._current_reward_sum[0] += reward
        self._current_episode_length[0] += 1

        # The runner's log function expects `infos['episode']` if an episode ends
        if done:
            info['episode'] = {
                'r': self._current_reward_sum[0].item(),
                'l': self._current_episode_length[0].item(),
            }
            # VecEnv typically auto-resets, simulate this for consistency
            obs_reset, _ = self.env.reset() # Get new obs
            self._current_obs_tensor = torch.tensor(obs_reset, dtype=torch.float32, device=self.device).unsqueeze(0)
            self._current_reward_sum[0] = 0
            self._current_episode_length[0] = 0
            self.episode_length_buf[0] = 0 # Reset for runner's perspective

        # privileged_obs is None for Hopper
        return self._current_obs_tensor, None, rewards_tensor, dones_tensor, info

    def get_observations(self):
        return self._current_obs_tensor

    def get_privileged_observations(self):
        return None

    def close(self):
        self.env.close()