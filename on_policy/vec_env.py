import torch
from typing import Tuple, Union
from abc import ABC, abstractmethod

class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buffer: torch.Tensor
    obs_buffer: torch.Tensor
    rew_buffer: torch.Tensor
    reset_buffer: torch.Tensor
    episode_length_buffer: torch.Tensor
    extras: dict
    device: torch.device

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass

    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass

