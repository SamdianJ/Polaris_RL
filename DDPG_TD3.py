import copy
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from gym import envs
import utils
from utils import Config, FileObject, ReplayBuffer

class AgentBase:
    '''Base class for DDPG and TD3'''
    def __init__(self, args: Config = Config()):
        self.net_dims = args.net_dims
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.max_action = args.max_action
        self.gamma = args.gamma
        self.policy_noise = args.policy_noise
        self.policy_noise_clip = args.policy_noise_clip
        self.policy_freq = args.policy_freq
        self.learning_rate = args.learning_rate
        self.tau = args.soft_update_tau
        self.device = args.device
        self.total_it = 0

'''net work factory'''

def build_mlp(dims: List[int]) -> nn.Sequential:
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])

    del net_list[-1]
    return nn.Sequential(*net_list)

def layer_init_with_orthogonal(layer, std=1.0, bias_const = 1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


'''AgentDDPG'''

class Actor(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = build_mlp(dims = [state_dim, *net_dims, action_dim])
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state).tanh()

class Critic(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim + action_dim, *net_dims, 1])

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim = 1)) #（batch, feature）


class AgentDDPG(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, args)
        self.explore_noise_std = 0.05

        self.actor = Actor(net_dims = net_dims, state_dim = state_dim, action_dim = action_dim).to(self.device)
        self.critic = Critic(net_dims = net_dims, state_dim = state_dim, action_dim = action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

    def select_action(self, state: np.array) -> np.array:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256):  
        self.total_it += 1

        state, action, reward, undone, next_state = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)
            
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (undone * self.gamma * target_q).detach()

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.get_q1_values(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Soft update of the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename: str):
        torch.save(self.actor.state_dict(), filename + "ddpg_actor")
        torch.save(self.critic.state_dict(), filename + "ddpg_critic")
        torch.save(self.actor_optim.state_dict(), filename + "ddpg_actor_optimizer")
        torch.save(self.critic_optim.state_dict(), filename + "ddpg_critic_optimizer")

    def load(self, filename: str):
        self.actor.load_state_dict(torch.load(filename + "ddpg_actor"))
        self.critic.load_state_dict(torch.load(filename + "ddpg_critic"))

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optim.load_state_dict(torch.load(filename + "ddpg_actor_optimizer"))
        self.critic_optim.load_state_dict(torch.load(filename + "ddpg_critic_optimizer"))


'''Agent TD3'''

class CriticTwin(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()   
        # Q1 net work
        self.net_q1 = build_mlp(dims = [state_dim + action_dim, *net_dims, 1])
        # Q2 net work
        self.net_q2 = build_mlp(dims = [state_dim + action_dim, *net_dims, 1])

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim = 1)
        q1 = self.net_q1(state_action)
        q2 = self.net_q2(state_action)
        return q1, q2

    def get_q1_values(self, state, action):
        state_action = torch.cat((state, action), dim = 1)
        q1 = self.net_q1(state_action)
        return q1  


class AgentTD3(AgentBase):
    def __init__(self, args: Config = Config()):
        super().__init__(args)

        self.actor = Actor(self.net_dims, self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticTwin(self.net_dims, self.state_dim, self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

    def select_action(self, state: np.array) -> np.array:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256):  
        self.total_it += 1

        state, action, reward, undone, next_state = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)
            
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (undone * self.gamma * target_q)

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.get_q1_values(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Soft update of the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename: str):
        torch.save(self.actor.state_dict(), filename + "td3_actor")
        torch.save(self.critic.state_dict(), filename + "td3_critic")
        torch.save(self.actor_optim.state_dict(), filename + "td3_actor_optimizer")
        torch.save(self.critic_optim.state_dict(), filename + "td3_critic_optimizer")

    def load(self, filename: str):
        self.actor.load_state_dict(torch.load(filename + "td3_actor"))
        self.critic.load_state_dict(torch.load(filename + "td3_critic"))

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optim.load_state_dict(torch.load(filename + "td3_actor_optimizer"))
        self.critic_optim.load_state_dict(torch.load(filename + "td3_critic_optimizer"))






