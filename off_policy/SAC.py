import copy
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import Config, FileObject, ReplayBuffer
from DDPG_TD3 import AgentBase, CriticTwin, build_mlp, layer_init_with_orthogonal

class ActorSAC(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_encoder = build_mlp(dims = [state_dim, *net_dims]) #encoder
        self.mean_variance = build_mlp(dims = [net_dims[-1], action_dim * 2]) #decoder for mean and variance
        layer_init_with_orthogonal(self.mean_variance[-1], std=0.01)

    def forward(self, state):
        state_code = self.state_encoder(state)
        # Assuming the first half is mean
        action_mean, _ = self.mean_variance(state_code).chunk(2, dim=-1)
        return torch.tanh(action_mean) * self.max_action

    def get_action(self, state):
        state_code = self.state_encoder(state)
        action_mean, action_variance_log = self.mean_variance(state_code).chunk(2, dim=-1)
        action_variance_log = action_variance_log.clamp(-16, 2).exp()

        distribution = torch.distributions.normal.Normal(action_mean, action_variance_log)
        return distribution.rsample().tanh()*self.max_action
    
    def sample(self, state):
        state_code = self.state_encoder(state)
        action_mean, action_log_std = self.mean_variance(state_code).chunk(2, dim=1)
        action_std = action_log_std.clamp(-16, 2).exp()
        distribution = torch.distributions.normal.Normal(action_mean, action_std)
        action_sample = distribution.rsample()
        action_sample_normalized = action_sample.tanh()
        action = action_sample_normalized * self.max_action

        log_prob_z = distribution.log_prob(action_sample)

        squash_correction = torch.log(1 - action_sample_normalized.pow(2) + 1e-6)
        log_prob = log_prob_z.sum(dim=-1, keepdim=True) - squash_correction.sum(dim=-1, keepdim=True)

        if self.max_action != 1.0:
            log_prob -= self.action_dim * torch.log(torch.tensor(self.max_action)).to(action.device)
        return action, log_prob

class AgentSAC(AgentBase):
    def __init__(self, config: Config, target_entropy = None):
        super().__init__(config)       

        init_alpha = config.alpha
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha), device=self.device))
        self.target_entropy = 1.0
        if target_entropy is None:
            self.target_entropy = -float(config.action_dim) 
        else:
            self.target_entropy = float(target_entropy)
        self.actor = ActorSAC(config.net_dims, config.state_dim, config.action_dim, config.max_action).to(self.device)
        self.critic = CriticTwin(self.net_dims, self.state_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            action = self.actor(state)
        else:
            action = self.actor.get_action(state) 
        return action.cpu().data.numpy().flatten()
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> Tuple[float, float]:
        self.total_it += 1

        state, action, reward, undone, next_state = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action, log_prob = self.actor.sample(next_state) #mean, log_pi
            
            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            alpha = self.log_alpha.exp()
            target_q = torch.min(target_q1, target_q2) - alpha * log_prob
            target_q = reward + undone * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

         # Freeze critic networks to avoid computing gradients for them during actor/alpha update
        for p in self.critic.parameters():
            p.requires_grad = False
 
        # optimize the actor and alpha
        new_action, new_log_prob = self.actor.sample(state)

        # optimize alpha
        alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        with torch.no_grad():
            self.log_alpha.data.clamp_(-20, 2)

        current_alpha_detached_for_actor = self.log_alpha.exp().detach()

        # optimize actor
        new_q1, new_q2 = self.critic(state, new_action)
        actor_loss = (current_alpha_detached_for_actor * new_log_prob - torch.min(new_q1, new_q2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Unfreeze critic networks
        for p in self.critic.parameters():
            p.requires_grad = True

        #soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()
    
    def save(self, filename: str):
        torch.save(self.actor.state_dict(), filename + "sac_actor")
        torch.save(self.critic.state_dict(), filename + "sac_critic")
        torch.save(self.actor_optim.state_dict(), filename + "sac_actor_optimizer")
        torch.save(self.critic_optim.state_dict(), filename + "sac_critic_optimizer")

    def load(self, filename: str):
        self.actor.load_state_dict(torch.load(filename + "sac_actor"))
        self.critic.load_state_dict(torch.load(filename + "sac_critic"))

        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optim.load_state_dict(torch.load(filename + "sac_actor_optimizer"))
        self.critic_optim.load_state_dict(torch.load(filename + "sac_critic_optimizer"))

    