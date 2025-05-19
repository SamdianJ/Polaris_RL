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
        self.state_encoder = build_mlp(dims = [state_dim, *net_dims]) #encoder
        self.mean_variance = build_mlp(dims = [*net_dims[-1], action_dim * 2]) #decoder for mean and variance
        layer_init_with_orthogonal(self.mean_variance[-1], gain=0.01)

    def forward(self, state):
        state_code = self.state_encoder(state)
        action_mean = self.mean_variance(state_code)[:, :self.action_dim]
        return action_mean.tanh()

    def get_action(self, state):
        state_code = self.state_encoder(state)
        action_mean, action_variance_log = self.mean_variance(state_code).chunk(2, dim=1)
        action_variance_log = action_variance_log.clamp(-16, 2).exp()

        distribution = torch.distributions.normal.Normal(action_mean, action_variance_log)
        return distribution.rsample().tanh()
    
    def sample(self, state):
        state_code = self.state_encoder(state)
        action_mean, action_variance_log = self.mean_variance(state_code).chunk(2, dim=1)
        action_variance_log = action_variance_log.clamp(-16, 2).exp()
        distribution = torch.distributions.normal.Normal(action_mean, action_variance_log)
        action = distribution.rsample().tanh()

        log_prob = distribution.log_prob(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1)
        return action, log_prob

class AgentSAC(AgentBase):
    def __init__(self, config: Config, target_entropy = None):
        super().__init__(config)       

        init_alpha = config.alpha
        self.log_alpha = nn.Parameter(torch.Tensor(np.log(init_alpha), device=self.device))
        self.target_entropy = -config.action_dim if target_entropy is None else target_entropy
        self.actor = ActorSAC(config.net_dims, config.state_dim, config.action_dim, config.max_action).to(self.device)
        self.critic = CriticTwin(self.net_dims, self.state_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor.get_action(state).cpu().data.numpy().flatten()
        return action
    
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
 
        with torch.no_grad():
            self.log_alpha.data.clamp_(16, 2)

        alpha = self.log_alpha.exp().detach()

        # optimize alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # optimize the actor
        new_action, log_prob = self.actor.sample(state)
        new_q1, new_q2 = self.critic(state, new_action)
        actor_loss = (alpha * log_prob - torch.min(new_q1, new_q2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        #soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    