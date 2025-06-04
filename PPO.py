import copy
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import Config, FileObject, ReplayBuffer
from DDPG_TD3 import AgentBase
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.utils.tensorboard import SummaryWriter

def layer_init_with_orthogonal(layer, std=np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def build_mlp(dims: List[int]) -> nn.Sequential:
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([layer_init_with_orthogonal(nn.Linear(dims[i], dims[i + 1])), nn.ReLU()])

    del net_list[-1]
    return nn.Sequential(*net_list)

class ActorPPO(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = build_mlp(dims = [state_dim, *net_dims, action_dim])
        self.max_action = max_action
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        return self.net(state)
    
    def get_action_and_value(self, state, action=None):
        if torch.isnan(state).any():
            print("NaN detected in state input to ActorPPO!")
    
        action_mean = self.net(state)
        action_logstd = self.log_std.expand_as(action_mean)

        action_std = torch.exp(action_logstd)
        
        base_dist = Normal(action_mean, action_std)
        
        # 采样动作
        if action is None:
            u = base_dist.sample()  # 在 (-1, 1) 范围内
            tanh_u = torch.tanh(u)
            action_to_env = tanh_u * self.max_action
            log_prob = base_dist.log_prob(u) - torch.log(1 - tanh_u.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1)
            entropy = base_dist.entropy().sum(dim=1)
            return action_to_env, log_prob, entropy
        else:
            normalized_action = action / self.max_action
            normalized_action = torch.clamp(normalized_action, -0.999, 0.999)
            u = torch.atanh(normalized_action)
            log_prob = base_dist.log_prob(u) - torch.log(1 - normalized_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1)
            entropy = base_dist.entropy().sum(dim=1)
            return action, log_prob, entropy

class CriticPPO(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims = [state_dim, *net_dims, 1])

    def forward(self, state):
        return self.net(state)
    
    def get_value(self, state):
        return self.net(state)
    
class ReplayBufferPPO:
    def __init__(self, config: Config):
        self.num_exploration_steps = config.num_exploration_steps
        self.num_mini_batch_size = config.on_policy_minibatch_size      
        self.num_envs = config.num_envs
        self.device = config.device
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.buffer_size = self.num_exploration_steps * self.num_envs
        self.states = torch.zeros((self.buffer_size, self.state_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.float32).to(self.device)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32).to(self.device)
        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32).to(self.device)   
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32).to(self.device)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32).to(self.device)
        self.sample_indices = np.arange(self.buffer_size)         

    def collect(self, state, action, log_prob, advantage, ret, value):
        flat_state = state.detach().reshape(-1, self.state_dim)
        self.states.copy_(flat_state)
        flat_action = action.detach().reshape(-1, self.action_dim)
        self.actions.copy_(flat_action) 
        self.log_probs.copy_(log_prob.detach().reshape(-1))
        self.advantages.copy_(advantage.detach().reshape(-1))
        self.returns.copy_(ret.detach().reshape(-1))
        self.values.copy_(value.detach().reshape(-1))
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def shuffle_indecies(self):
        np.random.shuffle(self.sample_indices)   

    def sample(self, minbatch_ind: int):
        start = minbatch_ind * self.num_mini_batch_size
        end = min(start + self.num_mini_batch_size, self.num_exploration_steps * self.num_envs)
        mb_inds = self.sample_indices[start: end]
        return (self.states[mb_inds],
                self.actions[mb_inds],
                self.log_probs[mb_inds],
                self.advantages[mb_inds], 
                self.returns[mb_inds],
                self.values[mb_inds])
    
    def num_minibatches(self):
        return (self.buffer_size + self.num_mini_batch_size - 1) // self.num_mini_batch_size

class AgentPPO(AgentBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.num_envs = config.num_envs
        self.num_steps = config.num_exploration_steps
        self.gamma = config.gamma
        self.clip_coef = config.clip_coef
        self.use_clipped_value_loss = True
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.gae_lambda = config.gae_lambda
        self.max_grad_norm = config.max_grad_norm
        self.use_tensorboard = config.use_tensorboard
        
        self.policy = ActorPPO(self.net_dims, self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.critic = CriticPPO(self.net_dims, self.state_dim, self.action_dim).to(self.device)
        self.global_step = 0

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _, _ = self.policy.get_action_and_value(state)
        return action.cpu().data.numpy().flatten()
    
    def explore(self, envs, num_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.zeros((num_steps, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((num_steps, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        log_probs = torch.zeros((num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        values = torch.zeros((num_steps, self.num_envs), dtype=torch.float32).to(self.device)  
        advantages = torch.zeros((num_steps, self.num_envs), dtype=torch.float32).to(self.device)  
        
        s, _ = envs.reset()
        next_obs = torch.tensor(s).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        ep_returns = [0.0]*self.num_envs
        finished_returns = []
        for step in range(num_steps):
            self.global_step += self.num_envs

            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, log_prob, _ = self.policy.get_action_and_value(next_obs)
                value = self.critic.get_value(next_obs).flatten()
                values[step] = value
            actions[step] = action
            log_probs[step] = log_prob
            
            next_obs, reward, term, trun, _ = envs.step(action.cpu().numpy())
            for i in range(self.num_envs):
                ep_returns[i] += reward[i]
                if term[i] or trun[i]:
                    finished_returns.append(ep_returns[i])
                    ep_returns[i] = 0.0

            rewards[step] = torch.tensor(reward).to(self.device).view(-1)

            next_obs = torch.tensor(next_obs).to(self.device)
            next_done = torch.logical_or(torch.tensor(term), torch.tensor(trun)).to(self.device)

        with torch.no_grad():
            lastgaelam = 0
            next_value = self.critic.get_value(next_obs).view(-1)

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = torch.logical_not(next_done)
                    next_values = next_value
                else:
                    nextnonterminal = torch.logical_not(dones[t + 1])
                    next_values = values[t + 1].view(-1)
                delta = rewards[t] + self.gamma * next_values * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values

        print(f"Exploration completed: {num_steps} steps, {self.num_envs} environments.")
        if len(finished_returns)>0:
            avg_return = sum(finished_returns)/len(finished_returns)
            print(f"Eval over {len(finished_returns)} episodes\tavg_return: {avg_return:.2f}")
        else:
            print("No episode finished in this evaluation window")
        return obs, actions, log_probs, advantages, returns, values        

    def update(self, replay_buffer: ReplayBufferPPO, num_epochs: int):
        # Update policy
        clipfracs = []
        for _ in range(num_epochs):
            replay_buffer.shuffle_indecies()
            for batch in range(replay_buffer.num_minibatches()):
                states, actions, log_probs, advantages, returns, values = replay_buffer.sample(batch)
                _, new_log_probs, entropy = self.policy.get_action_and_value(states, actions)
                new_value = self.critic.get_value(states).view(-1)
                logratio = new_log_probs - log_probs
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                
                # Policy loss
                policy_loss1 = -advantages * ratio
                policy_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # value loss
                if self.use_clipped_value_loss:
                    value_loss_upclipped = (new_value - returns) ** 2
                    value_pred_clipped = values + (new_value - values).clamp(-self.clip_coef, self.clip_coef)
                    value_loss_clipped = (value_pred_clipped - returns) ** 2
                    value_loss_max = torch.max(value_loss_upclipped, value_loss_clipped)
                    value_loss = 0.5 * value_loss_max.mean()
                else:   
                    value_loss = 0.5 * ((new_value - returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                policy_loss = policy_loss - self.entropy_coef * entropy_loss
                value_loss = value_loss * self.value_loss_coef
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.policy.log_std.data.clamp_(-10, 2)  # std ∈ [e⁻¹⁰, e²]

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        # Log training information
        if self.use_tensorboard:
            writer = SummaryWriter(self.config.file_object.log_dir)
            writer.add_scalar("loss/policy_loss", policy_loss.item(), self.global_step)
            writer.add_scalar("loss/value_loss", value_loss.item(), self.global_step)
            writer.add_scalar("loss/entropy_loss", entropy_loss.item(), self.global_step)
            writer.add_scalar("loss/approx_kl", approx_kl.item(), self.global_step)
            writer.add_scalar("loss/old_approx_kl", old_approx_kl.item(), self.global_step)
            writer.add_scalar("loss/clipfrac", np.mean(clipfracs), self.global_step)
            writer.close()
        else:
            print("Global Step:", self.global_step)
            print(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, "
                  f"Entropy Loss: {entropy_loss.item():.4f}, Approx KL: {approx_kl.item():.4f}, "
                  f"Old Approx KL: {old_approx_kl.item():.4f}, Clip Fraction: {np.mean(clipfracs):.4f}")

    def save(self, filename: str):
        torch.save(self.policy.state_dict(), filename + "ppo_actor")
        torch.save(self.critic.state_dict(), filename + "ppo_critic")
        torch.save(self.policy_optimizer.state_dict(), filename + "ppo_actor_optimizer")
        torch.save(self.critic_optimizer.state_dict(), filename + "ppo_critic_optimizer")

    def load(self, filename: str):
        self.policy.load_state_dict(torch.load(filename + "ppo_actor"))
        self.critic.load_state_dict(torch.load(filename + "ppo_critic"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "ppo_actor_optimizer"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "ppo_critic_optimizer"))