from typing import List, Tuple
import torch
import torch.nn as nn
import os
import numpy as np
import random
import gymnasium as gym

'''basic utils for DRL'''

class FileObject:
    def __init__(self, path: str, policy: str, env_name: str):
        '''Arguments for IO'''
        self.file_dir = path
        if not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            print("switching to default path...")
            self.file_dir = os.path.realpath(__file__)
        self.working_dir = os.path.join(self.file_dir, f"policy_{policy}_env_{env_name}")
        self.rb_dir = os.path.join(self.working_dir, "replay_buffer")
        self.model_dir = os.path.join(self.working_dir, "models")
        self.result_dir = os.path.join(self.working_dir, "results")

    def make_dir(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.make_rb_dir()
            self.make_model_dir()
            self.make_result_dir()

    def make_rb_dir(self):
        if not os.path.exists(self.rb_dir):
            os.makedirs(self.rb_dir)

    def make_model_dir(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def make_result_dir(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

class Config:
    def __init__(self, policy = 'TD3', xml = None):
        '''Arguments for policy'''
        self.policy_name = policy

        '''Arguments for environment''' 
        env_args = Env_Args(env_name="BipedalWalker-v3")
        self.env_name = env_args()['env_name']
        self.state_dim = env_args()['state_dim']
        self.action_dim = env_args()['action_dim']
        self.max_action = env_args()['max_action']
        self.num_envs = int(1)

        self.env_args = {
            'env_name':    self.env_name,
            'state_dim':   self.state_dim,
            'action_dim':  self.action_dim,
            'max_action':  self.max_action,
        }
        
        '''random seed'''
        self.random_seed = int(0)

        '''Arguments for agents'''
        self.gamma = 0.99

        # for exploration action
        self.exploration_noise = 0.1
        self.noise_decay = 0.999 #for enhanced convergence
        self.policy_noise = 0.2
        self.policy_noise_clip = 0.5
        if self.policy_name == 'TD3':
            self.policy_noise = self.policy_noise * self.max_action
            self.policy_noise_clip = self.policy_noise_clip * self.max_action

        # for SAC
        self.reward_scale = 1.0 
        self.alpha = 0.2

        '''Arguments for training'''
        self.net_dims = [256, 256]
        self.start_timesteps = 25000
        self.eval_frequency = 5000
        self.max_timesteps = 1000000
        self.learning_rate = 3e-4
        self.soft_update_tau = 5e-3
        self.policy_freq = 2

        '''Arguments for off-policy replay buffer'''
        self.batch_size = int(256)
        self.buffer_size = int(1e6)

        '''Arguments for on-policy training'''
        self.num_exploration_steps = int(2048)
        self.on_policy_minibatch_size = int(32)
        self.num_epochs = 10
        #PPO
        self.clip_coef = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95

        '''device settings'''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''Arguments for IO'''
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.file_object = FileObject(path=self.file_path, policy=self.policy_name, env_name=self.env_name)

        '''Auguments for logging'''
        self.use_wandb = False
        self.use_tensorboard = False

    def from_xml(self, config_file: str):
        '''Load config from xml file'''
        import xml.etree.ElementTree as ET
        
        try:
            # Parse the XML file
            tree = ET.parse(config_file)
            root = tree.getroot()
            
            # Load policy arguments
            policy = root.find('policy')
            if policy is not None:
                self.policy_name = policy.get('name', self.policy_name)
            
            # Load environment arguments
            env = root.find('environment')
            if env is not None:
                self.env_name = env.get('name', self.env_name)
                self.num_envs = int(env.get('num_envs', self.num_envs))
            
            # Random seed
            seed = root.find('random_seed')
            if seed is not None:
                self.random_seed = int(seed.text)
            
            # Agent arguments
            agent = root.find('agent')
            if agent is not None:
                self.gamma = float(agent.get('gamma', self.gamma))
                
                # Exploration noise
                noise = agent.find('noise')
                if noise is not None:
                    self.exploration_noise = float(noise.get('exploration', self.exploration_noise))
                    self.noise_decay = float(noise.get('decay', self.noise_decay))
                    self.policy_noise = float(noise.get('policy', self.policy_noise))
                    self.policy_noise_clip = float(noise.get('clip', self.policy_noise_clip))
                
                # SAC specific
                sac = agent.find('sac')
                if sac is not None:
                    self.reward_scale = float(sac.get('reward_scale', self.reward_scale))
                    self.alpha = float(sac.get('alpha', self.alpha))
            
            # Training arguments
            training = root.find('training')
            if training is not None:
                # Network architecture
                net = training.find('network')
                if net is not None:
                    dims_text = net.get('dimensions', None)
                    if dims_text:
                        self.net_dims = [int(dim.strip()) for dim in dims_text.split(',')]
                
                self.start_timesteps = int(training.get('start_timesteps', self.start_timesteps))
                self.eval_frequency = int(training.get('eval_frequency', self.eval_frequency))
                self.max_timesteps = int(training.get('max_timesteps', self.max_timesteps))
                self.learning_rate = float(training.get('learning_rate', self.learning_rate))
                self.soft_update_tau = float(training.get('soft_update_tau', self.soft_update_tau))
                self.policy_freq = int(training.get('policy_freq', self.policy_freq))
            
            # Off-policy buffer
            off_policy = root.find('off_policy')
            if off_policy is not None:
                self.batch_size = int(off_policy.get('batch_size', self.batch_size))
                self.buffer_size = int(off_policy.get('buffer_size', self.buffer_size))
            
            # On-policy arguments
            on_policy = root.find('on_policy')
            if on_policy is not None:
                self.num_exploration_steps = int(on_policy.get('exploration_steps', self.num_exploration_steps))
                self.on_policy_minibatch_size = int(on_policy.get('minibatch_size', self.on_policy_minibatch_size))
                self.num_epochs = int(on_policy.get('num_epochs', self.num_epochs))
                
                # PPO specific
                ppo = on_policy.find('ppo')
                if ppo is not None:
                    self.clip_coef = float(ppo.get('clip_coef', self.clip_coef))
                    self.entropy_coef = float(ppo.get('entropy_coef', self.entropy_coef))
                    self.value_loss_coef = float(ppo.get('value_loss_coef', self.value_loss_coef))
                    self.max_grad_norm = float(ppo.get('max_grad_norm', self.max_grad_norm))
                    self.gae_lambda = float(ppo.get('gae_lambda', self.gae_lambda))
            
            # Device settings
            device = root.find('device')
            if device is not None:
                use_cuda = device.get('cuda', '').lower() == 'true'
                if use_cuda and torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
            
            # Logging arguments
            logging = root.find('logging')
            if logging is not None:
                self.use_wandb = logging.get('wandb', '').lower() == 'true'
                self.use_tensorboard = logging.get('tensorboard', '').lower() == 'true'
            
            # Re-evaluate environment dimensions based on potentially new env_name
            self.re_eval_config()
            
            # File object
            self.file_object = FileObject(
                path=self.file_path,
                policy=self.policy_name,
                env_name=self.env_name
            )
            
            print(f"Configuration loaded from {config_file}")
        except Exception as e:
            print(f"Error loading config from {config_file}: {e}")
            print("Using default configuration")
        
    def re_eval_config(self):
        '''Re-evaluate the config'''
        env_args = Env_Args(env_name=self.env_name)
        self.state_dim = env_args()['state_dim']
        self.action_dim = env_args()['action_dim']
        self.max_action = env_args()['max_action']

        self.env_args = {
            'env_name':    self.env_name,
            'state_dim':   self.state_dim,
            'action_dim':  self.action_dim,
            'max_action':  self.max_action,
        }
        self.file_object = FileObject(path=self.file_path,
                                      policy=self.policy_name,
                                      env_name=self.env_name)
             
    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        torch.set_default_dtype(torch.float32)
        self.file_object.make_dir()

    def __str__(self):
        """Return all configuration attributes and their values."""
        info_lines = ["Config:"]
        for attr, val in sorted(self.__dict__.items()):
            info_lines.append(f"  {attr}: {val}")
        return "\n".join(info_lines)

class Env_Args:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode=None)
        obs_sp = self.env.observation_space
        act_sp = self.env.action_space
       
        if hasattr(obs_sp, 'shape'):
            self.state_dim = int(np.prod(obs_sp.shape))
        else:
            #if discrete
            self.state_dim = obs_sp.n
        if hasattr(act_sp, 'shape'):
            self.action_dim = int(np.prod(act_sp.shape))
        else:
            #if discrete
            self.action_dim = act_sp.n
        #max_action
        if hasattr(act_sp, 'high'):
            self.max_action = float(act_sp.high.flatten()[0])
        else:
            self.max_action = 1.0
        self.env.close()
        self.env = None

    def __call__(self, *args, **kwds):
        env_args = {
            'env_name': self.env_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_action': self.max_action,
        }
        return env_args
    
class ReplayBuffer:  
    '''Replay buffer for off-policy algorithms'''
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.pointer = 0
        self.is_full = False
        self.cur_size = 0
        self.buffer_size = buffer_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.undones = np.zeros((buffer_size, 1))
        self.next_state = np.zeros((buffer_size, state_dim))

    def add(self, state, action, reward, done, next_state):
        self.states[self.pointer] = state
        self.actions[self.pointer]= action
        self.rewards[self.pointer] = reward
        self.undones[self.pointer] = 1 - done
        self.next_state[self.pointer] = next_state

        self.pointer = (self.pointer + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)   

    def sample(self, batch_size: int):
        ids = np.random.randint(0, self.cur_size, size=batch_size)

        return (
            torch.FloatTensor(self.states[ids]).to(self.device),
            torch.FloatTensor(self.actions[ids]).to(self.device),
            torch.FloatTensor(self.rewards[ids]).to(self.device),
            torch.FloatTensor(self.undones[ids]).to(self.device),
            torch.FloatTensor(self.next_state[ids]).to(self.device), 
        )
    
    def save(self, filename: str):
        np.savez(filename, states=self.states, actions=self.actions, rewards=self.rewards, undones=self.undones, next_state=self.next_state)
    