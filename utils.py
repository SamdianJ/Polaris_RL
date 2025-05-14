from typing import List, Tuple
import torch
import torch.nn as nn
import os
import numpy as np
import random
import gymnasium as gym

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
    def __init__(self, xml = None):
        '''Arguments for policy'''
        self.policy_name = 'TD3'

        '''Arguments for environment''' 
        #env_args = Env_Args(env_name="BipedalWalker-v3")
        env_args = Env_Args(env_name="Pendulum-v1")
        self.env_name = env_args()['env_name']
        self.state_dim = env_args()['state_dim']
        self.action_dim = env_args()['action_dim']
        self.max_action = env_args()['max_action']

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
        self.policy_noise = 0.2
        self.policy_noise_clip = 0.5
        if self.policy_name == 'TD3':
            self.policy_noise = self.policy_noise * self.max_action
            self.policy_noise_clip = self.policy_noise_clip * self.max_action

        # for SAC
        self.reward_scale = 1.0 

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

        '''device settings'''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''Arguments for IO'''
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.file_object = FileObject(path=self.file_path, policy=self.policy_name, env_name=self.env_name)

    def from_xml(self, config_file: str):
        '''Load config from xml file'''
        import xml.etree.ElementTree as ET
        tree = ET.parse(config_file)
        root = tree.getroot()

        def get_text(tag, default=None):
            el = root.find(tag)
            return el.text.strip() if el is not None and el.text else default

        # policy config
        self.policy_name = get_text('policy_name', "TD3")

        # env config
        self.env_name = get_text('env_name', "BipedalWalker-v3")
        env_args = Env_Args(env_name=self.env_name)
        self.env_name = env_args()['env_name']
        self.state_dim = env_args()['state_dim']
        self.action_dim = env_args()['action_dim']
        self.max_action = env_args()['max_action']

        self.env_args = {
            'env_name':    self.env_name,
            'state_dim':   self.state_dim,
            'action_dim':  self.action_dim,
            'max_action':  self.max_action,
        }

        # agents
        self.gamma             = float(get_text('gamma', '0.99'))
        self.exploration_noise  = float(get_text('exploration_noise', '0.1'))
        self.policy_noise      = float(get_text('policy_noise', '0.2'))
        self.policy_noise_clip = float(get_text('policy_noise_clip', '0.5'))
        self.reward_scale      = float(get_text('reward_scale', '1.0'))

        if self.policy_name == 'TD3':
            self.policy_noise = self.policy_noise * self.max_action
            self.policy_noise_clip = self.policy_noise_clip * self.max_action

        # random seed
        self.random_seed = int(get_text('random_seed', 1017))

        # net_dims
        net_dims_txt = get_text('net_dims', '256,256')
        self.net_dims = [int(x) for x in net_dims_txt.split(',') if x]

        # training
        self.max_timesteps      = int(get_text('max_timesteps', '1000000'))
        self.eval_frequency    = int(get_text('eval_frequency', '5000'))
        self.start_timesteps    = int(get_text('start_timesteps', '25000'))
        self.policy_freq      = int(get_text('policy_freq', '2'))
        self.learning_rate    = float(get_text('learning_rate', '0.0003'))
        self.soft_update_tau  = float(get_text('soft_update_tau', '0.005'))
        self.policy_freq      = int(get_text('policy_freq', '2'))
        self.batch_size       = int(get_text('batch_size', '256'))
        self.buffer_size      = int(get_text('buffer_size', '1000000'))

        # file object
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
    