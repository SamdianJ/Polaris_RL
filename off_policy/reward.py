
'''
modules for doing reward modelings including reward shaping, reward scaling, reward normalization
'''

class RewardModel:
    def __init__(self, env_name):
        self.env = env_name

    def apply(self, state ,reward):
        return reward
    
class RewardShaping(RewardModel):
    def __init__(self, env_name, shaping_scale):
        super().__init__(env_name)
        self.shaping_scale = shaping_scale

    def apply(self, state ,reward):
        return reward
    
class BipedalWalkerHardcore_RewardShaping(RewardModel):
    def __init__(self, env_name, shaping_scale):
        super().__init__(env_name)

    def apply(self, state, reward):
        if (reward <= -100):
            reward = -1
        return reward
