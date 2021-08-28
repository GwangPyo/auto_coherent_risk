import torch as th
import numpy as np


class SAC(object):
    def __init__(self, env, policy, ):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

