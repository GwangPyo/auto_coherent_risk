from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from imitation.ppo.actor import AbstractActor
from rl_utils.utils import get_device
from imitation.env import NormalizedEnv
import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class Algorithm(ABC):
    actor: AbstractActor

    def __init__(self, env, device='auto', seed='auto', gamma=0.99):
        if isinstance(seed, int):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        env = self.wrap_env(env)
        self.learning_steps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.state_shape = np.prod(self.observation_space.shape)
        self.action_shape = np.prod(self.action_space.shape)
        if device == 'auto':
            device = get_device()
        self.device = device
        self.gamma = gamma

    def wrap_env(self, env):
        if not isinstance(env, VecEnv):
            if not isinstance(env, NormalizedEnv):
                env = NormalizedEnv(env)
            env = DummyVecEnv([lambda: env])
        else:
            if not isinstance(env.unwrapped, NormalizedEnv):
                env = NormalizedEnv(env)
        return env

    def explore(self, obs):
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi, _ = self.actor.sample(obs.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
