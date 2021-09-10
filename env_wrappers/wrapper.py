import gym
from abc import abstractmethod, ABCMeta
import numpy as np


class AbstractWrapper(gym.Env, metaclass=ABCMeta):
    def __init__(self, name: str):
        self.wrapped = gym.make(name)
        self.observation_space = self.wrapped.observation_space
        self.action_space = self.wrapped.action_space

    def reset(self):
        return self.wrapped.reset()

    @abstractmethod
    def step(self, action):
        pass

    def render(self, mode="human", **kwargs):
        return self.wrapped.render(mode, **kwargs)


class LunarLanderWrapper(AbstractWrapper):
    def __init__(self):
        super(LunarLanderWrapper, self).__init__("LunarLanderContinuous-v2")

    def step(self, action):
        next_obs, reward, done, info = self.wrapped.step(action)
        if done and reward != 100:
            info["is_success"] = False
        elif done and reward == 100:
            info["is_success"] = True

        return next_obs, reward, done, info


class BipedalWalkerWrapper(AbstractWrapper):
    def __init__(self):
        super(BipedalWalkerWrapper, self).__init__("BipedalWalker-v3")

    def step(self, action):
        next_obs, reward, done, info = self.wrapped.step(action)
        if done and reward == -100:
            info["is_success"] = False
        elif done and "TimeLimit.truncated" in info.keys():
            info["is_success"] = not (info["TimeLimit.truncated"])
        elif done:
            info["is_success"] = True
        return next_obs, reward, done, info


class BipedalWalkerHardcoreWrapper(AbstractWrapper):
    def __init__(self):
        super(BipedalWalkerHardcoreWrapper, self).__init__("BipedalWalkerHardcore-v3")

    def step(self, action):
        next_obs, reward, done, info = self.wrapped.step(action)
        if done and reward == -100:
            info["is_success"] = False
        elif done and "TimeLimit.truncated" in info.keys():
            info["is_success"] = not (info["TimeLimit.truncated"])
        elif done:
            info["is_success"] = True

        return next_obs, reward, done, info

