import gym
from abc import abstractmethod, ABCMeta
import numpy as np
from functools import wraps


def rescale_action(method):
    @wraps(method)
    def _impl(self, action):
        action = self.preprocess_action(action)
        method_output = method(self, action)
        return method_output
    return _impl


class AbstractWrapper(gym.Env, metaclass=ABCMeta):
    def __init__(self, name: str):
        self.wrapped = gym.make(name)
        self.observation_space = self.wrapped.observation_space
        self.wrapped_action_space = self.wrapped.action_space

        self.action_scale = (self.wrapped_action_space.high - self.wrapped_action_space.low)/2
        self.action_low = self.wrapped_action_space.low + 1
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.wrapped_action_space.shape)
        tanh_action_scale = ((self.wrapped_action_space.low == -1).all() and (self.wrapped_action_space.high == 1).all())
        self.preprocess_action = lambda x: x if tanh_action_scale else self._preprocess_action

    def reset(self):
        return self.wrapped.reset()

    def _preprocess_action(self, action):
        action = self.action_scale * action + self.action_low
        return action

    @abstractmethod
    def step(self, action):
        pass

    def render(self, mode="human", **kwargs):
        return self.wrapped.render(mode, **kwargs)


class LunarLanderWrapper(AbstractWrapper):
    def __init__(self):
        super(LunarLanderWrapper, self).__init__("LunarLanderContinuous-v2")

    @rescale_action
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

    @rescale_action
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

    @rescale_action
    def step(self, action):
        next_obs, reward, done, info = self.wrapped.step(action)
        if done and reward == -100:
            info["is_success"] = False
        elif done and "TimeLimit.truncated" in info.keys():
            info["is_success"] = not (info["TimeLimit.truncated"])
        elif done:
            info["is_success"] = True

        return next_obs, reward, done, info


if __name__ == "__main__":
    wrapped = BipedalWalkerHardcoreWrapper()
    sample = wrapped.action_space.sample()

