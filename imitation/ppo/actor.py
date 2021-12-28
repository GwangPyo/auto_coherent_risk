import gym
import numpy as np
from abc import ABCMeta, abstractmethod
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from typing import Tuple, Union
from net.utils import MLP
import torch as th
from imitation.network.utils import reparameterize, atanh, calculate_log_pi, TanhNormal

LOG_STD_MINMAX = (-8, 1)


def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


class AbstractActor(nn.Module, metaclass=ABCMeta):
    def __init__(self, feature_dim, action_space):
        super(AbstractActor, self).__init__()
        self.feature_dim = feature_dim
        self.action_space = action_space
        self.action_dim = self.get_action_dim()


    @abstractmethod
    def get_action_dim(self) -> Union[int, Tuple[int]]:
        pass

    @abstractmethod
    def forward(self, obs) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def distribution(self, obs):
        pass

    @abstractmethod
    def sample(self, obs) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def evaluate_action(self, obs, action) -> Tuple[Tensor, Tensor]:
        pass


class StateDependentPolicy(AbstractActor):
    def __init__(self, feature_dim, action_space, net_arch=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super(StateDependentPolicy, self).__init__(feature_dim, action_space)
        self.net_arch = net_arch
        self.net = MLP(
            feature_dim,
            self.net_arch,
            self.action_dim * 2
        )

    def get_action_dim(self):
        return np.prod(self.action_space.shape)

    def forward(self, states):
        return th.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        action, logprob = reparameterize(means, log_stds.clamp_(*LOG_STD_MINMAX))
        return action, logprob, th.tanh(means)

    def distribution(self, obs):
        means, log_stds = self.net(obs).chunk(2, dim=1)
        return TanhNormal(means, log_stds.exp())

    def evaluate_action(self, states, actions):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return evaluate_lop_pi(means, log_stds, actions), None


class StateIndependentPolicy(AbstractActor):
    def __init__(self, feature_dim, action_space, net_arch=(256, 256)):
        self.feature_dim = feature_dim
        self.net_arch = net_arch
        assert isinstance(action_space, gym.spaces.Box)
        super(StateIndependentPolicy, self).__init__(feature_dim, action_space)

        self.net = MLP(self.feature_dim, net_arch, self.action_dim)
        self.log_stds = nn.Parameter(th.zeros(1, self.action_dim))

    def distribution(self, obs):
        return TanhNormal(self.net(obs), self.log_stds.exp())

    def get_action_dim(self):
        return np.prod(self.action_space.shape)

    def forward(self, states):
        return th.tanh(self.net(states))

    def sample(self, states):
        mean = self.net(states)
        action, logprob = reparameterize(self.net(states), self.log_stds)
        self.clamp_logstd()
        return action, logprob, th.tanh(mean)

    def evaluate_action(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions), None

    def clamp_logstd(self):
        th.clamp(self.log_stds, *LOG_STD_MINMAX)

