import torch.nn as nn
from torch.distributions import Normal
from net.utils import Mlp
import gym
import torch as th


LOGSTD_MIN = -8.
LOGSTD_MAX = 2.


class RescaleAction(nn.Module):
    def __init__(self, action_space):
        super(RescaleAction, self).__init__()
        self.action_space = action_space
        assert isinstance(self.action_space, gym.spaces.Box)
        self.action_scale = 0.5 * th.FloatTensor(self.action_space.high - self.action_space.low)
        self.action_bias = 0.5 * th.FloatTensor(self.action_space.high + self.action_space.low)

    def forward(self, action):
        return self.action_scale * action + self.action_bias


class SACActor(nn.Module):
    def __init__(self, feature_dim, action_dim, action_scaler):
        super(SACActor, self).__init__()
        self.layers = Mlp(net_arch=[feature_dim, 64, 64],)
        self.mu = nn.utils.spectral_norm(nn.Linear(64, action_dim))
        self.log_sigma = nn.utils.spectral_norm(nn.Linear(64, action_dim))
        self.action_scaler = action_scaler

    def forward(self, obs):
        z = self.layers(obs)
        mean = self.mu(z)
        std = self.log_sigma(z)
        std = std.clamp(LOGSTD_MIN, LOGSTD_MAX)
        return mean, std

    def to(self, device):
        self.action_scaler.to(device)
        return nn.Module.to(self, device)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t)
        action = self.action_scaler(y_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= th.log(self.action_scaler.action_scale * (1 - y_t.pow(2)) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = self.action_scaler(mean)
        return action, log_prob, mean


