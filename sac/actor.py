import torch.nn as nn
from torch.distributions import Normal
from net.utils import Mlp
import gym
import torch as th


LOGSTD_MIN = -8.
LOGSTD_MAX = 1.


class RescaleAction(nn.Module):
    def __init__(self, action_space):
        super(RescaleAction, self).__init__()
        self.action_space = action_space
        assert isinstance(self.action_space, gym.spaces.Box)
        self.action_scale = th.FloatTensor(0.5 * (self.action_space.high - self.action_space.low))
        self.action_bias = th.FloatTensor(0.5 * (self.action_space.high + self.action_space.low))
        self.action_scale.requires_grad_(False)
        self.action_bias.requires_grad_(False)

    def forward(self, action):
        return self.action_scale * action + self.action_bias

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class SACActor(nn.Module):
    def __init__(self, feature_dim, action_dim, action_scaler):
        super(SACActor, self).__init__()
        self.layers = Mlp(net_arch=[feature_dim, 256, 64], spectral_norm=False, layer_norm=True )
        self.mu = nn.Linear(64, action_dim)
        self.log_sigma = nn.Linear(64, action_dim)
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

    def distribution(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
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


