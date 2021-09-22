from net.utils import Mlp
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from jax import jit


class Template(nn.Module):
    def __init__(self, input_dim):
        super(Template, self).__init__()
        self.layers = nn.Sequential(Mlp(net_arch=[input_dim, 64, 64], layer_norm=True, spectral_norm=True),
                                    nn.utils.spectral_norm(nn.Linear(64, 1, bias=False)))

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, observation_dim, gamma):
        super(Discriminator, self).__init__()
        self.g = Template(observation_dim)
        self.h = Template(observation_dim)
        self.gamma = gamma

    def f(self, obs, dones, next_obs):
        r = self.g(obs)
        v_t = self.h(obs)
        v_tp1 = self.h(next_obs)
        return r + (1. - dones) * v_tp1 - v_t

    def forward(self, obs, dones, logp_pi, next_obs):
        return self.f(obs, dones, next_obs) - logp_pi

    def calculate_reward(self, obs, dones, logp_pi, next_obs):
        with th.no_grad():
            logits= self.forward(obs, dones, logp_pi, next_obs)
            return -F.logsigmoid(-logits)

    def loss(self, policy_minibatch):
        """
        obs_pi, dones_pi, log_pi, next_obs_pi, success = policy_minibatch
        """
        obs_pi, dones_pi, log_pi, next_obs_pi, success = policy_minibatch
        success = (success - 0.5) * 2   # False |-> -1, True |-> 1
        success = success.detach()
        logits = self(obs_pi, dones_pi, log_pi, next_obs_pi)
        loss = -F.logsigmoid(success * logits).mean()
        return loss

    def clip_weight(self, low=-0.01, high=0.01):
        for param in self.parameters():
            param.data = param.data.clamp(low, high)
        return



