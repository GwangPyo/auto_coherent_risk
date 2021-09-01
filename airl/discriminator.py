from net.utils import Mlp
import torch.nn as nn
import torch as th
import torch.nn.functional as F


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

    def loss(self, policy_minibatch, expert_minibatch):
        """
        obs_pi, dones_pi, log_pi, next_obs_pi = policy_minibatch
        obs_exp, dones_exp, log_p_exp, next_obs_exp = expert_minibatch
        """
        logits_pi = self(*policy_minibatch)
        logits_expert = self(*expert_minibatch)
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(-logits_expert).mean()
        loss_disc = loss_pi + loss_exp
        with th.no_grad():
            acc_pi = (logits_pi < 0).float().mean().item()
            acc_exp = (logits_expert > 0).float().mean().item()
        return loss_disc, acc_pi, acc_exp




