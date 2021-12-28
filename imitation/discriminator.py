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
    def __init__(self, observation_dim, gamma, entropy_coef=1.):
        super(Discriminator, self).__init__()
        self.g = Template(observation_dim)
        self.h = Template(observation_dim)
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.optim_disc = th.optim.RMSprop(lr=3e-4, params=self.parameters())

    def f(self, obs, dones, next_obs):
        r = self.g(obs)
        v_t = self.h(obs)
        v_tp1 = self.h(next_obs)
        return r + (1. - dones) * v_tp1 - v_t

    def forward(self, obs, dones, logp_pi, next_obs):
        return self.f(obs, dones, next_obs) - self.entropy_coef * logp_pi

    def calculate_reward(self, obs, dones, logp_pi, next_obs):
        with th.no_grad():
            logits= self.forward(obs, dones, logp_pi, next_obs)
            return -F.logsigmoid(-logits)

    def clip_weight(self, low=-0.01, high=0.01):
        for param in self.parameters():
            param.data = param.data.clamp(low, high)
        return

    def update_disc(self, batch_trainee, batch_expert):
        obs, dones, log_pis, next_obs = batch_trainee
        obs_expert, dones_exp, log_pis_exp, next_obs_expert = batch_expert
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self(obs, dones, log_pis, next_obs)
        logits_exp = self(
            obs_expert, dones_exp, log_pis_exp, next_obs_expert)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_expert = -F.logsigmoid(logits_exp).mean()
        regularization_loss = th.square(loss_pi) + th.square(loss_expert)
        loss_disc = loss_pi + loss_expert + 0.0001 * regularization_loss

        with th.no_grad():
            acc_pi = (logits_pi < 0).float().mean().item()
            acc_exp = (logits_exp > 0).float().mean().item()

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()
        losses = {"disc_loss/expert_loss": loss_expert.item(), "disc_loss/pi_loss": loss_pi.item(),
                  "disc_loss/reg_loss": regularization_loss.item(),
                  "accuracy/pi_acc": acc_pi, "accuracy/expert_acc": acc_exp}
        return losses

"""
def loss(self, policy_minibatch):
    
    obs_pi, dones_pi, log_pi, next_obs_pi, success = policy_minibatch
    
    obs_pi, dones_pi, log_pi, next_obs_pi, success = policy_minibatch
    success = (success - 0.5) * 2   # False |-> -1, True |-> 1
    success = success.detach()
    logits = self(obs_pi, dones_pi, log_pi, next_obs_pi)
    loss = -F.logsigmoid(success * logits).mean()
    return loss
"""