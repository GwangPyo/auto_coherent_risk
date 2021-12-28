import torch as th
import torch.nn as nn
from torch.distributions import Distribution, Normal
import numpy as np
from torch.nn.functional import logsigmoid
from collections import OrderedDict
from net.utils import MLP
from UMNN.models.UMNN.MonotonicNN import ActorMonotonicNN
from torch.autograd import grad as torch_grad

LOG2 = np.log(2)
LOGSTD_MIN_MAX = (-5, 2)
LOG2PI = np.log(2 * np.pi)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * LOG2PI * log_stds.size(-1)
    return gaussian_log_probs - th.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    noises = th.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = th.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    return 0.5 * (th.log(1 + x + 1e-6) - th.log(1 - x + 1e-6))


class SACActor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(nn.Linear(feature_dim, 256),
                                 nn.Mish(inplace=True),
                                 nn.Linear(256, 256))
        self.mean = nn.Linear(256, action_dim)
        self.logstd = nn.Linear(256, action_dim)

    def forward(self, obs):
        z = self.net(obs)
        mean = self.mean(z)
        logstd = self.logstd(z)
        logstd = logstd.clamp(*LOGSTD_MIN_MAX)
        std = th.exp(logstd)
        return mean, std

    def distribution(self, obs):
        mean, std = self.forward(obs)
        return self._distribution(mean, std)

    @staticmethod
    def _distribution(mean, std):
        return TanhNormal(mean, std)

    def sample(self, obs):
        mean, logstd = self.forward(obs)
        tanh_normal = self._distribution(mean, logstd)
        action, pre_tanh = tanh_normal.rsample()
        logprob = tanh_normal.log_prob(pre_tanh)
        logprob = logprob.sum(dim=1, keepdim=True)
        return action, logprob, th.tanh(mean)

    def evaluate_action(self, obs, action):
        mean, logstd = self.forward(obs)
        tanh_normal = self._distribution(mean, logstd)
        atanh_action = th.atanh(action)
        return tanh_normal.log_prob(atanh_action).sum(dim=1, keepdim=True), None


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(th.zeros_like(self.normal_mean, device=normal_mean.device),
                                      th.ones_like(self.normal_std, device=normal_std.device))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self, sample_shape=th.Size()):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()

        return th.tanh(pretanh), pretanh


class AlphaEmbeddingNet(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.layer = nn.Sequential( nn.Linear(1, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, embed_dim))

    def forward(self, alpha):
        return self.layer(alpha)


class AlphaSACActor(nn.Module):
    def __init__(self, feature_dim, action_dim, rollout_alpha='random'):
        super().__init__()
        self.query_net = AlphaEmbeddingNet(embed_dim=64)
        self.key_net = MLP(feature_dim, hidden_sizes=(64, 64), output_size=64)
        self.value_net = MLP(feature_dim, hidden_sizes=(64, 64), output_size=64)
        self.key_net = MLP(feature_dim, hidden_sizes=(64, 64), output_size=64)
        self.mean = nn.Linear(64, action_dim)
        self.logstd = nn.Linear(64, action_dim)
        self._rollout_alpha = rollout_alpha
        self.sqrt_dk = 8.

    def get_rollout_alpha(self):
        if self._rollout_alpha == 'random':
            return th.rand((1, 1))
        else:
            return self._rollout_alpha.reshape(-1, 1)

    def set_rollout_alpha(self, alpha):
        if alpha is not None:
            if isinstance(alpha, float):
                alpha = th.Tensor([alpha])
            self._rollout_alpha = alpha
        else:
            self._rollout_alpha = 'random'

    def forward(self, obs, alpha):
        Q = self.query_net(alpha)
        K = self.key_net(obs)
        V = self.value_net(obs)
        Q = Q[:, None, :]
        K = K[:, None, :]
        V = V[:, :, None]
        QKt = th.bmm(Q.transpose(1, 2), K)/self.sqrt_dk
        attention = th.bmm(th.softmax(QKt, dim=-1), V)
        attention = attention.view(-1, 64)

        mean = self.mean(attention)
        logstd = self.logstd(attention)
        logstd = logstd.clamp(*LOGSTD_MIN_MAX)
        std = th.exp(logstd)
        return mean, std

    def distribution(self, obs, alpha):
        mean, std = self.forward(obs, alpha)
        return self._distribution(mean, std)

    @staticmethod
    def _distribution(mean, std):
        return TanhNormal(mean, std)

    def sample(self, obs, alpha=None):
        if alpha is None:
            alpha = self.get_rollout_alpha()
            alpha = th.repeat_interleave(alpha, repeats=obs.shape[0], dim=0)
            alpha = alpha.to(obs.device)
        mean, logstd = self.forward(obs, alpha)
        tanh_normal = self._distribution(mean, logstd)
        action, pre_tanh = tanh_normal.rsample()
        logprob = tanh_normal.log_prob(pre_tanh)
        logprob = logprob.sum(dim=1, keepdim=True)
        return action, logprob, th.tanh(mean)

    def evaluate_action(self, obs, action, alpha):
        mean, logstd = self.forward(obs, alpha)
        tanh_normal = self._distribution(mean, logstd)
        atanh_action = th.atanh(action)
        return tanh_normal.log_prob(atanh_action).sum(dim=1, keepdim=True), None

    def calculate_grad_norm(self, obs, alpha):
        alpha.requires_grad = True
        mean, logstd = self.forward(obs, alpha)
        gradient = torch_grad(outputs=mean,
                               inputs=alpha, grad_outputs=th.ones_like(mean),
                               create_graph=True, retain_graph=True)[0]
        grad_norm = th.sqrt(th.sum(gradient ** 2, dim=1, keepdim=True) + 1e-12)
        return grad_norm


class MonotonicSACActor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = ActorMonotonicNN(feature_dim, action_dim)

    def get_rollout_alpha(self):
        if self._rollout_alpha == 'random':
            return th.rand((1, 1))
        else:
            return self._rollout_alpha.reshape(-1, 1)

    def set_rollout_alpha(self, alpha):
        if alpha is not None:
            if isinstance(alpha, float):
                alpha = th.Tensor([alpha])
            self._rollout_alpha = alpha
        else:
            self._rollout_alpha = 'random'

    def forward(self, obs, alpha):
        mean, logstd = self.net(obs, alpha)
        logstd = logstd.clamp(*LOGSTD_MIN_MAX)
        std = th.exp(logstd)
        return mean, std

    def distribution(self, obs, alpha):
        mean, std = self.forward(obs, alpha)
        return self._distribution(mean, std)

    @staticmethod
    def _distribution(mean, std):
        return TanhNormal(mean, std)

    def sample(self, obs, alpha=None):
        if alpha is None:
            alpha = self.get_rollout_alpha()
            alpha = th.repeat_interleave(alpha, repeats=obs.shape[0], dim=0)
            alpha = alpha.to(obs.device)
        mean, logstd = self.forward(obs, alpha)
        tanh_normal = self._distribution(mean, logstd)
        action, pre_tanh = tanh_normal.rsample()
        logprob = tanh_normal.log_prob(pre_tanh)
        logprob = logprob.sum(dim=1, keepdim=True)
        return action, logprob, th.tanh(mean)

    def evaluate_action(self, obs, action, alpha):
        mean, logstd = self.forward(obs, alpha)
        tanh_normal = self._distribution(mean, logstd)
        atanh_action = th.atanh(action)
        return tanh_normal.log_prob(atanh_action).sum(dim=1, keepdim=True), None
