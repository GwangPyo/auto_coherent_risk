import torch as th
import torch.nn as nn
from torch.distributions import Distribution, Normal
import numpy as np
from torch.nn.functional import logsigmoid


LOG2 = np.log(2)
LOGSTD_MIN_MAX = (-5, 2)


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
