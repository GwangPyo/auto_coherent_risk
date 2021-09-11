from net.nets import IQN
import torch.nn as nn
import torch as th
from net.utils import IQNLosses, _jit_calculate_quantile_huber_loss
from net.nets import QFeatureNet, ValueFeatureNet, ODEIQN
import torch
import numpy as np


sqrt2 = np.sqrt(2)


def phi(x):
    return 0.5 * (1. + th.erf(x/sqrt2))


def phi_inverse(x):
    return sqrt2 * (th.erfinv(2 * x - 1))


class TauGenerator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, shape):
        pass


class CVaRTauGenerator(TauGenerator):
    def __init__(self, device, alpha):
        super().__init__(device)
        self.alpha = alpha

    def __call__(self, shape):
        return self.alpha * th.rand(size=shape, device=self.device)


class WangTauGenerator(TauGenerator):
    def __init__(self, device, eta):
        super().__init__(device)
        if eta >= 0:
            print("The eta parameter is positive. This is risk-seeking policy!")
        self.eta = eta

    def __call__(self, shape):
        taus = th.rand(size=shape, device=self.device)
        taus = phi(phi_inverse(taus) + self.eta)
        return taus.detach()


class PowerTauGenerator(WangTauGenerator):
    def __init__(self, device, eta):
        super(PowerTauGenerator, self).__init__(device, eta)
        self.eta_form = 1./(1 + np.abs(eta))

    def __call__(self, shape):
        taus = th.rand(size=shape, device=self.device)
        taus = 1 - th.pow(1. - taus, self.eta_form)
        return taus


def get_target_quantile(next_sa_quantiles, rewards, dones, gamma):
    with th.no_grad():
        next_q = next_sa_quantiles.mean(dim=-1)
        # Calculate target quantile values.
        rewards = rewards.flatten()
        rewards = th.stack([rewards] * next_sa_quantiles.shape[-1], dim=1)
        rewards = rewards[:, None, :]
        dones = dones.flatten()
        dones = th.stack([dones] * next_sa_quantiles.shape[-1], dim=1)
        dones = dones[:, None, :]
        target_sa_quantiles = rewards + (1.0 - dones) * gamma * next_sa_quantiles
        target_sa_quantiles = target_sa_quantiles.detach()
    return next_q, target_sa_quantiles


class IQNValueNet(nn.Module):
    def __init__(self,  feature_extractor_cls, feature_net_initializers,
                 N=16, N_dash=16, K=64, quantile_net_cls=IQN,
                 IQN_kwargs=None,):
        super(IQNValueNet, self).__init__()
        self.feature_extractor = feature_extractor_cls(**feature_net_initializers)
        assert  hasattr(self.feature_extractor, 'feature_dim')
        self.feature_dim = self.feature_extractor.feature_dim
        self.N = N
        self.N_dash = N_dash
        self.K = K
        if IQN_kwargs is None:
            IQN_kwargs = {}
        self.quantile_net = quantile_net_cls(self.feature_dim, num_actions=1, K=K, **IQN_kwargs)
        self._quantile_huber_loss = None
        self._get_target_quantile = None

    @staticmethod
    def get_default_tau(shape, device, dtype=th.float32,):
        taus = th.rand(
            size=shape, dtype=dtype,
            device=device)
        return taus

    def sac_calculate_iqn_loss(self, next_sa_quantiles, obs, actions, rewards,
                               dones, gamma):
        # Sample fractions.
        batch_size = obs.shape[0]
        taus = th.rand(
            batch_size, self.N, dtype=obs.dtype,
            device=obs.device)
        feature = self.feature_extractor(obs, actions)
        current_sa_quantiles = self.quantile_net.calculate_quantiles(taus, feature)
        assert current_sa_quantiles.shape == (batch_size, self.N, 1)
        gamma = th.Tensor([gamma]).to(obs.device)
        if self._get_target_quantile is None:
            self._get_target_quantile = th.jit.trace(get_target_quantile, (next_sa_quantiles, rewards, dones, gamma))

        with th.no_grad():
            if next_sa_quantiles.shape != (batch_size, 1,  self.N_dash):
                next_sa_quantiles = next_sa_quantiles.transpose(1, 2)
            assert next_sa_quantiles.shape == (batch_size, 1, self.N_dash)
            """
            next_q = next_sa_quantiles.mean(dim=-1)
            # Calculate target quantile values.
            rewards = rewards.flatten()
            rewards = th.stack([rewards] * self.N_dash, dim=1)
            rewards = rewards[:, None, :]
            dones = dones.flatten()
            dones = th.stack([dones] * self.N_dash, dim=1)
            dones = dones[:, None, :]
            target_sa_quantiles = rewards + (1.0 - dones) * gamma * next_sa_quantiles
            target_sa_quantiles = target_sa_quantiles.detach()
            """
            next_q, target_sa_quantiles = self._get_target_quantile(next_sa_quantiles, rewards, dones, gamma)
            assert target_sa_quantiles.shape == (batch_size, 1, self.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (batch_size, self.N, self.N_dash)
        if self._quantile_huber_loss is None:
            self._quantile_huber_loss = th.jit.trace(_jit_calculate_quantile_huber_loss, (td_errors.detach(), taus.detach()))
        quantile_huber_loss = self._quantile_huber_loss(td_errors, taus).mean()
        return quantile_huber_loss, next_q.detach().mean().item(), \
               td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)

    def quantile_huber_loss(self, predict_quantile, target_quantile, taus, reduce=True):

        assert predict_quantile.shape[-1] == 1 and len(predict_quantile.shape) == 3
        assert target_quantile.shape[1] == 1 and len(target_quantile.shape) == 3
        error = target_quantile - predict_quantile
        if self._quantile_huber_loss is None:
            self._quantile_huber_loss = th.jit.script(_jit_calculate_quantile_huber_loss)
        loss = self._quantile_huber_loss(error, taus)
        if reduce:
            loss = loss.mean()
        return loss


class IQNQNetwork(IQNValueNet):
    def __init__(self,  observation_space, action_space, N=16, N_dash=16, K=64, cvar_alpha=1.0):
        super(IQNQNetwork, self).__init__(
            feature_extractor_cls=QFeatureNet,
            feature_net_initializers={"observation_space": observation_space,
                                      "action_space": action_space,
                                      "normalize_action": True},
            N=N,
            N_dash=N_dash,
            K=K,
            quantile_net_cls=IQN,
            IQN_kwargs={"cvar_alpha":cvar_alpha}
        )

    def forward(self, obs, action, taus=None):
        z = self.feature_extractor(obs, action)
        return self.quantile_net(z, taus)

    def calculate_quantile(self, obs, actions, taus):
        feature = self.feature_extractor(obs, actions)
        return self.quantile_net.calculate_quantiles(taus=taus, feature=feature)

    def calculate_normalized_quantile(self, obs, actions, taus):
        feature = self.feature_extractor(obs, actions)
        return self.quantile_net.calculate_normalized_quantiles(taus=taus, feature=feature)


class ODEIQNQNetwork(IQNValueNet):
    def __init__(self,  observation_space, action_space, N=16, N_dash=16, K=64):
        super(ODEIQNQNetwork, self).__init__(
            feature_extractor_cls=QFeatureNet,
            feature_net_initializers={"observation_space": observation_space,
                                      "action_space": action_space,
                                      "normalize_action": True},
            N=N,
            N_dash=N_dash,
            K=K,
            quantile_net_cls=ODEIQN,

        )
    def forward(self, obs, action, taus=None):
        z = self.feature_extractor(obs, action)
        return self.quantile_net(z, taus)

    def calculate_quantile(self, obs, actions, taus):
        feature = self.feature_extractor(obs, actions)
        return self.quantile_net.calculate_quantiles(taus=taus, feature=feature)

    def calculate_normalized_quantile(self, obs, actions, taus):
        feature = self.feature_extractor(obs, actions)
        return self.quantile_net.calculate_normalized_quantiles(taus=taus, feature=feature)


class ODEIQNValueNetwork(IQNValueNet):
    def __init__(self, observation_space, N=16, N_dash=16, K=64):
        super(ODEIQNValueNetwork, self).__init__(
            feature_extractor_cls=QFeatureNet,
            feature_net_initializers={"observation_space": observation_space,
                                      "action_space": 1,
                                      "normalize_action": True},
            N=N,
            N_dash=N_dash,
            K=K,
            quantile_net_cls=ODEIQN,

        )

    def forward(self, obs, taus=None):
        z = self.feature_extractor(obs)
        return self.quantile_net(z, taus)

    def calculate_quantile(self, obs, taus):
        feature = self.feature_extractor(obs)
        return self.quantile_net.calculate_quantiles(taus=taus, feature=feature)

    def calculate_normalized_quantile(self, obs, taus):
        feature = self.feature_extractor(obs)
        return self.quantile_net.calculate_normalized_quantiles(taus=taus, feature=feature)


class RiskAversiveIQNQNetwork(IQNQNetwork):
    def __init__(self, observation_space, action_space, tau_generator_cls, tau_generator_kwargs, N=16, N_dash=16, K=64,):
        super(IQNQNetwork, self).__init__(
            feature_extractor_cls=QFeatureNet,
            feature_net_initializers={"observation_space": observation_space,
                                      "action_space": action_space,
                                      "normalize_action": True},
            N=N,
            N_dash=N_dash,
            K=K,
            quantile_net_cls=IQN,
            IQN_kwargs={"cvar_alpha": 1.0}
        )
        self.tau_generator = tau_generator_cls(**tau_generator_kwargs)

    def forward(self, obs, action, taus=None):
        if taus is None:
            taus = self.tau_generator()
        z = self.feature_extractor(obs, action)
        return self.quantile_net.forward(z, taus)


class IQNVNetwork(IQNValueNet):
    def __init__(self,  observation_space, N=16, N_dash=16, K=64, cvar_alpha=1.0):
        super(IQNVNetwork, self).__init__(
            feature_extractor_cls=ValueFeatureNet,
            feature_net_initializers={"observation_space": observation_space,
                                      "normalize_action": True},
            N=N,
            N_dash=N_dash,
            K=K,
            quantile_net_cls=IQN,
            IQN_kwargs={"cvar_alpha":cvar_alpha}
        )

    def forward(self, obs, taus=None):
        z = self.feature_extractor(obs)
        return self.quantile_net(z, taus)

    def calculate_quantile(self, obs, taus):
        feature = self.feature_extractor(obs)
        return self.quantile_net.calculate_quantiles(taus=taus, feature=feature)

    def calculate_normalized_quantile(self, obs, taus):
        feature = self.feature_extractor(obs)
        return self.quantile_net.calculate_normalized_quantiles(taus=taus, feature=feature)


class AutoriskIQNQNetwork(IQNValueNet):
    def __init__(self,  observation_space, action_space, N=16, N_dash=16, K=64):
        super(AutoriskIQNQNetwork, self).__init__(
            feature_extractor_cls=QFeatureNet,
            feature_net_initializers={"observation_space": observation_space,
                                      "action_space": action_space,
                                      "normalize_action": True},
            N=N,
            N_dash=N_dash,
            K=K,
            quantile_net_cls=IQN,
            IQN_kwargs={"init_uniform": False}
        )

    def forward(self, obs, action):
        z = self.feature_extractor(obs, action)
        return self.quantile_net(z)

    def calculate_quantile(self, obs, actions, taus):
        feature = self.feature_extractor(obs, actions)
        return self.quantile_net.calculate_quantiles(taus=taus, feature=feature)


class ProbQNet(nn.Module):
    def __init__(self, feature_net):
        super(ProbQNet, self).__init__()
        self.feature_net = feature_net
        self.feature_dim = 1

        self.linear = nn.Linear(self.feature_net.feature_dim, 1)

    def forward(self, obs, action):
        z = self.feature_net(obs, action)
        return self.linear(z)

    def to(self, device):
        self.linear.to(device)
        return super().to(device)

    def loss(self, target, obs, action, success, dones, next_obs, next_actions, next_logprob):
        current_q = self(obs, action)
        with th.no_grad():
            next_q = target(next_obs, next_actions)
            # entropy regularization
            next_q = next_q - next_logprob
            # NO discounting

            target_td = success + (1. - dones) * next_q
        assert current_q.shape == target_td.shape
        td_error = (0.5 * (target_td - current_q) ** 2).mean()
        return td_error



