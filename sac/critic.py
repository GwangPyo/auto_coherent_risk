from net.nets import IQN
import torch.nn as nn
import torch as th
from net.utils import IQNLosses
from net.nets import QFeatureNet, ValueFeatureNet


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

        with th.no_grad():
            if next_sa_quantiles.shape != (batch_size, 1,  self.N_dash):
                next_sa_quantiles = next_sa_quantiles.transpose(1, 2)
            assert next_sa_quantiles.shape == (batch_size, 1, self.N_dash)
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
            assert target_sa_quantiles.shape == (batch_size, 1, self.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (batch_size, self.N, self.N_dash)
        quantile_huber_loss = IQNLosses.calculate_quantile_huber_loss(td_errors, taus, None, 1.0)
        return quantile_huber_loss, next_q.detach().mean().item(), \
               td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)

    @staticmethod
    def quantile_huber_loss(predict_quantile, target_quantile, taus):
        assert predict_quantile.shape[-1] == 1 and len(predict_quantile.shape) == 3
        assert target_quantile.shape[1] == 1 and len(target_quantile.shape) == 3
        error = target_quantile - predict_quantile
        return IQNLosses.calculate_quantile_huber_loss(error, taus, None, 1.0)


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




