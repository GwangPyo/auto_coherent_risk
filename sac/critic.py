from net.nets import IQN
import torch.nn as nn
import torch as th
from net.utils import IQNLosses


class IQNHead(nn.Module):
    def __init__(self,  feature_net, N=8, N_dash=8, K=32,  IQN_kwargs=None, cvar_alpha=1.0):
        super(IQNHead, self).__init__()
        self.feature_dim = feature_net.feature_dim
        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.feature_extractor = feature_net
        self.cvar_alpha = cvar_alpha
        if IQN_kwargs is None:
            IQN_kwargs = {}
        self.Value = IQN(self.feature_dim, num_actions=1, cvar_alpha=self.cvar_alpha, **IQN_kwargs)

    def to(self, device):
        self.feature_extractor.to(device)
        return super().to(device)

    def forward(self, obs, action):
        z = self.feature_extractor(obs, action)
        return self.Value(z)

    def sac_calculate_iqn_loss(self, target, obs, actions, rewards, next_obs,
                               dones, next_actions, next_action_log_p_pi, ent_coeff, gamma):
        # Sample fractions.
        batch_size = obs.shape[0]
        taus = th.rand(
            batch_size, self.N, dtype=obs.dtype,
            device=obs.device)
        feature = self.feature_extractor(obs, actions)
        current_sa_quantiles = self.Value.calculate_quantiles(taus, feature)
        assert current_sa_quantiles.shape == (batch_size, self.N, 1)

        with th.no_grad():
            next_feature = target.feature_extractor(next_obs, next_actions.detach())
            next_q = target.Value.calculate_q(next_feature)

            # Sample next fractions.
            tau_dashes = self.cvar_alpha * th.rand(
                batch_size, self.N_dash, dtype=th.float32,
                device=obs.device)
            # Calculate quantile values of next states and next actions.

            next_sa_quantiles = (target.Value.calculate_quantiles(tau_dashes, next_feature)).transpose(1, 2)
            assert next_sa_quantiles.shape == (batch_size, 1, self.N_dash)

            # Calculate target quantile values.
            rewards = rewards.flatten()

            next_action_log_p_pi = next_action_log_p_pi.flatten().detach()
            next_action_log_p_pi = th.stack([next_action_log_p_pi] * self.N_dash, dim=1)
            next_action_log_p_pi = next_action_log_p_pi[:, None, :]
            next_sa_quantiles = next_sa_quantiles - ent_coeff * next_action_log_p_pi

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
