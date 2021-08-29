import torch as th
from typing import List
import torch.nn as nn


class Unit(nn.Module):
    def __init__(self, in_dim, out_dim, activation, layer_norm, spectral_norm):
        super(Unit, self).__init__()
        if spectral_norm:
            def Linear(in_dim, out_dim, *args, **kwargs):
                return nn.utils.spectral_norm(nn.Linear(in_dim, out_dim, *args, **kwargs))
        else:
            def Linear(in_dim, out_dim, *args, **kwargs):
                return nn.Linear(in_dim, out_dim, *args, **kwargs)
        if layer_norm:
            self.layers = nn.Sequential(Linear(in_dim, out_dim), nn.LayerNorm(out_dim), activation())
        else:
            self.layers = nn.Sequential(Linear(in_dim, out_dim), activation())

    def forward(self, x):
        return self.layers(x)


def Mlp(net_arch: List[int], activation=nn.Mish, layer_norm=True, spectral_norm=False):
    assert len(net_arch) > 1
    in_dimensions = net_arch[:-1]
    out_dimensions = net_arch[1:]
    sequences = []
    for indim, outdim in zip(in_dimensions, out_dimensions):
        sequences.append(Unit(indim, outdim, activation, layer_norm, spectral_norm))
    return nn.Sequential(*sequences)


class IQNLosses(object):
    @staticmethod
    def evaluate_quantile_at_action(s_quantiles, actions):
        assert s_quantiles.shape[0] == actions.shape[0]
        batch_size = s_quantiles.shape[0]
        N = s_quantiles.shape[1]
        # Expand actions into (batch_size, N, 1).
        action_index = th.stack([actions] * N, dim=1)
        action_index = action_index.reshape(batch_size, N, 1)
        # Calculate quantile values at specified actions.
        sa_quantiles = s_quantiles.gather(dim=2, index=action_index)
        return sa_quantiles

    @staticmethod
    def calculate_huber_loss(td_errors, kappa=1.0):
        return th.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa))

    @staticmethod
    def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
        assert not taus.requires_grad
        batch_size, N, N_dash = td_errors.shape

        # Calculate huber loss element-wisely.
        element_wise_huber_loss = IQNLosses.calculate_huber_loss(td_errors, kappa)
        assert element_wise_huber_loss.shape == (
            batch_size, N, N_dash)

        # Calculate quantile huber loss element-wisely.
        element_wise_quantile_huber_loss = th.abs(
            taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
        assert element_wise_quantile_huber_loss.shape == (
            batch_size, N, N_dash)

        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
            dim=1).mean(dim=1, keepdim=True)
        assert batch_quantile_huber_loss.shape == (batch_size, 1)

        if weights is not None:
            quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
        else:
            quantile_huber_loss = batch_quantile_huber_loss.mean()

        return quantile_huber_loss

    @staticmethod
    def dqn_calculate_iqn_loss(IQN_DQN, obs, actions, rewards, next_obs,
                               dones, weights=None, ):
        # Sample fractions.
        batch_size = obs.shape[0]
        taus = th.rand(
            batch_size, IQN_DQN.N, dtype=obs.dtype,
            device=obs.device)
        feature = obs  # self.q_network.dqn_network(state)
        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantiles = IQNLosses.evaluate_quantile_at_action(
            IQN_DQN.q_network.calculate_quantiles(taus, feature),
            actions)
        assert current_sa_quantiles.shape == (batch_size, IQN_DQN.N, 1)

        with th.no_grad():
            next_feature = next_obs  # self.target_q_network.dqn_network(next_states)
            next_q = IQN_DQN.target_q_network.calculate_q(next_feature)
            # Calculate greedy actions.
            next_actions = th.argmax(next_q, dim=1, keepdim=True)
            # Calculate features of next states.

            # Sample next fractions.
            tau_dashes = th.rand(
                batch_size, IQN_DQN.N_dash, dtype=th.float32,
                device=obs.device)
            # Calculate quantile values of next states and next actions.

            next_sa_quantiles = IQNLosses.evaluate_quantile_at_action(
                IQN_DQN.target_q_network.calculate_quantiles(tau_dashes, next_feature), next_actions).transpose(1, 2)
            assert next_sa_quantiles.shape == (batch_size, 1, IQN_DQN.N_dash)

            # Calculate target quantile values.
            rewards = rewards.flatten()
            rewards = th.stack([rewards] * IQN_DQN.N_dash, dim=1)
            rewards = rewards[:, None, :]
            dones = dones.flatten()
            dones = th.stack([dones] * IQN_DQN.N_dash, dim=1)
            dones = dones[:, None, :]
            target_sa_quantiles = rewards + (1.0 - dones) * IQN_DQN.gamma * next_sa_quantiles
            assert target_sa_quantiles.shape == (batch_size, 1, IQN_DQN.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (batch_size, IQN_DQN.N, IQN_DQN.N_dash)

        quantile_huber_loss = IQNLosses.calculate_quantile_huber_loss(td_errors, taus, weights, 1.0)

        return quantile_huber_loss, next_q.detach().mean().item(), \
               td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)
