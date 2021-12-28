import torch as th
from typing import List
import torch.nn as nn
from torch.jit import Final


class Unit(nn.Module):
    def __init__(self, in_dim, out_dim, activation, layer_norm, spectral_norm, batch_norm):
        super(Unit, self).__init__()
        if spectral_norm:
            def Linear(in_dim, out_dim, *args, **kwargs):
                return nn.utils.spectral_norm(nn.Linear(in_dim, out_dim, *args, **kwargs))
        else:
            def Linear(in_dim, out_dim, *args, **kwargs):
                return nn.Linear(in_dim, out_dim, *args, **kwargs)
        if layer_norm:
            self.layers = nn.Sequential(Linear(in_dim, out_dim), nn.LayerNorm(out_dim), activation())
        elif batch_norm:
            self.layers = nn.Sequential(Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), activation())
        else:
            self.layers = nn.Sequential(Linear(in_dim, out_dim), activation())

    def forward(self, x):
        return self.layers(x)


def Mlp(net_arch: List[int], activation=nn.Mish, layer_norm=True, batch_norm=False, spectral_norm=False):
    assert len(net_arch) > 1
    in_dimensions = net_arch[:-1]
    out_dimensions = net_arch[1:]
    sequences = []
    for indim, outdim in zip(in_dimensions, out_dimensions):
        sequences.append(Unit(indim, outdim, activation, layer_norm, spectral_norm, batch_norm))
    return nn.Sequential(*sequences)



element_wise_huberloss = nn.SmoothL1Loss(reduction='none')



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
    def calculate_huber_loss(td_errors):
        return th.where(
            td_errors.abs() <= 1.,
            0.5 * td_errors.pow(2),
            (td_errors.abs() - 0.5))

    @staticmethod
    def calculate_signed_huber_loss(td_errors):
        return th.where(
            td_errors.abs() <= 1.,
            0.5 * td_errors * th.abs(td_errors),
            (td_errors - 0.5) * (th.sign(td_errors)))


    @staticmethod
    def calculate_quantile_huber_loss(td_errors, taus, reduce=True):
        # assert not taus.requires_grad
        batch_size, N, N_dash = td_errors.shape
        # Calculate huber loss element-wisely.
        element_wise_huber_loss = IQNLosses.calculate_huber_loss(td_errors)
        assert element_wise_huber_loss.shape == (
            batch_size, N, N_dash)

        # Calculate quantile huber loss element-wisely.
        element_wise_quantile_huber_loss = th.abs(
            taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss
        assert element_wise_quantile_huber_loss.shape == (
            batch_size, N, N_dash)

        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
            dim=1).mean(dim=1, keepdim=True)
        assert batch_quantile_huber_loss.shape == (batch_size, 1)

        if reduce:
            quantile_huber_loss = batch_quantile_huber_loss.mean()
        else:
            quantile_huber_loss = batch_quantile_huber_loss
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

        quantile_huber_loss = IQNLosses.calculate_quantile_huber_loss(td_errors, taus, weights)

        return quantile_huber_loss, next_q.detach().mean().item(), \
               td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)


def calculate_huber_loss(td_errors, kappa=1.0):
    return th.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def quantile_huber_loss(
    current_quantiles: th.Tensor,
    target_quantiles: th.Tensor,
    taus: th.Tensor,
    sum_over_quantiles: bool = True,
) -> th.Tensor:

    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
    loss = th.abs(taus - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()
    return loss


def tqc_quantile_huber_loss(td_errors, taus):
    # assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape
    taus = taus.reshape(batch_size, 1, -1, 1)
    # Calculate huber loss element-wisely.
    element_wise_huber_loss = IQNLosses.calculate_huber_loss(td_errors)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = th.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
    ) * element_wise_huber_loss
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)
    loss = batch_quantile_huber_loss
    return loss


def calculate_fraction_loss(sa_quantiles, sa_quantile_hats, taus, weights=None):
    assert not sa_quantiles.requires_grad
    assert not sa_quantile_hats.requires_grad
    sa_quantiles = sa_quantiles[:, 1:-1]
    sa_quantiles = sa_quantiles.unsqueeze(-1)
    sa_quantile_hats = sa_quantile_hats.unsqueeze(-1)

    batch_size = sa_quantiles.shape[0]
    N = taus.shape[1] - 1
    assert sa_quantiles.shape == (batch_size, N-1, 1)

    # NOTE: Proposition 1 in the paper requires F^{-1} is non-decreasing.
    # I relax this requirements and calculate gradients of taus even when
    # F^{-1} is not non-decreasing.

    values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
    signs_1 = sa_quantiles > th.cat([
        sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
    assert values_1.shape == signs_1.shape

    values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
    signs_2 = sa_quantiles < th.cat([
        sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
    assert values_2.shape == signs_2.shape

    gradient_of_taus = (
        th.where(signs_1, values_1, -values_1)
        + th.where(signs_2, values_2, -values_2)
    ).view(batch_size, N-1)
    assert not gradient_of_taus.requires_grad
    assert gradient_of_taus.shape == taus[:, 1:-1].shape

    # Gradients of the network parameters and corresponding loss
    # are calculated using chain rule.
    if weights is not None:
        fraction_loss = ((
            (gradient_of_taus * taus[:, 1:-1]).sum(dim=1, keepdim=True)
        ) * weights).mean()
    else:
        fraction_loss = \
            (gradient_of_taus * taus[:, 1:-1]).sum(dim=1).mean()
    return fraction_loss


class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            activation=nn.Mish,
            layer_norm=True
    ):
        super().__init__()
        # TODO: initialization
        fcs = []
        in_size = input_size
        self.activation = activation
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            fcs.append(fc)
            if layer_norm:
                fcs.append(nn.LayerNorm(next_size))
            in_size = next_size
            fcs.append(self.activation(inplace=True))
        self.fcs = nn.Sequential(*fcs)
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = self.fcs(input)
        output = self.last_fc(h)
        return output

