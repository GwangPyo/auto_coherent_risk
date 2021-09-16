import torch.nn as nn
import torch as th
from net.utils import MLP
from net.odenet import ODEQuantileBlock
from net.nets import CosineEmbeddingNetwork


class ODEQfunction(nn.Module):
    def __init__(self, feature_dim, action_dim, net_arch=(64, 64)):
        super(ODEQfunction, self).__init__()
        self.linear = MLP(feature_dim + action_dim, net_arch[:-1], net_arch[-1])
        self.quantile_net = ODEQuantileBlock(net_arch[-1], )

    def forward(self, obs, action, taus):
        qf_input = th.cat((obs, action), dim=1)
        z = self.linear(qf_input)
        quantiles = self.quantile_net(z, taus)
        return th.squeeze(quantiles)


class Qfunction(nn.Module):
    def __init__(self, feature_dim, action_dim, net_arch=(256, 256), num_cosines=64):
        super(Qfunction, self).__init__()
        self.linear = MLP(feature_dim + action_dim, net_arch[:-1], net_arch[-1])
        self.cosine_embedding_net = CosineEmbeddingNetwork(num_cosines, embedding_dim=net_arch[-1])
        self.quantile_net = nn.Sequential(nn.Linear(net_arch[-1], 256), nn.Mish(), nn.Linear(256, 1))

    def forward(self, obs, action, taus):

        qf_input = th.cat((obs, action), dim=1)
        z = self.linear(qf_input)
        z = th.unsqueeze(z, dim=1)

        # z.shape == (batch_size, 1, embed_dim)
        embedded_tau = self.cosine_embedding_net(taus)
        # embedded_taus.shape == (batch_size, N, embed_dim)
        N = taus.shape[1]
        hadamard = z * embedded_tau

        quantiles = self.quantile_net(hadamard)

        quantiles = quantiles.view(-1, N)

        return quantiles


class Critics(nn.Module):
    def __init__(self, feature_dim, action_dim, n_nets, net_arch=(256, 256), num_cosines=64):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Qfunction(feature_dim, action_dim, net_arch=net_arch, num_cosines=num_cosines)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, obs, action, taus):
        quantiles = th.stack(tuple(net(obs, action, taus) for net in self.nets), dim=1)
        return quantiles


class ODECritics(nn.Module):
    def __init__(self, feature_dim, action_dim, n_nets, net_arch=(64, 64)):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = ODEQfunction(feature_dim, action_dim, net_arch=net_arch)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, obs, action, taus):
        quantiles = th.stack(tuple(net(obs, action, taus) for net in self.nets), dim=1)
        return quantiles

