import torch.nn as nn
from net.nets import CosineEmbeddingNetwork
from net.utils import MLP, quantile_huber_loss
import torch as th


class Vfunction(nn.Module):
    def __init__(self, feature_dim, net_arch=(256, 256), num_cosines=64, N=64):
        super(Vfunction, self).__init__()
        self.linear = MLP(feature_dim, net_arch[:-1], net_arch[-1])
        self.cosine_embedding_net = CosineEmbeddingNetwork(num_cosines, embedding_dim=net_arch[-1])
        self.quantile_net = nn.Sequential(nn.Linear(net_arch[-1], 256), nn.Mish(), nn.Linear(256, 1))
        self.N = N

    def forward(self, obs, taus=None):
        if taus is None:
            taus = th.randn(size=(obs.shape[0], self.N), device=obs.device)
        z = self.linear(obs)
        z = th.unsqueeze(z, dim=1)
        # z.shape == (batch_size, 1, embed_dim)
        embedded_tau = self.cosine_embedding_net(taus)
        # embedded_taus.shape == (batch_size, N, embed_dim)
        N = taus.shape[1]
        hadamard = z * embedded_tau
        quantiles = self.quantile_net(hadamard)
        quantiles = quantiles.view(-1, N)
        return quantiles

    @staticmethod
    def loss(predicts, targets, taus):
        predicts = predicts[:, :, None]
        targets = targets[:, None, :]
        td_error = th.abs(targets - predicts)
        return quantile_huber_loss(td_error, taus).mean()


class ScalarVfunction(nn.Module):
    def __init__(self, feature_dim, net_arch=(256, 256)):
        super().__init__()
        self.layers = MLP(feature_dim, net_arch, 1)

    def forward(self, obs):
        return self.layers(obs)

    @staticmethod
    def loss(predicts, targets, taus=None):
        return (predicts - targets).pow_(2).mean()


# Alias
QuantileCritic = Vfunction
Critic = ScalarVfunction

