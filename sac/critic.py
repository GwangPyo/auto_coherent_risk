import torch.nn as nn
import torch as th
from net.utils import MLP
from net.nets import CosineEmbeddingNetwork
from rl_utils.utils import differentiable_sort
import numpy as np
from UMNN.models.UMNN.MonotonicNN import MonotonicNN, CriticMonotonicNN

class AlphaEmbeddingNet(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.layer = nn.Sequential( nn.Linear(1, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, embed_dim))

    def forward(self, alpha):
        return self.layer(alpha)


class Qfunction(nn.Module):
    class Cumsum(nn.Module):
        def forward(self, x):
            return x.cumsum(dim=-1)

    def __init__(self, feature_dim, action_dim, net_arch, num_cosines):
        super(Qfunction, self).__init__()
        self.mlp = MLP(feature_dim + action_dim, net_arch, output_size=net_arch[-1])
        self.cosine_embedding_net = CosineEmbeddingNetwork(num_cosines, embedding_dim=net_arch[-1])
        self.qr_logits = nn.Sequential(nn.Linear(net_arch[-1], 256), nn.ReLU(), nn.Linear(256, 1))
        self.monotonize = nn.Sequential(nn.Softmax(dim=-1), Qfunction.Cumsum())
        self.embedding_dim = net_arch[-1]
        self.linear = nn.Linear(self.embedding_dim, num_cosines, bias=False)
        self.scale = nn.Sequential(nn.Linear(num_cosines, 1, bias=False), nn.Softplus())
        self.bias = nn.Sequential(nn.Linear(num_cosines, 1, bias=True))

    def embedding(self, feature, action):
        return self.mlp(th.cat((feature, action), dim=-1))

    def forward(self, feature, action, taus):
        batch_size = feature.shape[0]
        taus = differentiable_sort(taus)
        x = th.cat((feature, action), dim=1)
        z = self.mlp(x)
        embedded_tau = self.cosine_embedding_net(taus)
        N = taus.shape[1]
        z_ = z.view(-1, 1, self.embedding_dim)
        embedding = z_ * embedded_tau
        qr_logits = self.qr_logits(embedding)
        qr_logits = qr_logits.reshape(-1, N)
        qr_logits = self.monotonize(qr_logits)

        z_prime = self.linear(z)
        alpha = self.scale(z_prime)
        beta = self.bias(z_prime)
        quantiles = qr_logits * alpha + beta
        ret = quantiles.view(batch_size, N)
        return ret


class MonotonicIQN(nn.Module):
    def __init__(self, feature_dim, action_dim, net_arch):
        super().__init__()
        self.embed = MLP(feature_dim + action_dim, net_arch[:-1], net_arch[-1])
        self.monotone = MonotonicNN(net_arch[-1] + 1, list((net_arch[-1], net_arch[-1])))

    def forward(self, obs, action, taus):
        z = self.embed(th.cat((obs, action), dim=-1))
        n_quantile = taus.shape[-1]
        taus = taus.reshape(-1, 1)
        required = int( taus.shape[0]/ z.shape[0])
        z = th.repeat_interleave(z, required, dim=0)
        quantiles = self.monotone.forward(taus, z)
        return quantiles.reshape(-1, n_quantile)


class AlphaAttention(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super().__init__()
        self.query_net = AlphaEmbeddingNet(embed_dim=embedding_dim)
        self.key_net = MLP(feature_dim, hidden_sizes=(64, 64), output_size=embedding_dim)
        self.value_net = MLP(feature_dim, hidden_sizes=(64, 64), output_size=embedding_dim)
        self.sqrt_dk = np.sqrt(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, feature, alpha):
        Q = self.query_net(alpha)
        K = self.key_net(feature)
        V = self.value_net(feature)
        Q = Q[:, None, :]
        K = K[:, None, :]
        V = V[:, :, None]
        QKt = th.bmm(Q.transpose(1, 2), K)/self.sqrt_dk
        attention = th.bmm(th.softmax(QKt, dim=-1), V)
        attention = attention.view(-1, self.embedding_dim)
        return attention


class AlphaQfunction(nn.Module):
    class Cumsum(nn.Module):
        def forward(self, x):
            return x.cumsum(dim=-1)

    def __init__(self, feature_dim, action_dim, net_arch, num_cosines):
        super(AlphaQfunction, self).__init__()
        self.mlp = AlphaAttention(feature_dim + action_dim, net_arch[-1])
        self.cosine_embedding_net = CosineEmbeddingNetwork(num_cosines, embedding_dim=net_arch[-1])
        self.qr_logits = nn.Sequential(nn.Linear(net_arch[-1], 256), nn.ReLU(), nn.Linear(256, 1))
        self.monotonize = nn.Sequential(nn.Softmax(dim=-1), Qfunction.Cumsum())
        self.embedding_dim = net_arch[-1]
        self.linear = nn.Linear(self.embedding_dim, num_cosines, bias=False)
        self.scale = nn.Sequential(nn.Linear(num_cosines, 1, bias=False), nn.Softplus())
        self.bias = nn.Sequential(nn.Linear(num_cosines, 1, bias=True))

    def embedding(self, feature, action, alpha):
        return self.mlp(th.cat((feature, action), dim=-1), alpha)

    def forward(self, feature, action, taus, alpha):
        batch_size = feature.shape[0]
        taus = th.sort(taus, dim=0)[0]
        x = th.cat((feature, action), dim=1)
        z = self.mlp(x, alpha)
        embedded_tau = self.cosine_embedding_net(taus)
        N = taus.shape[1]
        z_ = z.view(-1, 1, self.embedding_dim)
        embedding = z_ * embedded_tau
        qr_logits = self.qr_logits(embedding)
        qr_logits = qr_logits.reshape(-1, N)
        qr_logits = self.monotonize(qr_logits)

        z_prime = self.linear(z)
        alpha = self.scale(z_prime)
        beta = self.bias(z_prime)
        quantiles = qr_logits * alpha + beta
        ret = quantiles.view(batch_size, N)
        return ret


class AlphaMonotonicIQN(nn.Module):
    def __init__(self, feature_dim, action_dim, net_arch, num_taus):
        super().__init__()
        self.embed = th.jit.script(AlphaAttention(int(feature_dim + action_dim), int(net_arch[-1])))
        self.monotone = MonotonicNN(net_arch[-1] + 1, list((net_arch[-1], net_arch[-1])), nb_steps=50)

    def forward(self, obs, action, taus, alpha):
        z = self.embed(th.cat((obs, action), dim=-1), alpha)
        with th.no_grad():
            n_quantile = taus.shape[-1]
            taus = th.sort(taus, dim=-1)[0]

            # taus_zero = th.zeros_like(taus)
            # taus_zero[:, 1:] = taus[:, :-1]
            # taus_zero = taus_zero.reshape(-1, 1)
            taus = taus.reshape(-1, 1)
            required = taus.shape[0] // z.shape[0]
            embed_shape = z.shape[-1]

        # z = th.repeat_interleave(z, required, dim=0)
        z = z[None].expand(required, -1, -1).reshape(-1, embed_shape)
        """
        quantiles_interval = self.monotone.forward(taus, z, taus_zero)
        quantiles_interval = quantiles_interval.reshape(-1, n_quantile)

        bias = th.zeros_like(quantiles_interval)
        bias[:, 1:] = th.cumsum(quantiles_interval[:, :-1], dim=-1)
        quantiles = quantiles_interval + bias
        """
        quantiles = self.monotone.forward(taus, z)
        return quantiles.reshape(-1, n_quantile)



class ScalarQfunction(nn.Module):
    def __init__(self, feature_dim, action_dim, net_arch=(64, 64)):
        super().__init__()
        self.layers = MLP(feature_dim + action_dim, net_arch, 1)

    def forward(self, obs, action):
        return self.layers(th.cat((obs, action), dim=-1))


class Discriminator(ScalarQfunction):
    def __init__(self, feature_dim, action_dim, net_arch=(256, 256)):
        super().__init__(feature_dim, action_dim, net_arch)
        self.layers = nn.Sequential(self.layers, nn.Sigmoid())


class Critics(nn.Module):
    def __init__(self, feature_dim, action_dim, n_nets, net_arch=(64, 64), num_cosines=64):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Qfunction(feature_dim, action_dim, net_arch=net_arch, num_cosines=num_cosines)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def embedding(self, obs, action):
        return self.nets[0].embedding(obs, action)

    def forward(self, obs, action, taus):
        quantiles = th.stack(tuple(net(obs, action, taus) for net in self.nets), dim=1)
        return quantiles


class ScalarCritic(nn.Module):
    def __init__(self, feature_dim, action_dim, n_nets, net_arch=(64, 64)):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = ScalarQfunction(feature_dim, action_dim, net_arch=net_arch)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def embedding(self, obs, action):
        return self.nets[0].embedding(obs, action)

    def forward(self, obs, action, taus):
        quantiles = th.stack(tuple(net(obs, action, taus) for net in self.nets), dim=1)
        return quantiles


class AlphaCritic(nn.Module):
    def __init__(self, feature_dim, action_dim, n_nets, net_arch=(64, 64), num_taus=64):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = th.jit.script(AlphaQfunction(int(feature_dim), int(action_dim), net_arch=net_arch, num_cosines=num_taus))
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def embedding(self, obs, action, alpha):
        return self.nets[0].embedding(obs, action, alpha)

    def forward(self, obs, action, taus, alpha):
        quantiles = th.stack(tuple(net(obs, action, taus, alpha) for net in self.nets), dim=1)
        return quantiles


