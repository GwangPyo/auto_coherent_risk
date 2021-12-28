import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch as th
from torch.nn import functional as F
from rl_utils.utils import differentiable_sort

class Cumsum(nn.Module):
    def forward(self, x):
        return x.cumsum(dim=-1)

class Negation(nn.Module):
    def forward(self, x):
        return -x


class FractionProposalNetwork(nn.Module):
    def __init__(self, device, N=32, embedding_dim=256):
        super(FractionProposalNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, N)
        )
        self.N = N
        self.embedding_dim = embedding_dim
        self.device = device

    def to(self, device, *args, **kwargs):
        self.device = device
        return super(FractionProposalNetwork, self).to(device, *args, **kwargs)

    def forward(self, state_embeddings):

        batch_size = state_embeddings.shape[0]

        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N)

        tau_0 = th.zeros(
            (batch_size, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        taus_1_N = th.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = th.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.N+1)

        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        assert tau_hats.shape == (batch_size, self.N)

        # Calculate entropies of value distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)
        return taus, tau_hats, entropies


class SpectralRiskNet(nn.Module):
    def __init__(self, in_features, n_bins=64, init_uniform=True, **kwargs):
        super(SpectralRiskNet, self).__init__()
        self.n_bins = n_bins
        self.float_n_bins_plus_eps = float(self.n_bins) + 1.
        self.layers = nn.Sequential(
                                    nn.Linear(in_features, 64, bias=False), nn.Tanh(),
                                    nn.Linear(64, n_bins))
        self.monotonize = nn.Sequential(nn.Softmax(dim=-1), Cumsum())
        self.linear = nn.Linear(in_features, n_bins, bias=False)
        self.scale = nn.Sequential(nn.Linear(n_bins, 1, bias=False), nn.ReLU()) # , Negation())
        self.bias = nn.Sequential(nn.Linear(n_bins, 1, bias=True))

        self.mid = 1/(2 * n_bins )
        self.maximum_entropy = np.log(self.n_bins)
        if init_uniform:
            self._init_weight_to_uniform_distr()

    def forward(self, feature):

        z = self.layers(feature)
        log_pdf = self.monotonize(z)
        z_prime = self.linear(feature)
        a = self.scale(z_prime)
        b = self.bias(z_prime)

        return a * log_pdf + b

    def _init_weight_to_uniform_distr(self):
        """
        The weight initialization to obtain uniform distribution (risk neutral sampling) at first.
        :return: None
        """
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                nn.init.uniform_(layer.weight, -self.n_bins, -3)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.fill_(-self.n_bins)

    def sample(self, feature, K):
        log_probs = self.forward(feature)
        distribution = Categorical(probs=th.softmax(log_probs, dim=-1))
        sample = distribution.sample((K, ))
        sample_ret = sample.transpose(0, 1)/self.n_bins + self.mid * th.rand((feature.shape[0], K), device=feature.device)
        sample_ret = differentiable_sort(sample_ret)
        log_prob_sample = distribution.log_prob(sample)
        return sample_ret + self.mid, log_prob_sample.transpose(0, 1)

    def entropy(self, feature):
        with th.no_grad():
            log_probs = self.forward(feature)
            distribution = Categorical(probs=th.softmax(log_probs, dim=-1))
        return distribution.entropy()

if __name__ == '__main__':
    net = SpectralRiskNet(4, 10, init_uniform=False)
    loss = net.sample(th.randn(64, 4), K=4)[1]
    print(loss.shape)
    optim = th.optim.Adam(net.parameters(), lr=3e-4)
    optim.zero_grad()
    loss.mean().backward()
    optim.step()
