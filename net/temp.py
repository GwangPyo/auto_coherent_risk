import matplotlib.pyplot as plt
import torch.nn as nn
from net.utils import Mlp
import torch as th
from torch.distributions import Categorical
import numpy as np


class SpectralRiskNet(nn.Module):
    def __init__(self, in_features, n_bins=10):
        super(SpectralRiskNet, self).__init__()
        self.n_bins = n_bins
        self.layers = nn.Sequential(Mlp(net_arch=[in_features, 64, 64], activation=nn.Mish, layer_norm=True, spectral_norm=True),
                                    nn.utils.spectral_norm(nn.Linear(64, n_bins)),
                                    nn.Softplus())
        # self._init_weight_to_uniform_distr()

    def forward(self, feature):
        neg_pdf = self.layers(feature)
        logits = 1. - neg_pdf.cumsum(dim=1)
        return Categorical(logits=logits)

    def _init_weight_to_uniform_distr(self):
        """
        The weight initialization to obtain uniform distribution (risk neutral sampling) at first.
        :return: None
        """
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                nn.init.uniform_(layer.weight, -1, 0)
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(-self.n_bins)

    def sample_from_feature(self, feature, sample_shape):
        distribution = self.forward(feature)
        samples = distribution.sample(sample_shape)
        return samples + 0.5

if __name__ == '__main__':
    net = SpectralRiskNet(64, 10)

    distribution = net.forward(th.randn((1, 64)))
    x = x.flatten().numpy()

    plt.hist(x.flatten())
    plt.show()