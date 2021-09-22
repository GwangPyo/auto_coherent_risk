import torch.nn as nn
from net.utils import Mlp
from torch.distributions import Categorical
import numpy as np
import torch as th


class SpectralRiskNet(nn.Module):
    def __init__(self, in_features, n_bins=10, init_uniform=True, **kwargs):
        super(SpectralRiskNet, self).__init__()
        self.n_bins = n_bins
        self.float_n_bins_plus_eps = float(self.n_bins) + 1.
        self.layers = nn.Sequential(
                                    nn.Linear(in_features, 64, bias=False), nn.Tanh(),
                                    nn.Linear(64, n_bins),
                                    nn.Softmax(dim=1))
        self.mid = 1/(2 * n_bins )
        self.maximum_entropy = np.log(self.n_bins)
        if init_uniform:
            self._init_weight_to_uniform_distr()

    def forward(self, feature):

        pdf = self.layers(feature)
        # differentiable sort
        indices = th.argsort(pdf, dim=1, descending=True)
        pdf = th.gather(pdf, dim=1, index=indices)

        return Categorical(probs=pdf)

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

    def sample(self, feature, sample_shape):
        distribution = self.forward(feature)
        sample = distribution.sample((sample_shape[-1],) )
        sample_ret = sample.transpose(0, 1)/(self.n_bins + 1) \
                  + self.mid * th.rand(size=sample_shape, device=feature.device)
        logprob = distribution.log_prob(sample).transpose(0, 1)

        return sample_ret + self.mid, logprob

    def entropy(self, feature):
        with th.no_grad():
            distribution = self.forward(feature)
        return distribution.entropy()
