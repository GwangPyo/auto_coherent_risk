import torch as th
from torch.distributions import Uniform, Normal, Categorical
from torch import nn
from net.spectral_risk_net import SpectralRiskNet


class EvaRUniform(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self._alpha = th.Tensor([alpha])
        self._uniform_01_evar = self._get_evar()
        print(self._uniform_01_evar)

    def _get_evar(self):
        logt = th.zeros(size=(1,))
        logt.requires_grad = True
        optim = th.optim.Adam([logt], lr=1e-3)
        for _ in range(10000):
            loss = self._evaluate_evar(th.exp(logt))
            optim.zero_grad()
            loss.backward()
            optim.step()
        return loss

    def _evaluate_evar(self, t):
        return t * th.log(t * (th.exp(1/t) - 1)) - t * th.log(self._alpha)

    @property
    def alpha(self):
        return self._alpha.item()

    def uniform_evar(self, low, high):
        assert low <= high
        return (high - low) * self._uniform_01_evar + low


