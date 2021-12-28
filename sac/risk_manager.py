import torch as th
import numpy as np

sqrt2 = np.sqrt(2)


def phi(x):
    return 0.5 * (1. + th.erf(x/sqrt2))


def phi_inverse(x):
    return sqrt2 * (th.erfinv(2 * x - 1))


class TauGenerator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, shape):
        return th.sort(th.rand(size=shape, device=self.device), dim=-1)[0]


class CVaRTauGenerator(TauGenerator):
    def __init__(self, device, alpha):
        super().__init__(device)
        self.alpha = alpha

    def __call__(self, shape):
        return th.sort(self.alpha * th.rand(size=shape, device=self.device), dim=-1)[0]


class RandomCVaRTauGenerator(TauGenerator):
    def __init__(self, device, max_alpha=1.):
        super().__init__(device)
        self.max_alpha = max_alpha

    def __call__(self, shape):
        current_alpha = th.rand(size=(1, ), device=self.device) * self.max_alpha
        return th.sort(current_alpha * th.rand(size=shape, device=self.device), dim=-1)[0]

    def sample(self, shape):
        current_alpha = th.rand(size=(1, ), device=self.device) * self.max_alpha
        return th.sort(current_alpha * th.rand(size=shape, device=self.device), dim=-1)[0], current_alpha


class WangTauGenerator(TauGenerator):
    def __init__(self, device, eta):
        super().__init__(device)
        if eta >= 0:
            print("The eta parameter is positive. This is risk-seeking policy!")
        self.eta = eta

    def __call__(self, shape):
        with th.no_grad():
            taus = th.rand(size=shape, device=self.device)
            taus = phi(phi_inverse(taus) + self.eta)
            return th.sort(taus, dim=-1)[0]


class PowerTauGenerator(WangTauGenerator):
    def __init__(self, device, eta):
        super(PowerTauGenerator, self).__init__(device, eta)
        self.eta_form = 1./(1 + np.abs(eta))

    def __call__(self, shape):
        with th.no_grad():
            taus = th.rand(size=shape, device=self.device)
            taus = 1 - th.pow(1. - taus, self.eta_form)
            return th.sort(taus, dim=-1)[0]
