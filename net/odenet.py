import torch as th
import math
import torch.nn as nn
import numpy as np


def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z


class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = th.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = th.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return th.cat(flat_parameters)


class ODEAdjoint(th.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with th.no_grad():
            z = th.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape).item()
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with th.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else th.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else th.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else th.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return th.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with th.no_grad():
            # Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = th.zeros(bs, n_dim).to(dLdz)
            adj_p = th.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = th.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = th.bmm(th.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = th.cat((z_i.view(bs, n_dim), adj_z, th.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            # Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = th.bmm(th.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None


class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=th.Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]


class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)


def add_time_mlp(in_tensor, t):
    bs, h = in_tensor.shape

    return th.cat((in_tensor, t.expand(bs, 1)), dim=1)


class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.activation = nn.Mish(inplace=True)

    def forward(self, x, t):
        if not self.time_invariant:
            x = add_time_mlp(x, t)
            # x = th.cat((x, t), dim=-1)

        h = self.activation(self.lin1(x))
        h = self.activation(self.lin2(h))
        out = self.lin3(h)
        return out


class ODELinear(ODEF):
    def __init__(self, h_dim):
        super(ODELinear, self).__init__()
        self.lin1 = nn.Linear(h_dim+1, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.activation = nn.Mish(inplace=True)

    def forward(self, x, t):
        x = th.cat([x, t], dim=1)
        h = self.activation(self.lin1(x))
        out = self.lin2(h)
        return out


class ODEBlock(nn.Module):
    def __init__(self, h_dim, t_start=0.0, t_end=1.0, dt=0.25):
        super(ODEBlock, self).__init__()
        self.layer = NeuralODE(ODELinear(h_dim))
        self.times = th.arange(start=t_start, end=t_end, step=dt, dtype=th.float32,)

    def to(self, device):
        self.times = self.times.to(device)
        return super().to(device)

    def forward(self, x):
        for dt in self.times:
            x = self.layer(x, dt.to(x.device))
        return x


@th.jit.script
def sort_taus(taus: th.Tensor):
    taus = th.sort(taus, dim=1)[0]
    taus = taus.transpose(0, 1)
    taus = th.unsqueeze(taus, dim=-1)
    return taus


@th.jit.script
def diff_sort_taus(taus: th.Tensor):
    indices = th.argsort(taus, dim=1)
    taus = th.gather(taus, dim=1, index=indices)
    taus = taus.transpose(0, 1)
    taus = th.unsqueeze(taus, dim=-1)
    return taus


@th.jit.script
def identity(taus: th.Tensor):
    taus = taus.transpose(0, 1)
    taus = th.unsqueeze(taus, dim=-1)
    return taus


class ODEQuantileBlock(nn.Module):
    sorting = {"default": sort_taus, "diff_sort": diff_sort_taus, "do_nothing": identity}

    def __init__(self, feature_dim, sorting='default'):
        super(ODEQuantileBlock, self).__init__()
        self.ode_layer = NeuralODE(ODELinear(feature_dim))
        self.sort = ODEQuantileBlock.sorting[sorting]
        self.pointwise = nn.Linear(feature_dim, 1)

    def forward(self, feature, taus):
        taus = self.sort(taus)
        features = []
        for t in taus:
            feature = self.ode_layer(feature, t[None])
            features.append(feature)
        batch = th.stack(features, dim=1)
        out = self.pointwise(batch)
        return out

