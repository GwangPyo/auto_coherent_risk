import torch as th
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class Cumsum(nn.Module):
    def forward(self, x):
        return x.cumsum(dim=-1)


class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            activation=nn.ReLU,
            layer_norm=True,
            spectral_norm=False,
            bias=True,
    ):
        super().__init__()
        # TODO: initialization
        fcs = []
        in_size = input_size
        self.activation = activation
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size, bias=bias)
            if spectral_norm:
                fc = nn.utils.spectral_norm(fc)
            self.add_module(f'fc{i}', fc)
            fcs.append(fc)
            if layer_norm:
                fcs.append(nn.LayerNorm(next_size))
            in_size = next_size
            try:
                fcs.append(self.activation(inplace=True))
            except TypeError:
                fcs.append(self.activation())
        self.fcs = nn.Sequential(*fcs)
        self.last_fc = nn.Linear(in_size, output_size, bias=bias)
        if spectral_norm:
            self.last_fc = nn.utils.spectral_norm(self.last_fc)

    def forward(self, input):
        h = self.fcs(input)
        output = self.last_fc(h)
        return output


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines=64, embedding_dim=7*7*64):
        super(CosineEmbeddingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * th.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)
        # Calculate cos(i * \pi * \tau).
        cosines = th.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)
        return tau_embeddings


class FractionProposalNetwork(nn.Module):
    def __init__(self, N=32, embedding_dim=7*7*64):
        super(FractionProposalNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, N)
        )
        self.N = N
        self.embedding_dim = embedding_dim

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


