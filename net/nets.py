import torch.nn as nn
import numpy as np
import torch as th
import gym
from net.utils import Mlp


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines=64, embedding_dim=64):
        super(CosineEmbeddingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.Mish(),
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


class QuantileNetwork(nn.Module):
    def __init__(self, num_actions, embedding_dim=64, dueling_net=False):
        super(QuantileNetwork, self).__init__()
        if not dueling_net:
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.LayerNorm(128),
                nn.Mish(),
                nn.Linear(128, num_actions),
            )
        else:
            self.advantage_net = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.LayerNorm(128),
                nn.Mish(),
                nn.Linear(128, num_actions),
            )
            self.baseline_net = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.LayerNorm(128),
                nn.Mish(),
                nn.Linear(128, 1),
            )

        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net

    def forward(self, state_embeddings, tau_embeddings):
        assert state_embeddings.shape[0] == tau_embeddings.shape[0]

        assert state_embeddings.shape[1] == tau_embeddings.shape[2]

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, N isn't neccesarily the same as fqf.N.
        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(
            batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * N, self.embedding_dim)

        # Calculate quantile values.
        if not self.dueling_net:
            quantiles = self.net(embeddings)
        else:
            advantages = self.advantage_net(embeddings)
            baselines = self.baseline_net(embeddings)
            quantiles =\
                baselines + advantages - advantages.mean(1, keepdim=True)
        return quantiles.view(batch_size, N, self.num_actions)


class IQN(nn.Module):
    def __init__(self, feature_dim, num_actions, K=32, num_cosines=32, dueling_net=False, cvar_alpha=1.0):
        super(IQN, self).__init__()
        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=feature_dim,)
        # Quantile network.
        self.quantile_net = QuantileNetwork(
            num_actions=num_actions, dueling_net=dueling_net, embedding_dim=feature_dim)

        self.K = K
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.num_cosines = num_cosines

        self.dueling_net = dueling_net
        self.cvar_alpha = cvar_alpha

    def calculate_quantiles(self, taus, feature):
        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(feature, tau_embeddings)

    def forward(self, feature):
        action = self.calculate_q(feature)
        return action

    def calculate_q(self, feature):
        batch_size = feature.shape[0]
        # Sample fractions.
        taus = self.cvar_alpha * th.rand(
            batch_size, self.K, dtype=th.float32,
            device=feature.device)

        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            taus, feature)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)

        # Calculate expectations of value distributions.

        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)
        return q


class AutoIQN(IQN):
    def __init__(self, feature_dim, num_actions, K=32, num_cosines=32, dueling_net=False, ):
        super(AutoIQN, self).__init__(feature_dim, num_actions, K, num_cosines, dueling_net, 1.0)


class QFeatureNet(nn.Module):
    def __init__(self, observation_space, action_space, normalize_action=True):
        assert isinstance(action_space, gym.spaces.Box)
        super(QFeatureNet, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_action = normalize_action

        self.obs_dim = np.prod(self.observation_space.shape)
        self.action_dim = np.prod(self.action_space.shape)
        self.linear = nn.Sequential(nn.Linear(self.obs_dim + self.action_dim, 64), nn.LayerNorm(64), nn.Mish())


    @property
    def feature_dim(self):
        return 64

    def forward(self, obs, action):
        obs = obs.reshape(-1, self.obs_dim)
        action = action.reshape(-1, self.action_dim)
        return self.linear(th.cat([obs, action], dim=-1))


class ObservationFeatureNet(nn.Module):
    def __init__(self, observation_space, ):
        super(ObservationFeatureNet, self).__init__()
        self.observation_space = observation_space
        self.obs_dim = np.prod(self.observation_space.shape)

    @property
    def feature_dim(self):
        return self.obs_dim

    def forward(self, obs):
        obs = obs.reshape(-1, self.obs_dim)
        return obs

