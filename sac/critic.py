from net.nets import IQN
import torch.nn as nn
from net.nets import QFeatureNet


class IQNCritic(nn.Module):
    def __init__(self,  feature_dim, N, N_dash,  IQN_kwargs=None):
        super(IQNCritic, self).__init__()
        self.feature_dim = feature_dim
        if IQN_kwargs is None:
            IQN_kwargs = {}
        self.Value = IQN(feature_dim, num_actions=1, **IQN_kwargs)

    def forward(self, obs, action):
        z = self.feature_extractor(obs, action)
        return self.Value(z)


