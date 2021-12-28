import torch as th
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(AutoEncoder, self).__init__()
        self.obs_layer = nn.Linear(observation_dim, 64)
        self.action_layer = nn.Linear(action_dim, 64)

        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 128),
            nn.Mish(inplace=True)
        )

        self.obs_decoder = nn.Linear(128, observation_dim)
        self.action_decoder = nn.Linear(128, action_dim)

    def build_discriminator(self):
        discriminator = nn.Sequential(
            nn.Linear(64, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        return discriminator

    def forward(self, obs, action):
        obs_z = self.obs_layer(obs)
        action_z = self.action_layer(action)
        cat = th.cat((obs_z, action_z), dim=1)
        z = self.encoder(cat)
        z_hat = self.decoder(z)
        obs_hat = self.obs_decoder(z_hat)
        action_hat = self.action_decoder(z_hat)
        return obs_hat, action_hat, z


class AAE(object):
    def __init__(self, obs_dim, action_dim):
        self.ae = AutoEncoder(obs_dim, action_dim)
        self.discriminator = self.ae.build_discriminator()
