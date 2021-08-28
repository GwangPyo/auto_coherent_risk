import torch.nn as nn
import torch as th
from abc import ABCMeta, abstractmethod
from rl_utils.utils import dump_state_dict, polyak_update, dimension
from sac.actor import SACActor, RescaleAction
from sac.critic import IQNCritic

from net.utils import IQNLosses


class AbstractSACPolicy(nn.Module, metaclass=ABCMeta):
    def __init__(self, env, ):
        super(AbstractSACPolicy, self).__init__()
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.observation_dim = dimension(self.observation_space)
        self.action_dim = dimension(self.action_space)
        self.target_entropy = -th.prod(th.Tensor(self.action_space.shape).to(self.device)).item()
        self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = th.optim.Adam([self.log_alpha], lr=3e-4)

    @abstractmethod
    def train_step(self, batch_data, critic_optim, actor_optim):
        pass


class MlpIQNSACPolicy(AbstractSACPolicy):
    def __init__(self, env, iqn_kwargs=None):
        self.iqn_kwargs = iqn_kwargs
        super(MlpIQNSACPolicy, self).__init__(env)
        scaler = RescaleAction(self.action_space)
        self.actor = SACActor(self.observation_dim, self.action_dim, scaler)
        self.qf = self._build_critic()
        self.qf_target = self._build_critic()

    def init_critic(self):
        self.qf_target.requires_grad_(False)
        dump_state_dict(self.qf, self.qf_target)

    def target_update(self, tau):
        return polyak_update(source=self.qf, target=self.qf_target, tau=tau)

    def _build_critic(self):
        return nn.Sequential(
            QFeatureNet(self.observation_space, self.action_space),
            IQNCritic((self.observation_dim + self.action_dim), **self.iqn_kwargs))

    def train_step(self, batch_data, critic_optim, actor_optim):
        obs, action, reward, done, next_obs, info = batch_data
        #######################################################
        # train critic                                        #
        #######################################################
        critic_loss = IQNLosses.sac_calculate_iqn_loss(IQNCritic)





