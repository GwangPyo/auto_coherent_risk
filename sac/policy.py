import torch.nn as nn
import torch as th
from abc import ABCMeta, abstractmethod
from rl_utils.utils import dump_state_dict, polyak_update, dimension
from sac.actor import SACActor, RescaleAction
from sac.critic import IQNHead
from net.nets import QFeatureNet
import numpy as np


class AbstractSACPolicy(nn.Module, metaclass=ABCMeta):
    def __init__(self, env, device):
        super(AbstractSACPolicy, self).__init__()
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.device = device

        self.observation_dim = dimension(self.observation_space)
        self.action_dim = dimension(self.action_space)
        self.target_entropy = -th.prod(th.Tensor(self.action_space.shape).to(self.device)).item()
        self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = th.optim.Adam([self.log_alpha], lr=3e-4)

    @abstractmethod
    def train_step(self, batch_data, critic_optim, actor_optim):
        pass

    def preprocess_obs(self, obs: np.ndarray) -> th.Tensor:
        obs = obs.reshape(-1, self.observation_dim)
        obs = th.from_numpy(obs).to(self.device)
        return obs

    def predict(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        th_obs = self.preprocess_obs(obs)
        action, _, mean_action = self.actor.sample(th_obs)
        if deterministic:
            action = mean_action
        action = action.detach().cpu().numpy()
        action = action.reshape((-1, ) + self.action_space.shape)

        return np.squeeze(action)

    def train_predict(self, obs: np.ndarray):
        return self.predict(obs, deterministic=False)


class MlpIQNSACPolicy(AbstractSACPolicy):
    def __init__(self, env, device,  gamma=0.99, iqn_kwargs=None, cvar_alpha=1.0):
        self.iqn_kwargs = iqn_kwargs
        super(MlpIQNSACPolicy, self).__init__(env, device)
        scaler = RescaleAction(self.action_space)
        self.actor = SACActor(self.observation_dim, self.action_dim, scaler)
        self.cvar_alpha = cvar_alpha
        self.qf1 = self._build_critic()
        self.qf2 = self._build_critic()
        self.qf_target = self._build_critic()
        self.to(device)
        self.gamma = gamma

    def to(self, device):
        self.actor.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.qf_target.to(device)
        return super().to(device)

    def build_optim(self, actor_optim_cls=th.optim.Adam, critic_optim_cls=th.optim.Adam, lr=3e-4):
        actor_optim = actor_optim_cls(self.actor.parameters(), lr)
        critic_optim = critic_optim_cls(nn.ModuleList([self.qf1, self.qf2]).parameters(), lr)
        return actor_optim, critic_optim

    def init_critic(self):
        self.qf_target.requires_grad_(False)
        dump_state_dict(self.qf, self.qf_target)

    def target_update(self, tau):
        polyak_update(source=self.qf2, target=self.qf_target, tau=tau)
        polyak_update(source=self.qf2, target=self.qf_target, tau=tau)

    def _build_critic(self):
        q_feature = QFeatureNet(self.observation_space, self.action_space)
        return IQNHead(feature_net=q_feature, )

    def train_step(self, batch_data, critic_optim, actor_optim):
        loss_summary = {}

        obs, actions, rewards, next_obs, dones, info = batch_data

        #######################################################
        # train critic                                        #
        #######################################################
        next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
        entropy_coefficient = self.log_alpha.exp().detach()
        qf1_loss, _, td_error = self.qf1.sac_calculate_iqn_loss(self.qf_target, obs, actions, rewards, next_obs,
                                                                  dones, next_actions, next_actions_log_p_pi,
                                                                  entropy_coefficient, self.gamma)
        qf2_loss, _, _ = self.qf2.sac_calculate_iqn_loss(self.qf_target, obs, actions, rewards, next_obs,
                                                                  dones, next_actions, next_actions_log_p_pi,
                                                                  entropy_coefficient, self.gamma)
        critic_loss = qf1_loss + qf2_loss
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        loss_summary["quantile_loss"] = critic_loss.item()
        loss_summary["td_error"] = td_error.mean().item()

        #######################################################
        # train actor                                         #
        #######################################################
        current_actions, log_p_pi, _ = self.actor.sample(obs)
        qf1_current = self.qf1(obs, current_actions)
        qf2_current = self.qf2(obs, current_actions)
        qf_current = th.min(qf1_current, qf2_current)
        assert qf_current.shape == log_p_pi.shape
        actor_loss = (-qf_current + entropy_coefficient * log_p_pi).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        loss_summary["policy loss"] = actor_loss.item()

        #######################################################
        # train entropy coeff                                 #
        #######################################################

        alpha_loss = -(self.log_alpha * (log_p_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        loss_summary["entropy_loss"] = alpha_loss.item()
        loss_summary["current entropy"] = log_p_pi.mean().item()
        loss_summary["ent_coef"] = entropy_coefficient.item()
        ret = {}
        for k in loss_summary.keys():
            ret[f'loss/{k}'] = loss_summary[k]
        ret["model/qf1"] = qf1_current.mean().item()
        ret["model/qf2"] = qf2_current.mean().item()
        return ret


policies = {"MlpIQNPolicy": MlpIQNSACPolicy}
