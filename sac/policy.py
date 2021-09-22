import torch.nn as nn
import torch as th
from abc import ABCMeta, abstractmethod
from rl_utils.utils import dump_state_dict, polyak_update, dimension
from sac.actor import SACActor
from sac.critic import Critics, ODECritics
from sac.risk_manager import *
from airl.discriminator import Discriminator
from net.spectral_risk_net import SpectralRiskNet
import numpy as np
from net.utils import quantile_huber_loss
from typing import Union, Type


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
        return obs.float()

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


class IQNPolicy(AbstractSACPolicy):
    def __init__(self, env, device,  gamma=0.99,
                 N=8, N_dash=8, K=32, n_critics=2,
                 qf_kwargs=None, tau_generator: Union[Type[TauGenerator], Type[SpectralRiskNet]] = TauGenerator, risk_kwargs=None):
        super(IQNPolicy, self).__init__(env, device)
        self.actor = SACActor(self.observation_dim, self.action_dim)
        assert risk_kwargs is None or isinstance(risk_kwargs, dict)
        if risk_kwargs is None:
            risk_kwargs = {}
        if qf_kwargs is None:
            qf_kwargs = {}
        self.n_critics = n_critics

        risk_kwargs["device"] = self.device
        self.base_tau_generator = TauGenerator(device=self.device)
        self.tau_generator = tau_generator(**risk_kwargs)
        self.critics = self.build_critics(qf_kwargs)
        self.critic_targets = self.build_critics(qf_kwargs)
        self.add_module("critic", self.critics)
        self.add_module("critic_target", self.critic_targets)
        self.add_module("actor", self.actor)
        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.init_critic()
        self.to(device)
        self.drop = 2 * self.n_critics
        self.gamma = gamma

    def build_critics(self, critic_kwargs):
        return Critics(self.observation_dim, self.action_dim, self.n_critics, **critic_kwargs)

    def build_optim(self, actor_optim_cls=th.optim.Adam, critic_optim_cls=th.optim.Adam, actor_lr=3e-4, critic_lr=3e-4):
        actor_optim = actor_optim_cls(self.actor.parameters(), actor_lr)
        critic_optim = critic_optim_cls(self.critics.parameters(), critic_lr)
        return actor_optim, critic_optim

    def init_critic(self):
        self.critic_targets.requires_grad_(False)
        dump_state_dict(self.critics, self.critic_targets)

    def target_update(self, tau):
        polyak_update(source=self.critics, target=self.critic_targets, tau=tau)

    def train_step(self, batch_data, critic_optim, actor_optim):
        loss_summary = {}
        obs, actions, rewards, next_obs, dones, info = batch_data
        batch_size = obs.shape[0]
        current_actions, log_p_pi, _ = self.actor.sample(obs)

        #######################################################
        # train entropy coeff                                 #
        #######################################################
        alpha_loss = -(self.log_alpha * (log_p_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        #######################################################
        # train critics                                       #
        #######################################################
        entropy_coefficient = self.log_alpha.exp().detach()

        with th.no_grad():
            tau_dash = self.base_tau_generator(shape=(batch_size, self.N_dash))
            next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
            next_qf_quantiles = self.critic_targets(next_obs, next_actions, tau_dash)
            target_quantile = rewards[:, None, :] + self.gamma * ((1. - dones)[:, None, :]) * (next_qf_quantiles
                              - entropy_coefficient * next_actions_log_p_pi[:, None, :])
        taus = self.base_tau_generator(shape=(batch_size, self.N))
        current_quantiles = self.critics(obs, actions, taus)
        critic_losses = []
        td_errors = []
        for i in range(self.n_critics):
            target = target_quantile[:, i, :]
            target = target.reshape(batch_size, -1, 1)
            current_q = current_quantiles[:, i, :]
            current_q = current_q.reshape(batch_size, 1, -1)
            td_error = target - current_q
            critic_loss = quantile_huber_loss(td_error, tau_dash)
            critic_loss = (critic_loss.sum(dim=-1)).mean()
            critic_losses.append(critic_loss)
            td_errors.append(th.abs(td_error).mean())
        critic_loss = sum(critic_losses)

        loss_summary["critic_loss"] = critic_loss.item()
        loss_summary["td_errors"] = sum(td_errors).item()/self.n_critics

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        #######################################################
        # train actor                                         #
        #######################################################

        taus_action = self.tau_generator(shape=(batch_size, self.K))
        qf_current = self.critics(obs, current_actions, taus_action)
        qf_current, _ = th.min(qf_current, dim=1)
        qf_current = qf_current.mean(dim=1, keepdim=True)

        assert qf_current.shape == log_p_pi.shape

        actor_loss = (-qf_current + entropy_coefficient * log_p_pi).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        loss_summary["policy loss"] = actor_loss.item()
        loss_summary["entropy_loss"] = alpha_loss.item()
        loss_summary["current entropy"] = log_p_pi.mean().item()
        loss_summary["ent_coef"] = entropy_coefficient.item()
        ret = {}
        for k in loss_summary.keys():
            ret[f'loss/{k}'] = loss_summary[k]

        return ret


class CVaRIQNPolicy(IQNPolicy):
    def __init__(self, env, device,  gamma=0.99,
                 N=8, N_dash=8, K=32, n_critics=2,
                 qf_kwargs=None, cvar_alpha=0.2):
        risk_kwargs = {"alpha": cvar_alpha}
        super(CVaRIQNPolicy, self).__init__(env, device, gamma, N, N_dash, K, n_critics,
                                qf_kwargs, tau_generator=CVaRTauGenerator, risk_kwargs=risk_kwargs)


class WangIQNPolicy(IQNPolicy):
    def __init__(self, env, device, gamma=0.99,
                 N=8, N_dash=8, K=32, n_critics=2,
                 qf_kwargs=None, eta=-0.75):
        risk_kwargs = {"eta": eta}
        super(WangIQNPolicy, self).__init__(env, device, gamma, N, N_dash, K, n_critics,
                                            qf_kwargs, tau_generator=WangTauGenerator, risk_kwargs=risk_kwargs)


class PowerIQNPolicy(IQNPolicy):
    def __init__(self, env, device, gamma=0.99,
                 N=8, N_dash=8, K=32, n_critics=2,
                 qf_kwargs=None, eta=-0.75):
        risk_kwargs = {"eta": eta}
        super(PowerIQNPolicy, self).__init__(env, device, gamma, N, N_dash, K, n_critics,
                                            qf_kwargs, tau_generator=PowerTauGenerator, risk_kwargs=risk_kwargs)


class ODEIQNPolicy(IQNPolicy):
    def build_critics(self, critic_kwargs):
        return ODECritics(self.observation_dim, self.action_dim, self.n_critics, **critic_kwargs)



class AutoRiskPolicy(IQNPolicy):
    def __init__(self, env, device, gamma=0.99,
                         N=8, N_dash=8, K=32, n_critics=2,
                         qf_kwargs=None, risk_kwargs=None):
        if risk_kwargs is None:
            risk_kwargs = {}
        risk_kwargs["in_features"] = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape)
        super(AutoRiskPolicy, self).__init__(env, device, gamma, N, N_dash, K,
                                             n_critics, tau_generator=SpectralRiskNet, qf_kwargs=qf_kwargs,
                                             risk_kwargs=risk_kwargs)
        self.pf = self.build_critics({})
        self.pf_target = self.build_critics({})
        self.pf_optim = th.optim.Adam(params=self.pf.parameters(), lr=3e-4)
        self.risk_optim = th.optim.Adam(params=self.tau_generator.parameters(), lr=3e-4)

        self.init_pf()

    def init_pf(self):
        self.add_module("pf", self.pf)
        self.add_module("pf_target", self.pf_target)
        self.to(self.device)
        self.pf_target.requires_grad_(False)
        dump_state_dict(self.pf, self.pf_target)

    def target_update(self, tau):
        polyak_update(source=self.critics, target=self.critic_targets, tau=tau)
        polyak_update(source=self.pf, target=self.pf_target, tau=tau)

    def train_step(self, batch_data, critic_optim, actor_optim):
        loss_summary = {}
        obs, actions, rewards, next_obs, dones, info = batch_data
        batch_size = obs.shape[0]
        current_actions, log_p_pi, _ = self.actor.sample(obs)

        #######################################################
        # train entropy coeff                                 #
        #######################################################
        alpha_loss = -(self.log_alpha * (log_p_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        entropy_coefficient = self.log_alpha.exp().detach()
        #######################################################
        # train critics                                       #
        #######################################################

        with th.no_grad():
            tau_dash = self.base_tau_generator(shape=(batch_size, self.N_dash))
            next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
            next_qf_quantiles = self.critic_targets(next_obs, next_actions, tau_dash)
            target_quantile = rewards[:, None, :] + self.gamma * ((1. - dones)[:, None, :]) * (next_qf_quantiles
                              - entropy_coefficient * next_actions_log_p_pi[:, None, :])

        taus = self.base_tau_generator(shape=(batch_size, self.N))
        current_quantiles = self.critics(obs, actions, taus)
        critic_losses = []
        td_errors = []
        for i in range(self.n_critics):
            target = target_quantile[:, i, :]
            target = target.reshape(batch_size, -1, 1)
            current_q = current_quantiles[:, i, :]
            current_q = current_q.reshape(batch_size, 1, -1)
            td_error = target - current_q
            critic_loss = quantile_huber_loss(td_error, tau_dash)
            critic_loss = (critic_loss.sum(dim=-1)).mean()
            critic_losses.append(critic_loss)
            td_errors.append(th.abs(td_error).mean())
        critic_loss = sum(critic_losses)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        loss_summary["critic_loss"] = critic_loss.item()
        loss_summary["td_errors"] = sum(td_errors).item()/self.n_critics
        #######################################################
        # train pfs                                           #
        #######################################################
        with th.no_grad():
            next_pf_quantiles = self.pf_target(next_obs, next_actions, tau_dash)
            target_pf_quantile = info[:, None, :] + self.gamma * ((1. - dones)[:, None, :]) * (next_pf_quantiles
                              - entropy_coefficient * next_actions_log_p_pi[:, None, :])
        current_pf = self.pf(obs, actions, taus)
        pf_losses = []
        td_errors = []
        for i in range(self.n_critics):
            target_p = target_pf_quantile[:, i, :]
            target_p = target_p.reshape(batch_size, -1, 1)
            current_p = current_pf[:, i, :]
            current_p = current_p.reshape(batch_size, 1, -1)
            td_error_p = target_p - current_p
            pf_loss = quantile_huber_loss(td_error_p, tau_dash)
            pf_loss = (pf_loss.sum(dim=-1)).mean()
            pf_losses.append(pf_loss)
            td_errors.append(th.abs(td_error_p).mean())
        pf_loss = sum(pf_losses)
        self.pf_optim.zero_grad()
        pf_loss.backward()
        self.pf_optim.step()
        loss_summary["pf_loss"] = pf_loss.item()
        #######################################################
        # train risk net                                      #
        #######################################################
        # step 1. Compute qf and pf. Normalize them
        tau_feature = th.cat((obs, current_actions), dim=1)
        taus_action, log_prob_taus = self.tau_generator.sample(tau_feature, sample_shape=(batch_size, self.N))
        qf_quantiles, _ = th.min(self.critics(obs, current_actions, taus_action), dim=1)
        # NORMALIZE
        qf_quantiles = qf_quantiles - qf_quantiles.mean(dim=1, keepdim=True)
        with th.no_grad():
            pf_quantiles, _ = th.min(self.pf(obs, current_actions, tau_dash), dim=1)
            # NORMALIZE
            pf_quantiles = pf_quantiles - pf_quantiles.mean(dim=1, keepdim=True)
            distribution_diff = qf_quantiles[:, None, :] - pf_quantiles[:, :, None]
            distribution_diff = quantile_huber_loss(th.abs(distribution_diff), tau_dash).mean().item()
        sign = th.sign(qf_quantiles - pf_quantiles)
        diff = qf_quantiles[:, None, :] - pf_quantiles[:, :, None]
        # qf_quantile > pf_quantile  - (qf_quantiles < pf_quantiles
        # log derivative trick for abs

        tau_loss = -quantile_huber_loss(th.abs(diff), taus_action) * ((log_prob_taus * sign).sum(dim=1, keepdim=False))

        tau_loss = tau_loss.mean()
        self.risk_optim.zero_grad()
        tau_loss.backward()
        self.risk_optim.step()
        loss_summary["w_distance_pq"] = distribution_diff
        loss_summary["risk_loss"] = tau_loss.item()
        loss_summary["tau_mean"] = taus_action.mean().item()
        #######################################################
        # train actor                                         #
        #######################################################
        current_actions, log_p_pi, _ = self.actor.sample(obs)
        entropy_tau = self.tau_generator.entropy(tau_feature)
        loss_summary["entropy_tau_per_max"] = entropy_tau.mean().item()
        with th.no_grad():
            tau_feature = th.cat((obs, current_actions), dim=1)
            taus_action, _ = self.tau_generator.sample(tau_feature, sample_shape=(batch_size, self.N))
        qf_current = self.critics(obs, current_actions, taus_action)
        qf_current, _ = th.min(qf_current, dim=1)
        qf_current = qf_current.mean(dim=1, keepdim=True)

        assert qf_current.shape == log_p_pi.shape

        actor_loss = (-qf_current + entropy_coefficient * log_p_pi).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        loss_summary["policy loss"] = actor_loss.item()
        loss_summary["entropy_loss"] = alpha_loss.item()
        loss_summary["current entropy"] = log_p_pi.mean().item()
        loss_summary["ent_coef"] = entropy_coefficient.item()

        ret = {}
        for k in loss_summary.keys():
            ret[f'loss/{k}'] = loss_summary[k]

        return ret





policies = {"IQNPolicy": IQNPolicy,
            "CVaRPolicy": CVaRIQNPolicy,
            "ODEIQNPolicy": ODEIQNPolicy,
            "WangPolicy": WangIQNPolicy,
            "PowerPolicy": PowerIQNPolicy,
            "AutoRiskPolicy": AutoRiskPolicy,
            }
"""
"ODEMlpoIQNPolicy": ODEMlpoIQNPolicy,
"AutoRiskIQNPolicy": AutoRiskSACPolicy}
"""


