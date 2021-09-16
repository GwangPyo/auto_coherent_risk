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


class AutoRiskSACPolicy(IQNPolicy):
    def __init__(self, env, device, gamma=0.99,
                         N=8, N_dash=8, K=32, n_critics=2,
                         qf_kwargs=None, risk_kwargs=None):
        super(AutoRiskSACPolicy, self).__init__(env, device, gamma, N, N_dash, K,
                                                n_critics, tau_generator=SpectralRiskNet, qf_kwargs=qf_kwargs,
                                                risk_kwargs=risk_kwargs)


"""
class AutoRiskSACPolicy(MlpIQNSACPolicy):
    def __init__(self, env, device,  gamma=0.99, iqn_kwargs=None):
        super(AutoRiskSACPolicy, self).__init__(env, device, gamma, iqn_kwargs, 1.0)
        self.pf1 = self._build_pf()
        self.pf1.to(self.device)
        self.pf_target = self._build_pf()
        self.pf_target.to(self.device)

        self.discriminator = Discriminator(self.observation_dim, self.gamma)
        self.discriminator.to(self.device)
        self.init_pf()

        self.rho = SpectralRiskNet(self.qf1.feature_dim, n_bins=10, init_uniform=False)
        self.rho.to(self.device)
        self.disc_optim = th.optim.Adam(self.discriminator.parameters(), lr=1e-9, weight_decay=1.)
        self.prob_optim = th.optim.Adam(self.pf1.parameters(), 1e-9)
        self.rho_optim = th.optim.Adam(self.rho.parameters(), lr=1e-2,)
        self.init_critic()

    def init_pf(self):
        self.pf_target.requires_grad_(False)
        dump_state_dict(self.pf1, self.pf_target)

    def _build_pf(self):
        return IQNQNetwork(observation_space=self.observation_space,
                           action_space=self.action_space,
                           cvar_alpha=self.cvar_alpha)

    def target_update(self, tau):
        polyak_update(source=self.vf, target=self.vf_target, tau=tau)
        polyak_update(source=self.pf1, target=self.pf_target, tau=tau)

    def _build_qf(self):
        return IQNQNetwork(observation_space=self.observation_space,
                           action_space=self.action_space,
                           cvar_alpha=self.cvar_alpha)

    def train_step(self, batch_data, critic_optim, actor_optim):
        loss_summary = {}
        obs, actions, rewards, next_obs, dones, succ = batch_data
        entropy_coefficient = self.log_alpha.exp().detach()

        #######################################################
        # train discriminator                                 #
        #######################################################
        with th.no_grad():
            action_distribution = self.actor.distribution(obs)
            logp_pi = action_distribution.log_prob(th.atanh(actions)).sum(dim=1, keepdim=True)
        discriminator_minibatch = (obs, dones, logp_pi, next_obs, succ)
        discriminator_loss = self.discriminator.loss(discriminator_minibatch)
        self.disc_optim.zero_grad()
        discriminator_loss.backward()
        self.disc_optim.step()
        loss_summary["discriminator loss"] = discriminator_loss.item()


        #######################################################
        # train pf                                            #
        #######################################################

        next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
        taus = self.vf_target.get_default_tau(shape=(obs.shape[0], self.vf_target.N), device=self.device)
        with th.no_grad():
            pf_target_quantile = self.pf_target.calculate_quantile(next_obs, next_actions, taus)
            pf_target_quantile = th.squeeze(pf_target_quantile)
            pf_target_quantile = pf_target_quantile[:, None, :]

            p_reward = self.discriminator.calculate_reward(obs, dones=dones, logp_pi=logp_pi, next_obs=next_obs)

        pf_loss, pf_next, pf_td_error = self.pf1.sac_calculate_iqn_loss(pf_target_quantile, obs, actions,
                                                                        p_reward, dones, gamma=1.)
        self.prob_optim.zero_grad()
        pf_loss.backward()
        self.prob_optim.step()
        loss_summary["pf_loss"] = pf_loss.item()

        #######################################################
        # train qf1, qf2                                      #
        #######################################################

        next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
        entropy_coefficient = self.log_alpha.exp().detach()
        with th.no_grad():
            tau_dash = self.vf_target.get_default_tau(shape=(obs.shape[0], self.vf_target.N_dash), device=self.device)
            vf_next_quantile = self.vf_target.calculate_quantile(next_obs, taus=tau_dash)
        qf1_loss, _, td_error = self.qf1.sac_calculate_iqn_loss(vf_next_quantile, obs, actions,
                                                                rewards, dones, self.gamma)
        qf2_loss, _, _ = self.qf2.sac_calculate_iqn_loss(vf_next_quantile, obs, actions,
                                                         rewards, dones, self.gamma)
        current_actions, log_p_pi, _ = self.actor.sample(obs)

        #######################################################
        # train vf                                            #
        #######################################################
        tau_dash = self.vf_target.get_default_tau(shape=(obs.shape[0], self.vf_target.N_dash), device=self.device)
        qf1_quantiles = th.squeeze(self.qf1.calculate_quantile(obs, current_actions, tau_dash))
        qf2_quantiles = th.squeeze(self.qf2.calculate_quantile(obs, current_actions, tau_dash))

        min_qf_quantiles = th.min(qf1_quantiles, qf2_quantiles)
        min_qf_quantiles = min_qf_quantiles - entropy_coefficient * log_p_pi
        min_qf_quantiles = min_qf_quantiles[:, None, :]

        taus = self.qf1.get_default_tau(shape=(obs.shape[0], self.qf1.K), device=self.device)

        vf_quantile = self.vf.calculate_quantile(obs, taus)
        vf_loss = self.vf.quantile_huber_loss(vf_quantile, min_qf_quantiles, taus)
        critic_loss = qf1_loss + qf2_loss + vf_loss
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        loss_summary["critic_loss"] = critic_loss.item()
        loss_summary["qf1_loss"] = qf1_loss.item()
        loss_summary["qf2_loss"] = qf2_loss.item()
        loss_summary["vf_loss"] = vf_loss.item()

        #######################################################
        # train actor                                         #
        #######################################################

        current_actions, log_p_pi, _ = self.actor.sample(obs)

        with th.no_grad():
            feature = self.qf1.feature_extractor(obs, actions)
            taus_policy, _ = self.rho.sample(feature, (obs.shape[0], self.qf1.K))
            rho_entropy = self.rho.entropy(feature)
        qf1_current = self.qf1(obs, current_actions, taus_policy.detach())
        qf2_current = self.qf2(obs, current_actions, taus_policy.detach())
        qf_current = th.min(qf1_current, qf2_current)
        assert qf_current.shape == log_p_pi.shape
        actor_loss = (-qf_current + entropy_coefficient * log_p_pi).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        loss_summary["policy loss"] = actor_loss.item()

        #######################################################
        # train risk_net                                      #
        #######################################################

        current_actions = current_actions.detach()
        taus_qf, logprobs = self.rho.sample(feature, (obs.shape[0], self.qf1.N_dash))
        with th.no_grad():
            taus_pf = self.pf1.get_default_tau(shape=(obs.shape[0], self.qf1.N), device=self.device)
            pf1_quantile_normalized = self.pf1.calculate_normalized_quantile(obs, current_actions, taus_pf)

        qf1_quantiles_rho = self.qf1.calculate_normalized_quantile(obs, current_actions, taus_qf)
        qf2_quantiles_rho = self.qf2.calculate_normalized_quantile(obs, current_actions, taus_qf)
        qf_qunatiles_rho = th.min(qf1_quantiles_rho, qf2_quantiles_rho)

        rho_loss = self.qf1.quantile_huber_loss(qf_qunatiles_rho, pf1_quantile_normalized.transpose(1, 2), taus_pf, reduce=False) * (-logprobs.sum(dim=1, keepdim=True))

        rho_loss = rho_loss.mean()

        self.rho_optim.zero_grad()

        rho_loss.backward()

        self.rho_optim.step()

        loss_summary["rho_loss"] = rho_loss.item()

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
        ret["model/qf1"] = qf1_quantiles.mean().item()
        ret["model/qf2"] = qf2_quantiles.mean().item()
        ret["model/p_reward"] = p_reward.mean().item()
        ret["model/rho_entropy_per_max"] = rho_entropy.mean().item()

        ret["model/risk_mean"] = taus_policy.mean().item()
        return ret
"""

policies = {"IQNPolicy": IQNPolicy,
            "CVaRPolicy": CVaRIQNPolicy,
            "WangPolicy": WangIQNPolicy,
            "PowerPolicy": PowerIQNPolicy,
            }
"""
"ODEMlpoIQNPolicy": ODEMlpoIQNPolicy,
"AutoRiskIQNPolicy": AutoRiskSACPolicy}
"""


