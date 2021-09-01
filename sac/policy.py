import torch.nn as nn
import torch as th
from abc import ABCMeta, abstractmethod
from rl_utils.utils import dump_state_dict, polyak_update, dimension
from sac.actor import SACActor, RescaleAction
from sac.critic import IQNQNetwork, IQNVNetwork, AutoriskIQNQNetwork, ProbQNet
from net.nets import Discriminator
from net.spectral_risk_net import SpectralRiskNet
import numpy as np

discriminator_criterion = nn.BCEWithLogitsLoss()
softplus = nn.Softplus()


class Rhat(nn.Module):
    def forward(self, x):
        return -softplus(-x)

rhat= Rhat()


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
        self.qf1 = self._build_qf()
        self.qf2 = self._build_qf()
        self.vf = self._build_vf()
        self.vf_target = self._build_vf()
        self.init_critic()
        self.to(device)
        self.gamma = gamma

    def to(self, device):
        self.actor.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.vf.to(device)
        self.vf_target.to(device)
        return super().to(device)

    def build_optim(self, actor_optim_cls=th.optim.Adam, critic_optim_cls=th.optim.Adam, actor_lr=3e-4, critic_lr=3e-4):
        actor_optim = actor_optim_cls(self.actor_parameters, actor_lr)
        critic_optim = critic_optim_cls(self.critic_trainable_parameters, critic_lr)
        return actor_optim, critic_optim

    @property
    def critic_trainable_parameters(self):
        key = 'params'
        return [{key: self.qf1.parameters()}, {key: self.qf2.parameters()}, {key: self.vf.parameters()}]

    @property
    def actor_parameters(self):
        key = 'params'
        return [{key: self.actor.parameters()}]

    def init_critic(self):
        self.vf_target.requires_grad_(False)
        dump_state_dict(self.vf, self.vf_target)

    def target_update(self, tau):
        polyak_update(source=self.vf, target=self.vf_target, tau=tau)

    def _build_qf(self):
        return IQNQNetwork(observation_space=self.observation_space,
                           action_space=self.action_space,
                           cvar_alpha=self.cvar_alpha)

    def _build_vf(self):
        return IQNVNetwork(observation_space=self.observation_space,
                           cvar_alpha=self.cvar_alpha)

    def train_step(self, batch_data, critic_optim, actor_optim):
        loss_summary = {}

        obs, actions, rewards, next_obs, dones, info = batch_data

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
        """
        qf1_quantiles = th.squeeze(self.qf1.calculate_quantile(obs, current_actions, tau_dash))
        qf2_quantiles = th.squeeze(self.qf2.calculate_quantile(obs, current_actions, tau_dash))

        min_qf_quantiles = th.min(qf1_quantiles, qf2_quantiles)
        min_qf_quantiles, _ = th.sort(min_qf_quantiles, dim=1)
        min_qf_quantiles = min_qf_quantiles - entropy_coefficient * log_p_pi
        min_qf_quantiles = min_qf_quantiles[:, None, :]
        taus = self.vf_target.get_default_tau(shape=(obs.shape[0], self.vf_target.N), device=self.device)
        """
        action_distribution = self.actor.distribution(obs)
        taus = self.qf1.get_default_tau(shape=(obs.shape[0], self.qf1.K), device=self.device)
        taus_pi_train = th.stack([th.pow(taus, 1/self.action_dim) for _ in range(self.action_dim)], dim=-1).transpose(0, 1)
        print(taus_pi_train.shape)
        action_samples = self.actor.action_scaler(th.tanh(action_distribution.icdf(taus_pi_train))).transpose(0, 2)

        vf_quantile = self.vf.calculate_quantile(obs, taus)
        vf_loss = self.vf.quantile_huber_loss(vf_quantile, min_qf_quantiles, taus)
        critic_loss = qf1_loss + qf2_loss + vf_loss
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        loss_summary["quantile_loss"] = critic_loss.item()
        loss_summary["td_error"] = td_error.mean().item()
        loss_summary["qf1_loss"] = qf1_loss.item()
        loss_summary["qf2_loss"] = qf2_loss.item()
        loss_summary["vf_loss"] = vf_loss.item()

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


class AutoRiskSACPolicy(MlpIQNSACPolicy):
    def __init__(self, env, device,  gamma=0.99, iqn_kwargs=None):
        super(AutoRiskSACPolicy, self).__init__(env, device, gamma, iqn_kwargs, 1.0)
        self.pf1 = self._build_pf()
        self.pf1.to(self.device)
        self.pf_target = self._build_pf()
        self.pf_target.to(self.device)

        self.discriminator = Discriminator(self.qf1.feature_dim)
        self.discriminator.to(self.device)
        self.init_pf()

        self.rho = SpectralRiskNet(self.vf_target.feature_dim, n_bins=10, init_uniform=False)
        self.rho.to(self.device)
        self.disc_optim = th.optim.Adam(self.discriminator.parameters(), lr=3e-4)
        self.prob_optim = th.optim.Adam(self.pf1.parameters(), 3e-4)
        self.rho_optim = th.optim.Adam(self.rho.parameters(), lr=3e-4)
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
            feature = self.qf1.feature_extractor(obs, actions)
        pred_success = self.discriminator(feature)
        discriminator_loss = discriminator_criterion(pred_success, succ)
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
            pf_target_quantile = th.squeeze(pf_target_quantile) - entropy_coefficient * next_actions_log_p_pi
            pf_target_quantile = pf_target_quantile[:, None, :]
            cost_hat = rhat(pred_success).detach()

        pf_loss, pf_next, pf_td_error = self.pf1.sac_calculate_iqn_loss(pf_target_quantile, obs, actions,
                                                                        cost_hat, dones, gamma=1.)
        self.prob_optim.zero_grad()
        pf_loss.backward()
        self.prob_optim.step()

        #######################################################
        # train qf1, qf2                                      #
        #######################################################
        next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
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
        qf1_quantiles = th.squeeze(self.qf1.calculate_quantile(obs, current_actions, tau_dash))
        qf2_quantiles = th.squeeze(self.qf2.calculate_quantile(obs, current_actions, tau_dash))

        min_qf_quantiles = th.min(qf1_quantiles, qf2_quantiles)
        min_qf_quantiles = min_qf_quantiles - entropy_coefficient * log_p_pi
        min_qf_quantiles = min_qf_quantiles[:, None, :]
        taus = self.vf_target.get_default_tau(shape=(obs.shape[0], self.vf_target.N), device=self.device)
        vf_quantile = self.vf.calculate_quantile(obs, taus)
        vf_loss = self.vf.quantile_huber_loss(vf_quantile, min_qf_quantiles, taus)
        critic_loss = qf1_loss + qf2_loss + vf_loss
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        loss_summary["pf_loss"] = pf_loss.item()
        loss_summary["quantile_loss"] = critic_loss.item()
        loss_summary["td_error"] = td_error.mean().item()
        loss_summary["qf1_loss"] = qf1_loss.item()
        loss_summary["qf2_loss"] = qf2_loss.item()
        loss_summary["vf_loss"] = vf_loss.item()

        #######################################################
        # train actor                                         #
        #######################################################
        current_actions, log_p_pi, _ = self.actor.sample(obs)
        with th.no_grad():
            feature = self.qf1.feature_extractor(obs, current_actions)
            taus_policy = self.rho.sample(feature, (obs.shape[0], self.qf1.K))
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

        taus_pf = self.rho.sample(feature, (obs.shape[0], self.qf1.N))
        taus_pi_train = self.rho.sample(feature, (obs.shape[0], self.qf1.K))
        pf1_quantile = th.squeeze(self.pf1.calculate_quantile(obs, current_actions, taus_pf))

        exp_pf1_quantile = th.exp(pf1_quantile).clamp(1e-8, 1)
        exp_pf1_quantile = exp_pf1_quantile/exp_pf1_quantile.mean(dim=1, keepdim=True)
        # with th.no_grad():
        action_distribution = self.actor.distribution(obs)
        """

        qf1_current = self.qf1(obs, current_actions, taus_pi_train)
        qf2_current = self.qf2(obs, current_actions, taus_pi_train)
        qf_current = th.min(qf1_current, qf2_current)

        actor_loss = (-qf_current + entropy_coefficient * log_p_pi.detach()).mean()
        exp_qf1 = th.exp(th.squeeze(qf1_quantile))
        exp_qf2 = th.exp(th.squeeze(qf2_quantile))
        exp_qf1 = exp_qf1/th.mean(exp_qf1, dim=1, keepdim=True)
        exp_qf2 = exp_qf2/th.mean(exp_qf2, dim=1, keepdim=True)
         rho_objective_2 = actor_loss
        """
        taus_pi_train = th.stack([taus_pi_train for _ in range(self.action_dim)], dim=-1).transpose(0, 1)
   
        # batch, n_dash,

        actor_icdf = self.actor.action_scaler(th.tanh(action_distribution.icdf(taus_pi_train))).transpose(0, 2)

        rho_objective_1 = [self.pf1.quantile_huber_loss(exp_pf1_quantile[:, :, None], a[:, None, :], taus_pf)
                           for a in actor_icdf]
        rho_objective_1 = sum(rho_objective_1)/self.action_dim
        rho_objective_1 = rho_objective_1.clamp(0, 1e+2)

        """
        0.5 * ((th.stack([exp_qf1 for _ in range(self.action_dim)], dim=-1) -actor_icdf) ** 2).sum(dim=1).mean() \
                            + 0.5 * ((th.stack([exp_qf2 for _ in range(self.action_dim)], dim=-1) - actor_icdf) ** 2).sum(dim=1).mean()
        """
        rho_loss = rho_objective_1.mean() #  + rho_objective_2
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
        ret["model/qf1"] = qf1_current.mean().item()
        ret["model/qf2"] = qf2_current.mean().item()
        ret["model/pf"] = pf1_quantile.mean().item()

        ret["model/risk_mean"] = taus_policy.mean().item()
        return ret


policies = {"MlpIQNPolicy": MlpIQNSACPolicy,
            "AutoRiskIQNPolicy": AutoRiskSACPolicy}



