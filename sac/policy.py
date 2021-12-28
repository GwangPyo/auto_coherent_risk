import torch.nn as nn
import torch as th
from abc import ABCMeta, abstractmethod
from rl_utils.utils import dump_state_dict, polyak_update, dimension
from sac.actor import SACActor, AlphaSACActor, MonotonicSACActor
from sac.critic import Critics, Discriminator, AlphaCritic
from sac.critic import  ScalarQfunction as RewardFunction
from sac.risk_manager import *
# from imitation.discriminator import Discriminator
from net.spectral_risk_net import SpectralRiskNet, FractionProposalNetwork
import numpy as np
from net.utils import quantile_huber_loss, calculate_fraction_loss
from typing import Union, Type
from misc.sched import SigmoidScheduler
from torch.autograd import grad


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
                 N=16, N_dash=16, K=32, n_critics=2,
                 qf_kwargs=None, tau_generator: Union[Type[TauGenerator], Type[SpectralRiskNet], Type[FractionProposalNetwork]] = TauGenerator, risk_kwargs=None):
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
            next_target_critic = next_qf_quantiles.mean(dim=-1, keepdim=True).argmin(dim=1, keepdim=True)
            next_target_critic = next_target_critic.expand(batch_size, 1, self.N_dash)
            next_qf_quantiles = th.gather(next_qf_quantiles, index=next_target_critic, dim=1)
            next_qf_quantiles = next_qf_quantiles - entropy_coefficient * next_actions_log_p_pi[:, None, :]
            target_quantile = rewards[:, None, :] + self.gamma * (1. - dones[:, None, :]) * next_qf_quantiles


        taus = self.base_tau_generator(shape=(batch_size, self.N))
        current_quantiles = self.critics(obs, actions, taus)
        critic_loss = quantile_huber_loss(current_quantiles, target_quantile, taus[:, None, :, None], sum_over_quantiles=True)

        loss_summary["critic_loss"] = critic_loss.item()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        #######################################################
        # train actor                                         #
        #######################################################

        taus_action = self.tau_generator(shape=(batch_size, self.K))
        qf_current = self.critics(obs, current_actions, taus_action.detach())
        qf_current = th.mean(qf_current, dim=1)
        qf_current, _ = qf_current.min(dim=1, keepdim=True)
        assert qf_current.shape == log_p_pi.shape

        actor_loss = (entropy_coefficient * log_p_pi - qf_current).mean()
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


class FQFPolicy(IQNPolicy):
    def __init__(self, env, device, gamma=0.99,
                         N=32, n_critics=2,
                         qf_kwargs=None, risk_kwargs=None):
        if risk_kwargs is None:
            risk_kwargs = {}
        risk_kwargs["N"] = N
        if qf_kwargs is None:
            risk_kwargs["embedding_dim"] = 256
        super(FQFPolicy, self).__init__(env, device, gamma, N, N, N,
                                             n_critics, tau_generator=FractionProposalNetwork, qf_kwargs=qf_kwargs,
                                             risk_kwargs=risk_kwargs)

        self.proposal_optim = th.optim.RMSprop(self.tau_generator.parameters(), lr=2.5e-9,  alpha=0.95, eps=0.0000)

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
            next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
            next_embedding = self.critic_targets.embedding(next_obs, next_actions)
            _, tau_dash, _ = self.tau_generator(next_embedding)
            next_qf_quantiles = self.critic_targets(next_obs, next_actions, tau_dash)

            target_quantile = rewards[:, None, :] + self.gamma * (1. - dones[:, None, :]) * (next_qf_quantiles
                              - entropy_coefficient * next_actions_log_p_pi[:, None, :])

            embedding = self.critics.embedding(obs, actions)
            _, taus, _ = self.tau_generator(embedding)

        current_quantiles = self.critics(obs, actions, taus)
        critic_loss = quantile_huber_loss(current_quantiles, target_quantile, taus, sum_over_quantiles=True)

        loss_summary["critic_loss"] = critic_loss.item()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        #######################################################
        # train actor                                         #
        #######################################################

        with th.no_grad():
            embedding = self.critics.embedding(obs, current_actions)
        taus_action, taus_hat, tau_ent = self.tau_generator(embedding)
        quantile_tau_hat = self.critics(obs, current_actions, taus_hat.detach())
        qf_current = ((taus_action[:, None, 1:] - taus_action[:, None, :-1]) * quantile_tau_hat).sum(dim=-1)
        qf_current, _ = qf_current.min(dim=1, keepdim=True)
        assert qf_current.shape == log_p_pi.shape

        actor_loss = (-qf_current + entropy_coefficient * log_p_pi).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        with th.no_grad():
            embedding = self.critics.embedding(obs, current_actions)

        taus_action, taus_hat, tau_ent = self.tau_generator(embedding)
        with th.no_grad():

            quantile_tau_hat = self.critics(obs, current_actions, taus_hat)
            quantile = self.critics(obs, current_actions, taus_action)
        fraction_loss = sum([calculate_fraction_loss(quantile[:, i, :], quantile_tau_hat[:, i, :], taus_action) for i in range(self.n_critics)])/self.n_critics
        self.proposal_optim.zero_grad()
        fraction_loss.backward(retain_graph=True)
        self.proposal_optim.step()


        loss_summary["policy loss"] = actor_loss.item()
        loss_summary["entropy_loss"] = alpha_loss.item()
        loss_summary["current entropy"] = -log_p_pi.mean().item()
        loss_summary["ent_coef"] = entropy_coefficient.item()
        loss_summary["frac_entropy"] = tau_ent.mean().item()
        loss_summary["frac_loss"] = fraction_loss.mean().item()

        ret = {}
        for k in loss_summary.keys():
            ret[f'loss/{k}'] = loss_summary[k]

        return ret


class ConsistCVaRSAC(AbstractSACPolicy):
    def __init__(self, env, device,  gamma=0.99,
                 N=16, N_dash=16, K=32, n_critics=2,
                 qf_kwargs=None, tau_generator=RandomCVaRTauGenerator , risk_kwargs=None):
        super(ConsistCVaRSAC, self).__init__(env, device)
        self.actor = AlphaSACActor(self.observation_dim, self.action_dim)
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

        taus_action, current_cvar_alpha = self.tau_generator.sample(shape=(batch_size, self.K))
        current_cvar_alpha = th.repeat_interleave(current_cvar_alpha.reshape(1, 1), dim=0, repeats=batch_size)
        current_actions, log_p_pi, _ = self.actor.sample(obs, current_cvar_alpha)
        #######################################################
        # train entropy coeff                                 #
        #######################################################

        ent_coeff_loss = -(self.log_alpha * (log_p_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        ent_coeff_loss.backward()
        self.alpha_optim.step()

        #######################################################
        # train critics                                       #
        #######################################################
        entropy_coefficient = self.log_alpha.exp().detach()

        with th.no_grad():
            tau_dash = self.base_tau_generator(shape=(batch_size, self.N_dash))
            next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
            next_qf_quantiles = self.critic_targets(next_obs, next_actions, tau_dash)
            next_target_critic = next_qf_quantiles.mean(dim=-1, keepdim=True).argmin(dim=1, keepdim=True)
            next_target_critic = next_target_critic.expand(batch_size, 1, self.N_dash)
            next_qf_quantiles = th.gather(next_qf_quantiles, index=next_target_critic, dim=1)
            next_qf_quantiles = next_qf_quantiles - entropy_coefficient * next_actions_log_p_pi[:, None, :]
            target_quantile = rewards[:, None, :] + self.gamma * (1. - dones[:, None, :]) * next_qf_quantiles

        taus = self.base_tau_generator(shape=(batch_size, self.N))
        current_quantiles = self.critics(obs, actions, taus)
        critic_loss = quantile_huber_loss(current_quantiles, target_quantile, taus[:, None, :, None],
                                          sum_over_quantiles=True)

        loss_summary["critic_loss"] = critic_loss.item()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        #######################################################
        # train actor                                         #
        #######################################################

        qf_current = self.critics(obs, current_actions, taus_action.detach())
        qf_current = th.mean(qf_current, dim=1)
        qf_current, _ = qf_current.min(dim=1, keepdim=True)
        assert qf_current.shape == log_p_pi.shape

        actor_loss = (entropy_coefficient * log_p_pi - qf_current)
        grad_norm = self.actor.calculate_grad_norm(obs, current_cvar_alpha)
        grad_penalty = ((grad_norm - 1) ** 2).mean(dim=1, keepdim=True)
        total_actor_loss = actor_loss + 10. * grad_penalty
        total_actor_loss = total_actor_loss.mean()
        actor_optim.zero_grad()
        total_actor_loss.backward()
        actor_optim.step()
        loss_summary["policy loss"] = actor_loss.mean().item()
        loss_summary["grad_penalty"] = grad_penalty.mean().item()/10
        loss_summary["entropy_loss"] = ent_coeff_loss.item()
        loss_summary["current entropy"] = -log_p_pi.mean().item()
        loss_summary["ent_coef"] = entropy_coefficient.item()
        ret = {}
        for k in loss_summary.keys():
            ret[f'loss/{k}'] = loss_summary[k]
        rand = np.random.uniform(0, 1)


        return ret

    # debug
    def check_dalpha_sign(self, obs):
        actions = []
        for alpha in [0.1, 0.2, 0.3, 0.5, 1.0]:
            _, _ , action= self.actor.sample(obs, th.repeat_interleave(th.Tensor([[alpha]]).to(self.device), dim=0,
                                                                    repeats=obs.shape[0]))
            actions.append(action)
        d_a = [0.1, 0.1, 0.2, 0.5]
        for (i, a), d_alpha in zip(enumerate(actions[:-1]), d_a):
            a_p1 = actions[i + 1]
            print(th.sign((a_p1 - a)))


class EVaRPolicy(IQNPolicy):
    def __init__(self, env, device, gamma=0.99,
                 N=8, N_dash=8, K=32, n_critics=2,
                 qf_kwargs=None, eta=0.75):

        super(EVaRPolicy, self).__init__(env, device, gamma, N, N_dash, K, n_critics,
                                            qf_kwargs, tau_generator=TauGenerator)
        self.logt = th.ones_like(self.log_alpha)
        self.logt.requires_grad = True
        self.t_optim = th.optim.Adam([self.logt], lr=3e-4)
        self.log_eta = np.log(eta)

    def train_step(self, batch_data, critic_optim, actor_optim):
        loss_summary = {}
        obs, actions, rewards, next_obs, dones, info = batch_data
        batch_size = obs.shape[0]

        taus_action = self.tau_generator(shape=(batch_size, self.K))
        current_actions, log_p_pi, _ = self.actor.sample(obs)
        #######################################################
        # train entropy coeff                                 #
        #######################################################

        ent_coeff_loss = -(self.log_alpha * (log_p_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        ent_coeff_loss.backward()
        self.alpha_optim.step()

        #######################################################
        # train critics                                       #
        #######################################################
        entropy_coefficient = self.log_alpha.exp().detach()

        with th.no_grad():
            tau_dash = self.base_tau_generator(shape=(batch_size, self.N_dash))
            next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
            next_qf_quantiles = self.critic_targets(next_obs, next_actions, tau_dash)
            next_target_critic = next_qf_quantiles.mean(dim=-1, keepdim=True).argmin(dim=1, keepdim=True)
            next_target_critic = next_target_critic.expand(batch_size, 1, self.N_dash)
            next_qf_quantiles = th.gather(next_qf_quantiles, index=next_target_critic, dim=1)
            next_qf_quantiles = next_qf_quantiles - entropy_coefficient * next_actions_log_p_pi[:, None, :]
            target_quantile = rewards[:, None, :] + self.gamma * (1. - dones[:, None, :]) * next_qf_quantiles

        taus = self.base_tau_generator(shape=(batch_size, self.N))
        current_quantiles = self.critics(obs, actions, taus)
        critic_loss = quantile_huber_loss(current_quantiles, target_quantile, taus[:, None, :, None],
                                          sum_over_quantiles=True)
        # consistency loss

        loss_summary["critic_loss"] = critic_loss.item()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        #######################################################
        # train actor                                         #
        #######################################################

        qf_current = self.critics(obs, current_actions, taus_action.detach())
        t = th.exp(self.logt)
        evar_estimated = ((th.logsumexp(self.log_eta -qf_current.detach() * t, dim=-1))/t).mean()
        self.t_optim.zero_grad()
        evar_estimated.backward()
        self.t_optim.step()
        t = t.detach()
        evar_current = (( th.logsumexp(self.log_eta - qf_current * t, dim=-1))/t)
        evar_current = evar_current.min(dim=-1, keepdim=True)[0]

        assert evar_current.shape == log_p_pi.shape

        actor_loss = (entropy_coefficient * log_p_pi - evar_current).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        loss_summary["t"] = t.item()
        loss_summary["evar"] = evar_estimated.item()
        loss_summary["policy loss"] = actor_loss.mean().item()
        loss_summary["entropy_loss"] = ent_coeff_loss.item()
        loss_summary["current entropy"] = -log_p_pi.mean().item()
        loss_summary["ent_coef"] = entropy_coefficient.item()
        ret = {}
        for k in loss_summary.keys():
            ret[f'loss/{k}'] = loss_summary[k]

        return ret


class DGPConsistCVaRSAC(AbstractSACPolicy):
    def __init__(self, env, device,  gamma=0.99,
                 N=8, N_dash=8, K=32, n_critics=2,
                 qf_kwargs=None, tau_generator=RandomCVaRTauGenerator , risk_kwargs=None):
        super(DGPConsistCVaRSAC, self).__init__(env, device)
        self.actor = MonotonicSACActor(self.observation_dim, self.action_dim)
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
        self.gp_scheduler = None

    def build_critics(self, critic_kwargs):
        return AlphaCritic(self.observation_dim, self.action_dim, self.n_critics, **critic_kwargs)

    def build_optim(self, actor_optim_cls=th.optim.Adam, critic_optim_cls=th.optim.Adam, actor_lr=3e-4, critic_lr=3e-4):
        actor_optim = actor_optim_cls(self.actor.parameters(), actor_lr)
        critic_optim = critic_optim_cls(self.critics.parameters(), critic_lr)
        return actor_optim, critic_optim

    def init_critic(self):
        self.critic_targets.requires_grad_(False)
        dump_state_dict(self.critics, self.critic_targets)

    def setup_scheduling(self, total_learning_timestep):
        self.gp_scheduler = SigmoidScheduler(total_learning_timestep)

    def target_update(self, tau):
        polyak_update(source=self.critics, target=self.critic_targets, tau=tau)

    def train_step(self, batch_data, critic_optim, actor_optim):
        loss_summary = {}
        obs, actions, rewards, next_obs, dones, info = batch_data
        batch_size = obs.shape[0]

        taus_action, current_cvar_alpha = self.tau_generator.sample(shape=(batch_size, self.K))
        current_cvar_alpha = th.repeat_interleave(current_cvar_alpha.reshape(1, 1), dim=0, repeats=batch_size)
        current_actions, log_p_pi, _ = self.actor.sample(obs, current_cvar_alpha)
        #######################################################
        # train entropy coeff                                 #
        #######################################################

        ent_coeff_loss = -(self.log_alpha * (log_p_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        ent_coeff_loss.backward()
        self.alpha_optim.step()

        #######################################################
        # train critics                                       #
        #######################################################
        entropy_coefficient = self.log_alpha.exp().detach()

        with th.no_grad():
            tau_dash = self.base_tau_generator(shape=(batch_size, self.N_dash))
            next_actions, next_actions_log_p_pi, _ = self.actor.sample(next_obs)
            next_qf_quantiles = self.critic_targets(next_obs, next_actions, tau_dash, current_cvar_alpha)
            next_target_critic = next_qf_quantiles.mean(dim=-1, keepdim=True).argmin(dim=1, keepdim=True)
            next_target_critic = next_target_critic.expand(batch_size, 1, self.N_dash)
            next_qf_quantiles = th.gather(next_qf_quantiles, index=next_target_critic, dim=1)
            next_qf_quantiles = next_qf_quantiles - entropy_coefficient * next_actions_log_p_pi[:, None, :]
            target_quantile = rewards[:, None, :] + self.gamma * (1. - dones[:, None, :]) * next_qf_quantiles

        taus = self.base_tau_generator(shape=(batch_size, self.N))
        current_quantiles = self.critics(obs, actions, taus, current_cvar_alpha.detach())
        critic_loss = quantile_huber_loss(current_quantiles, target_quantile, taus[:, None, :, None],
                                          sum_over_quantiles=True)
        # consistency loss

        loss_summary["critic_loss"] = critic_loss.item()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        #######################################################
        # train actor                                         #
        #######################################################

        qf_current = self.critics(obs, current_actions, taus_action.detach(), current_cvar_alpha.detach())
        qf_current = th.mean(qf_current, dim=1)
        qf_current, _ = qf_current.min(dim=1, keepdim=True)
        assert qf_current.shape == log_p_pi.shape

        actor_loss = (entropy_coefficient * log_p_pi - qf_current)
        # grad_norm = self.actor.calculate_grad_norm(obs, current_cvar_alpha)
        # grad_penalty = ((grad_norm - 1) ** 2).mean(dim=1, keepdim=True)
        total_actor_loss = actor_loss # + self.gp_scheduler() * grad_penalty
        total_actor_loss = total_actor_loss.mean()
        actor_optim.zero_grad()
        total_actor_loss.backward()
        actor_optim.step()

        loss_summary["policy loss"] = actor_loss.mean().item()
        # loss_summary["grad_penalty"] = grad_penalty.mean().item()/10
        loss_summary["entropy_loss"] = ent_coeff_loss.item()
        loss_summary["current entropy"] = -log_p_pi.mean().item()
        loss_summary["ent_coef"] = entropy_coefficient.item()
        ret = {}
        for k in loss_summary.keys():
            ret[f'loss/{k}'] = loss_summary[k]

        return ret

    # debug
    def check_dalpha_sign(self, obs):
        actions = []
        for alpha in [0.1, 0.2, 0.3, 0.5, 1.0]:
            _, _ , action= self.actor.sample(obs, th.repeat_interleave(th.Tensor([[alpha]]).to(self.device), dim=0,
                                                                    repeats=obs.shape[0]))
            actions.append(action)
        d_a = [0.1, 0.1, 0.2, 0.5]
        for (i, a), d_alpha in zip(enumerate(actions[:-1]), d_a):
            a_p1 = actions[i + 1]
            print(th.sign((a_p1 - a)))

policies = {"IQNPolicy": IQNPolicy,
            "CVaRPolicy": CVaRIQNPolicy,
            "WangPolicy": WangIQNPolicy,
            "PowerPolicy": PowerIQNPolicy,
            'FQFPolicy': FQFPolicy,
            "ConsistCVaRSAC": ConsistCVaRSAC,
            "DGPConsistCVaRSAC": DGPConsistCVaRSAC,
            "EVaRPolicy": EVaRPolicy,
            }
"""
"ODEMlpoIQNPolicy": ODEMlpoIQNPolicy,
"AutoRiskIQNPolicy": AutoRiskSACPolicy}
"""


