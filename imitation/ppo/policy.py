from imitation.ppo.critic import Critic, QuantileCritic
from sac.risk_manager import TauGenerator
from imitation.ppo.actor import BoxActor, StateIndependentPolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import numpy as np
from rl_utils.utils import get_device
from rl_utils.logger import logkvs, dumpkvs, logkv
import torch as th
from collections import deque
from net.utils import quantile_huber_loss
from torch import nn
from rl_utils.evaluation_utils import evaluate
from imitation.trajectory_dataset import TrajectoryDataSet
from torch.utils.data import DataLoader
from imitation.discriminator import Discriminator



class PPO(object):
    def __init__(self, env, n_steps=64, device='auto', gamma=0.99):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.feature_dim = np.prod(env.observation_space.shape)
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        self.env = env
        self.policy = BoxActor(feature_dim=self.feature_dim, action_space=self.action_space)
        self.critic = QuantileCritic(feature_dim=self.feature_dim,)
        self.n_steps = n_steps
        self.action_shape = (-1, ) + self.action_space.shape
        if device == 'auto':
            device = get_device()
        self.device = device
        self.policy.to(self.device)
        self.critic.to(self.device)
        self.tau_generator = TauGenerator(self.device)
        self.gamma = gamma
        self.vf_optim = th.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.policy_optim = th.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.n_epochs = 10

    def preprocess_obs(self, obs):
        with th.no_grad():
            obs = th.from_numpy(obs).float().to(self.device)
            obs = obs.reshape(-1, self.feature_dim)
            return obs

    def process_action(self, action):
        action = action.detach().cpu().numpy()
        action = action.reshape(self.action_shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def learn(self, steps):
        obs = self.env.reset()
        obs = self.preprocess_obs(obs)
        mean_episode_rewards = deque(maxlen=100)
        episode_reward = []
        epi_len = 0
        batch_buffer = {"obs":[], "action":[], "next_obs":[], "reward": [], "dones":[]}
        for s in range(steps):

            epi_len += 1
            with th.no_grad():
                policy_action, _, _ = self.policy.sample(obs)
                step_action = self.process_action(policy_action)
                next_obs, reward, done, info = self.env.step(step_action)
                episode_reward.append(reward[0])
                next_obs = self.preprocess_obs(next_obs)
                batch_buffer["obs"].append(obs.clone())
                batch_buffer["action"].append(policy_action)
                batch_buffer["reward"].append(reward)
                batch_buffer["dones"].append(done)
                batch_buffer["next_obs"].append(next_obs.clone())
                obs = next_obs
            if s > 0 and s % self.n_steps == 0:
                losses = self.train_step(batch_buffer)
                batch_buffer = {"obs":[], "action":[], "next_obs":[], "reward": [], "dones":[]}
                logkvs(losses)
            if done[0]:
                mean_episode_rewards.append(np.sum(episode_reward))
                logkv("episode_len", epi_len)
                logkv("episode reward", np.sum(episode_reward))
                episode_reward = []
                logkv("mean_episode_rewards", np.mean(mean_episode_rewards))
                epi_len = 0
                dumpkvs()

    def train_step(self, batch):
        obs, action, reward, dones, next_obs = batch["obs"], batch["action"], batch["reward"], \
                                              batch["dones"], batch['next_obs']
        obs = th.cat(obs, dim=0)
        action = th.cat(action, dim=0)
        next_obs = th.cat(next_obs, dim=0)
        reward = th.from_numpy(np.asarray(reward, dtype=np.float32)).to(self.device).reshape(-1, 1)
        dones = th.from_numpy(np.asarray(dones, dtype=np.float32)).to(self.device).reshape(-1, 1)
        with th.no_grad():
            old_log_probs, _ = self.policy.evaluate_action(obs, action)
        approx_kl = []
        pg_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.n_epochs):
            log_probs, entropy = self.policy.evaluate_action(obs, action)
            with th.no_grad():
                taus = self.tau_generator(shape=(obs.shape[0], 64))
                advantage = th.sort((reward + (1. - dones) * self.gamma * self.critic(next_obs, taus)), dim=1, )[0]\
                            - th.sort(self.critic(obs, taus), dim=1)[0]
                advantage = advantage.mean(dim=1, keepdim=True)
                advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)

            ratio = th.exp(log_probs - old_log_probs)
            policy_loss_1 = advantage * ratio
            policy_loss_2 = advantage * th.clamp(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            pg_losses.append(policy_loss.item())
            entropy = entropy.mean()
            entropies.append(entropy.item())
            policy_loss = policy_loss - 0.03 * entropy
            self.policy_optim.zero_grad()
            policy_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optim.step()

            taus = self.tau_generator(shape=(obs.shape[0], 32))
            # tau_dash = self.tau_generator(shape=(obs.shape[0], 32))

            value = self.critic(obs, taus)
            with th.no_grad():
                value_target = reward + self.gamma * (1. - dones) * self.critic(next_obs, taus)
                value_target = value_target[:, None, :]
            value = value[:, :, None]

            td_error = th.abs(value_target - value)
            value_loss = quantile_huber_loss(td_error, taus=taus).mean()
            value_losses.append(value_loss.item())
            with th.no_grad():
                log_ratio = log_probs - old_log_probs
                approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl.append(approx_kl_div)
            self.vf_optim.zero_grad()
            value_loss.backward()
            # th.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.vf_optim.step()

        return {
                "vf_loss": np.mean(value_losses),
                "policy_loss": np.mean(pg_losses),
                "entropy": np.mean(entropies),
                "approx_kl": np.mean(approx_kl),

                }


class AIRLPPO(nn.Module):
    def __init__(self, env, expert_trajectory: str, n_steps=64, device='auto', gamma=0.99, n_epochs=50, entropy_coef=0.003,
                 max_grad_norm=10.):
        super(AIRLPPO, self).__init__()

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.feature_dim = np.prod(env.observation_space.shape)
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        self.env = env
        self.gamma = gamma

        self.discriminator = Discriminator(self.feature_dim, self.gamma)
        self.policy = StateIndependentPolicy(feature_dim=self.feature_dim, action_space=self.action_space)
        self.critic = QuantileCritic(feature_dim=self.feature_dim,)
        self.n_steps = n_steps
        self.action_shape = (-1, ) + self.action_space.shape
        if device == 'auto':
            device = get_device()
        self.device = device

        self.expert_trajectory = TrajectoryDataSet(expert_trajectory, self.device)
        self.policy.to(self.device)
        self.critic.to(self.device)
        self.discriminator.to(self.device)
        self.discriminator.clip_weight(-0.01, 0.01)
        self.tau_generator = TauGenerator(self.device)

        self.vf_optim = th.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.policy_optim = th.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.n_epochs = n_epochs
        self.batch_size = n_steps * self.env.num_envs

    def preprocess_obs(self, obs):
        with th.no_grad():
            obs = th.from_numpy(obs).float().to(self.device)
            obs = obs.reshape(-1, self.feature_dim)
            return obs

    def process_action(self, action):
        action = action.detach().cpu().numpy()
        action = action.reshape(self.action_shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def learn(self, steps):
        obs = self.env.reset()
        obs = self.preprocess_obs(obs)
        mean_episode_rewards = deque(maxlen=100)
        episode_reward = []
        epi_len = 0
        batch_buffer = {"obs": [], "action": [], "next_obs": [], "dones": []}
        n_updates = 0
        n_episodes = 0
        for s in range(steps):
            epi_len += 1
            logkv("time/steps", s)

            with th.no_grad():
                policy_action, _, _ = self.policy.sample(obs)
                step_action = self.process_action(policy_action)
                next_obs, reward, done, info = self.env.step(step_action)
                episode_reward.append(reward[0])
                next_obs = self.preprocess_obs(next_obs)
                batch_buffer["obs"].append(obs.clone())
                batch_buffer["action"].append(policy_action)
                batch_buffer["dones"].append(done)
                batch_buffer["next_obs"].append(next_obs.clone())
                obs = next_obs

            if s > 0 and s % self.n_steps == 0:
                n_updates += 1
                logkv("time/n_upates", n_updates)
                batch_buffer["obs"] = th.cat(batch_buffer["obs"], dim=0)
                batch_buffer["action"] = th.cat(batch_buffer["action"], dim=0)
                batch_buffer["dones"] =   th.from_numpy(np.asarray(batch_buffer["dones"], dtype=np.float32)).to(self.device).reshape(-1, 1)
                batch_buffer["next_obs"] = th.cat(batch_buffer["next_obs"], dim=0)

                with th.no_grad():
                    logprob, _ = self.policy.evaluate_action(batch_buffer["obs"], batch_buffer["action"])

                trainee_batch = (batch_buffer["obs"], batch_buffer["dones"], logprob, batch_buffer["next_obs"])
                expert_obs, expert_action, expert_next_obs, expert_done, _, _ = self.expert_trajectory.sample(batch_size=self.batch_size)
                with th.no_grad():
                    expert_logp_pi, _ = self.policy.evaluate_action(expert_obs, expert_action)
                    expert_batch = (expert_obs, expert_done, expert_logp_pi, expert_next_obs)

                for _ in range(10):
                    loss_discriminator = self.discriminator.update_disc(trainee_batch, expert_batch)
                    logkvs(loss_discriminator)
                with th.no_grad():
                    rewards = self.discriminator.forward(*trainee_batch)
                batch_buffer["reward"] = rewards
                batch_buffer["log_prob"] = logprob
                losses = self.train_step(batch_buffer)
                logkv("rl_loss/discriminator_rewards", th.mean(rewards).item())
                batch_buffer = {"obs": [], "action": [], "next_obs": [], "reward": [], "dones": []}
                logkvs(losses)

            if done[0]:
                n_episodes += 1
                logkv("episode/num_episode", n_episodes)
                mean_episode_rewards.append(np.sum(episode_reward))
                logkv("episode/len", epi_len)
                logkv("episode/last_reward", np.sum(episode_reward))
                episode_reward = []
                logkv("episode/mean_episode_rewards", np.mean(mean_episode_rewards))
                epi_len = 0
                dumpkvs()

    def train_step(self, batch):
        obs, action, reward, dones, next_obs, old_log_probs = batch["obs"], batch["action"], batch["reward"], \
                                               batch["dones"], batch['next_obs'], batch["log_prob"]

        approx_kl = []
        pg_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.n_epochs):
            log_probs, entropy = self.policy.evaluate_action(obs, action)
            with th.no_grad():
                taus = self.tau_generator(shape=(obs.shape[0], 64))
                advantage = th.sort((reward + (1. - dones) * self.gamma * self.critic(next_obs, taus)), dim=1, )[0] \
                            - th.sort(self.critic(obs, taus), dim=1)[0]
                advantage = advantage.mean(dim=1, keepdim=True)
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            ratio = th.exp(log_probs - old_log_probs)
            policy_loss_1 = advantage * ratio
            policy_loss_2 = advantage * th.clamp(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            pg_losses.append(policy_loss.item())
            if entropy is not None:
                entropy = entropy.mean()
            else:
                entropy = -th.mean(log_probs)
            entropies.append(entropy.item())
            policy_loss = policy_loss - self.entropy_coef * entropy
            self.policy_optim.zero_grad()
            policy_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optim.step()

            taus = self.tau_generator(shape=(obs.shape[0], 32))
            # tau_dash = self.tau_generator(shape=(obs.shape[0], 32))

            value = self.critic(obs, taus)
            with th.no_grad():
                value_target = reward + self.gamma * (1. - dones) * self.critic(next_obs, taus)
                value_target = value_target[:, None, :]
            value = value[:, :, None]

            td_error = th.abs(value_target - value)
            value_loss = quantile_huber_loss(td_error, taus=taus).mean()
            value_losses.append(value_loss.item())
            with th.no_grad():
                log_ratio = log_probs - old_log_probs
                approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl.append(approx_kl_div)
            self.vf_optim.zero_grad()
            value_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.vf_optim.step()

        return {
            "rl_loss/vf_loss": np.mean(value_losses),
            "rl_loss/policy_loss": np.mean(pg_losses),
            "rl_loss/entropy": np.mean(entropies),
            "rl_loss/approx_kl": np.mean(approx_kl),

        }

if __name__ == "__main__":
    import gym
    from stable_baselines3.common.vec_env import SubprocVecEnv
    model = AIRLPPO(env=SubprocVecEnv([lambda: gym.make("LunarLanderContinuous-v2") for _ in range(1)]),
                    expert_trajectory='/home/yoo/auto_coherent_risk/episode_data.pkl',
                    n_steps=16)
    model.learn(10000000)


