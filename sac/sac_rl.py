import numpy as np
from rl_utils import ReplayBuffer, get_device
from rl_utils.utils import DummyEnv
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from rl_utils.logger import logkv, logkvs, logkv_mean, dump_tabular
from datetime import timedelta, datetime
import time
from collections import deque
from sac.policy import policies
from torch.optim import Adam
import torch as th
import pickle
from torch.utils.tensorboard import SummaryWriter


class SAC(object):
    def __init__(self, env, policy, buffer_size=int(1e+5), policy_kwargs=None, batch_size=256, tau=5e-3):
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])

        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.device = get_device()
        self.buffer = ReplayBuffer(buffer_size, self.device)
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_class = policies[policy]
        self.policy_name = policy
        self.policy_kwargs = policy_kwargs
        self.policy = policy_class(self.env, device=self.device, **policy_kwargs)
        self.batch_size = batch_size
        self.num_envs = self.env.num_envs
        self.actor_optim, self.critic_optim = self.policy.build_optim()
        self.tau = tau

        assert self.num_envs == 1

    def sample_random_actions(self):
        return np.asarray([self.action_space.sample() for _ in range(self.num_envs)])

    def set_env(self, env):
        if not isinstance(env, DummyEnv):
            env = DummyVecEnv([lambda: env])
        self.env = env

    def predict(self, obs, deterministic=True):
        return self.policy.predict(obs, deterministic), None

    @staticmethod
    def log(dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                SAC.log(v)
            else:
                logkv(k, v)

    def learn(self, steps=10000, learning_starts=1000):
        obs = self.env.reset()
        mean_episode_rewards = deque(maxlen=100)
        mean_episode_success = deque(maxlen=100)
        episode_rewards = []
        start_time = time.time()
        n_episodes = 0
        epilen = 0

        writer = SummaryWriter(log_dir="/home/yoo/risk_rl_tb_log")
        for s in range(steps):
            if s < learning_starts:
                actions = self.sample_random_actions()
            else:
                actions = self.policy.train_predict(obs)
                actions = actions[None]

            next_obs, reward, done, info = self.env.step(actions)
            episode_rewards.append(reward)
            epilen += 1
            if done[0]:
                obs = self.env.reset()
                mean_episode_rewards.append(np.sum(episode_rewards))
                episode_rewards = []
                current_time = time.time()
                n_episodes += 1
                logkv("episode/len", epilen)
                epilen = 0
                logkv("episode/n_episode", n_episodes)
                logkv(f"episode/mean_100 reward", np.mean(mean_episode_rewards))
                writer.add_scalar("episode/episode_reward", mean_episode_rewards[-1], s)
                logkv("time/time_elapsed", timedelta(seconds=int(current_time - start_time)))
                logkv("time/steps", s)
                if "is_success" in info[0].keys():
                    mean_episode_success.append(float(info[0]["is_success"]))
                    mean_succ = np.mean(mean_episode_success)
                    logkv("episode/100 epi succ_ratio", mean_succ)
                    writer.add_scalar("episode/episode_success", float(info[0]["is_success"]), s)

                fps = s /(current_time - start_time + 1e-12)
                logkv("time/fps", int(fps))
                eta = (steps - s)/fps
                logkv("time/eta", timedelta(seconds=int(eta)))
                dump_tabular()
                if n_episodes % 10 == 0:
                    writer.flush()
                    print("flush writer")
            self.buffer.add(obs.copy(), actions, reward, next_obs.copy(), done, info)
            obs = next_obs

            if s > learning_starts:
                batch_data = self.buffer.sample(self.batch_size)
                losses = self.policy.train_step(batch_data, critic_optim=self.critic_optim, actor_optim=self.actor_optim)
                logkvs(losses)
                for k, v in losses.items():
                    writer.add_scalar(k, v, s)
                self.policy.target_update(self.tau)

        writer.close()

    def save(self, path):

        dummy_env = DummyEnv(self.observation_space, self.action_space)
        meta_data = self.env.metadata
        initializers = {
            "env": dummy_env,
            "policy": self.policy_name,
            "policy_kwargs": self.policy_kwargs,
        }
        state_dict = self.policy.state_dict()
        with open(path, "wb") as f:
            pickle.dump([initializers, state_dict, meta_data], f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            [initializer, state_dict, meta_data] = pickle.load(f)
        setattr(initializer["env"], "metadata", meta_data)
        model = cls(**initializer)
        model.policy.load_state_dict(state_dict)
        return model


