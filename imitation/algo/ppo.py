import torch
from torch import nn
from torch.optim import Adam

from imitation.algo.base import Algorithm
from imitation.buffer import RolloutBuffer
from imitation.ppo.actor import StateIndependentPolicy
from imitation.ppo.critic import ScalarVfunction, Vfunction
from imitation.trainer import Trainer
from imitation.env import NormalizedEnv


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(values)
    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, env, test_env, device='auto', seed='auto', gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0, use_quantiles=True):
        super().__init__(env, device, seed, gamma=gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=self.observation_space.shape,
            action_shape=self.action_space.shape,
            device=self.device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            self.state_shape,
            self.action_space,
            units_actor
        ).to(self.device)
        self.use_quantiles = use_quantiles
        # Critic.
        if not use_quantiles:
            self.critic = ScalarVfunction(
                self.state_shape,
                units_critic
            ).to(self.device)
        else:
            self.critic = Vfunction(
                self.state_shape,
                units_critic
            ).to(self.device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        if not isinstance(env, NormalizedEnv):
            test_env = NormalizedEnv(env)
        self.test_env = test_env
        self.trainer = Trainer(self.wrap_env(env), self.test_env, algo=self, log_dir="/home/yoo")

    def is_update(self, step):
        return step % self.rollout_length == 0

    def learn(self, steps):
        self.trainer.num_steps = steps
        return self.trainer.train()

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = done
        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            if self.use_quantiles:
                taus = torch.randn(size=(states.shape[0], self.critic.N), device=self.device)
                value_arg = (states, taus)
                target_value_arg = (next_states, taus)
            else:
                value_arg = (states, )
                target_value_arg = (next_states, )
            values = self.critic(*value_arg)
            next_values = self.critic(*target_value_arg)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        if self.use_quantiles:
            taus = torch.randn(size=(states.shape[0], self.critic.N), device=self.device)
            value = self.critic(states, taus)
            loss_critic = self.critic.loss(value, targets, taus)
        else:
            value = self.critic(states)
            loss_critic = self.critic.loss(value, targets, None)
        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis, _  = self.actor.evaluate_action(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        pass


if __name__ == '__main__':
    import gym
    test = PPO(env=gym.make("BipedalWalker-v3"), test_env=gym.make("BipedalWalker-v3"))
    test.learn(100000)