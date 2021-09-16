from hrl_pybullet_envs.envs.ant_maze import ant_maze_bullet_env
from functools import wraps


def rescale_action(method):
    @wraps(method)
    def _impl(self, action):
        action = self.preprocess_action(action)
        method_output = method(self, action)
        return method_output
    return _impl

import gym

class AntMazeEnv(gym.Env):
    def __init__(self):
        self.wrapped = ant_maze_bullet_env.AntMazeBulletEnv(targ_dist_rew=True, sense_target=True, inner_rew_weight=0.1,
                                                            max_steps=-1)

        self.observation_space = self.wrapped.observation_space
        self.wrapped_action_space = self.wrapped.action_space
        self.action_space = gym.spaces.Box(-1, 1, shape=self.wrapped_action_space.shape)
        self.action_scale = (self.wrapped_action_space.high - self.wrapped_action_space.low)/2
        self.action_low = self.wrapped_action_space.low + 1

    def preprocess_action(self, action):
        action = self.action_scale * action + self.action_low
        return action

    def render(self, mode="human"):
        return self.wrapped.render(mode)

    def reset(self):
        return self.wrapped.reset()

    @rescale_action
    def step(self, action):
        obs, reward, done, info = self.wrapped.step(action)

        if done and self.wrapped.robot.walk_target_dist < self.wrapped.tol:
            info["is_success"] = True
            reward = 100
        elif done:
            info["is_success"] = False
            reward = -100
        return obs, reward, done, info



if __name__ == "__main__":
    env = AntMazeEnv()
    sample = env.wrapped_action_space.sample()
    print(env.preprocess_action(sample) - sample)