from hrl_pybullet_envs.envs.ant_maze import ant_maze_bullet_env

"""
    def step(self, a):
        self.t += 1
        if self.debug > 0:
            debug_draw_point(self.scene._p, *self.target, colour=[0.1, 0.5, 0.7])
        ant_obs, inner_rew, d, i = super().step(a)
        obs = self._get_obs(ant_obs)

        rew = inner_rew * self.inner_rew_weight

        if self.robot.walk_target_dist < self.tol:
            if self.done_at_target or (not self.done_at_target and self.t == self.max_steps - 1):
                rew += 1
                d = True

        if self.t == self.max_steps - 1:
            d = True

        if self.targ_dist_rew and d:  # rewarding based on distance to target on final step
            rew -= self.robot.walk_target_dist

        return obs, rew, d, i

"""
import gym

class AntMazeEnv(gym.Env):
    def __init__(self):
        self.wrapped = ant_maze_bullet_env.AntMazeBulletEnv(targ_dist_rew=True, sense_target=True, inner_rew_weight=0.1,
                                                            max_steps=-1)
        self.observation_space = self.wrapped.observation_space
        self.action_space = self.wrapped.action_space

    def render(self, mode="human"):
        return self.wrapped.render(mode)

    def reset(self):
        return self.wrapped.reset()

    def step(self, action):
        obs, reward, done, info = self.wrapped.step(action)

        if done and self.wrapped.robot.walk_target_dist < self.wrapped.tol:
            info["is_success"] = True
            reward = 100
        elif done:
            info["is_success"] = False
            reward = -100
        return obs, reward, done, info



