import gym


class LunarLanderWrapper(gym.Env):
    def __init__(self):
        self.wrapped = gym.make("LunarLanderContinuous-v2")
        self.observation_space = self.wrapped.observation_space
        self.action_space = self.wrapped.action_space

    def reset(self):
        return self.wrapped.reset()

    def step(self, action):
        next_obs, reward, done, info = self.wrapped.step(action)
        if done and reward != 100:
            info["is_success"] = False
        elif done and reward == 100:
            info["is_success"] = True

        return next_obs, reward, done, info

    def render(self, mode="human"):
        return self.wrapped.render(mode)