from sac import SAC
from misc.keras_progbar import Progbar
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
import pandas as pd


if __name__ == '__main__':

    models = [SAC.load(f"consist_cvar_dpg_sac_{i}") for i in range(6)]


    def generator():
        import navigation_2d
        import warnings
        warnings.filterwarnings("ignore")
        return gym.make("Navi-Acc-Full-Obs-Task0_easy-v0")

    env = generator()
    for _ in range(100):
        obs = env.reset()
        for _ in range(20):
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)
            if done:
                print("SAFSDFA")
                exit()
        actions = {i:[] for i in range(6)}
        for alpha in [0.1, 0.2, 0.3, 0.5, 1.0]:
            for i, model in enumerate(models):
                model.policy.actor.set_rollout_alpha(alpha)
                action, _ = model.predict(obs)
                actions[i].append(action)
        d_a = [0.1, 0.1, 0.2, 0.5]
        for j in range(6):
            print("j:", j)
            for (i, a), d_alpha in zip(enumerate(actions[j][:-1]), d_a):
                a_p1 = actions[j][i + 1]
                print(((a_p1 - a)/d_alpha))
            print("\n\n")