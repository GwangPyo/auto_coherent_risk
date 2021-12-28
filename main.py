# from sac import SAC
from env_wrappers import wrapped_lunar_lander
from rl_utils.evaluation_utils import evaluate, collect
import numpy as np
from misc.seed import fix_seed
import pandas as pd
# from sac import SAC
from misc.keras_progbar import Progbar
from evar_sac.evar_sac import EVaRSAC
# from stable_baselines3 import SAC
from sac import SAC
from imitation.trainer import Trainer


if __name__ == '__main__':
    import gym
    import navigation_2d

    from stable_baselines3.common.vec_env import SubprocVecEnv
   #  env = SubprocVecEnv([lambda: gym.make("LunarLanderContinuous-v2") for _ in range(10)])
   #  print(env.num_envs)
    class Wrapped(gym.Env):
        def __init__(self):
            self.wrapped = gym.make("LunarLanderContinuous-v2")
            self.observation_space = self.wrapped.observation_space
            self.action_space = self.wrapped.action_space

        def reset(self):
            return self.wrapped.reset()

        def step(self, action):
            obs, reward, done, info = self.wrapped.step(action)
            if done and reward != 100:
                info["is_success"] = False
            else:
                info["is_success"] = True
            return obs, reward, done, info


    def evaluate(model, env, step=10000):
        scores = []
        success = []
        progbar = Progbar(step)

        for _ in range(step):
            done = False
            obs = env.reset()
            score = 0
            transitions = []
            while not done:
                action, _ = model.predict(obs)
                next_obs, reward, done, info = env.step(action)
                transitions.append({"s": np.copy(np.squeeze(obs)), "a": np.squeeze(action), "r": reward, "done": done})
                score += reward
                obs = next_obs
                if done:
                    success.append(info[0]["is_success"])
                    scores.append(score)
                    progbar.add(1, [("score", score), ("success", success[-1])])

        print("mean success", np.mean(success))
        print("mean score", np.mean(scores))
        return pd.DataFrame.from_dict({"score": scores, "success":success})

    def evaluate_vec_env(model, env: SubprocVecEnv , step=10000):
        num_env = env.num_envs
        scores = []
        score = np.zeros(shape=(num_env, ))
        done = np.zeros(shape=(num_env, ), dtype=np.bool_)
        n_dones = 0
        progbar = Progbar(step)

        obs = env.reset()
        while n_dones < step:
            action, _ =model.predict(obs)
            obs, reward, done, info = env.step(action)
            score = score + np.asarray(reward)
            if done.any():
                idx = np.where(done)[0]
                for s in score[idx]:
                    scores.append(s)
                    progbar.add(1, [("score", s)])

                score[idx] = 0.
                n_dones += len(idx)
        print(len(scores))
        return pd.DataFrame.from_dict({"score": scores})


    def generator():
        import navigation_2d
        import warnings
        warnings.filterwarnings("ignore")
        return gym.make("Navi-Acc-Full-Obs-Task0_easy-v0")

    def func(verbose, alpha, i):
        model = EVaRSAC(env=generator(), policy="MlpPolicy", verbose=verbose, device='cpu', target_entropy=np.log(1 - alpha))
        model.learn(int(5e+5))
        model.save(f"evar_sac_{1 - alpha}_{i}")

    from multiprocessing import Process
    processes = []
    cnt = 0
    for i in range(30):
        for alpha in [0.05, 0.1, 0.5, 1]:
            model = SAC(env=generator(), policy="EVaRPolicy", policy_kwargs={"eta": 0.75})  # , verbose=1, target_entropy=np.log(1 - alpha))
            model.learn(int(3e+5))
            model.save(f"cvar_alpha_{alpha}")

