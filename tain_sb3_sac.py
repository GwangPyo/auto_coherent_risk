# from sac import SAC
from env_wrappers import wrapped_lunar_lander
from rl_utils.evaluation_utils import evaluate, collect
import numpy as np
from misc.seed import fix_seed
import pandas as pd
from sac import SAC
from misc.keras_progbar import Progbar


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
        model = SAC(env=generator(), policy="MlpPolicy")
        model.learn(int(5e+5))
        model.save(f"{1 - alpha}_{i}")

    from multiprocessing import Process
    processes = []
    cnt = 0
    for i in range(1):
        model = SAC.load(f"auto_cvar{i}")
        for alpha in [0.05, 0.1, 0.5, 1.]:
            model.policy.actor.set_rollout_alpha(alpha)
            env = SubprocVecEnv([generator for _ in range(32)])
            df = evaluate_vec_env(model, env, 10000)
            df = df.sort_values(by=["score"], ascending=True)
            print(i)
            print("cvar_0.05")
            print(df[:500].mean())
            print("cvar_0.1")
            print(df[:1000].mean())
            print("cvar_0.5")
            print(df[:5000].mean())
            print("Cvar_1")
            print(df.mean())


