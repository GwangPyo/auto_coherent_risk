from sac import SAC
from env_wrappers import wrapped_lunar_lander
from rl_utils.evaluation_utils import evaluate, collect
import numpy as np
from misc.seed import fix_seed
import pandas as pd


if __name__ == '__main__':
    learning_steps = int(3e+5)

    fix_seed(7777)
    """
    model = SAC.load(f"/home/yoo/risk_results/models/iqn_wang_{-0.75}")
    episodes = collect(env=wrapped_lunar_lander(), model=model, n_episodes=10000)
    import pickle
    with open("episode_data.pkl", "wb") as f:
        pickle.dump(episodes, f)

    evaluate(env=wrapped_lunar_lander(), model=model, steps=10000,
             save_path=f"/home/yoo/risk_results/iqn_wang_{-0.75}_10000.csv")
    """
    model = SAC(env=wrapped_lunar_lander(), policy="ODEIQNPolicy")
    model.learn(learning_steps)
    model.save(f"/home/yoo/risk_results/models/auto")
    evaluate(env=wrapped_lunar_lander(), model=model, steps=10000,
              save_path=f"/home/yoo/risk_results/auto_10000.csv")
    """
    rl_model = SAC(policy="IQNPolicy", env= wrapped_lunar_lander(),  )
    rl_model.learn(learning_steps, tb_log_dir="/home/yoo/risk_rl_tb_log", tb_log_name=f"iqn_lunarlander_ode", tb_log_option="Force")
    rl_model.save(f"/home/yoo/risk_results/models/neutral")

    evaluate(env=wrapped_lunar_lander(), model=rl_model, steps=10000,
             save_path=f"/home/yoo/risk_results/neutral_10000.csv")
    
    for alpha in np.arange(0.1, 0.6, 0.1):
        fix_seed(7777)
        model = SAC(env=wrapped_lunar_lander(), policy="CVaRPolicy", policy_kwargs={"cvar_alpha": alpha},
                    )
        model.learn(learning_steps, tb_log_dir="/home/yoo/risk_rl_tb_log", tb_log_name=f"iqn_cvar{alpha}_bipdeal", tb_log_option="Force")
        model.save(f"/home/yoo/risk_results/models/iqn_cvar_{alpha}")
        evaluate(env=wrapped_lunar_lander(), model=model, steps=10000,
                 save_path=f"/home/yoo/risk_results/cvar_{alpha}_10000.csv")
    """