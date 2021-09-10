from stable_baselines3 import SAC
from env_wrappers import BipedalWalkerHardcoreWrapper
from rl_utils.evaluation_utils import evaluate
import numpy as np
from misc.seed import fix_seed


if __name__ == '__main__':

    rl_model = SAC(policy="MlpPolicy", env=BipedalWalkerHardcoreWrapper(),  verbose=1)
    rl_model.learn(30000000)
    rl_model.save("sb3_bipedalwalker_hardcore")
    """
    rl_model.learn(300000, tb_log_dir="/home/yoo/risk_rl_tb_log", tb_log_name=f"iqn_auto_bipdeal", tb_log_option="Force")
    rl_model.save(f"/home/yoo/risk_results/models/auto_risk")
    evaluate(env=wrapped_lunar_lander(), model=rl_model, steps=10000,
             save_path=f"/home/yoo/risk_results/auto_risk_10000.csv")
    for alpha in np.arange(0.6, 1.1, 0.1):
        fix_seed(7777)
        model = SAC(env=wrapped_lunar_lander(), policy="MlpIQNPolicy", policy_kwargs={"cvar_alpha": alpha})
        model.learn(300000)
        model.save(f"/home/yoo/risk_results/models/iqn_cvar_{alpha}")
        evaluate(env=wrapped_lunar_lander(), model=model, steps=10000,
                 save_path=f"/home/yoo/risk_results/cvar_{alpha}_10000.csv")
    model = SAC(env=wrapped_lunar_lander(), policy="WangPolicy", policy_kwargs={"eta": -0.75})
    model.learn(300000)
    model.save(f"/home/yoo/risk_results/models/iqn_wang_{-0.75}")
    evaluate(env=wrapped_lunar_lander(), model=model, steps=10000,
             save_path=f"/home/yoo/risk_results/iqn_wang_{-0.75}_10000.csv")

    model = SAC(env=wrapped_lunar_lander(), policy="PowerPolicy", policy_kwargs={"eta": -0.75})
    model.learn(300000)
    model.save(f"/home/yoo/risk_results/models/iqn_power_{-0.75}")
    evaluate(env=wrapped_lunar_lander(), model=model, steps=10000,
             save_path=f"/home/yoo/risk_results/iqn_power_{-0.75}_10000.csv")
    
    """