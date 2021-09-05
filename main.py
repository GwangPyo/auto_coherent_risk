from sac import SAC
from env_wrappers import wrapped_bipdeal_walker_hardcore
from rl_utils.evaluation_utils import evaluate
import numpy as np
from misc.seed import fix_seed


if __name__ == '__main__':
    learning_steps = int(3e+6)
    fix_seed(7777)
    rl_model = SAC(policy="AutoRiskIQNPolicy", env= wrapped_bipdeal_walker_hardcore(), batch_size=256)
    rl_model.learn(learning_steps, tb_log_dir="/home/yoo/risk_rl_tb_log", tb_log_name=f"iqn_auto_bipdeal", tb_log_option="Force")
    rl_model.save(f"/home/yoo/risk_results/models/auto_risk")
    evaluate(env=wrapped_bipdeal_walker_hardcore(), model=rl_model, steps=10000,
             save_path=f"/home/yoo/risk_results/auto_risk_10000.csv")

    for alpha in np.arange(0.1, 0.6, 0.1):
        fix_seed(7777)
        model = SAC(env=wrapped_bipdeal_walker_hardcore(), policy="MlpIQNPolicy", policy_kwargs={"cvar_alpha": alpha})
        model.learn(learning_steps, tb_log_dir="/home/yoo/risk_rl_tb_log", tb_log_name=f"iqn_cvar{alpha}_bipdeal", tb_log_option="Force")
        model.save(f"/home/yoo/risk_results/models/iqn_cvar_{alpha}")
        evaluate(env=wrapped_bipdeal_walker_hardcore(), model=model, steps=10000,
                 save_path=f"/home/yoo/risk_results/cvar_{alpha}_10000.csv")
    model = SAC(env=wrapped_bipdeal_walker_hardcore(), policy="WangPolicy", policy_kwargs={"eta": -0.75})
    model.learn(learning_steps)
    model.save(f"/home/yoo/risk_results/models/iqn_wang_{-0.75}")
    evaluate(env=wrapped_bipdeal_walker_hardcore(), model=model, steps=10000,
             save_path=f"/home/yoo/risk_results/iqn_wang_{-0.75}_10000.csv")

    model = SAC(env=wrapped_bipdeal_walker_hardcore(), policy="PowerPolicy", policy_kwargs={"eta": -0.75})
    model.learn(learning_steps)
    model.save(f"/home/yoo/risk_results/models/iqn_power_{-0.75}")
    evaluate(env=wrapped_bipdeal_walker_hardcore(), model=model, steps=10000,
             save_path=f"/home/yoo/risk_results/iqn_power_{-0.75}_10000.csv")
