from sac import SAC
from env_wrappers import wrapped_lunar_lander
from rl_utils.evaluation_utils import evaluate
import numpy as np
from misc.seed import fix_seed


if __name__ == '__main__':


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
