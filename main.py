from sac import SAC
from lunarlander_wrapper import wrapped_lunar_lander
from rl_utils.utils import evaluate
import numpy as np
from misc.seed import fix_seed


if __name__ == '__main__':
    fix_seed(7777)
    rl_model = SAC(policy="AutoRiskIQNPolicy", env= wrapped_lunar_lander())
    rl_model.learn(300000, )
    rl_model.save(f"/home/yoo/risk_results/models/auto_risk")
    evaluate(env=wrapped_lunar_lander(), model=rl_model, steps=1000,
             save_path=f"/home/yoo/risk_results/auto_risk.csv")
    for alpha in np.arange(0.1, 0.6, 0.1):
        fix_seed(7777)
        model = SAC(env=wrapped_lunar_lander(), policy="MlpIQNPolicy", policy_kwargs={"cvar_alpha": alpha})
        model.learn(300000)
        model.save(f"/home/yoo/risk_results/models/iqn_cvar_{alpha}")
        evaluate(env=wrapped_lunar_lander(), model=model, steps=1000,
                 save_path=f"/home/yoo/risk_results/cvar_{alpha}.csv")
