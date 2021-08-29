from sac import SAC
from lunarlander_wrapper import wrapped_lunar_lander
from rl_utils.utils import evaluate
import numpy as np


if __name__ == '__main__':
    for alpha in np.arange(0.1, 1.1, 0.1):
        model = SAC(env=wrapped_lunar_lander(), policy="MlpIQNPolicy", policy_kwargs={"cvar_alpha": alpha})
        model.learn(300000)
        model.save(f"/home/yoo/risk_results/models/iqn_cvar_{alpha}")
        evaluate(env=wrapped_lunar_lander(), model=model, steps=1000,
                 save_path=f"/home/yoo/risk_results/cvar_{alpha}.csv")