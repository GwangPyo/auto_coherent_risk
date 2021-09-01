import numpy as np
import torch as th
import random


def fix_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)

