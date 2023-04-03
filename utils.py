import random

import numpy as np
import torch


def set_random_seed(seed=131, reproduce=False):
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    torch.cuda.manual_seed(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
