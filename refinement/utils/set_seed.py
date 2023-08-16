import torch
import numpy as np
import random

def set_seed(seed):
    # seed setting
    if seed == None:
        seed = np.random.randint(low=-2147483648, high=2147483647)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False