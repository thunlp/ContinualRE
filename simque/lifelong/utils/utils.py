import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

def set_seed(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['n_gpu'] > 0 and torch.cuda.is_available() and config['use_gpu']:
        torch.cuda.manual_seed_all(seed)