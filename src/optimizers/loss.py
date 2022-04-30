import torch.nn.functional as F
import torch.nn as nn


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def set_criterion(name, params=None):
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    elif name == 'NLLLoss':
        return nn.NLLLoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()