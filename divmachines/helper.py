import numpy as np
import numbers
import torch
from torch.autograd import Variable

from divmachines.torch_utils import gpu


def _prepare_for_prediction(x, feats):

    if type(x) is np.ndarray:
        if len(x.shape) == 2:
            if x.shape[1] == 1:
                raise ValueError("Array must have user and item columns")
            elif x.shape[1] == 2:
                return x
        elif len(x.shape) == 1 and len(x) == feats:
            return np.array([x])
        else:
            raise ValueError("Array dimensions not allowed")
    else:
        raise ValueError("Variable type not allowed")


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
