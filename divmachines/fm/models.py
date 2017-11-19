from __future__ import print_function

import torch
from torch.nn import Parameter
from divmachines import PointwiseModel
from divmachines.layers import ZeroEmbedding

FAST_VERSION = True
from divmachines.fm.second_order.fast.second_order_fast import SecondOrderInteraction


class PointwiseModelFM(PointwiseModel):
    """
    Base class for Pointwise model with Factorization Machine
    """
    pass


class FactorizationMachine(PointwiseModelFM):
    """
    Pointwise Factorization Machine Model

    Parameters
    ----------
    n_features: int
        Length of the input vector.
    n_factors: int, optional
        Number of factors of the factorized parameters
    """
    def __init__(self, n_features, n_factors=10):

        super(FactorizationMachine, self).__init__()
        if not FAST_VERSION:
            print('Slow version of {0} is being used'.format(__name__))
        self.n_features, self.factors = n_features, n_factors
        self.linear = Parameter(torch.Tensor(self.n_features))
        self.linear.data.uniform_(-0.01, 0.01)
        self.second_order = SecondOrderInteraction(self.n_features,
                                                   self.factors)

    def forward(self, x):
        linear = (x * self.linear).sum(1).unsqueeze(-1)
        interaction = self.second_order(x)
        res = linear + interaction

        return res
