from __future__ import print_function

from torch import FloatTensor
from torch.nn import Parameter
from divmachines import PointwiseModel
from divmachines.fm.second_order import SecondOrderInteraction as SOI


class FactorizationMachine(PointwiseModel):
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

        self.n_features, self.factors = n_features, n_factors
        self.linear = Parameter(FloatTensor(self.n_features))
        self.linear.data.uniform_(-0.01, 0.01)
        self.second_order = SOI(self.n_features,
                                     self.factors)

    def forward(self, x):
        linear = (x * self.linear).sum(1).unsqueeze(-1)
        interaction = self.second_order(x)
        res = linear + interaction

        return res
