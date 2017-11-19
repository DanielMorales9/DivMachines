from __future__ import print_function
import torch
from torch.nn import Module, Parameter
from torch.autograd import Variable


class SecondOrderInteraction(Module):
    """
    Factorized parameters for the Second Order Interactions

    Parameters
    ----------
    n_features: int
        Length of the input vector.
    n_factors: int, optional
        Number of factors of the factorized parameters
    """

    def __init__(self, n_features, n_factors):
        super(SecondOrderInteraction, self).__init__()
        self.batch_size = None
        self.n_features = n_features
        self.n_factors = n_factors

        self.v = Parameter(torch.Tensor(self.n_features, self.n_factors))
        self.v.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        self.batch_size = x.size()[0]
        pow_x = torch.pow(x, 2)
        pow_v = torch.pow(self.v, 2)
        pow_sum = torch.pow(torch.mm(x, self.v), 2)
        sum_pow = torch.mm(pow_x, pow_v)
        out = 0.5* (pow_sum - sum_pow).sum(1)

        return out.unsqueeze(-1)
