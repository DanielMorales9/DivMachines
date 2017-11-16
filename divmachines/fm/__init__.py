from __future__ import print_function

from torch import nn

from divmachines.models import PointwiseModel

try:
    from divmachines.fm.second_order.fast import SecondOrderInteraction
except ImportError:
    FAST_VERSION = False
    from divmachines.fm.second_order import SecondOrderInteraction


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
        if not FAST_VERSION:
            print('Slow version of {0} is being used'.format(__name__))
        self.input_features, self.factors = n_features, n_factors
        self.linear = nn.Linear(self.input_features, 1)
        self.second_order = SecondOrderInteraction(self.input_features,
                                                   self.factors)

    def forward(self, x):
        # make sure everything is on the CPU.
        self.linear.cpu()
        self.second_order.cpu()

        back_to_gpu = False

        if x.is_cuda:
            x = x.cpu()
            back_to_gpu = True

        linear = self.linear(x)
        interaction = self.second_order(x)
        res = linear + interaction

        if back_to_gpu:
            res = res.cuda()
            x = x.cuda()

        return res
