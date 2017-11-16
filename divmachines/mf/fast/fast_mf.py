import torch
from torch import nn

from .fast_inner import fast_forward, fast_backward


class MatrixFactorization(torch.nn.Module):

    def __init__(self, n_users, n_items, factors):
        '''
        - input_features (int): the length of the input vector.
        - factors (int): the dimension of the interaction terms.
        '''

        super(MatrixFactorization, self).__init__()
        self.n_items, self.n_users, self.factors = n_items, n_users, factors
        self.x = nn.Parameter(torch.Tensor(self.n_users, self.factors), requires_grad=True)
        self.y = nn.Parameter(torch.Tensor(self.n_items, self.factors), requires_grad=True)
        self.x.data.uniform_(-0.01, 0.01)
        self.y.data.uniform_(-0.01, 0.01)

    def forward(self, users, items):
        return ProductFunction()(users, items, self.x, self.y)


class ProductFunction(torch.autograd.Function):
    def forward(self, users, items, x, y):
        return fast_forward(self, users, items, x, y)

    def backward(self, grad_output):
        return fast_backward(self, grad_output)

