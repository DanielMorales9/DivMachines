import torch
from torch import nn

class PairwiseMatrixFactorization(nn.Module):

    def __init__(self, user_features, item_features, factors):
        '''
        - input_features (int): the length of the input vector.
        - factors (int): the dimension of the interaction terms.
        '''

        super(PairwiseMatrixFactorization, self).__init__()
        self.batch_size = None

        self.item_features, self.user_features, self.factors = item_features, user_features, factors
        self.x = nn.Parameter(torch.Tensor(self.user_features, self.factors), requires_grad=True)
        self.y = nn.Parameter(torch.Tensor(self.user_features, self.factors), requires_grad=True)
        self.x.data.uniform_(-0.01, 0.01)
        self.y.data.uniform_(-0.01, 0.01)

    def forward(self, user, item_i, item_j):
        return (self.x[user].mul(self.y[item_i]) -
                self.x[user].mul(self.y[item_j])).sum(1)

