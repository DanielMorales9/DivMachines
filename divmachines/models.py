from torch.nn import Module


class PointwiseModel(Module):
    """
    Base Model class Matrix Factorization
    """
    def __init__(self):
        super(PointwiseModel, self).__init__()

    def forward(self, *input):
        pass


class PairwiseModel(Module):
    """
    Base Pairwise Matrix Factorization Model
    """

    def forward(self, *input):
        pass