"""
Embedding layers useful for recommender models.
"""

from torch.nn import Embedding


class ScaledEmbedding(Embedding):
    """
    Embedding layer that initializes its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(Embedding):
    """
    Embedding layer that initializes its values
    to zero.
    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

#class TestEmbedding(Embedding):
