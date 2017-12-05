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


class TestEmbedding(Embedding):
    """
    Specific Embedding for Test

    Parameters
    ----------
    embedding_weights: ndarray
        Embedding's weights for the
    """
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 embedding_weights=None):
        self._embedding_weights = embedding_weights
        super(TestEmbedding, self).__init__(num_embeddings,
                                            embedding_dim,
                                            padding_idx,
                                            max_norm,
                                            norm_type,
                                            scale_grad_by_freq,
                                            sparse)

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        for i in range(self._embedding_weights.shape[0]):
            for j in range(self._embedding_weights.shape[1]):
                self.weight.data[i, j] = self._embedding_weights[i, j]
