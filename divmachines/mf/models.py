from divmachines import PointwiseModel
from divmachines.layers import ScaledEmbedding, ZeroEmbedding


class MatrixFactorizationModel(PointwiseModel):
    """
    Matrix Factorization Model with Bias Parameters
    Parameters
    ----------
    n_users: int
        Number of users to use in user latent factors
    n_items: int
        Number of items to use in item latent factors
    n_factors: int, optional
        Number of factors to use in user and item latent factors
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    """
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=10,
                 sparse=True):

        super(MatrixFactorizationModel, self).__init__()
        self._n_users = n_users
        self._n_items = n_items
        self._n_factors = n_factors
        self._sparse = sparse

        self.x = ScaledEmbedding(self._n_users,
                                 self._n_factors,
                                 sparse=self._sparse)
        self.y = ScaledEmbedding(self._n_items,
                                 self._n_factors,
                                 sparse=self._sparse)

        self.user_biases = ZeroEmbedding(self._n_users, 1, sparse=self._sparse)
        self.item_biases = ZeroEmbedding(self._n_items, 1, sparse=self._sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.
        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.
        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        biases_sum = user_bias + item_bias

        users_batch = self.x(user_ids).squeeze()
        items_batch = self.y(item_ids).squeeze()

        if len(users_batch.size()) > 2:
            dot = (users_batch * items_batch).sum(2)
        elif len(users_batch.size()) > 1:
            dot = (users_batch * items_batch).sum(1)
        else:
            dot = (users_batch * items_batch).sum()

        return biases_sum + dot


class SimpleMatrixFactorizationModel(PointwiseModel):
    """
    Matrix Factorization Model without Bias Parameters
    Parameters
    ----------
    n_users: int
        Number of users to use in user latent factors
    n_items: int
        Number of items to use in item latent factors
    n_factors: int, optional
        Number of factors to use in user and item latent factors
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    """

    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=10,
                 sparse=True):

        super(SimpleMatrixFactorizationModel, self).__init__()
        self._n_users = n_users
        self._n_items = n_items
        self._n_factors = n_factors
        self._sparse = sparse

        self.x = ScaledEmbedding(self._n_users,
                                 self._n_factors,
                                 sparse=self._sparse)
        self.y = ScaledEmbedding(self._n_items,
                                 self._n_factors,
                                 sparse=self._sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.
        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.
        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        users_batch = self.x(user_ids).squeeze()
        items_batch = self.y(item_ids).squeeze()

        if len(users_batch.size()) > 2:
            dot = (users_batch * items_batch).sum(2)
        elif len(users_batch.size()) > 1:
            dot = (users_batch * items_batch).sum(1)
        else:
            dot = (users_batch * items_batch).sum()

        return dot
