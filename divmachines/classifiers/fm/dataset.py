import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from divmachines.utility.indexing import vectorize_interactions, list2dic


class DenseDataset(Dataset):
    """
    Wrapper for Factorization Machines Dataset
    Parameter
    ----------
    x: ndarray
        transaction data.
    y: ndarray, optional
        target values for transaction data.
    dic: dict, optional
        Features dictionary, for each entry (k, v), k corresponds to a
        categorical feature to vectorize and v the corresponding index
        in the interactions array.
    n_users: int, optional
        Total number of users in the training samples: `x`.
        Default is None, `n_users` will be inferred from `x`.
    n_items: int, optional
        Total number of items in the training samples: `x`.
        Default is None, `n_items` will be inferred from `x`.
    """
    def __init__(self,
                 x,
                 y=None,
                 dic=None,
                 n_users=None,
                 n_items=None):
        super(DenseDataset, self).__init__()
        self._n_features = None
        self._ix = None
        self._n_users = n_users
        self._n_items = n_items
        self._initialize(x, y, dic)

    def _initialize(self, x, y, dic):
        self._len = len(x)
        self._dic = dic

        if dic is None:
            self._x = x.astype(np.float32)
            self._n_features = self._x.shape[1]
        else:
            users = len(np.unique(x[:, 0]))
            items = len(np.unique(x[:, 1]))

            self._n_users = self._n_users \
                if self._n_users is None else users
            self._n_items = self._n_items \
                if self._n_items is None else items

            data, rows, cols, self._ix, n_feats \
                = vectorize_interactions(x,
                                         dic=dic,
                                         ix=self._ix,
                                         n_users=self._n_users,
                                         n_items=self._n_items)
            self._n_features = self._n_features or n_feats
            try:
                coo = coo_matrix((data, (rows, cols)),
                                 shape=(self._len, self._n_features))
            except ValueError:
                raise ValueError("column index exceeds matrix dimensions\n "
                                 "You may want to specify the number of "
                                 "users and items.")

            self._x = coo.toarray().astype(np.float32)
        self._y = y.astype(np.float32) if y is not None else None

    @property
    def n_features(self):
        return self._n_features

    @n_features.getter
    def n_features(self):
        return self._n_features

    @property
    def n_users(self):
        return self._n_users

    @n_users.getter
    def n_users(self):
        return self._n_users

    @property
    def n_items(self):
        return self._n_items

    @n_items.getter
    def n_items(self):
        return self._n_items

    @property
    def index(self):
        return self._ix

    @index.getter
    def index(self):
        return self._ix

    @property
    def x(self):
        return self._x

    @x.getter
    def x(self):
        return self._x

    def __call__(self, x, y=None):
        self._initialize(x, y=y, dic=self._dic)
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self._y is None:
            return self._x[item, :]
        return self._x[item, :], self._y[item]


class SparseDataset(Dataset):
    """
    Wrapper for Factorization Machines Sparse Dataset
    Parameter
    ----------
    x: ndarray
        transaction data.
    y: ndarray, optional
        target values for transaction data.
    dic: dict, optional
        Features dictionary, for each entry (k, v), k corresponds to a
        categorical feature to vectorize and v the corresponding index
        in the interactions array.
    n_users: int, optional
        Total number of users in the training samples: `x`.
        Default is None, `n_users` will be inferred from `x`.
    n_items: int, optional
        Total number of items in the training samples: `x`.
        Default is None, `n_items` will be inferred from `x`.
    """
    def __init__(self,
                 x,
                 y=None,
                 dic=None,
                 n_users=None,
                 n_items=None):
        super(SparseDataset, self).__init__()
        self._n_features = None
        self._ix = None
        self._n_users = n_users
        self._n_items = n_items
        self._initialize(x, y, dic)

    def _initialize(self, x, y, dic):
        self._len = len(x)
        self._dic = dic
        self._y = y.astype(np.float32) if y is not None else None
        if dic is None:
            self._n_features = x.shape[1]
            self._sparse_x = dict()
            for r, row in enumerate(x):
                self._sparse_x[r] = \
                    [[col, d] for col, d in enumerate(row) if d != 0.]
            self._ix = None
        else:
            users = len(np.unique(x[:, 0]))
            items = len(np.unique(x[:, 1]))

            self._n_users = self._n_users \
                if self._n_users is None else users
            self._n_items = self._n_items \
                if self._n_items is None else items

            d, rows, cols, self._ix, n_feats \
                = vectorize_interactions(x,
                                         dic=self._dic,
                                         ix=self._ix,
                                         n_users=self._n_users,
                                         n_items=self._n_items)
            self._n_features = self._n_features or n_feats
            self._sparse_x = list2dic(d, rows, cols)
        self._zero = np.zeros(self._n_features, dtype=np.float32)

    @property
    def n_features(self):
        return self._n_features

    @n_features.getter
    def n_features(self):
        return self._n_features

    @property
    def index(self):
        return self._ix

    @index.getter
    def index(self):
        return self._ix

    @property
    def n_users(self):
        return self._n_users

    @n_users.getter
    def n_users(self):
        return self._n_users

    @property
    def n_items(self):
        return self._n_items

    @n_items.getter
    def n_items(self):
        return self._n_items

    @property
    def x(self):
        x = np.zeros((self._n_features, len(self)))

        for i in range(len(self)):
            x[i, :] = self[i]
        return x

    @x.getter
    def x(self):
        x = np.zeros((self._n_features, len(self)))

        for i in range(len(self)):
            x[i, :] = self[i]
        return x

    def __call__(self, x, y=None):
        self._initialize(x, y, self._dic)
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        copy = self._zero.copy()
        try:
            for d, col in self._sparse_x[item]:
                copy[col] = d
            if self._y is None:
                return copy
            else:
                return copy, self._y[item]
        except IndexError:
            raise IndexError('Index out of bound. '
                             'You may want to specify the number '
                             'of users and items.')

