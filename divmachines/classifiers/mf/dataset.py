import numpy as np
from torch.utils.data import Dataset
from divmachines.utility.indexing import make_indexable


class DenseDataset(Dataset):
    """Matrix Factorization Dataset
    Parameters
    ----------
    x: ndarray
        training samples
    y: ndarray, optional
        target values for corresponding samples.
        Default is None
    dic: dict, optional
        dic indicates the columns to make indexable
    n_users: int, optional
        Total number of users. The model will have `n_users` rows.
        Default is None, `n_users` will be inferred from `x`.
    n_items: int, optional
        Total number of items. The model will have `n_items` columns.
        Default is None, `n_items` will be inferred from `x`.
    """

    def __init__(self,
                 x,
                 y=None,
                 dic=None,
                 n_users=None,
                 n_items=None,
                 ix=None):
        super(Dataset, self).__init__()
        self._ix = ix
        self._n_users = n_users
        self._n_items = n_items
        self._initialize(x, y, dic)

    def _initialize(self, x, y, dic):

        self._dic = dic

        if dic is not None:
            self._x, self._ix = make_indexable(dic, x, self._ix)
        else:
            self._x = x

        self._len = self._x.shape[0]

        users = len(np.unique(self._x[:, 0]))
        items = len(np.unique(self._x[:, 1]))

        self._n_users = self._n_users if self._n_users is not None else users
        self._n_items = self._n_items if self._n_items is not None else items
        self._y = y

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

    def __call__(self, x, y=None):
        self._initialize(x, y, dic=self._dic)
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self._y is None:
            return self._x[item, :]
        return self._x[item, :], self._y[item]