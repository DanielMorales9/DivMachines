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
                 n_items=None):
        super(Dataset, self).__init__()
        self._initialize(x, y, dic, n_users=n_users, n_items=n_items)

    def _initialize(self,
                    x,
                    y,
                    dic,
                    ix=None,
                    n_users=None,
                    n_items=None):

        self._dic = dic

        if dic is not None:
            self._x, self._ix = make_indexable(dic, x, ix)
        else:
            self._x = x
            self._ix = None

        self._len = self._x.shape[0]

        users = len(np.unique(self._x[:, 0]))
        items = len(np.unique(self._x[:, 1]))

        if n_users is None and n_items is None:
            n_users = users
            n_items = items
        if n_users < users or n_items < items:
            raise ValueError("Number of users or items provided is "
                             "lower than the one detected")
        self._n_users = n_users if n_users > users else users
        self._n_items = n_items if n_items > items else items

        self._y = y

    def __call__(self,
                 x,
                 y=None,
                 dic=None,
                 n_users=None,
                 n_items=None):
        self._dic = dic or self._dic
        self._initialize(x, y,
                         self._dic,
                         ix=self._ix,
                         n_users=self._n_users,
                         n_items=self._n_items)
        return self

    def n_items(self):
        return self._n_items

    def n_users(self):
        return self._n_users

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self._y is None:
            return self._x[item, :]
        return self._x[item, :], self._y[item]