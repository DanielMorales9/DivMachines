import numpy as np
from torch.utils.data import Dataset
from divmachines.utility.indexing import make_indexable
import pandas as pd


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
        self.ix = ix
        self._n_users = n_users
        self._n_items = n_items
        self._initialize(x, y, dic)

    def _initialize(self, x, y, dic):

        self._dic = dic

        if dic is not None:
            self.x, self.ix = make_indexable(dic, x, self.ix)
        else:
            self.x = x

        self._len = self.x.shape[0]

        users = len(np.unique(self.x[:, 0]))
        items = len(np.unique(self.x[:, 1]))

        self._n_users = self._n_users if self._n_users is not None else users
        self._n_items = self._n_items if self._n_items is not None else items
        self.y = y

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
        return self.ix

    @index.getter
    def index(self):
        return self.ix

    def __call__(self, x, y=None):
        self._initialize(x, y, dic=self._dic)
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self.y is None:
            return self.x[item, :]
        return self.x[item, :], self.y[item]


class PairDataset(Dataset):
    """
    Wrapper for Factorization Machines PairwiseDataset
    Parameter
    ----------
    x: ndarray
        transaction data.
    y: ndarray, optional
        target values for transaction data.
    frac: float, optional
        Uniform user, uniform item negative sampling fraction
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
                 frac=0.8,
                 n_users=None,
                 n_items=None,
                 ix=None):
        super(PairDataset, self).__init__()
        self._frac = frac
        self._initialize(x, y)
        self._dataset = DenseDataset(x,
                                     y=y,
                                     dic=dic,
                                     n_users=n_users,
                                     n_items=n_items,
                                     ix=ix)

    def _initialize(self, x, y):
        if y is not None:
            t = np.zeros((x.shape[0], x.shape[1]+1), dtype=np.object)
            t[:, :-1] = x
            t[:, -1] = y
            df = pd.DataFrame(t, columns=['user', 'item', 'relevance'])
            df['index'] = df.index
            pos = df.loc[df['relevance'] == 1, ['user', 'item', 'index']]
            pos = pos.rename(columns={"index": "idx_pos"})
            neg = df.loc[df['relevance'] == 0, ['user', 'item', 'index']]
            neg = neg.rename(columns={"index": "idx_neg"})
            idx = pd.merge(pos, neg, on="user")[['user',
                                                 'idx_pos',
                                                 'idx_neg']]
            idx = idx.groupby(['user', 'idx_pos'], as_index=False)\
                     .apply(lambda g: g.sample(frac=self._frac))\
                     .reset_index(0, drop=True)
            self._idx = idx.values

    @property
    def n_users(self):
        return self._dataset.n_users

    @n_users.getter
    def n_users(self):
        return self._dataset.n_users

    @property
    def n_items(self):
        return self._dataset.n_items

    @n_items.getter
    def n_items(self):
        return self._dataset.n_items

    @property
    def index(self):
        return self._dataset.ix

    @index.getter
    def index(self):
        return self._dataset.ix

    @property
    def x(self):
        return self._dataset.x

    @x.getter
    def x(self):
        return self._dataset.x

    def __call__(self, x, y=None):
        self._initialize(x, y)
        self._dataset(x, y=y)
        return self

    def __len__(self):
        if self._dataset.y is None:
            return len(self._dataset)
        else:
            return len(self._idx)

    def __getitem__(self, item):
        if self._dataset.y is None:
            return self._dataset[item]
        else:
            left = self._idx[item, 0]
            right = self._idx[item, 1]
            ul, _ = self._dataset[left]
            u = ul[0]
            l = ul[1]
            r, _ = self._dataset[right]
            return u, l, r[1]
