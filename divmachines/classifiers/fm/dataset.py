import numpy as np
import pandas as pd
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
                 n_items=None,
                 lengths=None,
                 ix=None):
        super(DenseDataset, self).__init__()
        self._n_features = None
        self.ix = ix
        self._n_users = n_users
        self._n_items = n_items
        self._lengths = lengths
        self._initialize(x, y, dic)

    def _initialize(self, x, y, dic):
        self._len = len(x)
        self.dic = dic

        if dic is None:
            self._x = x.astype(np.float32)
            self._n_features = self._x.shape[1]
        else:
            users = len(np.unique(x[:, 0]))
            items = len(np.unique(x[:, 1]))

            self._n_users = self._n_users \
                if self._n_users is not None else users
            self._n_items = self._n_items \
                if self._n_items is not None else items

            data, rows, cols, self.ix, n_feats \
                = vectorize_interactions(x,
                                         dic=dic,
                                         ix=self.ix,
                                         n_users=self._n_users,
                                         n_items=self._n_items,
                                         lengths=self._lengths)
            self._n_features = self._n_features or n_feats
            try:
                coo = coo_matrix((data, (rows, cols)),
                                 shape=(self._len, self._n_features))
            except ValueError:
                raise ValueError("column index exceeds matrix dimensions\n "
                                 "You may want to specify the number of "
                                 "users and items.")

            self._x = coo.toarray().astype(np.float32)
        self.y = y.astype(np.float32) if y is not None else None

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
        return self.ix

    @index.getter
    def index(self):
        return self.ix

    @property
    def x(self):
        return self._x

    @x.getter
    def x(self):
        return self._x

    def __call__(self, x, y=None):
        self._initialize(x, y=y, dic=self.dic)
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self.y is None:
            return self._x[item, :]
        return self._x[item, :], self.y[item]


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
                 n_items=None,
                 lengths=None,
                 ix=None):
        super(SparseDataset, self).__init__()
        self._n_features = None
        self.ix = ix
        self._lengths = lengths
        self._n_users = n_users
        self._n_items = n_items
        self._initialize(x, y, dic)

    def _initialize(self, x, y, dic):
        self._len = len(x)
        self.dic = dic
        self.y = y.astype(np.float32) if y is not None else None
        if dic is None:
            self._n_features = x.shape[1]
            self._sparse_x = dict()
            for r, row in enumerate(x):
                self._sparse_x[r] = \
                    [[col, d] for col, d in enumerate(row) if d != 0.]
            self.ix = None
        else:
            users = len(np.unique(x[:, 0]))
            items = len(np.unique(x[:, 1]))

            self._n_users = self._n_users \
                if self._n_users is not None else users
            self._n_items = self._n_items \
                if self._n_items is not None else items

            d, rows, cols, self.ix, n_feats \
                = vectorize_interactions(x,
                                         dic=self.dic,
                                         ix=self.ix,
                                         n_users=self._n_users,
                                         n_items=self._n_items,
                                         lengths=self._lengths)
            self._n_features = self._n_features or n_feats
            self._sparse_x = list2dic(d, rows, cols)

    @property
    def n_features(self):
        return self._n_features

    @n_features.getter
    def n_features(self):
        return self._n_features

    @property
    def index(self):
        return self.ix

    @index.getter
    def index(self):
        return self.ix

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
        for i in range(len(self)):
            yield self[i]

    @x.getter
    def x(self):
        for i in range(len(self)):
            yield self[i]

    def __call__(self, x, y=None):
        self._initialize(x, y, self.dic)
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        copy = np.zeros(self._n_features, dtype=np.float32)
        for d, col in self._sparse_x[item]:
            copy[col] = d
        if self.y is None:
            return copy
        else:
            return copy, self.y[item]


class PairDataset(Dataset):
    """
    Wrapper for Factorization Machines PairwiseDataset
    Parameter
    ----------
    x: ndarray
        transaction data.
    y: ndarray, optional
        target values for transaction data.
    sparse: bool, optional
        Whether the underlying dataset should be an instance of SparseDataset
        or DenseDataset. Default False.
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
                 sparse=False,
                 frac=0.8,
                 dic=None,
                 n_users=None,
                 n_items=None,
                 lengths=None,
                 ix=None):
        super(PairDataset, self).__init__()
        self._frac = frac
        self._initialize(x, y)
        if sparse:
            self._dataset = SparseDataset(x,
                                          y=y,
                                          dic=dic,
                                          n_users=n_users,
                                          n_items=n_items,
                                          lengths=lengths,
                                          ix=ix)
        else:
            self._dataset = DenseDataset(x,
                                         y=y,
                                         dic=dic,
                                         n_users=n_users,
                                         n_items=n_items,
                                         lengths=lengths,
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
            idx = idx.groupby(['user', 'idx_pos'], as_index=False) \
                .apply(lambda g: g.sample(frac=self._frac)) \
                .reset_index(0, drop=True)[['idx_pos',
                                            'idx_neg']]
            self._idx = idx.values

    @property
    def n_features(self):
        return self._dataset.n_features

    @n_features.getter
    def n_features(self):
        return self._dataset.n_features

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
            l, _ = self._dataset[left]
            r, _ = self._dataset[right]
            return l, r

