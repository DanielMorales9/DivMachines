import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from divmachines.utility.data import vectorize_interactions, list2dic


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
    """
    def __init__(self, x, y=None, dic=None):
        super(DenseDataset, self).__init__()
        self._n_features = None
        self._initialize(x, y=y, dic=dic)

    def _initialize(self, x, y=None, dic=None, ix=None):
        self._len = len(x)
        self._dic = dic
        if dic is None:
            self._x = x.astype(np.float32)
            self._n_features = self._x.shape[1]
        else:
            data, rows, cols, self._ix \
                = vectorize_interactions(x, dic=dic, ix=ix, n_features=self._n_features)
            self._n_features = len(np.unique(cols)) if self._n_features is None else self._n_features
            coo = coo_matrix((data, (rows, cols)), shape=(self._len, self._n_features))
            self._x = coo.toarray().astype(np.float32)
        self._y = y.astype(np.float32) if y is not None else None

    def __call__(self, x, y=None, dic=None):
        self._dic = dic or self._dic
        self._initialize(x,
                         y=y,
                         dic=self._dic,
                         ix=self._ix)
        return self

    def n_features(self):
        return self._n_features

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
    """
    def __init__(self, x, y=None, dic=None):
        super(SparseDataset, self).__init__()
        self._n_features = None
        self._initialize(x, y=y, dic=dic)

    def _initialize(self, x, y, dic, ix=None):
        self._len = len(x)
        self._dic = dic
        self._y = y.astype(np.float32) if y is not None else None
        if dic is None:
            self._n_features = x.shape[1]
            self._sparse_x = dict()
            for r, row in enumerate(x):
                self._sparse_x[r] = [[col, d] for col, d in enumerate(row) if d != 0.]
        else:
            d, rows, cols, self._ix\
                = vectorize_interactions(x,
                                         dic=dic,
                                         ix=ix,
                                         n_features=self._n_features)
            self._n_features = len(np.unique(cols)) if self._n_features is None else self._n_features
            self._sparse_x = list2dic(d, rows, cols)
        self._zero = np.zeros(self._n_features, dtype=np.float32)

    def n_features(self):
        return self._n_features

    def __call__(self, x, y=None, dic=None):
        self._dic = dic or self._dic
        self._initialize(x, y=y, dic=self._dic, ix=self._ix)
        return self

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        copy = self._zero.copy()
        for d, col in self._sparse_x[item]:
            copy[col] = d
        if self._y is None:
            return copy
        else:
            return copy, self._y[item]
