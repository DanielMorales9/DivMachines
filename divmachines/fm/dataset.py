import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from divmachines.fm.utility import vectorize_interactions, list2dic


class DenseDataset(Dataset):
    """
    Wrapper for Factorization Machines
    Parameter
    ----------
    interactions: ndarray
        transaction data, triple users, item, rating
    feats_dic: dict, optional
        Features dictionary, for each entry (k, v), k corresponds to a
        categorical feature to vectorize and v the corresponding index
        in the interactions array.
    """
    def __init__(self, interactions, dic=None):
        super(DenseDataset, self).__init__()
        self._initialize(interactions, dic=dic)

    def __call__(self, interactions):
        self._initialize(interactions, dic=self._dic, ix=self._ix)
        return self

    def n_features(self):
        return self._n_features

    def n_items(self):
        return self._n_items

    def n_users(self):
        return self._n_users

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        return self._dataset[item, :]

    def _initialize(self, interactions, dic=None, ix=None):
        self._dic = dic or {'users': 0, 'items': 1}
        self._data, \
        self._rows, \
        self._cols, \
        self._ix = vectorize_interactions(interactions[:, :-1], dic=self._dic, ix=ix)
        ratings = interactions[:, -1]
        self._len = len(interactions)
        self._n_features = len(np.unique(self._cols))
        coo = coo_matrix((self._data, (self._rows, self._cols)), shape=(self._len, self._n_features))
        self._dataset = np.zeros(shape=(self._len, self._n_features + 1), dtype=np.float32)
        self._dataset[:, :-1] = coo.toarray()
        self._dataset[:, -1] = ratings
        self._n_users = len(np.unique(interactions[:, self._dic['users']]))
        self._n_items = len(np.unique(interactions[:, self._dic['items']]))


class SparseDataset(Dataset):
    """
    Wrapper for Factorization Machines
    Parameter
    ----------
    interactions: ndarray
        transaction data, triple users, item, rating
    feats_dic: dict, optional
        Features dictionary, for each entry (k, v), k corresponds to a
        categorical feature to vectorize and v the corresponding index
        in the interactions array.
    """
    def __init__(self, interactions, dic=None):
        super(SparseDataset, self).__init__()
        self._initialize(interactions, dic=dic)

    def _initialize(self, interactions, dic=None, ix=None):
        self._dic = dic or {'users': 0, 'items': 1}
        self._data, \
        self._rows, \
        self._cols, \
        self._ix = vectorize_interactions(interactions[:, :-1], dic=self._dic, ix=ix)
        self._ratings = interactions[:, -1]
        self._len = len(interactions)
        self._n_features = len(np.unique(self._cols))
        self._sparse_dataset = list2dic(self._data, self._rows, self._cols)
        self._zero = np.zeros(self._n_features+1, dtype=np.float32)
        self._n_users = len(np.unique(interactions[:, self._dic['users']]))
        self._n_items = len(np.unique(interactions[:, self._dic['items']]))

    def __call__(self, interactions):
        self._initialize(interactions, dic=self._dic, ix=self._ix)
        return self

    def __len__(self):
        return self._len

    def n_items(self):
        return self._n_items

    def n_users(self):
        return self._n_users

    def n_features(self):
        return self._n_features

    def __getitem__(self, item):
        copy = self._zero.copy()
        for data, col in self._sparse_dataset[item]:
            copy[col] = data
        copy[-1] = self._ratings[item]
        return copy
