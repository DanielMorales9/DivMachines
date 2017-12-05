import torch
import numpy as np
from torch.utils.data import Dataset
from ..torch_utils import gpu
from ..utility import make_indexable


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
    """

    def __init__(self, x, y=None, dic=None):
        super(Dataset, self).__init__()
        self._initialize(x, y, dic)

    def _initialize(self, x, y, dic, ix=None):
        self._dic = dic

        if dic is not None:
            self._x, self._ix = make_indexable(dic, x, ix)
        else:
            self._x = x

        self._len = self._x.shape[0]
        self._n_users = len(np.unique(self._x[:, 0]))
        self._n_items = len(np.unique(self._x[:, 1]))
        self._y = y

    def __call__(self, x, y=None, dic=None):
        self._dic = dic or self._dic
        self._initialize(x, y, self._dic, ix=self._ix)
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