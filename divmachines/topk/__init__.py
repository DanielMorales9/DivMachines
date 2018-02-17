import numpy as np
from torch.utils.data import Dataset


class Rank(Dataset):

    def __init__(self, x, rank, u, n_items, n_users):
        self._x = x
        self._rank = rank
        self._u = u
        self._n_items = n_items
        self._n_users = n_users
        self._len = len(self._rank)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        x = self._x[self._u *
                    self._n_items +
                    self._rank[self._u, i]]
        x[:self._n_users] = np.zeros(self._n_users,
                                     dtype=np.float32)
        return x, self._rank[self._u, i]

    def __call__(self, u):
        self._u = u