import numpy as np
from itertools import count
from collections import defaultdict


class ColumnDefaultFactory:

    def __init__(self, dic, old=False, prefix=None):
        self._dic = {}
        if not old:
            for k, _ in dic.items():
                self._dic[k] = defaultdict(lambda c=count(0): next(c))
        else:
            for k in prefix.keys():
                prefix[k] = -1

            for k, v in dic.items():
                for p in prefix.keys():
                    if k.startswith(p):
                        if v > prefix[p]:
                            prefix[p] = v

            for k in prefix.keys():
                self._dic[k] = defaultdict(lambda c=count(prefix[k] + 1):
                                           next(c))
            for k, v in dic.items():
                for p in prefix.keys():
                    if k.startswith(p):
                        self._dic[p][k] = v


    def __call__(self, key):
        for k in self._dic:
            if key.startswith(k):
                return self._dic[k][key]


class FeaturesFactory:

    def __init__(self, dic, *args, old=False, prefix=None):
        self._dic = {}
        if not old:
            args = [0] + list(args)
            for i in range(len(args)):
                if i > 0:
                    args[i] += args[i - 1]

            i = 0
            for k, _ in dic.items():
                self._dic[k] = defaultdict(lambda c=count(args[i]): next(c))
                i += 1
            self._dic['feats'] = defaultdict(lambda c=count(args[i]): next(c))
        else:
            for k in prefix.keys():
                prefix[k] = -1
            prefix['feats'] = 0

            for k, v in dic.items():
                for p in prefix.keys():
                    if k.startswith(p):
                        if v > prefix[p]:
                            prefix[p] = v

            for k in prefix.keys():
                self._dic[k] = defaultdict(lambda c=count(prefix[k]+1):
                                           next(c))
            for k, v in dic.items():
                for p in prefix.keys():
                    if k.startswith(p):
                        self._dic[p][k] = v

    def __call__(self, key):
        for k in self._dic:
            if key.startswith(k):
                return self._dic[k][key]


class IndexDictionary(defaultdict):

    def __init__(self, default_factory):
        self.default_factory = default_factory

    def __missing__(self, key):
        self[key] = self.default_factory(key)
        return self[key]


def make_indexable(dic, x, ix=None):
    """
    Makes the specified dictionary indexable.
    dic: dict
        Dictionary to make indexable
    x: ndarray
        Original transaction data
    ix: dict, optional
        Dictionary of indexes
    Returns
    -------
    x: ndarray
        Modified transaction data with indexable columns
    ix: dict, optional
        It maps the each column value to indexable columns
    """
    if ix is None:
        ix = IndexDictionary(ColumnDefaultFactory(dic))

    dic_lists = dict(dic)
    for k, v in dic_lists.items():
        dic_lists[k] = np.array([ix[str(k) + str(el)] for el in x[:, v]])

    new_x = np.zeros(shape=x.shape, dtype=np.int64)
    for k, v in dic.items():
        new_x[:, v] = dic_lists[k]
    return new_x, ix


def vectorize_dic(dic,
                  ix=None,
                  n_users=None,
                  n_items=None,
                  lengths=None):
    """
    Creates a scipy csr matrix from a list of lists
    (each inner list is a set of values corresponding to a feature)
    """
    args = None
    if ix is None:
        if n_users and n_items and lengths:
            lista = [n_users, n_items]
            lista.extend(lengths.values())
            args = tuple(lista)
            ix = IndexDictionary(FeaturesFactory(dic, *args))
        elif n_users and n_items:
            ix = IndexDictionary(FeaturesFactory(dic, n_users, n_items))
        else:
            ix = defaultdict(lambda c=count(0): next(c))

    n = len(list(dic.values())[0])  # num samples
    g = len(dic.keys())  # num groups
    nz = n * g  # number of non-zeros

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        # append index el with k in order to prevent mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(k) + str(el)] for el in lis]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)
    if args:
        p = sum(args)
    else:
        p = len(ix)
    ixx = np.where(col_ix < p)

    return data[ixx], row_ix[ixx], col_ix[ixx], ix


def non_zero(interactions, cols):
    nz = 0
    for r in range(len(interactions)):
        for c in cols:
            if interactions[r, c] != 0:
                nz += 1
    return nz


def vectorize_interactions(interactions,
                           dic=None,
                           ix=None,
                           n_users=None,
                           n_items=None,
                           lengths=None):
    # TODO not elegant solution
    if dic is not None:
        vec_dic = dic.copy()
        keys = []
        # changes the dic into a dic of lists
        for k in dic:
            keys.append(dic[k])
            vec_dic[k] = interactions[:, dic[k]]

        d, r, c, ix = vectorize_dic(vec_dic,
                                    ix=ix,
                                    n_users=n_users,
                                    n_items=n_items,
                                    lengths=lengths)

        real_valued_cols = list(set(range(interactions.shape[1])) - set(keys))
        cat_nz = len(d)
    else:
        cat_nz = 0
        real_valued_cols = range(interactions.shape[1])
        ix = defaultdict(lambda c=count(0): next(c))

    nz = non_zero(interactions, real_valued_cols) + cat_nz
    data = np.empty(nz)
    rows = np.empty(nz, dtype=int)
    cols = np.empty(nz, dtype=int)
    if dic is not None:
        data[:len(d)] = d
        rows[:len(d)] = r
        cols[:len(d)] = c

    i = cat_nz
    for r in range(len(interactions)):
        for col in real_valued_cols:
            if interactions[r, col] != 0:
                data[i] = interactions[r, col]
                cols[i] = ix['feats' + str(col)]
                rows[i] = r
                i += 1

    if n_users is None or n_items is None:
        n_features = len(np.unique(cols))
    else:
        if lengths:
            n_features = n_users + n_items + len(real_valued_cols)
            for v in lengths.values():
                n_features += v
        else:
            n_features = n_users + n_items + len(real_valued_cols)
    return data, rows, cols, ix, n_features


def list2dic(data, rows, cols):
    dic = defaultdict(lambda: [])
    for d, r, c in zip(data, rows, cols):
        dic[r].append([d, c])
    return dic

