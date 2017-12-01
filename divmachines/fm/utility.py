import numpy as np
from itertools import count
from collections import defaultdict


def vectorize_dic(dic, ix=None, p=None):
    """
    Creates a scipy csr matrix from a list of lists (each inner list is a set of values corresponding to a feature)

    Parameters:
    -----------
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    """

    if ix is None:
        ix = defaultdict(lambda c=count(0): next(c))

    n = len(list(dic.values())[0])  # num samples
    g = len(dic.keys())  # num groups
    nz = n * g  # number of non-zeros

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        # append index el with k in order to prevent mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)

    if p is None:
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


def vectorize_interactions(interactions, dic=None, ix=None):
    if dic is not None:

        vec_dic = dic.copy()
        keys = []
        for k in dic:
            keys.append(dic[k])
            vec_dic[k] = interactions[:, dic[k]]
        d, r, c, ix = vectorize_dic(vec_dic, ix=ix)
        real_valued_cols = list(set(range(interactions.shape[1])) - set(keys))
        cat_nz = len(d)
    else:
        cat_nz = 0
        real_valued_cols = range(interactions.shape[1])
        ix = defaultdict(lambda c = count(0): next(c))

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
                cols[i] = ix[str(col)]
                rows[i] = r
                i += 1
    return data, rows, cols, ix


def list2dic(data, rows, cols):
    dic = defaultdict()
    for d, r, c in zip(data, rows, cols):
        if dic.get(r, None) is None:
            dic[r] = [[d, c]]
        else:
            dic[r].append([d, c])
    return dic