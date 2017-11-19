import numpy as np
from itertools import count
from collections import defaultdict
from scipy.sparse import csr


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
        ix = defaultdict(count(0).next)

    n = len(dic.values()[0])  # num samples
    g = len(dic.keys())  # num groups
    nz = n * g  # number of non-zeros

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.iteritems():
        # append index el with k in order to prevet mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)

    if p is None:
        p = len(ix)

    ixx = np.where(col_ix < p)

    return data[ixx], row_ix[ixx], col_ix[ixx], ix
