import numpy as np
import numbers


def _prepare_for_prediction(x, feats):

    if type(x) is np.ndarray:
        if len(x.shape) == 2:
            if x.shape[1] == 1:
                raise ValueError("Array must have user and item columns")
            elif x.shape[1] == 2:
                return x
        elif len(x.shape) == 1 and len(x) == feats:
            return np.array([x])
        else:
            raise ValueError("Array dimensions not allowed")
    else:
        raise ValueError("Variable type not allowed")


def cartesian(*arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in np.arange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def cartesian2D(*arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        2-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (n, ) containing cartesian products
        formed of input arrays.
    """
    def cartesian_rec(arrays,
                      lengths=None,
                      cols=None,
                      n=None,
                      cum_col=None,
                      repeat=True,
                      out=None):
        arrays = [np.asarray(x) for x in arrays]
        lengths = lengths if lengths is not None else [len(x) for x in arrays]
        cols = cols if cols is not None else [x.shape[1] for x in arrays]
        n = n if n else np.prod(lengths)
        if out is None:
            m = sum(cols)
            dtype = arrays[0].dtype
            out = np.zeros([n, m], dtype=dtype)

        cum_col = cum_col or 0

        a, l, c = arrays[0], lengths[0], cols[0]
        times = int(n / l)
        if repeat:
            out[:, cum_col:cum_col + c] = \
                np.repeat(a, times, axis=0)
        else:
            out[:, cum_col:cum_col+c] = \
                np.tile(a, (times,1))
        cum_col += c
        if len(arrays) == 1:
            return out
        else:
            return cartesian_rec(arrays[1:],
                                 lengths[1:],
                                 cols[1:],
                                 n,
                                 cum_col,
                                 (not repeat),
                                 out)
    out = cartesian_rec(arrays)
    return out


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def shape_for_mf(_item_catalog, x):
    if type(x) == int or type(x) == np.int_:
        test = cartesian([x], _item_catalog)
        n_items = len(_item_catalog)
        n_users = 1
        update_dataset = False
    elif len(x.shape) == 1:
        test = cartesian(x, _item_catalog)
        n_items = len(_item_catalog)
        n_users = len(x)
        update_dataset = False
    elif x.shape[1] == 1:
        test = cartesian(x.reshape(-1), _item_catalog)
        n_items = len(_item_catalog)
        n_users = len(x)
        update_dataset = False
    elif x.shape[1] == 2:
        test = x
        n_users = len(np.unique(x[:, 0]))
        n_items = len(np.unique(x[:, 1]))
        update_dataset = True
    else:
        raise ValueError("Shape of x is not valid")
    return n_items, n_users, test, update_dataset


def _swap_k(index, k, matrix):
    for r, c in enumerate(index):
        temp = matrix[r, c + k]
        matrix[r, c + k] = matrix[r, k]
        matrix[r, k] = temp


def _tensor_swap_k(index, k, matrix, multi=False):
    for r, c in enumerate(index):
        if multi:
            temp = matrix[r, c + k].clone()
        else:
            temp = matrix[r, c + k]
        matrix[r, c + k] = matrix[r, k]
        matrix[r, k] = temp


def _tensor_swap(index, tensor):
    # Ogni riga è un ranking dell'utente
    for r, cols in enumerate(index):
        for col, i in zip(cols, range(len(index))):
            temp = tensor[r, i].clone()
            tensor[r, i] = tensor[r, col]
            tensor[r, col] = temp
    return tensor


def index(rank, idx):
    re_idx = np.vectorize(lambda x: idx[str(x)])
    return np.array([re_idx(lis) for lis in rank])


def re_index(items, rank):
    for i, arr in enumerate(rank):
        for j, r in enumerate(arr):
            rank[i, j] = items[r]