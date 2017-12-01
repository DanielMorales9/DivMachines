import numpy as np
from torch import LongTensor, FloatTensor
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle as sshuffle
import torch


def gpu(tensor, gpu=False):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):
    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def sparse_mini_batch(rows, cols, data, ratings, batch_size=None, n_features=None):
    batch_size = batch_size
    n_features = n_features

    n_samples = len(ratings)
    if batch_size is None:
        batch_size = n_samples-1

    if n_features is None:
        raise ValueError("Number of features must be provided: n_features")

    data_tensor = FloatTensor(coo_matrix((data, (rows, cols)), shape=(n_samples,n_features)).toarray())
    rating_tensor = FloatTensor(torch.from_numpy(ratings.astype(np.float32)))
    for i in range(0, n_samples, batch_size):
        yield data_tensor[i:i + batch_size], rating_tensor[i:i + batch_size]


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", None)

    if batch_size is None:
        batch_size = len(tensors[0])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def sparse_shuffle(rows, cols, data, ratings, **kwargs):
    random_state = kwargs.get('random_state')

    if random_state is None:
        random_state = np.random.RandomState()

    coo = coo_matrix((data, (rows, cols)),
                        shape=(len(np.unique(rows)), len(np.unique(cols))))

    csr, ratings = sshuffle(coo, ratings, random_state=random_state)
    coo = csr.tocoo()
    return coo.row, coo.col, coo.data, ratings


def assert_no_grad(variable):
    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients")


def set_seed(seed, cuda=False):
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
