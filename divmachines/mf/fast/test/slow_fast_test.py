from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from mfpytorch.fast.fast_mf import MatrixFactorization as Fast
from mfpytorch.mf import MatrixFactorization as Slow

N_USERS = 50
N_ITEMS = 50
BATCH_SIZE = 32
N_FACTORS = 5
N_TESTS = 10


class SlowMF(torch.nn.Module):
    def __init__(self):
        super(SlowMF, self).__init__()
        self.mf = Slow(N_USERS, N_ITEMS, N_FACTORS)

    def forward(self, users, items):
        return self.mf(users, items)


class FastMF(torch.nn.Module):

    def __init__(self):
        super(FastMF, self).__init__()
        self.mf = Fast(N_USERS, N_ITEMS, N_FACTORS)

    def forward(self, users, items):
        return self.mf(users, items)


def _forward_backward_check(dtype, index_dtype):
    np.random.seed(1)
    torch.manual_seed(1)
    slow = SlowMF()

    np.random.seed(1)
    torch.manual_seed(1)
    fast = FastMF()

    if dtype is np.float64:
        slow.double()
        fast.double()

    for i in range(N_TESTS):
        users = Variable(torch.from_numpy(np.random.randint(0, 50, size=BATCH_SIZE).astype(index_dtype)))
        items = Variable(torch.from_numpy(np.random.randint(0, 50, size=BATCH_SIZE).astype(index_dtype)))
        y = Variable(torch.from_numpy(np.random.random(BATCH_SIZE).astype(dtype)))

        out_slow = slow(users, items)
        out_fast = fast(users, items)

        assert np.allclose(out_slow.data.numpy(),
                           out_fast.data.numpy()), "Forward passes differed for {}".format(dtype)

        loss_slow = F.mse_loss(out_slow, y)
        loss_fast = F.mse_loss(out_fast, y)
        loss_slow.backward()
        loss_fast.backward()

        for var_slow, var_fast in zip(slow.parameters(), fast.parameters()):
            assert np.allclose(var_slow.grad.data.numpy(),
                               var_fast.grad.data.numpy()), "Backward passes differed for {}".format(dtype)


def test_forward_backward_float_long():
        _forward_backward_check(np.float32, np.int64)


def test_forward_backward_double_long():
        _forward_backward_check(np.float64, np.int64)


test_forward_backward_double_long()