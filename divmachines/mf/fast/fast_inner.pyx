from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable

cimport cython
cimport numpy as np

from cython cimport floating
from cython cimport integral
from cython.view cimport array as cvarray

@cython.boundscheck(False)
def _compute_mul(floating[:] mul,
                floating[:,:] x,
                floating[:,:] y,
                integral[:] user,
                integral[:] item,
                int batch_size,
                int factors):
    cdef int f, u, i
    cdef double r
    for b in range(batch_size):
        r = 0.0
        u = user[b]
        i = item[b]
        for f in range(factors):
            r = r + x[u, f] * y[u, f]
        mul[b] = r


def fast_forward(self, user, item, x, y):
    self.x = x
    self.y = y
    self.user = user
    self.item = item

    self.factors = x.size()[-1]
    self.batch_size = user.size()[0]
    self.mul = np.zeros(self.batch_size, dtype=self.x.numpy().dtype)
    _compute_mul(self.mul,
                self.x.numpy(),
                self.y.numpy(),
                self.user.numpy(),
                self.item.numpy(),
                self.batch_size,
                self.factors)

    return torch.from_numpy(self.mul).unsqueeze(-1)

@cython.boundscheck(False)
def _compute_grad(integral[:] user,
                    integral[:] item,
                    floating[:,:] x,
                    floating[:, :] y,
                    floating[:, :] dldy,
                    int batch_size,
                    int factors,
                    floating[:,:] grad_x,
                    floating[:,:] grad_y):
    cdef int b, f, u, i
    for b in range(batch_size):
        u = user[b]
        i = item[b]
        for f in range(factors):
            grad_x[u, f] = dldy[b, 0] * y[i, f]
            grad_y[i, f] = dldy[b, 0] * x[u, f]

def fast_backward(self, grad_output):
    grad_input_users = grad_output.new(self.batch_size).zero_()
    grad_input_items = grad_output.new(self.batch_size).zero_()

    grad_x = grad_output.new(self.x.size()[0], self.factors).zero_()
    grad_y = grad_output.new(self.y.size()[0], self.factors).zero_()

    _compute_grad(self.user.numpy(),
                    self.item.numpy(),
                    self.x.numpy(),
                    self.y.numpy(),
                    grad_output.numpy(),
                    self.batch_size,
                    self.factors,
                    grad_x.numpy(),
                    grad_y.numpy())

    return grad_input_users, grad_input_items, grad_x, grad_y