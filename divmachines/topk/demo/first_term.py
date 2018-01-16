import numpy as np
import torch
from torch.autograd.variable import Variable
from divmachines.models.layers import TestEmbedding


def _compute_f(x, y, k, rank, users):
    # Initialize Index Vars
    u_idx = Variable(torch.from_numpy(users))
    i_idx = Variable(torch.from_numpy(rank))
    users_batch = x(u_idx).squeeze()
    print(users_batch)
    items_batch = y(i_idx[:, k:]).squeeze().transpose(0, 1)
    print(items_batch)
    dot = (users_batch * items_batch)
    print(dot)
    return dot


users = 3
items = 5
factors = 2

x_weights = np.array([[0, 1],
                      [2, 3],
                      [4, 5]], dtype=np.float)
y_weights = np.array([[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7],
                      [8, 9]], np.float)

x = TestEmbedding(users, factors, embedding_weights=x_weights)
y = TestEmbedding(items, factors, embedding_weights=y_weights)

users = np.array([0, 1, 2])
rank = np.array([[0, 1, 2, 3, 4],
                 [0, 1, 2, 3, 4],
                 [0, 1, 2, 3, 4]])
k = 1

print(_compute_f(x, y, k, rank, users))
