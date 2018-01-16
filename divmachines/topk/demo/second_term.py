import numpy as np
import torch
from torch.autograd.variable import Variable
from divmachines.models.layers import TestEmbedding

def _compute_f(x, y, k, rank, users):
    # Initialize Index Vars
    u_idx = Variable(torch.from_numpy(users))
    users_batch = x(u_idx).squeeze()
    i_idx = Variable(torch.from_numpy(rank))
    items_batch = y(i_idx[:, k:]).squeeze().transpose(0, 1)
    dot = (users_batch * items_batch)
    return dot.sum(2).data.numpy()


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
users = 3
var_numpy = np.zeros((users, factors), dtype=np.float32)
N = [1, 1, 1]
for i in range(3):
    user_idx = Variable(torch.from_numpy(np.array([i])))
    item_idx = Variable(torch.from_numpy(np.array([0, 1, 2, 3, 4])))
    diff = x(user_idx) - y(item_idx)
    prod = torch.pow(diff, 2).sum(0)
    var_numpy[i, :] = torch.mul(prod, N[i]).data.numpy()

var_t = torch.from_numpy(var_numpy)
var = torch.nn.Embedding(var_t.size(0), var_t.size(1))
var.weight = torch.nn.Parameter(var_t)

idx = Variable(torch.from_numpy(np.array([0, 1, 2])))

var(idx)

i_idx = Variable(torch.from_numpy(rank))
items_batch = y(i_idx[:, k:]).squeeze().transpose(0, 1)
print(items_batch)
print(var(idx))
print(items_batch * var(idx))

