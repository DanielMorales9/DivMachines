import numpy as np
import torch
from torch.autograd.variable import Variable
from divmachines.models.layers import TestEmbedding

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
                      [8, 9]], dtype=np.float)

x = TestEmbedding(users, factors, embedding_weights=x_weights)
y = TestEmbedding(items, factors, embedding_weights=y_weights)

var = np.zeros((users, factors), dtype=np.float)
N = [1, 1, 1]
for i in range(3):
    user_idx = Variable(torch.from_numpy(np.array([i])))
    item_idx = Variable(torch.from_numpy(np.array([0, 1, 2, 3])))
    diff = x(user_idx) - y(item_idx)
    print(diff)
    prod = torch.pow(diff, 2).sum(0)
    print(torch.pow(diff, 2))
    var[i, :] = torch.mul(prod, N[i]).data.numpy()
var_t = torch.from_numpy(var)
embedding = torch.nn.Embedding(var_t.size(0), var_t.size(1))
embedding.weight = torch.nn.Parameter(var_t)

idx = Variable(torch.from_numpy(np.array([0, 1, 2])))
print(embedding(idx))