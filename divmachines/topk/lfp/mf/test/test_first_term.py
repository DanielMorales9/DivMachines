import numpy as np
import torch
from torch.autograd.variable import Variable
from divmachines.models.layers import TestEmbedding

users = 3
items = 5
factors = 2
k = 1

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

print(y.weight)


users = np.array([0, 1, 2])
rank = np.array([[0, 1, 2, 3, 4],
                 [0, 1, 2, 3, 4],
                 [0, 1, 2, 3, 4]])

u_idx = Variable(torch.from_numpy(users))
i_idx = Variable(torch.from_numpy(rank))
print(y(i_idx[:, 2:]))

i_unranked = y(i_idx[:, k:]).transpose(0, 1)

users_batch = x(u_idx)
term0 = (users_batch * i_unranked)
t0 = term0.data.numpy()

print(t0.shape == (4, 3, 2))

print(np.all(t0 == np.array([[[0.,  3.],
                              [4.,   9.],
                              [8., 15.]],

                             [[0.,   5.],
                              [8.,  15.],
                              [16., 25.]],
                             [[0.,  7.],
                              [12.,  21.],
                              [24.,  35.]],
                             [[0.,  9.],
                              [16.,  27.],
                              [32.,  45.]]])))