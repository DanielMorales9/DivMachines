import numpy as np
import torch
from torch.autograd.variable import Variable
from divmachines.models.layers import TestEmbedding
n_users = 3
n_items = 5
n_factors = 2

x_weights = np.array([[0, 1],
                      [2, 3],
                      [4, 5]], dtype=np.float)
y_weights = np.array([[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7],
                      [8, 9]], np.float)

x = TestEmbedding(n_users, n_factors, embedding_weights=x_weights)
y = TestEmbedding(n_items, n_factors, embedding_weights=y_weights)

users = np.array([0, 1, 2])
rank = np.array([[0, 1, 2, 3, 4],
                 [0, 1, 2, 3, 4],
                 [0, 1, 2, 3, 4]])

k = 1
wm = Variable(torch.from_numpy(
                np.array([1 / (2 ** m) for m in range(k)],
                         dtype=np.float32)))\
                .unsqueeze(1).expand(k, n_factors)

print(wm)
# Initialize Index Vars
i_idx = Variable(torch.from_numpy(rank))
i_ranked = (y(i_idx[:, :k]) * wm).transpose(0, 1).unsqueeze(0)
i_unranked = y(i_idx[:, k:]).transpose(0, 1).unsqueeze(0)
e_ranked = i_ranked.expand(n_items-k, k, n_users, n_factors).transpose(0, 1)
e_unranked = i_unranked.expand(k, n_items-k, n_users, n_factors)
print(i_ranked)
print(e_ranked)
print(i_unranked)
print(e_unranked)
term1 = (e_ranked * e_unranked).sum(0)
print(term1)





