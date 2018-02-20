import timeit
import pandas as pd
import numpy as np
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)


n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

r = """
from divmachines.classifiers import FM
model = FM(n_iter=10,
           batch_size=1000,
           learning_rate=1e-1,
           n_jobs=3,
           sparse=True,
           verbose=True)

interactions = train[['user', 'item', 'rating']].values

x = interactions[:, :-1]
y = interactions[:, -1]


model.fit(x, y, {'users': 0, 'items': 1})"""

print(timeit.timeit(stmt=r, number=5, globals=globals()))


s = """
from divmachines.classifiers import FM
model = FM(n_iter=10, 
           batch_size=1000,
           learning_rate=1e-1,
           n_jobs=3, 
           verbose=True,
           sparse=False)

interactions = train[['user', 'item', 'rating']].values

x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x, y, {'users': 0, 'items': 1})"""

print(timeit.timeit(stmt=s, number=5, globals=globals()))
