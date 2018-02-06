import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from divmachines.topk.lfp import MF_LFP
from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../../data/ua.base', delimiter='\t', names=cols)

interactions = train[['user', 'item', 'rating']].values

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)
logger = TLogger()

model = MF_LFP(n_iter=10,
               n_jobs=8,
               batch_size=1000,
               n_factors=10,
               learning_rate=1,
               logger=logger,
               verbose=True)

x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x, y, n_users=n_users, n_items=n_items)

users = np.unique(x[:, 0])
values = cartesian(users, np.unique(x[:, 1]))

table = np.zeros((users.shape[0], 6), dtype=np.int)
table[:, 0] = users
table[:, 1:] = model.predict(values, top=5)
print(table)
plt.plot(logger.epochs, logger.losses)
plt.show()
