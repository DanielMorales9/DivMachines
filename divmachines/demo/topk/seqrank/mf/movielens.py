import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from divmachines.topk.seqrank import MF_SeqRank
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

model = MF_SeqRank(n_iter=10,
                   n_jobs=8,
                   n_factors=10,
                   learning_rate=1,
                   use_cuda=False,
                   verbose=True,
                   logger=logger)

x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x, y, dic={'users': 0, 'items': 1},
          n_users=n_users, n_items=n_items)

plt.plot(logger.epochs, logger.losses)
plt.show()
users = np.unique(x[:, 0])
values = cartesian(users, np.unique(x[:, 1]))

table = np.zeros((users.shape[0], 6), dtype=np.int)
table[:, 0] = users
table[:, 1:] = model.predict(values, top=5)
print(table)
