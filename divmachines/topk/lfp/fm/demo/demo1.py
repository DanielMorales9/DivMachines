import numpy as np
import matplotlib.pyplot as plt
from divmachines.topk.lfp import LFP_FM
from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian2D

interactions = np.array([[100, 10, 0, 1, 5],
                         [200, 10, 0, 1, 1],
                         [100, 20, 1, 1, 2],
                         [200, 20, 1, 1, 4],
                         [100, 30, 1, 0, 3],
                         [200, 30, 1, 0, 2],
                         [100, 40, 1, 1, 4],
                         [200, 40, 1, 1, 4]])

n_users = 2
n_items = 4

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

logger = TLogger()

model = LFP_FM(n_iter=45,
               n_jobs=2,
               n_factors=4,
               learning_rate=.3,
               logger=logger)

x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x, y, n_users=n_users, n_items=n_items)

plt.plot(logger.epochs, logger.losses)
plt.show()
users = np.unique(x[:, 0]).reshape(-1, 1)[::-1, :].copy()
items = np.unique(x[:, 1:], axis=0)
values = cartesian2D(users, items)
top = 3
table = np.zeros((users.shape[0], top+1), dtype=np.int)
table[:, 0] = users[:, 0]
table[:, 1:] = model.predict(values, top=top, b=-1)
print(table)
