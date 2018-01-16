import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from divmachines.topk import LatentFactorPortfolio as LFP
from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../data/ua.base', delimiter='\t', names=cols).sample(10000)

logger = TLogger()

model = LFP(n_iter=1000,
            n_jobs=4,
            n_factors=10,
            batch_size=1000,
            learning_rate=1,
            logger=logger)


interactions = train[['user', 'item', 'rating']].values

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x, y, n_users=n_users, n_items=n_items)

plt.plot(logger.epochs, logger.losses)
plt.show()

values = cartesian(np.unique(x[:, 0]), np.unique(x[:, 1]))

print(model.predict(values, top=5))
