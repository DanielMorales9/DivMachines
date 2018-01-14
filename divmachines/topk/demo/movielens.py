import numpy as np
import pandas as pd
from divmachines.topk import LatentFactorPortfolio as LFP
from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../data/ua.base', delimiter='\t', names=cols).sample(10)

logger = TLogger()

model = LFP(n_iter=10, n_jobs=2, learning_rate=0.60653066, logger=logger)

interactions = train[['user', 'item', 'rating']].values

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x[:5, :], y[:5], n_users=n_users, n_items=n_items)

values = cartesian(np.unique(x[:, 0]), np.unique(x[:, 1]))

print(model.predict(values, top=5))
