import pandas as pd
import numpy as np
from divmachines.classifiers import FM
import matplotlib.pyplot as plt
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)
logger = TLogger()

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

model = FM(n_iter=10,
           learning_rate=1e-1,
           logger=logger,
           batch_size=1000,
           n_jobs=4,
           sparse=True,
           use_cuda=True)

interactions = train[['user', 'item', 'rating']].values
x = interactions[:, :-1]
y = interactions[:, -1]
model.fit(x, y, {'users': 0, 'items': 1})
plt.plot(logger.epochs, logger.losses)
plt.show()

print(model.predict(x))
