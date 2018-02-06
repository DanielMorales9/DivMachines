import pandas as pd
import numpy as np
from divmachines.classifiers import FM
from divmachines.logging import TrainingLogger as TLogger
logger = TLogger()
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

model = FM(n_iter=20,
           batch_size=1000,
           learning_rate=1e-1,
           sparse_num=1,
           sparse=True,
           n_jobs=2,
           logger=logger)

interactions = train[['user', 'item', 'rating']].values
np.random.shuffle(interactions)
x = interactions[:, :-1]
y = interactions[:, -1]


model.fit(x, y, {'users': 0, 'items': 1})

print(model.predict(x))
