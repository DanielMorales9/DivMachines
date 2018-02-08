import pandas as pd
import numpy as np
from divmachines.classifiers import FM
from divmachines.logging import TrainingLogger as TLogger
import torch
logger = TLogger()
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

model = FM(n_iter=1,
           batch_size=1000,
           learning_rate=1e-1,
           sparse=True,
           n_jobs=4,
           verbose=True,
           logger=logger,
           early_stopping=True)

interactions = train[['user', 'item', 'rating']].values
np.random.shuffle(interactions)
x = interactions[:, :-1]
y = interactions[:, -1]


model.fit(x, y, {'users': 0, 'items': 1})

print(model.predict(x))

model.save("./time.pth.tar")

saved_model = FM(model="./time.pth.tar",
                 n_iter=5,
                 batch_size=1000,
                 learning_rate=1e-1,
                 sparse=True,
                 n_jobs=2,
                 logger=logger)


print(saved_model.predict(x))
