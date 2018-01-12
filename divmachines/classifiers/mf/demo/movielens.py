import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from divmachines.classifiers.mf import MF
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)

map_user = train.groupby('user').count().reset_index()[['user']].reset_index()
map_user.columns = ['u_idx', 'user']
map_item = train.groupby('item').count().reset_index()[['item']].reset_index()
map_item.columns = ['i_idx', 'item']
train = pd.merge(pd.merge(train, map_user, on="user"), map_item, on="item")

logger = TLogger()

model = MF(n_iter=100, learning_rate=0.60653066, logger=logger)

interactions = train[['u_idx', 'i_idx', 'rating']].values

x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x, y)

print(model.predict(x))

plt.plot(logger.epochs, logger.losses)
plt.show()
