import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from divmachines.mf import MatrixFactorization
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.test', delimiter='\t', names=cols)

map_user = train.groupby('user').count().reset_index()[['user']].reset_index()
map_user.columns = ['u_idx', 'user']
map_item = train.groupby('item').count().reset_index()[['item']].reset_index()
map_item.columns = ['i_idx', 'item']
train = pd.merge(pd.merge(train, map_user, on="user"), map_item, on="item")

logger = TLogger()

model = MatrixFactorization(n_iter=100, learning_rate=0.60653066, logger=logger)

interactions = train[['u_idx', 'i_idx', 'rating']].values

model.fit(interactions)

print(len(model.predict(np.unique(train[['u_idx']].values))))

plt.plot(logger.epochs, logger.losses)
plt.show()
