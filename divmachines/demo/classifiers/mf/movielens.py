import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from divmachines.classifiers import MF
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)

# map_user = train.groupby('user').count().reset_index()[['user']].reset_index()
# map_user.columns = ['u_idx', 'user']
# map_item = train.groupby('item').count().reset_index()[['item']].reset_index()
# map_item.columns = ['i_idx', 'item']
# train = pd.merge(pd.merge(train, map_user, on="user"), map_item, on="item")

logger = TLogger()

model = MF(n_iter=100,
           n_jobs=2,
           batch_size=1000,
           learning_rate=0.60653066,
           use_cuda=False,
           logger=logger,
           early_stopping=True,
           verbose=True)

interactions = train[['user', 'item', 'rating']].values

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

x = interactions[:100, :-1]
y = interactions[:100, -1]

model.fit(x,
          y,
          dic={'users': 0, 'items': 1},
          n_users=n_users, n_items=n_items)

print(model.predict(x))
model.save("./time.pth.tar")

model = MF(n_iter=1,
           n_jobs=8,
           batch_size=10,
           learning_rate=0.60653066,
           use_cuda=False,
           logger=logger,
           early_stopping=True,
           model="./time.pth.tar",
           verbose=True)

x = interactions[:100, :-1]
y = interactions[:100, -1]
print(model.predict(x))
plt.plot(logger.epochs, logger.losses)
plt.show()
