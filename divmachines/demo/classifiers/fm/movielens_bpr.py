from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian2D
from divmachines.classifiers import BPR_FM
from torch.optim.adam import Adam
import pandas as pd
import numpy as np

logger = TLogger()
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base',
                    delimiter='\t',
                    names=cols)[['user',
                                 'item',
                                 'rating']]
n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

item_cat = train[['item']].drop_duplicates().values

train.loc[train['rating'] < 4, 'rating'] = 0
train.loc[train['rating'] >= 4, 'rating'] = 1
users = train.user.unique()
users_len = users.shape[0]
idx = np.arange(users_len)
np.random.shuffle(idx)
user_take = idx[:int(users_len*0.8)]
ut = users[user_take]

test = train.loc[~train['user'].isin(ut)]
train = train.loc[train['user'].isin(ut)]

dr = test.groupby('user', as_index=False) \
    .apply(lambda g: g.sample(10)).reset_index(0, drop=True)

# Create Ground Truth
ground = test[~test.index.isin(dr.index)].dropna()

train = pd.concat((train, dr))
users = ground.user.unique()

model = BPR_FM(n_iter=100,
               learning_rate=1e-1,
               sparse=False,
               batch_size=4096,
               n_jobs=8,
               use_cuda=True,
               optimizer_func=Adam,
               verbose=True,
               stopping=True,
               early_stopping=True)

interactions = train[['user', 'item', 'rating']].values
np.random.shuffle(interactions)
x = interactions[:, :-1]
y = interactions[:, -1]


model.fit(x, y, {'users': 0, 'items': 1},
          n_users=n_users, n_items=n_items)


x = cartesian2D(ground.user.unique().reshape(-1, 1), item_cat.reshape(-1, 1))
pred = model.predict(x).reshape(ground.user.unique().shape[0], item_cat.shape[0])

np.save("predictions.txt", pred)
arg = np.argsort(-pred, 1)
np.save("rankings.txt", arg)
