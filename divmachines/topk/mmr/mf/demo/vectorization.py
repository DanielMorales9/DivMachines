import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from divmachines.topk.mmr import MF_MMR
from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian

PATH = "/home/daniel/Desktop/MSD/triplets_short.csv"
FEATS_PATH = "/home/daniel/Desktop/MSD/features.csv"
df = pd.read_csv(PATH, sep=",", header=0)

feats = pd.read_csv(FEATS_PATH, sep=",", header=0)[['tps_id', 'artist_id', 'mode']]
train = pd.merge(df, feats, on="tps_id")

n_users = df.user_id.unique().shape[0]
n_items = df.tps_id.unique().shape[0]
print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)
n_artists = feats.artist_id.unique().shape[0]
interactions = train[['user_id', 'tps_id','playcounts']].values

logger = TLogger()
model = MF_MMR(n_iter=1,
               learning_rate=1e-1,
               logger=logger,
               n_jobs=4,
               use_cuda=False)

x = interactions[:, :-1]
y = interactions[:, -1]
model.fit(x, y,
          n_users=n_users,
          n_items=n_items)
logger = TLogger()

plt.plot(logger.epochs, logger.losses)
plt.show()
users = np.unique(x[:, 0])[:10]
items = np.unique(x[:, 1])
values = cartesian(users, items)
top = 3
table = np.zeros((users.shape[0], top+1), dtype=np.object)
table[:, 0] = users
table[:, 1:] = model.predict(values, top=top, b=1)
print(table)