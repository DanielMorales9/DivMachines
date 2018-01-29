import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from divmachines.topk.seqrank import FM_SeqRank
from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian2D

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
interactions = train[['user_id', 'tps_id', 'artist_id', 'mode', 'playcounts']].values

logger = TLogger()
model = FM_SeqRank(n_iter=1,
                   learning_rate=1e-1,
                   logger=logger,
                   n_jobs=4,
                   use_cuda=False)

x = interactions[:, :-1]
y = interactions[:, -1]
model.fit(x, y, dic={'users': 0, 'items': 1, 'artists': 2},
          n_users=n_users,
          n_items=n_items,
          lengths={"n_artists": n_artists})
logger = TLogger()

plt.plot(logger.epochs, logger.losses)
plt.show()
users = np.unique(x[:, 0]).reshape(-1, 1)[:10, :]
items = train[['tps_id', 'artist_id', 'mode']].drop_duplicates('tps_id').values
values = cartesian2D(users, items)
top = 3
table = np.zeros((users.shape[0], top+1), dtype=np.object)
table[:, 0] = users[:, 0]
table[:, 1:] = model.predict(values, top=top, b=1)
print(table)