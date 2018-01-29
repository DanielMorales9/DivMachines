import pandas as pd
from divmachines.classifiers import MF
from divmachines.logging import TrainingLogger as TLogger

PATH = "/home/daniel/Desktop/MSD/triplets_short.csv"
FEATS_PATH = "/home/daniel/Desktop/MSD/features.csv"
df = pd.read_csv(PATH, sep=",", header=0)

feats = pd.read_csv(FEATS_PATH, sep=",", header=0)[['artist_id', 'mode', 'tps_id']]
train = pd.merge(df, feats, on="tps_id")

n_users = df.user_id.unique().shape[0]
n_items = df.tps_id.unique().shape[0]
n_artists = feats.artist_id.unique().shape[0]
interactions = train[['user_id', 'tps_id', 'playcounts']].values

logger = TLogger()
model = MF(n_iter=10,
           learning_rate=1e-1,
           logger=logger,
           n_jobs=4,
           use_cuda=False)

x = interactions[:, :-1]
y = interactions[:, -1]
model.fit(x, y, {'users': 0, 'items': 1},
          n_users=n_users,
          n_items=n_items)

print(model.predict(x))
