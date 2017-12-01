from divmachines.fm.dataset import DenseDataset, SparseDataset
from divmachines.logging import TrainingLogger as TLogger
from divmachines.fm import Pointwise
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
DATASET_PATH = '/home/daniel/Desktop/ml-100k/u.data'
GENRE_PATH = '/home/daniel/Desktop/ml-100k/u.item'
data = pd.read_csv(DATASET_PATH, sep="\t", header=None)
data.columns = ['user', 'item', 'rating', 'time']
header = "item | movie_title | release_date | video_release_date | " \
         "IMDb_URL | unknown | Action | Adventure | Animation | Children's | " \
         "Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | " \
         "Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western "
header = header.replace(" |", "")
header = header.split()
items = pd.read_csv(GENRE_PATH, sep="|", header=None, encoding='iso-8859-2')
items.columns = header
proj = ['user', 'item']
proj.extend(header[5:])
proj.append('rating')
train = pd.merge(data, items, on='item', how='inner')[proj].values

d = DenseDataset(train, dic={'users': 0, 'items': 1})
logger = TLogger()

model = Pointwise(n_iter=20,
                  learning_rate=1,
                  logger=logger,
                  batch_size=1000,
                  n_workers=4)
model.fit(d)
epochs, losses = logger.epochs, logger.losses
logs = np.zeros(shape=(len(epochs), 2))
logs[:, 0] = epochs
logs[:, 1] = losses

df = pd.DataFrame(logs, columns=['epochs', 'losses'])
logs = df.groupby('epochs').mean().values

plt.plot(logger.epochs, logger.losses)
plt.show()

print(model.predict(train))

