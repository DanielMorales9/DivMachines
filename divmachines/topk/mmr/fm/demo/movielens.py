from divmachines.topk.mmr import FM_MMR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from divmachines.logging import TrainingLogger as TLogger
from divmachines.utility.helper import cartesian2D

DATASET_PATH = './../../../../../data/ua.base'
GENRE_PATH = './../../../../../data/u.item'
data = pd.read_csv(DATASET_PATH, sep="\t", names=['user', 'item', 'rating', 'time'])
header = "item | movie_title | release_date | video_release_date | " \
         "IMDb_URL | unknown | Action | Adventure | Animation | Children's | " \
         "Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | " \
         "Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western "
header = header.replace(" |", "")
header = header.split()
items = pd.read_csv(GENRE_PATH, sep="|", names=header, encoding='iso-8859-2')
proj = ['user', 'item']
proj.extend(header[5:])
proj.append('rating')
train = pd.merge(data, items, on='item', how='inner')[proj]

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

logger = TLogger()

model = FM_MMR(n_iter=10,
               n_jobs=8,
               n_factors=100,
               learning_rate=.1,
               use_cuda=False,
               batch_size=1000,
               verbose=True,
               logger=logger)

interactions = train.values
x = interactions[:, :-1]
y = interactions[:, -1]

model.fit(x, y, dic={'users': 0, 'items': 1}, n_users=n_users, n_items=n_items)
plt.plot(logger.epochs, logger.losses)
plt.show()
users = np.unique(x[:10, 0]).reshape(-1, 1)
items = np.unique(x[:, 1:], axis=0)
values = cartesian2D(users, items)
top = 3
table = np.zeros((users.shape[0], top+1), dtype=np.int)
table[:, 0] = users[:, 0]
table[:, 1:] = model.predict(values, top=top, b=1)
print(table)
