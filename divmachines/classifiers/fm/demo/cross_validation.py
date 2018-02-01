import pandas as pd
import numpy as np
from divmachines.model_selection import cross_validate
from divmachines.model_selection.split import KFold
from divmachines.classifiers import FM

DATASET_PATH = './../../../../data/ua.base'
GENRE_PATH = './../../../../data/u.item'
data = pd.read_csv(DATASET_PATH, sep="\t", names=['user', 'item', 'rating', 'time']).sample(1000)
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

model = FM(n_iter=10,
           learning_rate=1e-1,
           use_cuda=False)

cv = KFold(folds=10, shuffle=True)

interactions = train.values
x = interactions[:, :-1]
y = interactions[:, -1]

for k, v in cross_validate(model, x, y,
                           cv=cv,
                           fit_params={'dic':
                                           {'users': 0, 'items': 1},
                                       'n_users': n_users,
                                       'n_items': n_items},
                           metrics='mean_square_error',
                           verbose=10,
                           n_jobs=8,
                           return_times=True,
                           return_train_score=True).items():
    print("%s\t %s" % (k, v))
