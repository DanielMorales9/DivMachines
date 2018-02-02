from divmachines.model_selection import GridSearchCV
from divmachines.classifiers import FM
import pandas as pd
import numpy as np

DATASET_PATH = './../../../../data/ua.base'
GENRE_PATH = './../../../../data/u.item'
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
train = pd.merge(data, items, on='item', how='inner')[proj].sample(1000)

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

model = FM(verbose=True)

interactions = train.values
x = interactions[:, :-1]
y = interactions[:, -1]

gSearch = GridSearchCV(model,
                       param_grid={"n_iter": [1000], "learning_rate": [1], "l2": [0, 0.1, 0.2]},
                       cv='kFold',
                       metrics='mean_square_error',
                       n_jobs=8,
                       verbose=10,
                       return_train_score=True)

gSearch.fit(x, y, fit_params={'dic':
                                  {'users': 0, 'items': 1},
                              'n_users': n_users,
                              'n_items': n_items})

print(gSearch.get_scores(pretty=True))
