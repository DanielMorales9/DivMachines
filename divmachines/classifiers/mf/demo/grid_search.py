import pandas as pd
import numpy as np
from divmachines.classifiers.mf import MF
from divmachines.model_selection.search import GridSearchCV

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

train = train[["user", "item", "rating"]].values
x = train[:, :-1]
y = train[:, -1]

model = MF()

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

gSearch = GridSearchCV(model,
                       param_grid={"iter": [1], "learning_rate": [0.1]},
                       cv='userHoldOut',
                       metrics='mean_square_error',
                       verbose=10,
                       n_jobs=4,
                       return_train_score=True)

gSearch.fit(x, y, fit_params={'dic':
                                  {'users': 0, 'items': 1},
                              'n_users': n_users,
                              'n_items': n_items
                              })

print(gSearch.get_scores(pretty=True))
