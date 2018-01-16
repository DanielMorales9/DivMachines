import pandas as pd
import numpy as np
from divmachines.classifiers import MF
from divmachines.model_selection import cross_validate
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols).head(100)
logger = TLogger()

model = MF(n_iter=10,
           learning_rate=1e-1)
n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

train = train[["user", "item", "rating"]].values
x = train[:, :-1]
y = train[:, -1]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

for k, v in cross_validate(model,
                           x,
                           y,
                           cv='userHoldOut',
                           fit_params={'dic': {'users': 0, 'items': 1},
                                       'n_users': n_users,
                                       'n_items': n_items},
                           metrics='mean_square_error',
                           verbose=10,
                           n_jobs=4,
                           return_times=False,
                           return_train_score=False).items():
    print("%s\t %s" % (k, v))
