import pandas as pd
from divmachines.classifiers.mf import MF
from divmachines.model_selection import cross_validate
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)
logger = TLogger()

model = MF(n_iter=10,
           learning_rate=1e-1)
train = train[["user", "item", "rating"]].values
x = train[:, :-1]
y = train[:, -1]

for k, v in cross_validate(model, x, y,
                           cv='userHoldOut',
                           fit_params={'dic': {'users': 0, 'items': 1}},
                           metrics='mean_square_error',
                           verbose=10,
                           n_jobs=2,
                           return_times=True,
                           return_train_score=True).items():
    print("%s\t %s" % (k, v))
