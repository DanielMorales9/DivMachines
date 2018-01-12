import pandas as pd
from divmachines.classifiers.mf import MF
from divmachines.model_selection.search import GridSearchCV

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)

model = MF()
train = train[["user", "item", "rating"]].values
x = train[:, :-1]
y = train[:, -1]

gSearch = GridSearchCV(model,
                       param_grid={"iter": [1], "learning_rate": [0.1, 0.3]},
                       cv='userHoldOut',
                       metrics='mean_square_error',
                       verbose=10,
                       n_jobs=4,
                       return_train_score=True)

gSearch.fit(x, y, fit_params={'dic': {'users': 0, 'items': 1}})

print(gSearch.get_scores(pretty=True))
