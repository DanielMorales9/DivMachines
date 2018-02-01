import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from divmachines.classifiers import FM
import matplotlib.pyplot as plt
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)
logger = TLogger()

n_users = np.unique(train[["user"]].values).shape[0]
n_items = np.unique(train[["item"]].values).shape[0]

print("Number of users: %s" % n_users)
print("Number of items: %s" % n_items)

model = FM(n_iter=1,
           learning_rate=1e-1,
           n_jobs=5,
           batch_size=100,
           logger=logger)

interactions = train[['user', 'item', 'timestamp', 'rating']].values

x = interactions[:, :-1]
y = interactions[:, -1]
sc = StandardScaler()
x[:, -1] = sc.fit_transform(x[:, -1].reshape(-1, 1).astype(np.float32)).reshape(1, -1)

model.fit(x, y, {'users': 0, 'items': 1})

print(model.predict(x))
