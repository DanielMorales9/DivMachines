import matplotlib.pyplot as plt
import pandas as pd
from divmachines.classifiers.fm import FM
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)
logger = TLogger()

model = FM(n_iter=100,
           learning_rate=1e-1,
           logger=logger,
           batch_size=100,
           n_jobs=2)
model.fit(train[['user', 'item']].values,
          train['rating'].values,
          {'users': 0, 'items': 1})

plt.plot(logger.epochs, logger.losses)
plt.show()

#print(models.predict(np.array([0, 1])))
