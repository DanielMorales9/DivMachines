
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from divmachines.fm import Pointwise
from divmachines.logging import TrainingLogger as TLogger

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.test', delimiter='\t', names=cols)

logger = TLogger()

model = Pointwise(n_iter=1,
                  learning_rate=1e-1,
                  logger=logger,
                  batch_size=1000,
                  n_workers=2)
model.fit(train[['user', 'item', 'rating']].values)

plt.plot(logger.epochs, logger.losses)
plt.show()

print(model.predict(np.array([0, 1])))
