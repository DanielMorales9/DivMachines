import timeit

NUMBER = 10


def one_worker():
    SETUP_CODE = '''
import pandas as pd
from divmachines.fm import PointwiseClassifierFM


cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.test', delimiter='\t', names=cols)
    '''
    TEST_CODE = '''
model = PointwiseClassifierFM(n_iter=1,
                              learning_rate=1e-1,
                              batch_size=1000, 
                              n_workers=1)
model.fit(train[['user', 'item', 'rating']].values)'''

    times = timeit.timeit(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          number=NUMBER)


    # priniting avg exec. time
    print('One worker time: {}'.format(times))

def two_workers():
    SETUP_CODE = '''
import pandas as pd
from divmachines.fm import PointwiseClassifierFM

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.test', delimiter='\t', names=cols)'''
    TEST_CODE = '''
model = PointwiseClassifierFM(n_iter=1,
                      learning_rate=1e-1,
                      batch_size=1000, 
                      n_workers=2)
model.fit(train[['user', 'item', 'rating']].values)'''

    times = timeit.timeit(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          number=NUMBER)

    # priniting avg exec. time
    print('Two workers time: {}'.format(times))

def four_worker():
    SETUP_CODE = '''
import pandas as pd
from divmachines.fm import PointwiseClassifierFM


cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('~/Stuff/fm_tensorflow-master/data/ua.test', delimiter='\t', names=cols)
    '''
    TEST_CODE = '''
model = PointwiseClassifierFM(n_iter=1,
                              learning_rate=1e-1,
                              batch_size=1000, 
                              n_workers=1)
model.fit(train[['user', 'item', 'rating']].values)'''

    times = timeit.timeit(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          number=NUMBER)


    # priniting avg exec. time
    print('Four worker time: {}'.format(times))

one_worker()
two_workers()
four_worker()