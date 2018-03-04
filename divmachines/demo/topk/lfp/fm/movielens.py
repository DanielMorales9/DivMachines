from torch.utils.data import DataLoader, Dataset
from divmachines.logging import TrainingLogger as TLogger
from torch.optim.adam import Adam
from tqdm import tqdm
from divmachines.topk.lfp import FM_LFP
from divmachines.classifiers import FM
import numpy as np
import pandas as pd
from divmachines.utility.helper import cartesian2D
import os

UPL = 3
N_JOBS = 3
N_ITER = 10
TOP = 5
FACTORS = 10
USERS_BATCH = 10
LEARNING_RATE = .001
BATCH_SIZE = 2048
VERBOSE = True
USE_CUDA = False
SPARSE = False
STOP = True
TRIPLETS_PATH = './../../../../../data/ua.base'
GENRE_PATH = './../../../../../data/u.item'
HEADER = "item | movie_title | release_date | video_release_date | " \
         "IMDb_URL | unknown | Action | Adventure | Animation | Children's | " \
         "Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | " \
         "Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western "
MODEL_PATH = "./saveme.pth.tar"
ground_path = "ground-upl" + str(UPL) + ".csv"
train_path = "train-upl" + str(UPL) + ".csv"


HEADER = HEADER.replace(" |", "")
HEADER = HEADER.split()
proj = ['user', 'item']
proj.extend(HEADER[5:])
proj.append('rating')

if not os.path.exists(MODEL_PATH):
    data = pd.read_csv(TRIPLETS_PATH, sep="\t", names=['user', 'item', 'rating', 'time'])
    items = pd.read_csv(GENRE_PATH, sep="|", names=HEADER, encoding='iso-8859-2')
    train = pd.merge(data, items, on='item', how='inner')[proj]
    n_users = np.unique(train[["user"]].values).shape[0]
    n_items = np.unique(train[["item"]].values).shape[0]

    # Train-test Split
    users = train.user.unique()
    np.random.shuffle(users)
    train_users_mask = int(users.shape[0]*0.8)
    train_users = users[0:train_users_mask]
    test = train.loc[~train.user.isin(train_users)]
    train = train.loc[train.user.isin(train_users)]

    dr = test.groupby('user', as_index=False) \
        .apply(lambda g: g.sample(2)).reset_index(0, drop=True)

    # Create Ground Truth
    # ground = test[test.user_id != dr.user_id & (test.tps_id != dr.tps_id)]
    ground = test[~test.index.isin(dr.index)].dropna()
    train = pd.concat((train, dr))

    print("Number of users: %s" % n_users)
    print("Number of items: %s" % n_items)

    logger = TLogger()

    model = FM_LFP(n_iter=N_ITER,
                   optimizer_func=Adam,
                   n_jobs=N_JOBS,
                   n_factors=FACTORS,
                   batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE,
                   use_cuda=USE_CUDA,
                   verbose=VERBOSE,
                   sparse=SPARSE,
                   logger=logger,
                   early_stopping=STOP)

    interactions = train.values
    x = interactions[:, :-1]
    y = interactions[:, -1]

    model.fit(x, y, dic={'users': 0, 'items': 1},
              n_users=n_users, n_items=n_items)

    model.save(MODEL_PATH)
    train.to_csv(train_path, index=USE_CUDA)
    ground.to_csv(ground_path, index=USE_CUDA)

model = FM_LFP(n_iter=N_ITER,
               model=MODEL_PATH,
               n_jobs=N_JOBS,
               batch_size=BATCH_SIZE,
               n_factors=FACTORS,
               learning_rate=LEARNING_RATE,
               use_cuda=USE_CUDA,
               verbose=VERBOSE,
               sparse=SPARSE)

train = pd.read_csv(train_path, header=0)
ground = pd.read_csv(ground_path, header=0)

dataset = pd.concat((train, ground))
users = dataset.user.unique()
item_catalogue = dataset[proj[1:-1]].drop_duplicates().values

values = cartesian2D(users.reshape(-1, 1), item_catalogue)

model0 = FM(n_iter=N_ITER,
            model=MODEL_PATH,
            n_jobs=N_JOBS,
            batch_size=BATCH_SIZE,
            n_factors=FACTORS,
            learning_rate=LEARNING_RATE,
            use_cuda=USE_CUDA,
            verbose=VERBOSE,
            sparse=SPARSE)

rank = model0.predict(values) \
    .reshape(users.shape[0], item_catalogue.shape[0])


for b in tqdm([1.0], desc="sys.div.", leave=False):
    table = np.zeros((users.shape[0], TOP + 1), dtype=np.object)
    table[:, 0] = users
    table[:, 1:] = model.predict(values, top=TOP, b=b, rank=rank)
    np.savetxt("./results-b"+str(b), table, fmt="%s")
