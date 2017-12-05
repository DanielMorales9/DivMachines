from divmachines.validate import cross_validate
from divmachines.validate import NaiveHoldOut
from divmachines.mf import MF
from divmachines.fm import FM
from divmachines.utility import make_indexable
import pandas as pd

DATASET_PATH = '/home/daniel/Desktop/ml-100k/u.data'
GENRE_PATH = '/home/daniel/Desktop/ml-100k/u.item'
data = pd.read_csv(DATASET_PATH, sep="\t", header=None)
data.columns = ['user', 'item', 'rating', 'time']
header = "item | movie_title | release_date | video_release_date | " \
         "IMDb_URL | unknown | Action | Adventure | Animation | Children's | " \
         "Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | " \
         "Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western "
header = header.replace(" |", "")
header = header.split()
items = pd.read_csv(GENRE_PATH, sep="|", header=None, encoding='iso-8859-2')
items.columns = header
proj = ['user', 'item']
#proj.extend(header[5:])
proj.append('rating')
train = pd.merge(data, items, on='item', how='inner').sample(frac=0.005)[proj].values

model = FM(n_iter=100,
           learning_rate=1e-1)
x = train[:, :-1]
y = train[:, -1]

print(cross_validate(model,
                     x,
                     y,
                     cv='naiveHoldOut',
                     fit_params={'dic':{'users':0, 'items':1}},
                     metrics='mean_square_error',
                     verbose=10,
                     n_jobs=1))
