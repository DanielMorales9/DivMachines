from divmachines.model_selection.search import GridSearchCV
from divmachines.classifiers.fm import FM
import pandas as pd
ITEM_PATH = '../../../../data/u.item'

cols = ['user', 'item', 'rating', 'timestamp']
data = pd.read_csv('../../../../data/ua.base', delimiter='\t', names=cols)

header = "item | movie_title | release_date | video_release_date | " \
         "IMDb_URL | unknown | Action | Adventure | Animation | Children's | " \
         "Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | " \
         "Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western "
header = header.replace(" |", "")
header = header.split()
items = pd.read_csv(ITEM_PATH, sep="|", header=None, encoding='iso-8859-2')
items.columns = header
proj = ['user', 'item']
proj.extend(header[5:])
proj.append('rating')
train = pd.merge(data, items, on='item', how='inner')[proj].values

model = FM()
x = train[:, :-1]
y = train[:, -1]

gSearch = GridSearchCV(model,
                       param_grid={"iter": [1], "learning_rate": [0.1, 0.3]},
                       cv='userHoldOut',
                       metrics='mean_square_error',
                       verbose=10,
                       n_jobs=2,
                       return_train_score=True)

gSearch.fit(x, y, fit_params={'dic': {'users': 0, 'items': 1}})

print(gSearch.get_scores(pretty=True))
