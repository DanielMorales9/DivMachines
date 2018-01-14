import numpy as np
from itertools import count
from divmachines.classifiers import Classifier
from divmachines.classifiers.mf import MF
from divmachines.utility.helper import shape_prediction


ITEMS = 'items'
USERS = 'users'


class LatentFactorPortfolio(Classifier):
    """
    Latent Factor Portfolio implementation based on the following work

    `Shi, Y., Zhao, X., Wang, J., Larson, M., & Hanjalic, A. (2012, August).
    Adaptive diversification of recommendation results via latent factor portfolio.
    In Proceedings of the 35th international ACM SIGIR conference on Research
    and development in information retrieval (pp. 175-184). ACM.`

    Parameters
    ----------
    model: classifier, optional
        An instance of `divmachines.classifier.mf.MF`.
        Default is None
    n_factors: int, optional
        Number of factors to use in user and item latent factors
    sparse: boolean, optional
        Use sparse dataset
    loss: function, optional
        an instance of a Pytorch optimizer or a custom loss.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a Pytorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use SGD by default.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Mini batch size.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    use_cuda: boolean, optional
        Run the models on a GPU.
    logger: :class:`divmachines.logging`, optional
        A logger instance for logging during the training process
    n_jobs: int, optional
        Number of jobs for data loading.
        Default is 0, it means that the data loader runs in the main process.
    """

    def __init__(self,
                 model=None,
                 n_factors=10,
                 sparse=False,
                 n_iter=10,
                 loss=None,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer_func=None,
                 batch_size=None,
                 random_state=None,
                 use_cuda=False,
                 logger=None,
                 n_jobs=0):
        self._model = model
        self._n_factors = n_factors
        self._sparse = sparse
        self._n_iter = n_iter
        self._loss = loss
        self._l2 = l2
        self._learning_rate = learning_rate
        self._optimizer_func = optimizer_func
        self._batch_size = batch_size
        self._random_state = random_state
        self._use_cuda = use_cuda
        self._logger = logger
        self._n_jobs = n_jobs

        self._initialized = False

    @property
    def n_users(self):
        return self._model.n_users

    @n_users.getter
    def n_users(self):
        return self._model.n_users

    @property
    def n_items(self):
        return self._model.n_items

    @n_items.getter
    def n_items(self):
        return self._model.n_items

    def _initialize(self):
        self._init_model()

    def _init_model(self):
        if self._model is None:
            self._model = MF(n_factors=self._n_factors,
                             sparse=self._sparse,
                             n_iter=self._n_iter,
                             loss=self._loss,
                             l2=self._l2,
                             learning_rate=self._learning_rate,
                             optimizer_func=self._optimizer_func,
                             batch_size=self._batch_size,
                             random_state=self._random_state,
                             use_cuda=self._use_cuda,
                             logger=self._logger,
                             n_jobs=self._n_jobs)
        elif not isinstance(self._model, MF):
            raise ValueError("Model must be an instance of "
                             "divmachines.classifiers.mf.MF class")

    def _init_dataset(self, x):
        self._item_index = {}
        # self._user_index = {}
        for k, v in self._model.index.items():
            if k.startswith(ITEMS):
                try:
                    self._item_index[v] = int(k[len(ITEMS):])
                except ValueError:
                    raise ValueError("You may want to provide an integer "
                                     "index for the items")
            # elif k.startswith(USERS):
            #     try:
            #         self._user_index[v] = int(k[len(USERS):])
            #     except ValueError:
            #         raise ValueError("You may want to provide an integer "
            #                          "index for the users")
            # else:
            #     raise ValueError("Not possible")
        self._item_catalog = np.array(list(self._item_index.values()))

    def fit(self, x, y, user_col=0, item_col=1, n_users=None, n_items=None):
        """
        Fit the underlying classifier.
        When called repeatedly, models fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        x: ndarray
            Training samples
        y: ndarray
            Target values for samples
        user_col: int, optional
            user column in training samples
        item_col: int, optional
            item column in training samples
        n_users: int, optional
            Total number of users. The model will have `n_users` rows.
            Default is None, `n_users` will be inferred from `x`.
        n_items: int, optional
            Total number of items. The model will have `n_items` columns.
            Default is None, `n_items` will be inferred from `x`.
        """
        if not self._initialized:
            self._initialize()

        dic = {USERS: user_col, ITEMS: item_col}

        self._model.fit(x, y, dic=dic, n_users=n_users, n_items=n_items)
        self._init_dataset(x)

    def predict(self, x, top=10, b=0.5):
        """
        Predicts

        Parameters
        ----------
        x: ndarray or int
            array of users, user item interactions matrix (for each user
            you must provide the whole item_catalog) or a single user
            to which predict the item ranking.
        top: int, optional
            Length of the ranking
        b: float, optional
            System-level Diversity.
            It controls the trade-off for all users between
            accuracy and diversity.
        Returns
        -------
        topk: ndarray
            `top` items for each user supplied
        """

        n_items, n_users, test, update_dataset = \
            shape_prediction(self._item_catalog, x)

        try:
            rank = self._model.predict(test).reshape(n_users, n_items)
        except ValueError:
            raise ValueError("You may want to provide for each user "
                             "the item catalog as transaction matrix.")

        # Rev-indexes and other data structures are updated
        # if new items and new users have been provided.
        if update_dataset:
            self._init_dataset(x)

        users = np.unique(test[:, 0])

        re_ranking = self._sequential_re_ranking(rank, users, top, b)

        return re_ranking

    def _sequential_re_ranking(self, rank, users, top, b):
        rank = np.argsort(-rank, 1)
        print(self.rev_index(rank[:, :top]))

        rows = np.arange(self.n_users)
        model = self._model.model

        for k in range(1, top):
            values = np.zeros((self.n_users, self.n_items-k))
            arg_max_per_user = np.argsort(values, 1)[:, 0]
            substitute(arg_max_per_user, k, rank, rows)

        return self.rev_index(rank[:, :top])

    def _compute_f(self, rank, users, k, b):
        a = np.zeros((self.n_users, self.n_items-k))
        return a

    def rev_index(self, rank):
        re_idx = np.vectorize(lambda x: self._item_index[x])
        return np.array([re_idx(lis) for lis in rank])


def substitute(arg_max_per_user, k, rank, rows):
    for r, c in zip(rows, arg_max_per_user):
        temp = rank[r, c + k]
        rank[r, c + k] = rank[r, k]
        rank[r, k] = temp
