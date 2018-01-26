import torch
import numpy as np
import pandas as pd
from torch.autograd.variable import Variable
from divmachines.classifiers import Classifier
from divmachines.classifiers.mf import MF
from divmachines.utility.helper import shape_for_mf, \
    _swap_k, _tensor_swap_k, index, re_index
from divmachines.utility.torch import gpu


ITEMS = 'items'
USERS = 'users'


class MF_MMR(Classifier):
    """
    Maximal Marginal Relevance with Matrix Factorization Correlation measure

    Parameters
    ----------
    model: classifier, optional
        An instance of `divmachines.classifier.lfp.MF`.
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

    @property
    def logger(self):
        return self._logger

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
                             "divmachines.classifiers.lfp.MF class")

    def _init_dataset(self, x):
        self._rev_item_index = {}
        self._user_index = {}
        self._item_index = {}
        for k, v in self._model.index.items():
            if k.startswith(ITEMS):
                try:
                    self._rev_item_index[v] = int(k[len(ITEMS):])
                    self._item_index[int(k[len(ITEMS):])] = v
                except ValueError:
                    raise ValueError("You may want to provide an integer "
                                     "index for the items")
            elif k.startswith(USERS):
                try:
                    self._user_index[int(k[len(USERS):])] = v
                except ValueError:
                    raise ValueError("You may want to provide an integer "
                                     "index for the users")
            else:
                raise ValueError("Not possible")
        self._item_catalog = np.array(list(self._rev_item_index.values()))

    def fit(self, x, y, n_users=None, n_items=None):
        """
        Fit the underlying classifier.
        When called repeatedly, models fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        x: ndarray
            Training samples. User column must be 0 while item column
            must be 1
        y: ndarray
            Target values for samples
        n_users: int, optional
            Total number of users. The model will have `n_users` rows.
            Default is None, `n_users` will be inferred from `x`.
        n_items: int, optional
            Total number of items. The model will have `n_items` columns.
            Default is None, `n_items` will be inferred from `x`.
        """
        if not self._initialized:
            self._initialize()

        if x.shape[1] != 2:
            raise ValueError("x must have two columns: users and items cols")

        dic = {USERS: 0, ITEMS: 1}

        self._model.fit(x, y, dic=dic, n_users=n_users, n_items=n_items)
        self._init_dataset(x)

    def predict(self, x, top=10, b=0.5):
        """
        Predicts

        Parameters
        ----------
        x: ndarray or int
            array of users, user item interactions matrix
            (you must provide the same items for all user
            listed) or instead a single user to which
            predict the item ranking.
        top: int, optional
            Length of the ranking
        b: float, optional
            System-level Diversity.
            It controls the trade-off for all users between
            accuracy and diversity.
        Returns
        -------
        top-k: ndarray
            `top` items for each user supplied
        """

        n_items, n_users, test, update_dataset = \
            shape_for_mf(self._item_catalog, x)

        try:
            pred = self._model.predict(test).reshape(n_users, n_items)
        except ValueError:
            raise ValueError("You may want to provide for each user "
                             "the item catalog as transaction matrix.")

        # Rev-indexes and other data structures are updated
        # if new items and new users are provided.
        if update_dataset:
            self._init_dataset(x)

        users = index(np.array([x[i, 0] for i in sorted(
            np.unique(x[:, 0], return_index=True)[1])]), self._user_index)
        items = index(np.array([x[i, 1] for i in sorted(
            np.unique(x[:, 1], return_index=True)[1])]), self._item_index)

        re_ranking = self._mmr(pred, users, items, top, b)

        return re_ranking

    def _mmr(self, predictions, users, items, top, b):
        rank = np.argsort(-predictions, 1)
        predictions = np.sort(predictions)[:, ::-1].copy()
        pred = gpu(torch.from_numpy(predictions),
                   self._use_cuda).float()
        re_index(items, rank)

        y = self._model.y

        for k in range(1, top):
            values = self._mmr_objective(b, k, pred, rank, y)
            # TODO it may not work with GPUs
            # TODO if GPU enabled, arg_max_per_user should go to gpu as well
            arg_max_per_user = np.argsort(values, 1)[:, -1].copy()
            _swap_k(arg_max_per_user, k, rank)
            _tensor_swap_k(arg_max_per_user, k, pred, multi=False)

        return index(rank[:, :top], self._rev_item_index)

    def _mmr_objective(self, b, k, pred, rank, y):
        corr = self._correlation(y, k, rank)
        max_corr_per_user = np.sort(corr, 1)[:, -1, :].copy()
        max_corr = gpu(torch.from_numpy(max_corr_per_user), self._use_cuda)
        values = torch.mul(pred[:, k:], b) - torch.mul(max_corr, 1 - b)
        values = values.cpu().numpy()
        return values

    def _correlation(self, y, k, rank):
        i_idx = Variable(gpu(torch.from_numpy(rank), self._use_cuda))

        i_ranked = (y(i_idx[:, :k])).unsqueeze(2)
        i_unranked = y(i_idx[:, k:]).unsqueeze(1)

        n_users = rank.shape[0]
        n_items = rank.shape[1]

        e_ranked = i_ranked.expand(n_users,
                                   k,
                                   n_items - k,
                                   self._n_factors)
        e_unranked = i_unranked.expand(n_users,
                                       k,
                                       n_items - k,
                                       self._n_factors)
        corr = (e_ranked * e_unranked).sum(3)

        return corr.cpu().data.numpy()
