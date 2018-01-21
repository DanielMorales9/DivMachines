import torch
import numpy as np
from divmachines.classifiers import Classifier
from divmachines.classifiers import FM
from divmachines.utility.helper import index, \
    _tensor_swap, re_index, _swap_k, _tensor_swap_k
from divmachines.utility.torch import gpu

ITEMS = 'items'
USERS = 'users'


class MMR_FM(Classifier):
    """
    MMR implementation based on Factorization Machine Model

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

    @logger.getter
    def logger(self):
        return self._logger

    @property
    def n_features(self):
        return self._model.n_features

    @n_features.getter
    def n_features(self):
        return self._model.n_features

    @property
    def dataset(self):
        return self._model.dataset

    @dataset.getter
    def dataset(self):
        return self._model.dataset

    def _initialize(self):
        self._init_model()

    def _init_model(self):
        if self._model is None:
            self._model = FM(n_factors=self._n_factors,
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
        elif not isinstance(self._model, FM):
            raise ValueError("Model must be an instance of "
                             "divmachines.classifiers.FM class")

    def _init_dataset(self, x):
        self._user_index = {}
        for k, v in self._model.index.items():
            if k.startswith(USERS):
                try:
                    self._user_index[int(k[len(USERS):])] = v
                except ValueError:
                    raise ValueError("You may want to provide an integer "
                                     "index for the users")

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

        dic = {USERS: 0, ITEMS: 1}

        self._model.fit(x, y, dic=dic, n_users=n_users, n_items=n_items)
        self._init_dataset(x)

    def predict(self, x, top=10, b=0.5):
        """
        Predicts

        Parameters
        ----------
        x: ndarray or int
            An array of feature vectors.
            The User-item-feature matrix must have the same items
            in the same order for all user
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

        n_users = np.unique(x[:, 0]).shape[0]
        n_items = np.unique(x[:, 1]).shape[0]

        # prediction of the relevance of all the item catalog
        # for the users supplied
        predictions = self._model.predict(x).reshape(n_users, n_items)

        items = np.array([x[i, 1] for i in sorted(
            np.unique(x[:, 1], return_index=True)[1])])
        users = index(np.array([x[i, 0] for i in sorted(
            np.unique(x[:, 0], return_index=True)[1])]), self._user_index)

        x = self.dataset.copy()

        re_ranking = self._mmr(x, n_users, n_items, top, b, predictions, items)
        return re_ranking

    def _mmr(self, x, n_users, n_items, top, b,
             predictions, items):
        rank = np.argsort(-predictions, 1)
        # zeroing users cols
        self.zero_users(x)

        x = x.reshape(n_users, n_items, self.n_features)
        x = gpu(_tensor_swap(rank, torch.from_numpy(x)),
                self._use_cuda)
        predictions = np.sort(predictions)[:, ::-1].copy()
        pred = gpu(torch.from_numpy(predictions),
                   self._use_cuda).float()

        re_index(items, rank)

        v = self._model.v.data

        n_items = pred.shape[1]
        n_users = pred.shape[0]

        for k in range(1, top):

            values = self.mmr_objective(b, k, n_items, n_users, pred, v, x)
            # TODO it may not work with GPUs
            # TODO if GPU enabled, arg_max_per_user should go to gpu as well
            arg_max_per_user = np.argsort(values, 1)[:, -1].copy()
            _swap_k(arg_max_per_user, k, rank)
            _tensor_swap_k(arg_max_per_user, k, pred, multi=False)
            _tensor_swap_k(arg_max_per_user, k, x, multi=True)

        return rank[:, :top]

    def mmr_objective(self, b, k, n_items, n_users, pred, v, x):
        corr = self._correlation(v, k, x, n_users, n_items)
        max_corr_per_user = np.sort(corr, 1)[:, -1, :].copy()
        max_corr = gpu(torch.from_numpy(max_corr_per_user), self._use_cuda)
        values = torch.mul(pred[:, k:], b) - torch.mul(max_corr, 1 - b)
        values = values.cpu().numpy()
        return values

    def zero_users(self, x):
        x[:, :self.n_users] = np.zeros((x.shape[0], self.n_users),
                                       dtype=np.float)

    def _correlation(self, v, k, x, n_users, n_items):
        # dim (u, i, n, f) -> (u, i, f)
        prod = (x.unsqueeze(-1).expand(n_users,
                                       n_items,
                                       self.n_features,
                                       self._n_factors) * v).sum(2)

        unranked = prod[:, k:, :]
        ranked = prod[:, :k, :]
        e_corr = unranked.unsqueeze(1).expand(n_users,
                                              k,
                                              n_items-k,
                                              self._n_factors) * \
                 ranked.unsqueeze(2).expand(n_users,
                                            k,
                                            n_items-k,
                                            self._n_factors)
        corr = e_corr.sum(3)

        return corr.cpu().numpy()
