import torch
import numpy as np
from tqdm import tqdm
from torch.autograd.variable import Variable
from divmachines.classifiers import Classifier
from divmachines.classifiers.mf import MF
from divmachines.utility.helper import shape_for_mf, \
    _swap_k, index, re_index
from divmachines.utility.torch import gpu


ITEMS = 'items'
USERS = 'users'


class MF_SeqRank(Classifier):
    """
    Seq Rank for Matrix Factorization Model
    with MF-Correlation

   Parameters
    ----------
    model: classifier, str or optional
        An instance of `divmachines.classifier.lfp.FM`.
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
    device_id: int, optional
        GPU device ID to which the tensors are sent.
        If set use_cuda must be True.
        By Default uses all GPU available.
    logger: :class:`divmachines.logging`, optional
        A logger instance for logging during the training process
    n_jobs: int, optional
        Number of jobs for data loading.
        Default is 0, it means that the data loader runs in the main process.
    early_stopping: bool, optional
        Performs a dump every time to enable early stopping.
        Default False.
        n_iter_no_change : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'
    n_iter_no_change : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'
    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        convergence is considered to be reached and training stops.
    stopping: bool, optional
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
                 device_id=None,
                 logger=None,
                 n_jobs=0,
                 pin_memory=False,
                 verbose=False,
                 early_stopping=False,
                 n_iter_no_change=10,
                 tol=1e-4,
                 stopping=False):
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
        self._pin_memory = pin_memory
        self._verbose = verbose
        self._early_stopping = early_stopping
        self._tol = tol
        self._n_iter_no_change = n_iter_no_change
        self._stopping = stopping
        if device_id is not None and not self._use_cuda:
            raise ValueError("use_cuda flag must be true")
        self._device_id = device_id

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
                             device_id=self._device_id,
                             logger=self._logger,
                             n_jobs=self._n_jobs,
                             pin_memory=self._pin_memory,
                             verbose=self._verbose,
                             early_stopping=self._early_stopping,
                             n_iter_no_change=self._n_iter_no_change,
                             tol=self._tol,
                             stopping=self._stopping)
        elif isinstance(self._model, str):
            self._model = MF(model=self._model,
                             n_factors=self._n_factors,
                             sparse=self._sparse,
                             n_iter=self._n_iter,
                             loss=self._loss,
                             l2=self._l2,
                             learning_rate=self._learning_rate,
                             optimizer_func=self._optimizer_func,
                             batch_size=self._batch_size,
                             random_state=self._random_state,
                             use_cuda=self._use_cuda,
                             device_id=self._device_id,
                             logger=self._logger,
                             n_jobs=self._n_jobs,
                             pin_memory=self._pin_memory,
                             verbose=self._verbose,
                             early_stopping=self._early_stopping,
                             n_iter_no_change=self._n_iter_no_change,
                             tol=self._tol,
                             stopping=self._stopping)
        elif not isinstance(self._model, MF):
            raise ValueError("Model must be an instance of "
                             "divmachines.classifiers.lfp.MF class")

    def _init_dataset(self, x):
        self._rev_item_index = {}
        self._user_index = {}
        self._item_index = {}
        for k, v in self._model.index.items():
            if k.startswith(ITEMS):
                self._rev_item_index[str(v)] = k[len(ITEMS):]
                self._item_index[k[len(ITEMS):]] = v
            elif k.startswith(USERS):
                self._user_index[k[len(USERS):]] = v

    def fit(self, x, y, dic=None, n_users=None, n_items=None):
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
        dic: dict, optional
            dic indicates the columns to vectorize
            if training samples are in raw format.
        n_users: int, optional
            Total number of users. The model will have `n_users` rows.
            Default is None, `n_users` will be inferred from `x`.
        n_items: int, optional
            Total number of items. The model will have `n_items` columns.
            Default is None, `n_items` will be inferred from `x`.
        """
        self._initialize()

        if x.shape[1] != 2:
            raise ValueError("x must have two columns: users and items cols")

        self._model.fit(x, y,
                        dic=dic,
                        n_users=n_users,
                        n_items=n_items)
        self._init_dataset(x)
        if self._early_stopping:
            self._prepare()

    def _prepare(self):
        dump = self._model.dump
        self.dump = dump

    def save(self, path):
        torch.save(self.dump, path)

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

        load = isinstance(self._model, str)
        if load:
            self._initialize()

        n_users = np.unique(x[:, 0]).shape[0]
        n_items = np.unique(x[:, 1]).shape[0]

        try:
            rank = self._model.predict(x).reshape(n_users, n_items)
        except ValueError:
            raise ValueError("You may want to provide for each user "
                             "the item catalog as transaction matrix.")

        # Rev-indexes and other data structures are updated
        # if new items and new users are provided.
        self._init_dataset(x)

        users = index(np.array([x[i, 0] for i in sorted(
            np.unique(x[:, 0], return_index=True)[1])]), self._user_index)
        items = index(np.array([x[i, 1] for i in sorted(
            np.unique(x[:, 1], return_index=True)[1])]), self._item_index)

        re_ranking = self._sequential_re_ranking(rank, users, items, top, b)

        return re_ranking

    def _sequential_re_ranking(self, rank, users, items, top, b):
        rank = np.argsort(-rank, 1)

        re_index(items, rank)

        x = self._model.x
        y = self._model.y

        for k in tqdm(range(1, top),
                      desc="Sequential Re-ranking",
                      leave=False,
                      disable=not self._verbose):
            values = self._compute_delta_f(x, y, k, b, rank, users)

            arg_max_per_user = np.argsort(values, 1)[:, -1].copy()
            _swap_k(arg_max_per_user, k, rank)

        return index(rank[:, :top], self._rev_item_index)

    def _compute_delta_f(self, x, y, k, b, rank, users):
        # Initialize Variables
        # and other coefficients
        u_idx = Variable(gpu(torch.from_numpy(users),
                             self._use_cuda,
                             self._device_id))
        i_idx = Variable(gpu(torch.from_numpy(rank),
                             self._use_cuda,
                             self._device_id))

        n_users = rank.shape[0]
        n_items = rank.shape[1]

        wk = 1/(2**k)
        wm = Variable(gpu(torch.from_numpy(
            np.array([1 / (2 ** m) for m in range(k)],
                     dtype=np.float32)), self._use_cuda,
            self._device_id)) \
            .unsqueeze(1).expand(k, self._n_factors)

        i_ranked = (y(i_idx[:, :k]) * wm).unsqueeze(2)
        i_unranked = y(i_idx[:, k:]).transpose(0, 1)

        users_batch = x(u_idx)
        term0 = (users_batch * i_unranked).sum(2).transpose(0, 1)

        # This block of code computes the third term of the DeltaF
        e_ranked = i_ranked.expand(n_users,
                                   k,
                                   n_items - k,
                                   self._n_factors)
        e_unranked = i_unranked \
            .transpose(0, 1) \
            .unsqueeze(1).expand(n_users,
                                 k,
                                 n_items - k,
                                 self._n_factors)

        term2 = torch.mul((e_ranked * e_unranked).sum(3).sum(1), 2*b)

        delta_f = torch.mul(term0 - term2, wk)
        return delta_f.cpu().data.numpy()


def index_dataset(x, idx0, idx1):
    user_profile = x.copy()
    users = index(user_profile[:, 0], idx0)
    items = index(user_profile[:, 1], idx1)
    user_profile[:, 0] = users
    user_profile[:, 1] = items
    return user_profile

