import torch
import numpy as np
import pandas as pd
from torch.autograd.variable import Variable
from divmachines.classifiers import Classifier
from divmachines.classifiers.mf import MF
from divmachines.utility.helper import shape_for_mf, \
    _swap_k, index, re_index
from divmachines.utility.torch import gpu
from tqdm import tqdm

ITEMS = 'items'
USERS = 'users'


class MF_LFP(Classifier):
    """
    Latent Factor Portfolio implementation based on the following work

    `Shi, Y., Zhao, X., Wang, J., Larson, M., & Hanjalic, A. (2012, August).
    Adaptive diversification of recommendation results via latent factor portfolio.
    In Proceedings of the 35th international ACM SIGIR conference on Research
    and development in information retrieval (pp. 175-184). ACM.`

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
    pin_memory bool, optional:
        If ``True``, the data loader will copy tensors
        into CUDA pinned memory before returning them.
    verbose: bool, optional:
        If ``True``, it will print information about iterations.
        Default False.
    save_path: string, optional
        Path name to save the model. Default None.
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
                 n_jobs=0,
                 pin_memory=False,
                 verbose=False,
                 save_path=None):
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
        self._save_path = save_path

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
                             n_jobs=self._n_jobs,
                             verbose=self._verbose,
                             pin_memory=self._pin_memory,
                             save_path=self._save_path)
        elif not isinstance(self._model, MF):
            raise ValueError("Model must be an instance of "
                             "divmachines.classifiers.lfp.MF class")

    def _init_dataset(self, x, train=True):
        self._rev_item_index = {}
        self._user_index = {}
        self._item_index = {}
        for k, v in self._model.index.items():
            if k.startswith(ITEMS):
                self._rev_item_index[str(v)] = k[len(ITEMS):]
                self._item_index[k[len(ITEMS):]] = v
            elif k.startswith(USERS):
                self._user_index[k[len(USERS):]] = v

            else:
                raise ValueError("Not possible")
        self._item_catalog = np.array(list(self._rev_item_index.values()))
        if train:
            self._estimate_variance(x)

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
            rank = self._model.predict(test).reshape(n_users, n_items)
        except ValueError:
            raise ValueError("You may want to provide for each user "
                             "the item catalog as transaction matrix.")

        # Rev-indexes and other data structures are updated
        # if new items and new users are provided.
        if update_dataset:
            self._init_dataset(x, train=False)

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
        var = self.torch_variance()

        for k in tqdm(range(1, top),
                      desc="Sequential Re-ranking",
                      leave=False,
                      disable=not self._verbose):
            values = self._compute_delta_f(x, y, k, b, var, rank, users)

            arg_max_per_user = np.argsort(values, 1)[:, -1].copy()
            _swap_k(arg_max_per_user, k, rank)

        return index(rank[:, :top], self._rev_item_index)

    def _compute_delta_f(self, x, y, k, b, var, rank, users):
        # Initialize Variables
        # and other coefficients
        u_idx = Variable(gpu(torch.from_numpy(users), self._use_cuda))
        i_idx = Variable(gpu(torch.from_numpy(rank), self._use_cuda))

        wk = 1/(2**k)
        wm = Variable(gpu(torch.from_numpy(
            np.array([1 / (2 ** m) for m in range(k)],
                     dtype=np.float32)), self._use_cuda)) \
            .unsqueeze(1).expand(k, self._n_factors)

        i_ranked = (y(i_idx[:, :k]) * wm).transpose(0, 1).unsqueeze(0)
        i_unranked = y(i_idx[:, k:]).transpose(0, 1)

        term0 = self._first_term(i_unranked, u_idx, x)

        term1 = self._second_term(b, i_unranked, u_idx, var, wk)

        term2 = self._third_term(b,
                                 k,
                                 u_idx,
                                 var,
                                 i_ranked,
                                 i_unranked,
                                 len(users),
                                 rank.shape[1])

        delta_f = torch.mul(term0 - term1 - term2, wk)\
            .sum(2).transpose(0, 1)

        return delta_f.cpu().data.numpy()

    def _third_term(self,
                    b, k, u_idx, var,
                    i_ranked, i_unranked,
                    n_users, n_items):
        # This block of code computes the third term of the DeltaF
        e_ranked = i_ranked.expand(n_items - k,
                                   k,
                                   n_users,
                                   self._n_factors) \
            .transpose(0, 1)
        e_unranked = i_unranked.unsqueeze(0).expand(k,
                                                    n_items - k,
                                                    n_users,
                                                    self._n_factors)

        coeff1 = torch.mul(var(u_idx), 2. * b)
        term2 = (e_ranked * e_unranked).sum(0) * coeff1
        return term2

    def _second_term(self, b, i_unranked, u_idx, var, wk):
        # This block of code computes the second term of the DeltaF
        term1 = torch.pow(i_unranked, 2) * var(u_idx)
        coeff0 = b * wk
        term1 = torch.mul(term1, coeff0)
        return term1

    def _first_term(self, i_unranked, u_idx, x):
        # This block of code computes the first term of the DeltaF.
        users_batch = x(u_idx)
        term0 = (users_batch * i_unranked)
        return term0

    def torch_variance(self):
        var_v = torch.from_numpy(self._var)
        var = torch.nn.Embedding(var_v.size(0), var_v.size(1))
        var.weight = torch.nn.Parameter(var_v)
        var = gpu(var, self._use_cuda)
        return var

    def _estimate_variance(self, x):
        train = index_dataset(x, self._user_index, self._item_index)
        x = self._model.x
        y = self._model.y

        self._var = np.zeros((self.n_users, self._n_factors),
                             dtype=np.float32)

        for i, (u, g) in tqdm(enumerate(pd.DataFrame(train).groupby(0)),
                              desc="Var. Estimate",
                              leave=False,
                              disable=not self._verbose):
            user_profile = g.values[:, 1].astype(np.int64)
            upl = user_profile.shape[0]
            user_idx = Variable(gpu(torch.from_numpy(np.array([u])),
                                    self._use_cuda))
            item_idx = Variable(gpu(torch.from_numpy(user_profile),
                                    self._use_cuda))
            diff = x(user_idx) - y(item_idx)

            prod = torch.pow(diff, 2).sum(0)

            var = torch.div(prod, upl)
            self._var[i, :] = var.cpu().data.numpy()


def index_dataset(x, idx0, idx1):
    user_profile = x.copy()
    users = index(user_profile[:, 0], idx0)
    items = index(user_profile[:, 1], idx1)
    user_profile[:, 0] = users
    user_profile[:, 1] = items
    return user_profile
# TODO add save API