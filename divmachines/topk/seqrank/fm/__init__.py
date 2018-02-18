import torch
import numpy as np
from tqdm import tqdm
from divmachines.classifiers import Classifier
from torch.utils.data import DataLoader
from divmachines.topk import Rank
from divmachines.classifiers import FM
from divmachines.utility.helper import index, \
    _swap_k, _tensor_swap_k, _tensor_swap, re_index
from divmachines.utility.torch import gpu

ITEMS = 'items'
USERS = 'users'


class FM_SeqRank(Classifier):
    """
    Seq Rank with Factorization Machines model

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
    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        convergence is considered to be reached and training stops.
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
                 early_stopping=False,
                 n_iter_no_change=10,
                 tol=1e-4):
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
        self._n_iter_no_change = n_iter_no_change
        self._tol = tol

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
                             n_jobs=self._n_jobs,
                             pin_memory=self._pin_memory,
                             verbose=self._verbose,
                             early_stopping=self._early_stopping,
                             n_iter_no_change=self._n_iter_no_change,
                             tol=self._tol)
        elif isinstance(self._model, str):
            self._model = FM(model=self._model,
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
                             logger=self._logger,
                             n_jobs=self._n_jobs,
                             pin_memory=self._pin_memory,
                             verbose=self._verbose,
                             early_stopping=self._early_stopping,
                             n_iter_no_change=self._n_iter_no_change,
                             tol=self._tol)
        elif not isinstance(self._model, FM):
            raise ValueError("Model must be an instance of "
                             "divmachines.classifiers.FM class")

    def _init_dataset(self, x):
        pass

    def fit(self, x, y,
            dic=None,
            n_users=None,
            n_items=None,
            lengths=None):
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
        dic: dict, optional
            dic indicates the columns to vectorize
            if training samples are in raw format.
        n_users: int, optional
            Total number of users. The model will have `n_users` rows.
            Default is None, `n_users` will be inferred from `x`.
        n_items: int, optional
            Total number of items. The model will have `n_items` columns.
            Default is None, `n_items` will be inferred from `x`.
        lengths: dic, optional
            Dictionary of lengths of each feature in dic except for
            users and items.
        """
        self._initialize()

        self._model.fit(x,
                        y,
                        dic=dic,
                        n_users=n_users,
                        n_items=n_items,
                        lengths=lengths)

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
        if isinstance(self._model, str):
            self._initialize()

        # prediction of the relevance of all the item catalog
        # for the users supplied
        rank = self._model.predict(x).reshape(n_users, n_items)

        items = np.array([x[i, 1] for i in sorted(
            np.unique(x[:, 1], return_index=True)[1])])

        if self._sparse:
            x = self._model._dataset
        else:
            x = self.dataset.copy()

        re_ranking = self._sequential_re_ranking(x, n_users,
                                                 n_items, top,
                                                 b, rank, items)
        return re_ranking

    def _sequential_re_ranking(self, x, n_users, n_items, top, b,
                               predictions, items):
        rank = np.argsort(-predictions, 1)
        # zeroing users cols
        rank = rank.astype(dtype=np.object)
        if not self._sparse:
            self.zero_users(x)
            x = x.reshape(n_users, n_items, self.n_features)
            x = gpu(_tensor_swap(rank, torch.from_numpy(x)),
                    self._use_cuda)

        predictions = np.sort(predictions)[:, ::-1].copy()
        pred = gpu(torch.from_numpy(predictions),
                   self._use_cuda).float()

        v = self._model.v.data

        for k in tqdm(range(1, top),
                      desc="Sequential Re-ranking",
                      leave=False,
                      disable=not self._verbose):
            if not self._sparse:
                values = self._compute_delta_f(v, k, b, pred, x)
            else:
                values = self._sparse_compute_delta_f(v, k, b, pred, x, rank)
            arg_max_per_user = np.argsort(values, 1)[:, -1].copy()
            _swap_k(arg_max_per_user, k, rank)
            _tensor_swap_k(arg_max_per_user, k, pred, multi=False)
            if not self._sparse:
                _tensor_swap_k(arg_max_per_user, k, x, multi=True)

        re_index(items, rank)
        return rank[:, :top]

    def zero_users(self, x):
        x[:, :self.n_users] = np.zeros((x.shape[0], self.n_users),
                                       dtype=np.float)

    def _compute_delta_f(self, v, k, b, pred, x):

        term0 = pred[:, k:]
        n_items = pred.shape[1]
        n_users = pred.shape[0]
        delta = np.zeros((n_users, n_items - k),
                         dtype=np.float32)

        wk = 1 / (2 ** k)
        for u in tqdm(range(n_users),
                      desc="MMR Re-ranking",
                      leave=False,
                      disable=not self._verbose):
            prod = (x[u, :, :].squeeze()
                    .unsqueeze(-1).expand(n_items,
                                          self.n_features,
                                          self._n_factors) * v) \
                .sum(1)
            wm = gpu(torch.from_numpy(
                np.array([1 / (2 ** m) for m in range(k)],
                         dtype=np.float32)), self._use_cuda) \
                .unsqueeze(-1) \
                .expand(k, self._n_factors)
            unranked = prod[k:, :]
            ranked = (prod[:k, :] * wm)
            t2 = unranked.unsqueeze(0) \
                     .expand(k, n_items - k, self._n_factors) * \
                 ranked.unsqueeze(1) \
                     .expand(k, n_items - k, self._n_factors)
            term2 = t2.sum(2).sum(0)
            term2 = torch.mul(term2, 2)
            delta[u, :] = torch.mul(term0[u, :] - torch.mul(term2, 2*b), wk) \
                .cpu().numpy()
        return delta

    def _sparse_compute_delta_f(self, v, k, b, pred, x, rank):

        term0 = pred[:, k:]
        n_items = pred.shape[1]
        n_users = pred.shape[0]
        delta = np.zeros((n_users, n_items - k),
                         dtype=np.float32)
        ranking = None
        wk = 1 / (2 ** k)
        for u in tqdm(range(n_users),
                      desc="MMR Re-ranking",
                      leave=False,
                      disable=not self._verbose):
            prod_numpy = np.zeros((n_items, self._n_factors),
                                  dtype=np.float32)
            prod = gpu(torch.from_numpy(prod_numpy), self._use_cuda)

            if ranking:
                ranking(u)
            else:
                ranking = Rank(x, rank, u, n_items, self.n_users)

            data_loader = DataLoader(ranking,
                                     pin_memory=self._pin_memory,
                                     batch_size=self._batch_size,
                                     num_workers=self._n_jobs)

            for batch, i in tqdm(data_loader,
                                 disable=not self._verbose,
                                 desc="Rank", leave=False):
                batch = gpu(batch, self._use_cuda)
                i = gpu(i, self._use_cuda)
                batch_size = list(batch.shape)[0]
                prod[i, :] = (batch.squeeze()
                              .unsqueeze(-1)
                              .expand(batch_size,
                                      self.n_features,
                                      self._n_factors) * v).sum(1)

            wm = gpu(torch.from_numpy(
                np.array([1 / (2 ** m) for m in range(k)],
                         dtype=np.float32)), self._use_cuda) \
                .unsqueeze(-1) \
                .expand(k, self._n_factors)
            unranked = prod[k:, :]
            ranked = (prod[:k, :] * wm)
            t2 = unranked.unsqueeze(0) \
                     .expand(k, n_items - k, self._n_factors) * \
                 ranked.unsqueeze(1) \
                     .expand(k, n_items - k, self._n_factors)
            term2 = t2.sum(2).sum(0)
            term2 = torch.mul(term2, 2)
            delta[u, :] = torch.mul(term0[u, :] - torch.mul(term2, 2 * b), wk) \
                .cpu().numpy()
        return delta

        # dim (u, i, n, f) -> (u, i, f)
        # prod = (x.unsqueeze(-1).expand(n_users,
        #                                n_items,
        #                                self.n_features,
        #                                self._n_factors) * v).sum(2)
        #
        # wm = gpu(torch.from_numpy(
        #     np.array([1 / (2 ** m) for m in range(k)],
        #              dtype=np.float32)), self._use_cuda) \
        #     .unsqueeze(0).unsqueeze(2) \
        #     .expand(n_users, k, self._n_factors)
        #
        # unranked = prod[:, k:, :]
        # ranked = prod[:, :k, :] * wm
        # t2 = unranked.unsqueeze(1).expand(n_users,
        #                                   k,
        #                                   n_items-k,
        #                                   self._n_factors) * \
        #      ranked.unsqueeze(2).expand(n_users,
        #                                 k,
        #                                 n_items-k,
        #                                 self._n_factors)
        # term2 = t2.sum(3).sum(1)
        # term2 = torch.mul(term2, 2)
        # delta = torch.mul(term0 - torch.mul(term2, 2*b), wk)
        # return delta.cpu().numpy()
