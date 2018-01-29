import torch
import numpy as np
from torch.autograd.variable import Variable
from torch.nn import Embedding, Parameter
from divmachines.classifiers import Classifier
from divmachines.classifiers import FM
from divmachines.utility.helper import index, \
    _swap_k, _tensor_swap_k, _tensor_swap, re_index
from divmachines.utility.torch import gpu

ITEMS = 'items'
USERS = 'users'


class FM_LFP(Classifier):
    """
    Latent Factor Portfolio implementation based on Factorization Machine Model

    My personal research founding.

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
                self._user_index[k[len(USERS):]] = v

        x = self.dataset.copy()
        nz = list(filter(lambda k: k[1] < self.n_users,
                         [(r, c) for r, c in zip(*np.nonzero(x))]))
        self.zero_users(x)
        d = {}
        for r, c in nz:
            d.setdefault(c, []).append(r)

        # getting parameter from the model
        v = self._model.v.cpu().data.numpy()

        # (t, n)
        x = gpu(torch.from_numpy(x), self._use_cuda)
        X = gpu(Embedding(x.size(0), x.size(1)), self._use_cuda)
        X.weight = Parameter(x)

        # (n, f)
        V = Variable(gpu(torch.from_numpy(v), self._use_cuda))

        var = np.zeros((self.n_users, self._n_factors),
                       dtype=np.float)
        var = gpu(torch.from_numpy(var), self._use_cuda)

        for k, val in d.items():
            idx = Variable(gpu(torch.from_numpy(np.array(val)),
                               self._use_cuda))
            i = X(idx).size(0)

            prod = (X(idx)
                    .unsqueeze(2)
                    .expand(i,
                            self.n_features,
                            self._n_factors) *
                    V.unsqueeze(0)
                    .expand(i,
                            self.n_features,
                            self._n_factors)).sum(1)
            diff = V[k, :] - prod
            a = torch.pow(diff, 2).sum(0)
            var[k, :] = torch.div(a, len(val)).data
        self._var = var.cpu().numpy()

    def fit(self,
            x,
            y,
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
        if not self._initialized:
            self._initialize()

        self._model.fit(x,
                        y,
                        dic=dic,
                        n_users=n_users,
                        n_items=n_items,
                        lengths=lengths)
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
        rank = self._model.predict(x).reshape(n_users, n_items)

        items = np.array([x[i, 1] for i in sorted(
            np.unique(x[:, 1], return_index=True)[1])])
        users = index(np.array([x[i, 0] for i in sorted(
            np.unique(x[:, 0], return_index=True)[1])]), self._user_index)

        x = self.dataset.copy()

        re_ranking = self._sequential_re_ranking(x,
                                                 n_users,
                                                 n_items,
                                                 top,
                                                 b,
                                                 rank,
                                                 items,
                                                 users)
        return re_ranking

    def _sequential_re_ranking(self, x, n_users, n_items, top, b,
                               predictions, items, users):
        rank = np.argsort(-predictions, 1)
        # zeroing users cols
        self.zero_users(x)
        rank = rank.astype(dtype=np.object)
        x = x.reshape(n_users, n_items, self.n_features)
        x = gpu(_tensor_swap(rank, torch.from_numpy(x)),
                self._use_cuda)
        predictions = np.sort(predictions)[:, ::-1].copy()
        pred = gpu(torch.from_numpy(predictions),
                   self._use_cuda).float()

        re_index(items, rank)

        v = self._model.v.data

        u_idx = Variable(gpu(torch.from_numpy(users),
                             self._use_cuda))
        var = self.torch_variance()
        variance = var(u_idx).data

        for k in range(1, top):

            values = self._compute_delta_f(v, k, b, variance, pred, x)
            arg_max_per_user = np.argsort(values, 1)[:, -1].copy()
            _swap_k(arg_max_per_user, k, rank)
            _tensor_swap_k(arg_max_per_user, k, pred, multi=False)
            _tensor_swap_k(arg_max_per_user, k, x, multi=True)

        return rank[:, :top]

    def zero_users(self, x):
        x[:, :self.n_users] = np.zeros((x.shape[0], self.n_users),
                                       dtype=np.float)

    def _compute_delta_f(self, v, k, b, var, pred, x):

        term0 = pred[:, k:]
        n_items = pred.shape[1]
        n_users = pred.shape[0]
        delta = np.zeros((n_users, n_items-k),
                         dtype=np.float32)
        wk = 1 / (2 ** k)
        for u in range(n_users):
            prod = (x[u, :, :].squeeze()
                    .unsqueeze(-1).expand(n_items,
                                          self.n_features,
                                          self._n_factors) * v).sum(1)
            t1 = torch.pow(prod, 2)
            term1 = torch.mul((t1 * var[u, :]).sum(1)[k:], wk)
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
            term2 = torch.mul((t2 * var[u, :]).sum(2).sum(0), 2)
            delta[u, :] = torch.mul(term0[u, :] - torch.mul(term1 + term2, b), wk).cpu().numpy()

        return delta

        # dim (u, i, n, f) -> (u, i, f)
        # prod = (x.unsqueeze(-1).expand(n_users,
        #                                n_items,
        #                                self.n_features,
        #                                self._n_factors) * v).sum(2)
        #
        # t1 = torch.pow(prod, 2)
        #
        # term1 = torch.mul((t1.transpose(0, 1) * var) \
        #             .transpose(0, 1).sum(2)[:, k:], wk)
        #
        # wm = gpu(torch.from_numpy(
        #     np.array([1 / (2 ** m) for m in range(k)],
        #              dtype=np.float32)), self.use_cuda) \
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
        # term2 = torch.mul((t2.transpose(0, 2) * var)
        #                   .sum(3).sum(1)
        #                   .transpose(0, 1), 2)
        #
        # delta = torch.mul(term0 - torch.mul(term1 + term2, b), wk)

    def torch_variance(self):
        var = gpu(torch.from_numpy(self._var).float(), self._use_cuda)
        e = Embedding(var.size(0), var.size(1))
        e.weight = Parameter(var)
        return e
