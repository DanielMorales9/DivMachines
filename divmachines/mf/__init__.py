# coding=utf-8
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from divmachines import Classifier
from divmachines.helper import _prepare_for_prediction
from divmachines.logging import Logger
from divmachines.mf.models import PairwiseMatrixFactorizationModel, MatrixFactorizationModel, PointwiseModelMF, \
    PairwiseModelMF
from divmachines.torch_utils import set_seed, gpu, shuffle, minibatch, cpu
from torch.utils.data import TensorDataset, DataLoader


class MatrixFactorizationDataset(TensorDataset):
    """Matrix Factorization Dataset
    Parameters
    ----------
    interactions: ndarray
        triple samples user, item, rating
    use_cuda: boolean, optional
        flag for using cuda
    """
    def __init__(self, interactions, use_cuda=False):
        self._n_users = len(np.unique(interactions[:, 0]))
        self._n_items = len(np.unique(interactions[:, 1]))
        target = gpu(torch.from_numpy(interactions[:, -1].astype(np.float32)), use_cuda)
        data = gpu(torch.from_numpy(interactions[:, :-1]), use_cuda)

        super(MatrixFactorizationDataset, self).__init__(data, target)

    def n_items(self):
        return self._n_items

    def n_users(self):
        return self._n_users


class MF(Classifier):
    """
    Base class for all MF Classifier
    """
    def __init__(self):
        super(MF, self).__init__()

    def _initialize(self, interactions):

        self._init_model()

        self._init_optimization_function()

        self._initialized = True


class Pointwise(MF):
    """
    Pointwise Classifier. Uses a classic
    matrix factorization approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.
    The model is trained through a set of observations of user-item
    pairs.

    Parameters
    ----------
    n_factors: int, optional
        Number of factors to use in user and item latent factors
    model: :class: div.machines.models, optional
        A matrix Factorization model
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
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
        Run the model on a GPU.
    logger: :class:`divmachines.logging`, optional
        A logger instance for logging during the training process
    n_workers: int, optional
        Number of workers for data loading
    """

    def __init__(self,
                 n_factors=10,
                 model=None,
                 sparse=True,
                 n_iter=10,
                 loss=None,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer_func=None,
                 batch_size=None,
                 random_state=None,
                 use_cuda=False,
                 logger=None,
                 n_workers=1):

        super(Pointwise, self).__init__()
        self._n_factors = n_factors
        self._model = model
        self._n_iter = n_iter
        self._sparse = sparse
        self._batch_size = batch_size
        self._random_state = random_state or np.random.RandomState()
        self._use_cuda = use_cuda
        self._l2 = l2
        self._n_workers = n_workers
        self._learning_rate = learning_rate
        self._optimizer_func = optimizer_func
        self._loss_func = loss or torch.nn.MSELoss()
        self._logger = logger or Logger()
        self._dataset = None
        self._optimizer = None
        self._initialized = False

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    def _init_optimization_function(self):
        if self._optimizer_func is None:
            self._optimizer = optim.SGD(self._model.parameters(),
                                        weight_decay=self._l2,
                                        lr=self._learning_rate)
        else:
            self._optimizer = self._optimizer_func(self._model.parameters(),
                                                   weight_decay=self._l2,
                                                   lr=self._learning_rate)

    def _init_model(self):
        if self._model is not None:
            if isinstance(self._model, PointwiseModelMF):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of PointwiseModel")

        else:
            self._model = gpu(MatrixFactorizationModel(self._n_users,
                                                       self._n_items,
                                                       self._n_factors,
                                                       self._sparse),
                              self._use_cuda)

    def fit(self, interactions):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        interactions: ndarray
            user, item, rating triples
        """

        if not self._initialized:
            self._initialize(interactions)

        self._dataset = MatrixFactorizationDataset(interactions,
                                                   self._use_cuda)

        self._n_users = self._dataset.n_users()
        self._n_items = self._dataset.n_items()
        self._check_input(interactions[:, 0], interactions[:, 1])

        loader = DataLoader(self._dataset,
                            shuffle=True,
                            batch_size=self._batch_size,
                            num_workers=self._n_workers)

        for epoch in range(self._n_iter):

            for (mini_batch_num,
                 (batch_data, batch_rating)) in enumerate(loader):

                user_var = Variable(batch_data[:, 0])
                item_var = Variable(batch_data[:, 1])
                rating_var = Variable(batch_rating)

                # forward step
                predictions = self._model(user_var, item_var)

                # Zeroing Embeddings' gradients
                self._optimizer.zero_grad()

                # Compute Loss
                loss = self._loss_func(predictions, rating_var)

                self._logger.log(loss, epoch=epoch, batch=mini_batch_num)

                # backward step
                loss.backward()

                # optimization step
                self._optimizer.step()

    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.
        Parameters
        ----------
        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        Returns
        -------
        predictions: np.array
            Predicted scores for all items in item_ids.
        """

        self._check_input(user_ids, item_ids, allow_items_none=True)
        self._model.train(False)

        user_ids, item_ids = _prepare_for_prediction(user_ids, item_ids,
                                                     self._n_items,
                                                     self._use_cuda)

        out = self._model(user_ids, item_ids)

        return cpu(out.data).numpy().flatten()


class Pairwise(MF):
    """
    A pairwise matrix factorization model.
    Uses a classic matrix factorization approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.
    The model is trained through a set of observations of user-item
    pairs.

    Based on:
            Jahrer, M. & Töscher, A.. (2012). Collaborative Filtering
     Ensemble for Ranking. Proceedings of KDD Cup 2011, in PMLR 18:153-167

    Parameters
    ----------
    n_factors: int, optional
        Number of factors to use in user and item latent factors
    model: :class: div.machines.models, optional
        A matrix Factorization model
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    loss: function, optional
        an instance of a Pytorch optimizer or a custom loss.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Mini batch size.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    use_cuda: boolean, optional
        Run the model on a GPU.
    logger: :class:`divmachines.logging`, optional
        A logger instance for logging during the training process
    """

    def __init__(self,
                 n_factors=10,
                 model=None,
                 sparse=True,
                 n_iter=10,
                 loss=None,
                 l2=0.0,
                 learning_rate=1e-2,
                 batch_size=None,
                 random_state=None,
                 use_cuda=False,
                 logger=None):

        super(Pairwise, self).__init__()
        self._n_factors = n_factors
        self._model = model
        self._n_iter = n_iter
        self._sparse = sparse
        self._batch_size = batch_size
        self._random_state = random_state or np.random.RandomState()
        self._use_cuda = use_cuda
        self._l2 = l2
        self._learning_rate = learning_rate
        self._optimizer_func = None
        self._loss_func = loss or torch.nn.MSELoss()
        self._logger = logger or Logger()

        self._optimizer = None
        self._initialized = False

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    def _init_model(self):
        if self._model is not None:
            if isinstance(self._model, PairwiseModelMF):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of PairwiseModel")

        else:
            self._model = gpu(PairwiseMatrixFactorizationModel(
                                                            self._n_users,
                                                            self._n_items,
                                                            self._n_factors,
                                                            self._sparse),
                                   self._use_cuda)

    def _init_optimization_function(self):
        if self._optimizer_func is None:
            self._optimizer = optim.SGD(self._model.parameters(),
                                        weight_decay=self._l2,
                                        lr=self._learning_rate)
        else:
            self._optimizer = self._optimizer_func(self._model.parameters(),
                                                   weight_decay=self._l2,
                                                   lr=self._learning_rate)

    def fit(self, interactions):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        interactions: ndarray
            user, item, rating triples
        """
        user_ids = interactions[:, 0]
        item_ids = interactions[:, 1]

        if not self._initialized:
            self._initialize(interactions)

        self._check_input(user_ids, item_ids)

        # compute s for all j
        pop_dict = self._items_popularity(item_ids)

        # compute |I_u^+| for each user u
        sizes_dict = self._positive_item_set_size(user_ids)

        # compute I \ I_u^+ for each user u
        neg_dict = self._negative_item_set(user_ids, item_ids, pop_dict)

        # start iterations
        for epoch in range(self._n_iter):

            new_interactions = self._draw_negative_item_sets(interactions,
                                                             neg_dict,
                                                             sizes_dict)
            ratings_ids = new_interactions[:, 2]
            ratings_ids = ratings_ids.astype(np.float32)
            users, pos, neg, ratings = shuffle(
                                            new_interactions[:, 0],
                                            new_interactions[:, 1],
                                            new_interactions[:, 3],
                                            ratings_ids,
                                            random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users), self._use_cuda)

            pos_item_ids_tensor = gpu(torch.from_numpy(pos), self._use_cuda)
            neg_item_ids_tensor = gpu(torch.from_numpy(neg), self._use_cuda)

            rating_ids_tensor = gpu(torch.from_numpy(ratings),
                                    self._use_cuda)

            for (mini_batch_num,
                    (batch_user,
                     batch_pos,
                     batch_neg,
                     batch_rating)) in enumerate(
                                    minibatch(user_ids_tensor,
                                              pos_item_ids_tensor,
                                              neg_item_ids_tensor,
                                              rating_ids_tensor,
                                              batch_size=self._batch_size)):

                user_var = Variable(batch_user)
                pos_var = Variable(batch_pos)
                neg_var = Variable(batch_neg)
                rating_var = Variable(batch_rating)

                # forward step
                predictions = self._model(user_var, pos_var, neg_var)

                # Zeroing Embeddings' gradients
                self._optimizer.zero_grad()

                # Compute Loss
                loss = self._loss_func(predictions, rating_var)

                self._logger.log(loss, epoch=epoch, batch=mini_batch_num)

                # backward step
                loss.backward()

                # optimization step
                self._optimizer.step()

    @staticmethod
    def _draw_negative_item_sets(interactions, neg_dict, sizes_dict):
        interactions.view('i8,i8,i8').sort(order=['f0', 'f1'], axis=0)
        pred = None
        new_interactions = np.zeros(shape=(interactions.shape[0],
                                           interactions.shape[1]+1), dtype=np.int64)
        new_interactions[:, 0:interactions.shape[1]] = interactions
        for c, u in enumerate(interactions[:, 0]):
            if pred:
                if u != pred:
                    items = np.random.shuffle(
                                np.random.choice([a for a, _ in neg_dict[u]],
                                     sizes_dict[u],
                                     p=[b for _, b in neg_dict]))\
                        .reshape(-1, 1)
                    new_interactions[c:len(items), -1] = items
                    pred = u
        return new_interactions

    @staticmethod
    def _negative_item_set(user_ids, item_ids, pop_dict):
        item_catalog = np.array(range(len(np.unique(item_ids))))
        dic = {}
        for u, i in zip(user_ids, item_ids):
            if u in dic:
                dic[u].append(i)
            else:
                dic[u] = [i]
        for u in user_ids:
            dic[u] = np.setdiff1d(item_catalog, np.array(dic[u]))
            dic[u] = np.array([(a, pop_dict[a]) for a in dic[u]])
        return dic

    @staticmethod
    def _positive_item_set_size(user_ids):
        unique, counts = np.unique(user_ids, return_counts=True)
        return dict(zip(unique, counts))

    @staticmethod
    def _items_popularity(item_ids):
        unique, counts = np.unique(item_ids, return_counts=True)
        most_pop = max(counts)
        pop = counts / float(most_pop)
        return dict(zip(unique, pop))

    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.
        Parameters
        ----------
        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        Returns
        -------
        predictions: np.array
            Predicted scores for all items in item_ids.
        """

        self._check_input(user_ids, item_ids, allow_items_none=True)
        self._model.train(False)

        user_ids, item_ids = _prepare_for_prediction(user_ids, item_ids,
                                                     self._n_items,
                                                     self._use_cuda)

        out = self._model(user_ids, item_ids, None)

        return cpu(out.data).numpy().flatten()

