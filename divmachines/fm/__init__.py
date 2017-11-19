# coding=utf-8
import numpy as np
import torch
from torch import optim
from torch.sparse import FloatTensor as SparseFTensor
from torch.autograd import Variable

from divmachines import Classifier
#from divmachines.helper import _predict_process_ids
from divmachines.logging import Logger
from divmachines.fm.models import PointwiseModelFM, FactorizationMachine
from divmachines.torch_utils import set_seed, gpu, sparse_shuffle, sparse_mini_batch, cpu
from divmachines.fm.utility import vectorize_dic


class ClassifierFM(Classifier):
    """
    Base Classifier for Factorization Machines
    """

    def __init__(self):
        super(ClassifierFM, self).__init__()
        self.data = None
        self._rows = None
        self._cols = None
        self._ix = None
        self._n_features = None
        self._ratings = None
        self._initialized = False

    def _initialize(self, interactions):
        # TODO:
        # 1) What happens if we update the data (new users, new items)?
        # 2) Update function for context features!
        self._data, \
        self._rows, \
        self._cols, \
        self._ix = vectorize_dic({"users": interactions[:, 0],
                                  "items": interactions[:, 1]})

        self._ratings = interactions[:, -1]
        self._n_features = len(np.unique(self._cols))

        self._init_model()

        self._init_optimization_function()

        self._initialized = True


class PointwiseClassifierFM(ClassifierFM):
    """
    Pointwise Classifier. Uses a classic
     factorization approach, with latent vectors used
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
                 logger=None):

        super(PointwiseClassifierFM, self).__init__()
        self._n_factors = n_factors
        self._model = model
        self._n_iter = n_iter
        self._sparse = sparse
        self._batch_size = batch_size
        self._random_state = random_state or np.random.RandomState()
        self._use_cuda = use_cuda
        self._l2 = l2
        self._learning_rate = learning_rate
        self._optimizer_func = optimizer_func
        self._loss_func = loss or torch.nn.MSELoss()
        self._logger = logger or Logger()
        self._optimizer = None

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
            if isinstance(self._model, PointwiseModelFM):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of PointwiseModelFM")

        else:
            self._model = gpu(FactorizationMachine(self._n_features,
                                                   self._n_factors),
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
        for epoch in range(self._n_iter):
            rows, cols, data, ratings = sparse_shuffle(np.copy(self._rows),
                                                       np.copy(self._cols),
                                                       np.copy(self._data),
                                                       np.copy(self._ratings))
            print(epoch)
            for (mini_batch_num,
                 (sparse_batch_tensor,
                  batch_rating)) in enumerate(
                            sparse_mini_batch(rows,
                                  cols,
                                  data,
                                  ratings,
                                  batch_size=self._batch_size,
                                  n_features=self._n_features)):

                observations_var = Variable(gpu(sparse_batch_tensor.to_dense(),
                                                self._use_cuda))
                rating_var = Variable(gpu(batch_rating),
                                      self._use_cuda)

                # forward step
                predictions = self._model(observations_var)

                # Zeroing Embeddings' gradients
                self._optimizer.zero_grad()

                # Compute Loss
                loss = self._loss_func(predictions, rating_var)

                # logging
                self._logger.log(loss, epoch, batch=mini_batch_num)

                # backward step
                loss.backward()
                # optimization step
                self._optimizer.step()

                # def predict(self, user_ids, item_ids=None):
                #     """
                #     Make predictions: given a user id, compute the recommendation
                #     scores for items.
                #     Parameters
                #     ----------
                #     user_ids: int or array
                #        If int, will predict the recommendation scores for this
                #        user for all items in item_ids. If an array, will predict
                #        scores for all (user, item) pairs defined by user_ids and
                #        item_ids.
                #     item_ids: array, optional
                #         Array containing the item ids for which prediction scores
                #         are desired. If not supplied, predictions for all items
                #         will be computed.
                #     Returns
                #     -------
                #     predictions: np.array
                #         Predicted scores for all items in item_ids.
                #     """
                #
                #     self._check_input(user_ids, item_ids, allow_items_none=True)
                #     self._model.train(False)
                #
                #     user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                #                                               self._n_items,
                #                                               self._use_cuda)
                #
                #     out = self._model(user_ids, item_ids)
                #
                #     return cpu(out.data).numpy().flatten()
