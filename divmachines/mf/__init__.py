import torch
from torch import optim
import numpy as np
from torch.autograd import Variable
from divmachines.torch_utils import shuffle, minibatch, gpu, cpu, set_seed
from divmachines.logging import Logger
from divmachines.mf.helper import _predict_process_ids
from divmachines.mf.models import MatrixFactorizationModel, BiasMatrixFactorizationModel
from divmachines import Classifier


class MatrixFactorization(Classifier):
    """
    A matrix factorization model. Uses a classic
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
        
        super(MatrixFactorization, self).__init__()
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

        self._n_users = None
        self._n_items = None
        self.x = None
        self.y = None
        self.user_biases = None
        self.item_biases = None
        self._optimizer = None
        self._has_representation = False
        self._initialized = False

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    def _initialize(self, interactions):

        self._n_users = len(np.unique(np.squeeze(interactions[:, 0])))
        self._n_items = len(np.unique(np.squeeze(interactions[:, 1])))

        if self._model is not None:
            if isinstance(self._model, MatrixFactorizationModel):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of MatrixFactorizationModel")

        else:
            self._model = gpu(BiasMatrixFactorizationModel(self._n_users,
                                                            self._n_items,
                                                            self._n_factors,
                                                            self._sparse),
                               self._use_cuda)

        if self._optimizer_func is None:
            self._optimizer = optim.SGD(self._model.parameters(),
                                        weight_decay=self._l2,
                                        lr=self._learning_rate)
        else:
            self._optimizer = self._optimizer_func(self._model.parameters(),
                                                   weight_decay=self._l2,
                                                   lr=self._learning_rate)
            self._initialized = True

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._n_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._n_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

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
        ratings_ids = interactions[:, 2]
        ratings_ids = ratings_ids.astype(np.float32)

        if not self._initialized:
            self._initialize(interactions)

        self._check_input(user_ids, item_ids)

        for epoch in range(self._n_iter):

            users, items, ratings = shuffle(user_ids,
                                            item_ids,
                                            ratings_ids,
                                            random_state=self._random_state)

            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)

            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)

            rating_ids_tensor = gpu(torch.from_numpy(ratings),
                                    self._use_cuda)

            for (mini_batch_num,
                 (batch_user,
                 batch_item,
                 batch_rating)) in enumerate(
                                    minibatch(user_ids_tensor,
                                              item_ids_tensor,
                                              rating_ids_tensor,
                                              batch_size=self._batch_size)):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
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

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._n_items,
                                                  self._use_cuda)

        out = self._model(user_ids, item_ids)

        return cpu(out.data).numpy().flatten()
