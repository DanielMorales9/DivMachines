import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .dataset import DenseDataset, SparseDataset
from .models import FactorizationMachine
from .utility import vectorize_dic
from .. import Classifier
from ..logging import Logger
from ..torch_utils import set_seed, gpu, cpu

class FM(Classifier):
    """
    Base Classifier for Factorization Machines
    """
    def __init__(self):
        super(FM, self).__init__()
        self._sparse = False
        self._initialized = False

    def _initialize(self, train):
        if isinstance(train, Dataset):
            self._dataset = train
        elif type(train).__module__ == np.__name__:
            if self._sparse:
                self._dataset = SparseDataset(train)
            else:
                self._dataset = DenseDataset(train)
        else:
            raise TypeError("Training set must be of type dataset or of type ndarray")

        self._n_users = self._dataset.n_users()
        self._n_items = self._dataset.n_items()

        self._n_features = self._dataset.n_features()

        self._init_model()

        self._init_optimization_function()

        self._initialized = True


class Pointwise(FM):
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
        self._learning_rate = learning_rate
        self._optimizer_func = optimizer_func
        self._loss_func = loss or torch.nn.MSELoss()
        self._logger = logger or Logger()
        self._n_workers = n_workers
        self._optimizer = None
        self._dataset = None

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
            if isinstance(self._model, FactorizationMachine):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of FactorizationMachine")

        else:
            self._model = gpu(FactorizationMachine(self._n_features,
                                                   self._n_factors),
                              self._use_cuda)

    def __deepcopy__(self, memo):
        model = copy.deepcopy(self._model, memo)
        random_state = copy.deepcopy(self._random_state, memo)
        optimizer_func = copy.deepcopy(self._optimizer_func, memo)
        logger = copy.deepcopy(self._logger, memo)
        return Pointwise(n_factors=self._n_factors,
                         model=model,
                         sparse=self._sparse,
                         n_iter=self._n_iter,
                         batch_size=self._batch_size,
                         random_state=random_state,
                         use_cuda=self._use_cuda,
                         n_workers=self._n_workers,
                         learning_rate=self._learning_rate,
                         l2=self._l2,
                         optimizer_func=optimizer_func,
                         logger=logger)

    def fit(self, train):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        train: ndarray or :class:`divmachines.fm.dataset`
            Train samples
        """

        if not self._initialized:
            self._initialize(train)

        loader = DataLoader(self._dataset,
                            batch_size=self._batch_size,
                            num_workers=self._n_workers,
                            shuffle=True)
        for epoch in range(self._n_iter):
            for mini_batch_num, batch in enumerate(loader):

                batch_tensor = gpu(batch[:, :-1], self._use_cuda)
                batch_ratings = gpu(batch[:, -1], self._use_cuda)

                observations_var = Variable(batch_tensor)
                rating_var = Variable(batch_ratings)

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

    def predict(self, x):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.
        Parameters
        ----------
        x: ndarray
            samples for which predict the ratings/rank score
        Returns
        -------
        predictions: np.array
            Predicted scores for all items in item_ids.
        """

        # if the users do not belong to the user set it raise error
        # self._check_input(user_ids, item_ids, allow_items_none=True)
        self._model.train(False)
        self._dataset = self._dataset(x)

        loader = DataLoader(self._dataset,
                            batch_size=len(x),
                            shuffle=False,
                            num_workers=self._n_workers)

        for samples in loader:
            var = Variable(gpu(samples, self._use_cuda))
            out = self._model(var)

        return cpu(out.data).numpy().flatten()
