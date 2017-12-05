# coding=utf-8
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from divmachines import Classifier
from divmachines.helper import _prepare_for_prediction
from divmachines.logging import Logger
from divmachines.mf.models import MatrixFactorizationModel
from divmachines.torch_utils import set_seed, gpu, cpu
from torch.utils.data import DataLoader
from .dataset import DenseDataset


class MF(Classifier):
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
    n_jobs: int, optional
        Number of workers for data loading.
        Default is 0, it means that the data loader runs in the main process.
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
                 n_jobs=0):

        super(MF, self).__init__()
        self._n_factors = n_factors
        self._model = model
        self._n_iter = n_iter
        self._sparse = sparse
        self._batch_size = batch_size
        self._random_state = random_state or np.random.RandomState()
        self._use_cuda = use_cuda
        self._l2 = l2
        self._n_jobs = n_jobs
        self._learning_rate = learning_rate
        self._optimizer_func = optimizer_func
        self._loss_func = loss or torch.nn.MSELoss()
        self._logger = logger or Logger()
        self._dataset = None
        self._optimizer = None
        self._initialized = False

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    def _initialize(self, x, y=None, dic=None):

        self._init_dataset(x, y=y, dic=dic)

        self._init_model()

        self._init_optim_fun()

        self._initialized = True

    def _init_dataset(self, x, y=None, dic=None):
        self._dataset = DenseDataset(x, y=y, dic=dic)
        self._n_users = self._dataset.n_users()
        self._n_items = self._dataset.n_items()

    def _init_optim_fun(self):
        if self._optimizer_func is None:
            self._optimizer = \
                optim.SGD(self._model.parameters(),
                          weight_decay=self._l2,
                          lr=self._learning_rate)
        else:
            self._optimizer = \
                self._optimizer_func(self._model.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def _init_model(self):
        if self._model is not None:
            if issubclass(self._model, torch.nn.Module):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of "
                                 "torch.nn.Module class")

        else:
            self._model = gpu(MatrixFactorizationModel(self._n_users,
                                                       self._n_items,
                                                       self._n_factors,
                                                       self._sparse),
                              self._use_cuda)

    def fit(self, x, y, dic=None):
        """
        Fit the model.
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        x: ndarray
            Training samples
        y: ndarray
            Target values for samples
        dic: dict, optional
            dic indicates the columns to make indexable.
        """

        if not self._initialized:
            self._initialize(x, y=y, dic=dic)

        loader = DataLoader(self._dataset,
                            shuffle=True,
                            batch_size=self._batch_size,
                            num_workers=self._n_jobs)

        for epoch in range(self._n_iter):

            for (mini_batch_num,
                 (batch_data, batch_rating)) in enumerate(loader):
                user_var = Variable(gpu(batch_data[:, 0], self._use_cuda))
                item_var = Variable(gpu(batch_data[:, 1], self._use_cuda))
                rating_var = Variable(gpu(batch_rating), self._use_cuda).float()

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

    def predict(self, x, **kwargs):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.
        Parameters
        ----------
        x: ndarray
           It will predict scores for all (user, item) pairs defined in x
        Returns
        -------
        predictions: ndarray
            Predicted scores for all elements
        """

        self._model.train(False)

        x = _prepare_for_prediction(x, 2)

        self._dataset = self._dataset(x)

        loader = DataLoader(self._dataset,
                            batch_size=len(self._dataset),
                            num_workers=self._n_jobs)

        out = None
        for batch_data in loader:
            user_var = Variable(gpu(batch_data[:, 0], self._use_cuda))
            item_var = Variable(gpu(batch_data[:, 1], self._use_cuda))
            out = self._model(user_var, item_var)

        return cpu(out.data).numpy().flatten()