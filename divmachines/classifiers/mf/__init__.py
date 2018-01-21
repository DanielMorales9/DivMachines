# coding=utf-8
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from divmachines.classifiers import Classifier
from divmachines.utility.helper import _prepare_for_prediction
from divmachines.logging import Logger
from divmachines.models import SimpleMatrixFactorizationModel
from divmachines.utility.torch import set_seed, gpu
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
    model: :class: div.machines.models, optional
        A matrix Factorization model
    n_factors: int, optional
        Number of factors to use in user and item latent factors
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
                 model=None,
                 n_factors=10,
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
        self.n_factors = n_factors
        self.iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self._model = model
        self._sparse = sparse
        self._random_state = random_state or np.random.RandomState()
        self._use_cuda = use_cuda
        self._n_jobs = n_jobs
        self._optimizer_func = optimizer_func
        self._loss_func = loss or torch.nn.MSELoss()
        self._logger = logger or Logger()
        self._dataset = None
        self._optimizer = None
        self._initialized = False

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    @property
    def n_users(self):
        return self._dataset.n_users

    @n_users.getter
    def n_users(self):
        return self._dataset.n_users

    @property
    def n_items(self):
        return self._dataset.n_items

    @n_items.getter
    def n_items(self):
        return self._dataset.n_items

    @property
    def index(self):
        return self._dataset.index

    @index.getter
    def index(self):
        return self._dataset.index

    @property
    def x(self):
        return self._model.x

    @x.getter
    def x(self):
        return self._model.x

    @property
    def y(self):
        return self._model.y

    @x.getter
    def y(self):
        return self._model.y

    def _initialize(self,
                    x,
                    y=None,
                    dic=None,
                    n_users=None,
                    n_items=None):

        self._init_dataset(x,
                           y=y,
                           dic=dic,
                           n_users=n_users,
                           n_items=n_items)

        self._init_model()

        self._init_optim_fun()

        self._initialized = True

    def _init_dataset(self,
                      x,
                      y=None,
                      dic=None,
                      n_users=None,
                      n_items=None):
        if type(x).__module__ == np.__name__:
            if y is None or type(x).__module__ == np.__name__:
                if self._dataset is not None:
                    self._dataset = self._dataset(x, y=y)
                else:
                    self._dataset = DenseDataset(x,
                                                 y=y,
                                                 dic=dic,
                                                 n_users=n_users,
                                                 n_items=n_items)
        else:
            raise TypeError("Training set must be of type dataset or of type ndarray")

    def _init_optim_fun(self):
        if self._optimizer_func is None:
            self._optimizer = \
                optim.SGD(self._model.parameters(),
                          weight_decay=self.l2,
                          lr=self.learning_rate)
        else:
            self._optimizer = \
                self._optimizer_func(self._model.parameters(),
                                     weight_decay=self.l2,
                                     lr=self.learning_rate)

    def _init_model(self):
        if self._model is not None:
            if issubclass(self._model, torch.nn.Module):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of "
                                 "torch.nn.Module class")

        else:
            self._model = gpu(SimpleMatrixFactorizationModel(self.n_users,
                                                             self.n_items,
                                                             self.n_factors,
                                                             self._sparse),
                              self._use_cuda)

    def fit(self, x, y, dic=None, n_users=None, n_items=None):
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
        n_users: int, optional
            Total number of users. The model will have `n_users` rows.
            Default is None, `n_users` will be inferred from `x`.
        n_items: int, optional
            Total number of items. The model will have `n_items` columns.
            Default is None, `n_items` will be inferred from `x`.
        """

        if not self._initialized:
            self._initialize(x,
                             y=y,
                             dic=dic,
                             n_users=n_users,
                             n_items=n_items)

        loader = DataLoader(self._dataset,
                            shuffle=True,
                            batch_size=self.batch_size,
                            num_workers=self._n_jobs)

        for epoch in range(self.iter):

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

        self._init_dataset(x)
        if self.batch_size is None:
            self.batch_size = len(self._dataset)

        loader = DataLoader(self._dataset,
                            shuffle=False,
                            batch_size=self.batch_size,
                            num_workers=self._n_jobs)

        out = np.zeros(len(x))
        for i, batch_data in enumerate(loader):
            user_var = Variable(gpu(batch_data[:, 0], self._use_cuda))
            item_var = Variable(gpu(batch_data[:, 1], self._use_cuda))
            out[(i*self.batch_size):((i+1)*self.batch_size)] = self._model(user_var, item_var) \
                .cpu().data.numpy()

        return out.flatten()
