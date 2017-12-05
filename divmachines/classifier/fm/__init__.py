import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .dataset import DenseDataset, SparseDataset
from divmachines.model import FactorizationMachine
from divmachines.classifier import Classifier
from divmachines.logging import Logger
from divmachines.torch_utils import set_seed, gpu, cpu


class FM(Classifier):
    """
    Factorization Machine

    Parameters
    ----------
    n_factors: int, optional
        Number of factors to use in user and item latent factors
    model: :class: div.machines.model, optional
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
    n_jobs: int, optional
        Number of jobs for data loading.
        Default is 0, it means that the data loader runs in the main process.
    """

    def __init__(self,
                 n_factors=10,
                 model=None,
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

        super(FM, self).__init__()
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
        self._n_jobs = n_jobs
        self._optimizer = None
        self._dataset = None
        self._sparse = sparse
        self._initialized = False

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self._use_cuda)

    def _initialize(self, x, y=None, dic=None):

        self._init_dataset(x, y=y, dic=dic)

        self._init_model()

        self._init_optim_fun()

        self._initialized = True

    def _init_dataset(self, x, y=None, dic=None):
        if type(x).__module__ == np.__name__:
            if y is None or type(x).__module__ == np.__name__:
                if self._dataset is not None:
                    self._dataset = self._dataset(x, y=y, dic=dic)
                elif self._sparse:
                    self._dataset = SparseDataset(x, y=y, dic=dic)
                else:
                    self._dataset = DenseDataset(x, y=y, dic=dic)
        else:
            raise TypeError("Training set must be of type dataset or of type ndarray")

        self._n_features = self._dataset.n_features()

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
            if isinstance(self._model, FactorizationMachine):
                self._model = gpu(self._model, self._use_cuda)
            else:
                raise ValueError("Model must be an instance of FactorizationMachine")

        else:
            self._model = gpu(FactorizationMachine(self._n_features,
                                                   self._n_factors),
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
            dic indicates the columns to vectorize
            if training samples are in raw format.
        """

        if not self._initialized:
            self._initialize(x, y=y, dic=dic)

        loader = DataLoader(self._dataset,
                            batch_size=self._batch_size,
                            num_workers=self._n_jobs,
                            shuffle=True)

        for epoch in range(self._n_iter):
            for mini_batch_num, (batch_tensor, batch_ratings) in enumerate(loader):
                batch_tensor = gpu(batch_tensor, self._use_cuda)
                batch_ratings = gpu(batch_ratings, self._use_cuda)

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

    def predict(self, x, **kwargs):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.
        Parameters
        ----------
        x: ndarray or :class:`divmachines.fm.dataset`
            samples for which predict the ratings/rank score
        Returns
        -------
        predictions: np.array
            Predicted scores for each sample in x
        """

        self._model.train(False)
        if len(x.shape) == 1:
            x = np.array([x])

        self._init_dataset(x)

        loader = DataLoader(self._dataset,
                            batch_size=len(x),
                            shuffle=False,
                            num_workers=self._n_jobs)
        out = None
        for samples in loader:
            var = Variable(gpu(samples, self._use_cuda))
            out = self._model(var)

        return cpu(out.data).numpy().flatten()
