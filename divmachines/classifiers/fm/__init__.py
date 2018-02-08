import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .dataset import DenseDataset, SparseDataset
from divmachines.models import FactorizationMachine
from divmachines.classifiers import Classifier
from divmachines.logging import Logger
from divmachines.utility.torch import set_seed, gpu, cpu
from tqdm import tqdm


class FM(Classifier):
    """
    Factorization Machine

    Parameters
    ----------
    n_factors: int, optional
        Number of factors to use in user and item latent factors
    model: :class: div.machines.models, string, optional
        A Factorization Machine model or
        the pathname of the saved model. Default None
    sparse: boolean, optional
        Use sparse dataset
    loss: function, optional
        an instance of a PyTorch optimizer or a custom loss.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
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
    early_stopping: bool, optional
        Performs a dump every time to enable early stopping.
        Default False.
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
                 n_jobs=0,
                 shuffle=True,
                 pin_memory=False,
                 verbose=False,
                 sparse_num=0,
                 early_stopping=True):

        super(FM, self).__init__()
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self._model = model
        self._sparse = sparse
        self._random_state = random_state or np.random.RandomState()
        self.use_cuda = use_cuda
        self._optimizer_func = optimizer_func
        self._loss_func = loss or torch.nn.MSELoss()
        self._logger = logger or Logger()
        self._n_jobs = n_jobs
        self._optimizer = None
        self._dataset = None
        self._sparse = sparse
        self._n_items = None
        self._n_users = None
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self._disable = not verbose
        self._sparse_num = sparse_num
        self._early_stopping = early_stopping

        set_seed(self._random_state.randint(-10 ** 8, 10 ** 8),
                 cuda=self.use_cuda)

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
    def n_features(self):
        return self._dataset.n_features

    @n_features.getter
    def n_features(self):
        return self._dataset.n_features

    @property
    def index(self):
        return self._dataset.index

    @index.getter
    def index(self):
        return self._dataset.index

    @property
    def model(self):
        if self.use_cuda and torch.cuda.device_count() > 1:
            return self._model.module
        return self._model

    @model.getter
    def model(self):
        if self.use_cuda and torch.cuda.device_count() > 1:
            return self._model.module
        return self._model

    @property
    def dataset(self):
        return self._dataset.x

    @dataset.getter
    def dataset(self):
        return self._dataset.x

    @property
    def v(self):
        if self.use_cuda and torch.cuda.device_count() > 1:
            return self._model.module.v
        return self._model.v

    @v.getter
    def v(self):
        if self.use_cuda and torch.cuda.device_count() > 1:
            return self._model.module.v
        return self._model.v

    def _initialize(self,
                    x,
                    y=None,
                    dic=None,
                    n_users=None,
                    n_items=None,
                    lengths=None):
        self._init_dataset(x,
                           y=y,
                           dic=dic,
                           n_users=n_users,
                           n_items=n_items,
                           lengths=lengths)
        self._init_model()

        self._init_optim_fun()

        self._initialized = True

    def _init_dataset(self,
                      x,
                      y=None,
                      dic=None,
                      n_users=None,
                      n_items=None,
                      lengths=None):
        ix = None
        if isinstance(self._model, str):
            dic = torch.load(self._model)['dic']
            n_users = torch.load(self._model)['n_users']
            n_items = torch.load(self._model)['n_items']
            lengths = torch.load(self._model)['lengths']
            ix = torch.load(self._model)['ix']

        if type(x).__module__ == np.__name__:
            if y is None or type(x).__module__ == np.__name__:

                if self._dataset is not None:
                    self._dataset = self._dataset(x, y=y)
                elif self._sparse:
                    self._dataset = SparseDataset(x,
                                                  y=y,
                                                  dic=dic,
                                                  n_users=n_users,
                                                  n_items=n_items,
                                                  lengths=lengths,
                                                  ix=ix)
                else:
                    self._dataset = DenseDataset(x,
                                                 y=y,
                                                 dic=dic,
                                                 n_users=n_users,
                                                 n_items=n_items,
                                                 lengths=lengths,
                                                 ix=ix)

        else:
            raise TypeError("Training set must be of type "
                            "dataset or of type ndarray")

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
            if isinstance(self._model, str):
                model_dict = torch.load(self._model)
                n_features = model_dict["n_features"]
                n_factors = model_dict["n_factors"]
                self._model = FactorizationMachine(n_features,
                                                   n_factors=n_factors)
                self._model.load_state_dict(model_dict['state_dict'])
            elif not isinstance(self._model, FactorizationMachine):
                raise ValueError("Model must be an instance "
                                 "of FactorizationMachine")
        else:
            self._model = FactorizationMachine(self.n_features,
                                               n_factors=self.n_factors)
        if self.use_cuda and torch.cuda.device_count() > 1:
            self._model = torch.nn.DataParallel(gpu(self._model,
                                                    self.use_cuda))
        else:
            self._model = gpu(self._model,
                              self.use_cuda)

    def fit(self,
            x,
            y,
            dic=None,
            n_users=None,
            n_items=None,
            lengths=None):
        """
        Fit the models.
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

        self._initialize(x, y=y, dic=dic,
                         n_users=n_users,
                         n_items=n_items,
                         lengths=lengths)

        disable_batch = self._disable or self.batch_size is None

        loader = DataLoader(self._dataset,
                            pin_memory=self._pin_memory,
                            batch_size=self.batch_size,
                            num_workers=self._n_jobs,
                            shuffle=True)

        for epoch in tqdm(range(self.n_iter),
                          desc='Fitting',
                          leave=False,
                          disable=self._disable):
            for mini_batch_num, \
                (batch_tensor, batch_ratings) in tqdm(enumerate(loader),
                                                      desc='Batches',
                                                      leave=False,
                                                      disable=disable_batch):

                batch_tensor = gpu(batch_tensor, self.use_cuda)
                batch_ratings = gpu(batch_ratings, self.use_cuda)

                observations_var = Variable(batch_tensor)
                rating_var = Variable(batch_ratings)

                # forward step
                predictions = self._model(observations_var)

                # Zeroing Embeddings' gradients
                self._optimizer.zero_grad()

                # Compute Loss
                loss = self._loss_func(predictions, rating_var)

                # logging
                self._logger.log(loss, epoch, batch=mini_batch_num,
                                 cpu=self.use_cuda)

                # backward step
                loss.backward()

                # optimization step
                self._optimizer.step()

        if self._early_stopping:
            self._prepare(dic, n_users, n_items, lengths)

    def _prepare(self, dic, n_users, n_items, lengths):
        idx = None
        if self._dataset.index:
            idx = {}
            for k, v in self._dataset.index.items():
                idx[k] = v
        self.dump = {'state_dict': self._model.state_dict(),
                     'n_features': self.n_features,
                     'n_factors': self.n_factors,
                     'dic': dic,
                     'n_users': n_users,
                     'n_items': n_items,
                     'lengths': lengths,
                     'ix': idx}

    def save(self, path):
        torch.save(self.dump, path)

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
        if len(x.shape) == 1:
            x = np.array([x])

        if isinstance(self._model, str):
            self._initialize(x)
        else:
            self._init_dataset(x)

        disable_batch = self._disable or self.batch_size is None
        if self.batch_size is None:
            self.batch_size = len(self._dataset)
        loader = DataLoader(self._dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self._n_jobs)

        out = np.zeros(len(x))
        for i, batch_data in tqdm(enumerate(loader),
                                  desc="Prediction",
                                  leave=False,
                                  disable=disable_batch):
            var = Variable(gpu(batch_data, self.use_cuda))
            out[(i*self.batch_size):((i+1)*self.batch_size)] = \
                cpu(self._model(var), self.use_cuda).data.numpy()

        return out.flatten()
