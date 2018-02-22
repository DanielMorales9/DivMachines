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
from divmachines.utility.indexing import FeaturesFactory, IndexDictionary


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
    device_id: int, optional
        GPU device ID to which the tensors are sent.
        If set use_cuda must be True.
        By Default uses all GPU available.
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
    n_iter_no_change : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'
    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        convergence is considered to be reached and training stops.
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
                 device_id=None,
                 logger=None,
                 n_jobs=0,
                 shuffle=True,
                 pin_memory=False,
                 verbose=False,
                 sparse_num=0,
                 early_stopping=False,
                 n_iter_no_change=10,
                 tol=1e-4):

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
        self._n_iter_no_change = n_iter_no_change
        self._tol = tol
        if device_id is not None and not self.use_cuda:
            raise ValueError("use_cuda flag must be true")
        self._device_id = device_id
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
        if self.use_cuda and torch.cuda.device_count() > 1 and \
                self._device_id is None:
            return self._model.module
        return self._model

    @model.getter
    def model(self):
        if self.use_cuda and torch.cuda.device_count() > 1 and \
                self._device_id is None:
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
        if self.use_cuda and torch.cuda.device_count() > 1 and \
                self._device_id is None:
            return self._model.module.v
        return self._model.v

    @v.getter
    def v(self):
        if self.use_cuda and torch.cuda.device_count() > 1 and \
                self._device_id is None:
            return self._model.module.v
        return self._model.v

    def _initialize(self,
                    x,
                    y=None,
                    dic=None,
                    n_users=None,
                    n_items=None,
                    lengths=None):

        self._best_loss = np.inf
        self._no_improvement_count = 0

        self._init_dataset(x,
                           y=y,
                           dic=dic,
                           n_users=n_users,
                           n_items=n_items,
                           lengths=lengths)
        self._init_model()

        self._init_optim_fun()

    def _init_dataset(self,
                      x,
                      y=None,
                      dic=None,
                      n_users=None,
                      n_items=None,
                      lengths=None):
        ix = None
        if isinstance(self._model, str):
            self._load = torch.load(self._model, map_location=lambda storage, loc: storage)
            dic = self._load['dic']
            n_users = self._load['n_users']
            n_items = self._load['n_items']
            lengths = self._load['lengths']
            ixx = self._load['ix']
            ix = IndexDictionary(FeaturesFactory(ixx,
                                                 old=True,
                                                 prefix=dic.copy()))
            # feeding the dictionary
            for k in ixx:
                f = ix[k]

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
                # map the tensor to cpu
                model_dict = self._load
                n_features = model_dict["n_features"]
                n_factors = model_dict["n_factors"]
                self._model = FactorizationMachine(n_features,
                                                   n_factors=n_factors)
                dic = {}
                for m in model_dict['state_dict']:
                    if m.startswith('module.'):
                        dic[m[7:]] = model_dict['state_dict'][m]
                    else:
                        dic[m] = model_dict['state_dict'][m]
                self._model.load_state_dict(dic)
            elif not (isinstance(self._model, FactorizationMachine) or
                      isinstance(self._model, torch.nn.DataParallel)):
                raise ValueError("Model must be an instance "
                                 "of FactorizationMachine")
        else:
            self._model = FactorizationMachine(self.n_features,
                                               n_factors=self.n_factors)
        if not isinstance(self._model, torch.nn.DataParallel):
            if self.use_cuda and torch.cuda.device_count() > 1 and \
                    self._device_id is None:
                self._model = torch.nn.DataParallel(gpu(self._model,
                                                        self.use_cuda))
            else:
                self._model = gpu(self._model,
                                  self.use_cuda,
                                  self._device_id)

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
            mini_batch_num = 0
            acc_loss = 0.0
            for batch_tensor, batch_ratings in tqdm(loader,
                                                    desc='Batches',
                                                    leave=False,
                                                    disable=disable_batch):
                batch_size = batch_tensor.shape[0]
                batch_tensor = gpu(batch_tensor,
                                   self.use_cuda,
                                   self._device_id)
                batch_ratings = gpu(batch_ratings,
                                    self.use_cuda,
                                    self._device_id)

                observations_var = Variable(batch_tensor)
                rating_var = Variable(batch_ratings)

                # forward step
                predictions = self._model(observations_var)

                # Zeroing Embeddings' gradients
                self._optimizer.zero_grad()

                # Compute Loss
                loss = self._loss_func(predictions, rating_var)

                acc_loss += loss.data.cpu().numpy()[0] * batch_size

                # logging
                self._logger.log(loss, epoch, batch=mini_batch_num,
                                 cpu=self.use_cuda)

                # backward step
                loss.backward()

                # optimization step
                self._optimizer.step()

                mini_batch_num += 1

            acc_loss /= len(self._dataset)

            self._update_no_improvement_count(acc_loss)

            if self._no_improvement_count > self._n_iter_no_change:
                break

        if self._early_stopping:
            self._prepare(dic, n_users, n_items, lengths)

    def _update_no_improvement_count(self, acc_loss):
        if acc_loss > self._best_loss - self._tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        if acc_loss < self._best_loss:
            self._best_loss = acc_loss

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

    def init_predict(self, x):
        if isinstance(self._model, str):
            self._initialize(x)
        else:
            self._init_dataset(x)

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

        self.init_predict(x)

        disable_batch = self._disable or self.batch_size is None
        if self.batch_size is None:
            self.batch_size = len(self._dataset)
        loader = DataLoader(self._dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self._n_jobs)

        out = np.zeros(len(x))
        i = 0
        for batch_data in tqdm(loader,
                               desc="Prediction",
                               leave=False,
                               disable=disable_batch):
            var = Variable(gpu(batch_data,
                               self.use_cuda,
                               self._device_id))
            out[(i*self.batch_size):((i+1)*self.batch_size)] = \
                cpu(self._model(var), self.use_cuda).data.numpy()
            i += 1

        return out.flatten()
