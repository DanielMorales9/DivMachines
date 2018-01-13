import numpy as np
import pandas as pd
from divmachines.classifiers import Classifier
from divmachines.classifiers.mf import MF


class LatentFactorPortfolio(Classifier):
    """
    Latent Factor Portfolio implementation based on the following work

    `Shi, Y., Zhao, X., Wang, J., Larson, M., & Hanjalic, A. (2012, August).
    Adaptive diversification of recommendation results via latent factor portfolio.
    In Proceedings of the 35th international ACM SIGIR conference on Research
    and development in information retrieval (pp. 175-184). ACM.`

    Parameters
    ----------
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

    def _initialize(self, x, y=None, dic=None):
        self._init_model()
        self._init_dataset(x, y, dic=dic)

    def _init_model(self):
        self._classifier = MF(n_factors=self._n_factors,
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

    def _init_dataset(self, x, y=None, dic=None):
        item_col = dic.get('items', None) or 1
        self._item_catalogue = x[:, item_col].unique()

    def fit(self, x, y, dic=None):
        """
        Fit the underlying classifiers.
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
        """

        if self._initialized:
            self._initialize(x, y=y, dic=dic)

        self._classifier.fit(x, y, dic=dic)

    def predict(self, x, top=10, b=0.5):
        """
        Predicts

        Parameters
        ----------
        x: ndarray
            users to predict the ranking
        top: int, optional
            Length of the ranking
        b: float, optional
            System-level Diversity.
            `b` controls the trade-off for all users between
            accuracy and diversity.
        Returns
        -------
        top-k: ndarray
            Ranking list for all submitted users
        """

        pass
