import copy
import time
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from .helper import check_random_state
from .metrics import create_scorers


def cross_validate(classifier,
                   x,
                   y,
                   metrics,
                   cv="kFold",
                   fit_params=None,
                   verbose=0,
                   n_jobs=1,
                   pre_dispatch='2*n_jobs'):
    """
    Cross validation function
    Parameters
    ----------
    classifier: :class:`divmachines.Classifier`
        Classifier that has fit method.
        In order to run classifiers in parallel make sure that
        the n_jobs parameters in classifier is 0.
    x: ndarray
        Training samples
    y: ndarray
        The target variable to try to predict in the case of
        supervised learning.
    cv: string, :class:`divmachines.validate`, optional
        Determines the cross-validation splitting strategy.
        Default strategy is KFold with 3 splits.
        Consider to use Hold-Out cross-validation for model-based classifiers.
        KFold will lead the classifier to fail during prediction phase
        because of the different parameter dimensions.
    metrics: str, list, set, tuple
        Metric to use in the cross validation phase
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    n_jobs: int or optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    pre_dispatch: int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    verbose: integer, optional
        The verbosity level.

    """

    if metrics is None:
        raise ValueError("Metrics must be provided")
    parallel = Parallel(n_jobs=n_jobs,
                        pre_dispatch=pre_dispatch,
                        verbose=verbose)

    cv = _create_cross_validator(cv)

    scores = parallel(
        delayed(_fit_and_score)(clone(classifier),
                                x,
                                y,
                                metrics,
                                fit_params,
                                train_idx,
                                test_idx,
                                verbose,
                                return_train_score=True,
                                return_times=True)
        for train_idx, test_idx in cv.split(x, y))
    return scores


def _fit_and_score(classifier,
                   x,
                   y,
                   metrics,
                   fit_params,
                   train_idx,
                   test_idx,
                   verbose=0,
                   return_train_score=False,
                   return_times=False):
    """
    Fit classifier and compute scores for a given dataset split.
    Parameters
    ----------
    classifier : estimator object implementing 'fit'
        The object to use to fit the data.
    x : array-like of shape at least 2D
        The data to fit.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    metrics: tuple
        metrics for computing the score
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    train_idx : array-like, shape (n_train_samples,)
        Indices of training samples.
    test_idx : array-like, shape (n_test_samples,)
        Indices of test samples.
    verbose: int, optional
        #TODO implement: Verbosity level
    return_train_score : boolean, optional, default: False
        Compute and return score on training set.
    return_times : boolean, optional, default: False
        Whether to return the fit/score times.
    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.
    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).
    n_test_samples : int
        Number of test samples.
    fit_time : float
        Time spent for fitting in seconds.
    score_time : float
        Time spent for scoring in seconds.
    """

    fit_params = fit_params or {}
    train_scores = None
    start_time = time.time()

    x_train, y_train = x[train_idx, :], y[train_idx]
    x_test, y_test = x[test_idx, :], y[test_idx]

    classifier.fit(x_train, y_train, **fit_params)

    fit_time = time.time() - start_time

    scorers = create_scorers(metrics)

    test_scores = _score(classifier, x_test, y_test, scorers)
    score_time = time.time() - start_time - fit_time
    if return_train_score:
        train_scores = _score(classifier, x_train, y_train, scorers)
    #TODO: Human Readable scores
    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_times:
        ret.extend([fit_time, score_time])
    return ret


def _score(classifier, x, y, scorers):
    """
    Performs the scores on the classifier
    Parameters
    ----------
    classifier: :class:`divmachines.Classifier`
        An instance of a classifier class
    x: ndarray
        samples to predict
    y: ndarray
        true target values
    scores: arraylike
        list of callables, each element is a score function
    Returns
    -------
    scores: arralike
        list of scores
    """
    predictions = classifier.predict(x)

    scores = []
    for scorer in scorers:
        scores.append(scorer(predictions, y))

    return scores


def clone(obj):
    """
    Function to clone instances
    Parameters
    ---------
    obj: object
        instance to clone
    """
    return copy.deepcopy(obj)


def _get_cv(cv):
    cv = CROSS_VALIDATOR[cv]
    if cv is None:
        raise ValueError("Cross Validator must be provided")
    return cv


def _create_cross_validator(cv):
    """
    Return an instance of a cross validator.

    Parameters
    ----------
    cv: string, :class:`divmachines.validate`, optional
        Determines the cross-validation splitting strategy.
    Returns
    -------
    cv: :class:`divmachines.validate.CrossValidator`
        An instance of a Cross Validation strategy
    """
    if isinstance(cv, str):
        cv = _get_cv(cv)()
    elif not hasattr(cv, "split"):
        raise ValueError("Input Cross Validator must be an "
                         "instance child of CrossValidator class")
    return cv


class CrossValidator(metaclass=ABCMeta):
    """
    Base class of the Data Partitioning strategy or
    Cross Validation strategy
    """

    def __init__(self):
        pass

    def split(self, x, y):
        """
        Data partitioning function,
        it returns the training and the test indexes
        Parameters
        ----------
        x: ndarray
            training samples
        y: ndarray
            target value for samples
        Returns
        -------
        train_index : ndarray
            The training set indices for that split.
        test_index : ndarray
            The testing set indices for that split.
        """
        indices = np.arange(len(x))
        for test_index in self._iter_indices_mask(x, y, indices):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    @abstractmethod
    def _iter_indices_mask(self, x, y, indices):
        raise NotImplementedError


class KFold(CrossValidator):
    """
    K-fold cross validation strategy
    It divides the dataset into k independent fold
    and at each iteration considers one fold as the
    test set and the remaining folds as the training sets
    Parameters
    ----------
    folds: int, optional
        Number of fold to use for the k-fold cross validation.
        Minimum is 2 an default is 3.
    shuffle: boolean, optional
        Whether to shuffle the data before splitting into batches.
        By default it shuffle the data
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    """
    def __init__(self, folds=3, shuffle=True, random_state=None):
        super(KFold, self).__init__()

        if folds < 2:
            raise ValueError("Number of folds too low, minimum value is 2")
        else:
            self._folds = folds
        self._shuffle = shuffle
        self._random_state = random_state

    def _iter_indices_mask(self, x, y, indices):
        if self._shuffle:
            check_random_state(self._random_state).shuffle(indices)

        n_splits = self._folds
        fold_sizes = (len(x) // n_splits) * np.ones(n_splits, dtype=np.int)
        fold_sizes[:len(x) % n_splits] += 1
        current = 0
        mask = np.zeros(len(indices), dtype=np.bool)
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            copy_mask = np.copy(mask)
            copy_mask[indices[start:stop]] = True
            yield copy_mask
            current = stop


class LeaveOneOut(CrossValidator):
    """
    Leave-One-Out cross-validator
    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.
    Parameters
    ----------
    shuffle: boolean, optional
        Whether to shuffle the data before splitting into batches.
        By default it shuffle the data
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    """

    def __init__(self, shuffle=True, random_state=None):
        super(LeaveOneOut, self).__init__()
        self._shuffle = shuffle
        self._random_state = random_state

    def _iter_indices_mask(self, x, y, indices):
        if self._shuffle:
            check_random_state(self._random_state).shuffle(indices)

        mask = np.zeros(len(indices), dtype=np.bool)
        for i in indices:
            new_mask = mask.copy()
            new_mask[i] = True
        yield new_mask


class NaiveHoldOut(CrossValidator):
    """
    Hold-Out cross-validator
    Provides train/test indices to split data in train/test sets.
    The partitioning is performed by randomly withholding some ratings
    for all or some of the users.
    The Naive version of the Hold-Out cross validation removes from the test set
    all the user that are not present in the
    Parameters
    ----------
    ratio: float, optional
        Ratio between the train set .
        For instance, 0.7 means that the train set is 70% of the
        original dataset, while the test set is 30% of it.
        Default is 80% for the train set and 20% for the test set.
    times: int, optional
        Number of times to run Hold-out cross validation.
        Higher values of it result in less variance in the result score.
        Default is 10.
    user_idx: int, optional
        Indicates the user index in the transaction data.
        Default is 0.
    item_idx: int, optional
        Indicates the item index in the transaction data
        Default is 1.
    shuffle: boolean, optional
        Whether to shuffle the data before splitting into batches.
        By default it shuffle the data
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    """
    def __init__(self,
                 ratio=0.8,
                 times=10,
                 user_idx=0,
                 item_idx=1,
                 shuffle=True,
                 random_state=None):
        super(NaiveHoldOut, self).__init__()
        self._times = times
        self._ratio = ratio
        self._user_idx = user_idx
        self._item_idx = item_idx
        self._shuffle = shuffle
        self._random_state = random_state

    def split(self, x, y):
        data = pd.DataFrame(x)
        n_samples = data.shape[0]
        indices = np.arange(len(x))

        for i in range(self._times):
            if self._shuffle:
                check_random_state(self._random_state).shuffle(indices)

            train_size = int(n_samples * self._ratio)
            # split data according to the shuffled index and the holdout size
            train_idx = indices[:train_size]
            train_split = data.ix[indices[:train_size]]
            test_split = data.ix[indices[train_size:]]

            # remove new user and new items from the test split
            train_users = train_split[self._user_idx].unique()
            train_items = train_split[self._item_idx].unique()
            test_idx = test_split.index[(test_split[self._user_idx].isin(train_users)) &
                                    (test_split[self._item_idx].isin(train_items))]
            yield train_idx, test_idx

    def _iter_indices_mask(self, x, y, indices):
        pass


CROSS_VALIDATOR = dict(
    kFold=KFold,
    leaveOneOut=LeaveOneOut,
    naiveHoldOut=NaiveHoldOut
)