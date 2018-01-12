from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from divmachines.utility.helper import check_random_state


def _get_cv(cv):
    cv = CROSS_VALIDATOR[cv]
    if cv is None:
        raise ValueError("Cross Validator must be provided")
    return cv


class CrossValidator(metaclass=ABCMeta):
    """
    Base class of the Data Partitioning strategy or
    Cross Validation strategy
    """

    def __init__(self):
        pass

    @abstractmethod
    def _iter_indices_mask(self, x, y, indices):
        raise NotImplementedError

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
    Naive Hold-Out cross-validator
    Provides train/test indices to split data in train/test sets.
    The partitioning is performed by randomly withholding some ratings
    for some of the users.
    The naive hold-out cross validation removes from the test set all
    the user that are not present in the train set.
    The classifiers could not handle the cold start problem.
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
                 random_state=None):
        super(NaiveHoldOut, self).__init__()
        self._times = times
        self._ratio = ratio
        self._user_idx = user_idx
        self._item_idx = item_idx
        self._random_state = random_state

    def split(self, x, y):
        data = pd.DataFrame(x)
        n_samples = data.shape[0]
        indices = np.arange(len(x))

        for i in range(self._times):
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


class UserHoldOut(CrossValidator):
    """
    User Hold-Out cross-validator
    Provides train/test indices to split data in train/test sets.
    The partitioning is performed by randomly withholding some ratings
    for all or some of the users.
    The User Hold-Out cross validation removes from the test set
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
                 random_state=None):
        super(UserHoldOut, self).__init__()
        self._times = times
        self._ratio = ratio
        self._user_idx = user_idx
        self._item_idx = item_idx
        self._random_state = random_state

    def _iter_indices_mask(self, x, y, indices):
        data = pd.DataFrame(x)[[self._user_idx, self._item_idx]]
        mask = np.zeros(data.shape[0], dtype=np.bool)
        for i in range(self._times):
            copy_mask = np.copy(mask)
            grouped = data.groupby(0)
            for user, g in grouped:
                idx_shuffled = g.index.values.reshape(-1)
                n_observed = int((1 - self._ratio) * len(idx_shuffled))
                check_random_state(self._random_state).shuffle(idx_shuffled)
                copy_mask[idx_shuffled[0:n_observed]] = True

            # cleaning
            train_split = data.ix[indices[np.logical_not(copy_mask)]]
            test_split = data.ix[indices[copy_mask]]

            # remove new user and new items from the test split
            train_items = train_split[self._item_idx].unique()
            test_idx = test_split.index[~test_split[self._item_idx].isin(train_items)]
            copy_mask[test_idx] = False
            yield copy_mask


CROSS_VALIDATOR = dict(
    kFold=KFold,
    leaveOneOut=LeaveOneOut,
    naiveHoldOut=NaiveHoldOut,
    userHoldOut=UserHoldOut
)


def create_cross_validator(cv):
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
