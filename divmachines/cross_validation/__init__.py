import copy
import time
import numpy as np
from joblib import Parallel, delayed
from divmachines.metrics import create_scorers
from validation.split import _get_cv, _create_cross_validator


def cross_validate(classifier,
                   x,
                   y,
                   metrics,
                   cv="kFold",
                   fit_params=None,
                   verbose=0,
                   n_jobs=1,
                   pre_dispatch='2*n_jobs',
                   return_train_score=False,
                   return_times=False):
    """
    Cross validation function
    Parameters
    ----------
    classifier: :class:`divmachines.classifiers.Classifier`
        Classifier that has fit method.
        In order to run classifier in parallel make sure that
        the n_jobs parameters in classifier is 0.
    x: ndarray
        Training samples
    y: ndarray
        The target variable to try to predict in the case of
        supervised learning.
    cv: string, :class:`divmachines.validate`, optional
        Determines the cross-validation splitting strategy.
        Default strategy is KFold with 3 splits.
        Consider to use Hold-Out cross-validation for model-based classifier.
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
    return_train_score: bool, optional
        Whether to return train scores or not.
        Default is False.
    return_times: bool, optional
        Whether to return times scores or not.
        Default is False.
    Returns
    -------
    scores : dict of float arrays of shape=(n_splits,)
        Array of scores of the estimator for each run of the cross validation.
        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:
            ``test_score``
                The score array for test scores on each cv split.
            ``train_score``
                The score array for train scores on each cv split.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``

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
                                return_train_score=return_train_score,
                                return_times=return_times)
        for train_idx, test_idx in cv.split(x, y))

    train_scores = []
    test_scores = []
    fit_times = []
    score_times = []
    for score in scores:
        i = 0
        if return_train_score:
            train_scores.append(score[i])
            i += 1
        test_scores.append(score[i])
        i += 1
        if return_times:
            fit_times.append(score[i])
            score_times.append(score[i+1])

    if return_train_score:
        train_scores = _aggregate_score_dicts(train_scores)

    test_scores = _aggregate_score_dicts(test_scores)

    if return_times:
        ret = {'fit_time': np.array(fit_times), 'score_time': np.array(score_times)}
    else:
        ret = {}

    for name in test_scores.keys():
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])

    return ret


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
    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_times:
        ret.extend([fit_time, score_time])
    return ret


def _score(classifier, x, y, scorers):
    """
    Performs the scores on the classifier
    Parameters
    ----------
    classifier: :class:`divmachines.classifiers.Classifier`
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

    scores = {}
    for scorer_name, scorer in scorers.items():
        scores[scorer_name] = scorer(predictions, y)
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


def _aggregate_score_dicts(scores):
    """
    Aggregate the list of dict to dict of np ndarray
    The aggregated output of _fit_and_score will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}
    Parameters
    ----------
    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.
    Example
    -------
    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    out = {}
    for key in scores[0]:
        out[key] = np.asarray([score[key] for score in scores])
    return out
