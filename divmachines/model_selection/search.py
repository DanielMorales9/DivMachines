from abc import ABCMeta, abstractmethod
from collections import Mapping
from itertools import product
from functools import partial, reduce
import operator
import numpy as np
from .split import create_cross_validator
from . import clone, _fit_and_score, _aggregate_score_dicts
from joblib import Parallel, delayed


class ParameterGrid(object):
    """Grid of parameters with a discrete number of values for each.
    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    Read more in the :ref:`User Guide <search>`.
    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        An empty dict signifies default parameters.
        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.
    """

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of string to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


class BaseSearchCV(metaclass=ABCMeta):
    """Base class for hyper parameter search with cross-validation."""

    @abstractmethod
    def __init__(self,
                 classifier,
                 metrics=None,
                 n_jobs=1,
                 cv=None,
                 verbose=0,
                 pre_dispatch='2*n_jobs',
                 return_train_score=True):

        self.metrics = metrics
        self.classifier = classifier
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score
        self._scores = None

    def fit(self, x, y=None, fit_params=None):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """

        cv = create_cross_validator(self.cv)

        candidate_params = list(self._get_param_iterator() or [])

        base_classifier = clone(self.classifier)

        scores = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )(delayed(_fit_and_score)(clone(base_classifier),
                                  x,
                                  y,
                                  self.metrics,
                                  fit_params,
                                  train_idx,
                                  test_idx,
                                  parameters,
                                  verbose=self.verbose,
                                  return_train_score=self.return_train_score,
                                  return_times=True,
                                  return_parameters=True)
          for parameters, (train_idx, test_idx) in product(candidate_params, cv.split(x, y)))

        if self.return_train_score:
            (train_scores, test_scores, fit_times,
             score_times, params) = zip(*scores)
            self._scores = _aggregate_score_dicts_per_params(test_scores,
                                                             fit_times,
                                                             score_times,
                                                             params,
                                                             train_scores=train_scores)
        else:
            (test_scores, fit_times,
             score_times, params) = zip(*scores)
            self._scores = _aggregate_score_dicts_per_params(test_scores,
                                                             fit_times,
                                                             score_times,
                                                             params,
                                                             train_scores=None)

    def get_scores(self, pretty=False):
        if pretty:
            pretty_scores = []
            for score in self._scores:
                el = {}
                for k, v in score.items():
                    if isinstance(v, list):
                        el[k] = np.mean(v)
                    else:
                        el[k] = v
                pretty_scores.append(el)
        return self._scores

    def _get_param_iterator(self):
        pass


class GridSearchCV(BaseSearchCV):
    """
    Exhaustive search over specified parameter values for an estimator.
    Important members are fit, predict.
    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.
    Read more in the :ref:`User Guide <grid_search>`.
    Parameters
    ----------
    classifier : classifier object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    metrics : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    pre_dispatch : int, or string, optional
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
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <model_selection>` for the various
        cross-validation strategies that can be used here.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    return_train_score : boolean, optional
        If ``False``, the ``cv_results_`` attribute will not include training
        scores. Default is True.
    """

    def __init__(self, classifier,
                 param_grid,
                 metrics=None,
                 n_jobs=1,
                 cv='kFold',
                 verbose=0,
                 pre_dispatch='2*n_jobs',
                 return_train_score=True):
        super(GridSearchCV, self).__init__(
            classifier=classifier,
            metrics=metrics,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            return_train_score=return_train_score)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return ParameterGrid(self.param_grid)


def _aggregate_score_dicts_per_params(test_scores,
                                      fit_times,
                                      score_times,
                                      params,
                                      train_scores=None):
    scores = []
    i = -1
    if train_scores is not None:
        for p, f, s, t, tr in zip(params,
                                  fit_times,
                                  score_times,
                                  test_scores,
                                  train_scores):
            _fill_scores(f, i, p, s, scores, t, tr)
    else:
        for p, f, s, t in zip(params,
                              fit_times,
                              score_times,
                              test_scores):
            _fill_scores(f, i, p, s, scores, t)
    return scores


def _fill_scores(f, i, p, s, scores, t, tr=None):
    j = _get_index_score(scores, p)
    if j == -1:
        i += 1
        scores.append(p)
    j = i
    if scores[j].get("fit_times", None) is None:
        scores[j]["fit_times"] = [f]
    else:
        scores[j]["fit_times"].append(f)
    if scores[j].get("score_times", None) is None:
        scores[j]["score_times"] = [s]
    else:
        scores[j]["score_times"].append(s)

    for k, v in t.items():
        if scores[j].get("test_%s" % k, None) is None:
            scores[j]["test_%s" % k] = [v]
        else:
            scores[j]["test_%s" % k].append(v)

    if tr is not None:
        for k, v in tr.items():
            if scores[j].get("train_%s" % k, None) is None:
                scores[j]["train_%s" % k] = [v]
            else:
                scores[j]["train_%s" % k].append(v)


def _get_index_score(scores, param):
    for i, score in enumerate(scores):
        all = True
        for k, v in param.items():
            if score.get(k, None) is None:
                all = False
                break
            else:
                all = all and score[k] == v
        if all:
            return i
    return -1
