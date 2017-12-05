import numpy as np


def create_scorers(metrics):
    """
    Function that create list of scorers

    Parameters
    ----------
    metrics: string, callable, list/tuple, dict
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
    Returns
    -------
    scoring : list of callable
        A list of scorer function
    """
    if callable(metrics):
        return metrics
    elif isinstance(metrics, str):
        return [_get_scores(metrics)]
    elif isinstance(metrics, (list, tuple, set)):
        scorers = []
        for metric in metrics:
            scorers.append(create_scorers(metric))
        return scorers
    elif isinstance(metrics, dict):
        raise ValueError("Dictionaries are not allowed")
    elif metrics is None:
        raise ValueError("At least one metric must be provided")


def mean_absolute_error(y_pred, y_true):
    """
    Mean Absolute Error (MAE) function
    Parameters:
    ----------
    y_pred: ndarray
        predicted target
    y_true: ndarray
        true target
    Returns
    -------
    loss: float
        Loss value
    """
    return np.mean(y_pred - y_true)


def mean_square_error(y_pred, y_true):
    """
    Mean Square Error (MSE) function
    Parameters:
    ----------
    y_pred: ndarray
        predicted target
    y_true: ndarray
        true target
    Returns
    -------
    loss: float
        Loss value
    """
    return np.mean(np.power(y_pred - y_true, 2))


def root_mean_square_error(y_pred, y_true):
    """
    Root Mean Square Error (RMSE) function
    Parameters:
    ----------
    y_pred: ndarray
        predicted target
    y_true: ndarray
        true target
    Returns
    -------
    loss: float
        Loss value
    """
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))


def _get_scores(metrics):
    return SCORERS[metrics]


SCORERS = dict(
    mean_absolute_error=mean_absolute_error,
    mean_square_error=mean_square_error,
    root_mean_square_error=root_mean_square_error,
)