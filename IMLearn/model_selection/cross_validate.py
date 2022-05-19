from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score, validation_score = 0, 0
    separator = np.remainder(np.arange(y.size), cv)  # [0, 1, .., cv-1, 0,1, .., cv-1, 0, ....]
    for k in range(cv):
        # separate the data to k-folds
        train_fold_X, train_fold_y = X[separator != k], y[separator != k]
        validate_fold_X, validate_fold_y = X[separator == k], y[separator == k]
        estimator.fit(train_fold_X, train_fold_y)
        train_score += scoring(train_fold_y, estimator.predict(train_fold_X))
        validation_score += scoring(validate_fold_y, estimator.predict(validate_fold_X))
    return train_score/cv, validation_score/cv
