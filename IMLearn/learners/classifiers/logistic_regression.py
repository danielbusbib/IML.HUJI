from typing import NoReturn

import numpy as np

from IMLearn import BaseEstimator
from IMLearn.desent_methods import GradientDescent
from IMLearn.desent_methods.modules import LogisticModule, RegularizedModule, L1, L2


class LogisticRegression(BaseEstimator):

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]
        init = np.random.randn(X.shape[1]) / np.sqrt(X.shape[1])
        # Create module for the Gradient-Descent solver:
        if self.penalty_ == "none":
            model = LogisticModule(init.copy())
        else:
            l = L1() if self.penalty_ == "l1" else L2()
            model = RegularizedModule(LogisticModule(), l, self.lam_, init.copy(), self.include_intercept_)

        self.coefs_ = self.solver_.fit(model, X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        res = self.predict_proba(X)
        return np.where(res > self.alpha_, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]
        return 1 / (1 + np.exp(-X @ self.coefs_))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        from IMLearn.metrics.loss_functions import misclassification_error
        return misclassification_error(y, self._predict(X))
