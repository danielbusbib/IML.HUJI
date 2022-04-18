from typing import NoReturn
# from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv, LinAlgError

from IMLearn.learners import MultivariateGaussian

from IMLearn import BaseEstimator


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # separate class
        self.classes_ = np.unique(y)

        # mu vector for each i in {1,...,k}
        mu = []
        for k in self.classes_:
            mu.append(X[y == k].mean(axis=0))
        self.mu_ = np.array(mu)
        # covariance matrix
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            self.cov_ += (X[y == self.classes_[i]] - self.mu_[i]).T @ (X[y == self.classes_[i]] - self.mu_[i])
        self.cov_ /= y.size

        self._cov_inv = np.linalg.inv(self.cov_)

        self.pi_ = np.array([(y == label).mean() for label in self.classes_])
        self.fitted_ = True

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
        res = []
        for i in range(X.shape[0]):
            lst = []
            for k in range(self.classes_.shape[0]):
                p = (X[i].T @ self._cov_inv @ self.mu_[k]) - 0.5 * self.mu_[k] @ \
                    self._cov_inv @ self.mu_[k]
                p += np.log(self.pi_[k])
                lst.append(p)
            # argmax k
            res.append(self.classes_[lst.index(max(lst))])
        return np.array(res)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        res = []

        for i in range(X.shape[0]):
            lst = []
            for k in range(self.classes_.shape[0]):
                p = np.exp(-0.5 * ((X[i] - self.mu_[k]) @ np.linalg.inv(self.cov_) @ (X[i] - self.mu_[k]).T))\
                    / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.cov_))
                p *= self.pi_[k]
                lst.append(p)

            # argmax k
            res.append(lst)

        return np.array(res)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(self._predict(X), y)
