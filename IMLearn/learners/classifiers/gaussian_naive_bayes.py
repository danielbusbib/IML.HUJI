import math
from typing import NoReturn

from IMLearn import BaseEstimator
import numpy as np

from IMLearn.learners import MultivariateGaussian, UnivariateGaussian


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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

        # vars
        vars = []
        for k in self.classes_:
            vars.append(X[y == k].var(axis=0))
        self.vars_ = np.array(vars)

        # pi
        self.pi_ = np.array([(y == label).mean() for label in self.classes_])
        self.fitted_ = True

    def gaussian_pdf(self, X, mu, var):
        """
        calculate pdf on x
        :param x: sample
        :return:
        """
        return np.exp(- (X - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

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
            for j in range(self.classes_.shape[0]):

                product = self.pi_[j]
                for k in range(X[i].shape[0]):
                    product *= self.gaussian_pdf(X[i, k], self.mu_[j, k], self.vars_[j, k])

                lst.append(product)
            # argmax y
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
            for j in range(self.classes_.shape[0]):

                product = self.pi_[j]
                for k in range(X[i].shape[0]):
                    product *= self.gaussian_pdf(X[i, k], self.mu_[j, k], self.vars_[j, k])
                lst.append(product)

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
        return misclassification_error(y, self._predict(X))
