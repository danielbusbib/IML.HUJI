from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso, Ridge

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(-1.2, 2, n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    epsilon = np.random.normal(0, noise, n_samples)
    y = f(X) + epsilon
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=2 / 3)
    train_X = train_X.to_numpy().reshape((train_X.shape[0],))
    train_y = train_y.to_numpy().reshape((train_y.shape[0],))
    test_X = test_X.to_numpy().reshape((test_X.shape[0],))
    test_y = test_y.to_numpy().reshape((test_y.shape[0],))
    fig = go.Figure([go.Scatter(x=X, y=f(X), mode="markers", name="true model (noiseless)"),
                     go.Scatter(x=train_X, y=train_y, mode="markers", name="train model"),
                     go.Scatter(x=test_X, y=test_y, mode="markers", name="test model")])
    fig.update_layout(title=f"Polynomial model| Num Samples: {n_samples} | noise: {noise}")
    fig.write_image(f"polynomial.model.selection.{n_samples}.samples.and.{noise}.noise.png")
    # fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k = 10
    x = list(range(k + 1))
    trains_s, valid_s = [], []
    for i in range(k + 1):
        train_score, validation_error = cross_validate(
            PolynomialFitting(i), train_X, train_y, mean_square_error, cv=5)
        trains_s.append(train_score)
        valid_s.append(validation_error)

    plt.plot(x, valid_s, label="Validation Error")
    plt.plot(x, trains_s, label="Training Error")
    plt.title(f"POLYNOMIAL | VALID ERROR AND TRAINING ERROR | NUM SAMPLES = {n_samples}"
              f" | NOISE = {noise}")
    plt.legend(), plt.xlabel("K"), plt.ylabel("MSE error")
    plt.grid()
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = valid_s.index(min(valid_s))
    poly_train = PolynomialFitting(k_star).fit(train_X, train_y)
    print(f"num of samples:{n_samples} | noise:{noise}")
    print("k* = ", k_star)
    print("test error = ", round(mean_square_error(test_y, poly_train.predict(test_X)), 2))


def run_ridge(lamdas, train_X, train_y, n):
    trains_s, valid_s = [], []
    for i in lamdas:
        train_score, validation_error = cross_validate(
            RidgeRegression(lam=i), train_X, train_y, mean_square_error, cv=5)
        trains_s.append(train_score)
        valid_s.append(validation_error)
    plt.plot(lamdas, valid_s, label="Validation Error")
    plt.plot(lamdas, trains_s, label="Training Error")
    plt.title(f"Ridge Model| VALID- AND TRAINING- ERROR | NUM SAMPLES = {n}")
    plt.legend(), plt.xlabel("Lamda"), plt.ylabel("MSE error")
    plt.grid()
    plt.show()
    return valid_s


def run_lasso(lamdas, train_X, train_y, n):
    trains_s, valid_s = [], []
    for i in lamdas:
        train_score, validation_error = cross_validate(
            Lasso(alpha=i), train_X, train_y, mean_square_error, cv=5)
        trains_s.append(train_score)
        valid_s.append(validation_error)
    plt.plot(lamdas, valid_s, label="Validation Error")
    plt.plot(lamdas, trains_s, label="Training Error")
    plt.title(f"Lasso Model | VALID- AND TRAINING- ERROR | NUM SAMPLES = {n}")
    plt.legend(), plt.xlabel("Lamda"), plt.ylabel("MSE error")
    plt.grid()
    plt.show()
    return valid_s


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    n = 50
    train_X, train_y, test_X, test_y = X[:n], y[:n], X[n:], y[n:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamdas = np.linspace(0, 1.8, num=n_evaluations)
    valid_ridge = run_ridge(lamdas, train_X, train_y, n_samples)
    valid_lasso = run_lasso(lamdas, train_X, train_y, n_samples)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = lamdas[np.argmin(valid_ridge)]
    best_lasso = lamdas[np.argmin(valid_lasso)]
    print("Best Regularization Term:")
    print(f"Lasso: {best_lasso}")
    print(f"Ridge: {best_ridge}")
    ridge = RidgeRegression(lam=best_ridge).fit(train_X, train_y)
    lasso = Lasso(alpha=best_lasso).fit(train_X, train_y)
    linear_regression = LinearRegression().fit(train_X, train_y)
    print("---------------------------")
    print("Test Error of fitted model:")
    print(f"Lasso: {mean_square_error(test_y, lasso.predict(test_X))}")
    print(f"Ridge: {mean_square_error(test_y, ridge.predict(test_X))}")
    print(f"Linear Regression: {linear_regression.loss(test_X, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    # part 1
    select_polynomial_degree(noise=5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    # part 2
    select_regularization_parameter()
