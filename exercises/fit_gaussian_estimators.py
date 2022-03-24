from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from matplotlib import pyplot as plt

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)
    un = UnivariateGaussian()
    un.fit(samples)
    print("(", un.mu_, ",", un.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    lst, n = [], 0
    for i in range(10, 1010, 10):
        # samples = np.random.normal(mu, sigma, i)
        un.fit(samples[:i])
        lst.append([i, abs(un.mu_ - mu)])
        n += 1
    data = np.array(lst[:])
    x, y = data.T
    plt.title("Absolute distance between the estimated- and true value of the expectation")
    plt.xlabel("Sample Size")
    plt.ylabel("Abs distance")
    plt.scatter(x, y)
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    y = un.pdf(samples)
    plt.title("Empirical PDF of fitted model")
    plt.xlabel("Sample Value")
    plt.ylabel("Gaussian PDF")
    plt.scatter(samples, y)
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4.0, 0]
    cov = np.array([[1.0, 0.2, 0.0, 0.5],
                    [0.2, 2.0, 0, 0],
                    [0, 0, 1.0, 0],
                    [0.5, 0, 0, 1.0]])
    samples_mn = np.random.multivariate_normal(mu, cov, 1000)
    mn = MultivariateGaussian()
    mn.fit(samples_mn)
    print(mn.mu_)
    print(mn.cov_)

    # Question 5 - Likelihood evaluation
    f_line = np.linspace(-10, 10, 200)
    S = 200 * 200
    data = np.ndarray(S)
    x, y = np.zeros(S), np.zeros(S)
    c = 0
    max_val = MultivariateGaussian.log_likelihood(np.array([f_line[0], 0, f_line[0], 0]), cov, samples_mn)
    max_f = (f_line[0], f_line[0])
    for i in range(200):
        for j in range(200):
            data[c] = MultivariateGaussian.log_likelihood(np.array([f_line[j], 0, f_line[i], 0]), cov, samples_mn)
            if data[c] > max_val:
                max_val = data[c]
                max_f = f_line[i], f_line[j]
            x[c], y[c] = f_line[i], f_line[j]
            c += 1

    plt.scatter(x, y, c=data)
    plt.title("Multivariate Likelihood evaluation - mu[f1, 0, f3, 0]")
    plt.xlabel("f3")
    plt.ylabel("f1")
    plt.colorbar()
    plt.show()

    # Question 6 - Maximum likelihood
    print(max_f)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
