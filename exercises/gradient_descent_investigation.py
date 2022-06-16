import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly
import warnings
from utils import custom


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def call_back(model, **kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
        return

    return call_back, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    def plot_norm(x, y, tit):
        plt.plot(x, y)
        plt.title(tit), plt.xlabel("Num of iteration"), plt.ylabel("Norm")
        plt.grid()
        plt.show()

    for eta in etas:
        # L2 MODULE
        l2 = L2(init.copy())
        c = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(eta), out_type="best", callback=c[0])
        w = gd.fit(l2, X=None, y=None)
        plotly.offline.plot(plot_descent_path(L2, np.array(c[2]), title=f"| module L2 | eta={eta}"))
        plot_norm(list(range(len(c[1]))), c[1],
                  f"L2  norm as a function of the GD iteration | eta={eta}")
        l2.weights = w
        print(f"eta: {eta}")
        print(f"module: L2, lowest error: {l2.compute_output()}")

        # L1 MODULE
        l1 = L1(init.copy())
        c = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(eta), out_type="best", callback=c[0])
        w = gd.fit(l1, X=None, y=None)
        plotly.offline.plot(plot_descent_path(L1, np.array(c[2]), title=f"| module L1 | eta={eta}"))
        plot_norm(list(range(len(c[1]))), c[1],
                  f"L1 norm as a function of the GD iteration | eta={eta}")
        l1.weights = w
        print(f"module: L1, lowest error: {l2.compute_output()}", end='\n')


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    c_rate = []
    d = []
    for g in gammas:
        c = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=g),
                             out_type="best", callback=c[0])
        l1 = L1(init.copy())
        w = gd.fit(l1, X=None, y=None)
        c_rate.append(c[1])
        if g == .95:
            d = c[2]

    # Plot algorithm's convergence for the different values of gamma
    plt.plot(list(range(len(c_rate[0]))), c_rate[0])
    plt.plot(list(range(len(c_rate[1]))), c_rate[1])
    plt.plot(list(range(len(c_rate[2]))), c_rate[2])
    plt.plot(list(range(len(c_rate[3]))), c_rate[3])
    plt.title("convergence rate for all decay rates"), plt.xlabel("Num of iteration"), plt.ylabel("Norm")
    plt.legend(["gamma 0.9", "gamma 0.95", "gamma 0.99", "gamma 1"])
    plt.grid()
    plt.show()

    print(f"exponentially decay - l1 lowest norm: {np.min([np.min(c_rate[i]) for i in range(4)])}")
    # Plot descent path for gamma=0.95
    plotly.offline.plot(plot_descent_path(L1, np.array(d)))
    plotly.offline.plot(plot_descent_path(L2, np.array(d)))


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    from IMLearn.model_selection.cross_validate import cross_validate
    from IMLearn.metrics.loss_functions import misclassification_error
    from sklearn.metrics import roc_curve, auc

    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    # Plotting convergence rate of logistic regression over SA heart disease data
    lg = LogisticRegression()
    lg._fit(X_train, y_train)
    # plot ROC curve - taken from LAB 04
    fpr, tpr, thresholds = roc_curve(y_train, lg.predict_proba(np.c_[np.ones(len(X_train)), X_train]))
    c = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - Logistic Regression}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    a_star = round(thresholds[np.argmax(tpr - fpr)], 2)
    lg.alpha_ = a_star
    print("-----------")
    print(f"a* = {a_star}")
    print(f"model test error: {lg._loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lg1 = LogisticRegression(penalty="l1", alpha=0.5)
    lg2 = LogisticRegression(penalty="l2", alpha=0.5)

    lamdas = np.linspace(0.0001, 2, num=50)
    trains_err_1, valid_err_1 = [], []
    for i in lamdas:
        lg1.lam_, lg2.lam_ = i, i
        t1, v1 = cross_validate(lg1, X_train, y_train, misclassification_error)
        trains_err_1.append(t1), valid_err_1.append(v1)
    best_lam1 = lamdas[np.argmin(valid_err_1)]
    lg1.lam = best_lam1
    lg1._fit(X_train, y_train)
    test_error = lg1.loss(X_test, y_test)
    print("-----------")
    print(f"module L1 | lambda chosen: {best_lam1}")
    print(f"model test error: {test_error}")
    trains_err_2, valid_err_2 = [], []
    for i in lamdas:
        lg2.lam_ = i
        t2, v2 = cross_validate(lg2, X_train, y_train, misclassification_error)
        trains_err_2.append(t2), valid_err_2.append(v2)
    best_lam2 = lamdas[np.argmin(valid_err_2)]
    lg2.lam_ = best_lam2
    test_error = lg2.loss(X_test, y_test)
    print("-----------")
    print(f"module L2 | lambda chosen: {best_lam2}")
    print(f"model test error: {test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    warnings.filterwarnings('ignore')
    fit_logistic_regression()
