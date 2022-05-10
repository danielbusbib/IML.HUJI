import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    test_err = []
    train_err = []
    for max_t in range(n_learners):
        test_err.append(adaboost.partial_loss(test_X, test_y, max_t))
        train_err.append(adaboost.partial_loss(train_X, train_y, max_t))
    # plotting
    x = [i for i in range(n_learners)]
    plt.plot(x, test_err)
    plt.plot(x, train_err)
    plt.title(f"Train and Test Errors as a function of the number of fitted learners"
              f" | model noise ratio:{noise}"), plt.xlabel("Fitted Learners"), plt.ylabel("Error")
    plt.legend(["Test Error", "Train Error"])
    plt.grid()
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,horizontal_spacing=0.01, vertical_spacing=0.03,
                        subplot_titles=[f"{t} Classifiers" for t in T])
    # each T:
    err=[]
    for j, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, t),
                                         *lims, showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=custom,
                                               line=dict(width=0.5, color="DarkSlateGrey"))
                                   ),
                        ],
                       rows=(j // 2) + 1, cols=(j % 2) + 1)
        err.append(adaboost.partial_loss(test_X, test_y, t))
    print(err)
    fig.update_layout(title=f"Decision Boundary , noise: {noise}",
                      margin_t=100)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"adaboost.decision.boundary.all.iterations.noise.{noise}.png")
    # fig.show()

    # Question 3: Decision surface of best performing ensemble
    min_err = min(test_err)
    min_size = test_err.index(min_err) + 1
    fig = go.Figure([decision_surface(lambda X: adaboost.partial_predict(X, min_size),
                                      *lims, showscale=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                showlegend=False,
                                marker=dict(color=test_y,
                                            colorscale=custom,
                                            )
                                ),
                     ])
    fig.update_layout(title=f"Decision Boundary | Best ensemble size: {min_size} | noise: {noise}"
                            f" | accuracy = {1 - min_err}")
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"adaboost.decision.boundary.best.size.ensemble.noise.{noise}.png")
    # fig.show()

    # Question 4: Decision surface with weighted samples
    mk_size = 45 if noise == 0 else 10
    fig1 = go.Figure([decision_surface(adaboost.predict, *lims, showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color=train_y,
                                             colorscale=[custom[0], custom[-1]],
                                             size=adaboost.D_,
                                             sizeref=2 * np.max(adaboost.D_) / (mk_size ** 2),
                                             sizemode="area",
                                             sizemin=0.5,
                                             line=dict(width=1, color="DarkSlateGrey")
                                             )
                                 ),
                      ])
    fig1.update_layout(title=f"noise={noise}: "
                             f"Decision Boundary with weighted train ensemble")
    fig1.update_xaxes(visible=False)
    fig1.update_yaxes(visible=False)
    fig1.write_image(f"adaboost.decision.boundary.weighted.noise.{noise}.png")
    # fig1.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
