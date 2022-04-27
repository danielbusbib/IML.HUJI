from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

from matplotlib import pyplot as plt
import plotly.express as px


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_func(fit: Perceptron, x: np.ndarray, y_: int):
            losses.append(fit._loss(X, y))


        d = list(range(1, 1001))
        p = Perceptron(callback=callback_func)
        p._fit(X, y)

        # Plot figure of loss as function of fitting iteration
        if len(d) != len(losses):
            for _ in range(len(d) - len(losses)):
                losses.append(0)

        fig = px.line(x=d, y=losses, markers=True,
                      title=f"Perceptron Model - Loss as function of fitting iteration"
                            f"on {f} Dataset").\
            update_layout(xaxis_title="NUMBER OF ITERATIONS", yaxis_title="LOSS")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA()
        gne = GaussianNaiveBayes()
        gne._fit(X, y)
        lda._fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            horizontal_spacing=0.01, vertical_spacing=.9,
                            subplot_titles=[f"Classifier: {m[0]}"
                                            f"  ||  Accuracy: {m[1]}"
                                            for m in [("Gaussian Naive Bayes", accuracy(y, gne._predict(X))),
                                                      ("LDA", accuracy(y, lda._predict(X)))]])

        # Gaussian Naive Bayes PLOT:
        pr = gne._predict(X)
        for c in ['0', '1', '2']:
            V = X[y == int(c)]
            fig.add_trace(
                go.Scatter(x=V[:, 0], y=V[:, 1], mode="markers", name=f"Shape True value - {c}",
                           marker=dict(color='white', symbol=c, size=10, line=dict(color="black", width=2)),
                           ),
                row=1, col=1
            )

        for c in ['0', '1', '2']:
            P = X[pr == int(c)]
            fig.add_trace(
                go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", name=f"Color Predicted value - {c}",
                           marker=dict(color=['blue', 'yellow', 'red'][int(c)], size=10,
                                       line=dict(color="black", width=2),
                                       ),
                           ),
                row=1, col=1
            )

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=gne.mu_[:, 0], y=gne.mu_[:, 1], mode="markers", name='mean',
                       marker=dict(color='black', symbol=['x', 'x', 'x'], size=13,
                                   line=dict(color="black", width=2),
                                   ),
                       ),
            row=1, col=1
        )

        # Add ellipses depicting the covariances of the fitted Gaussians
        covs = np.zeros((2, 2))
        covs[0, 0] = gne.vars_[0, 0]
        covs[1, 1] = gne.vars_[0, 1]
        fig.add_trace(get_ellipse(gne.mu_[0], covs), row=1, col=1)
        covs[0, 0] = gne.vars_[1, 0]
        covs[1, 1] = gne.vars_[1, 1]
        fig.add_trace(get_ellipse(gne.mu_[1], covs), row=1, col=1)
        covs[0, 0] = gne.vars_[2, 0]
        covs[1, 1] = gne.vars_[2, 1]
        fig.add_trace(get_ellipse(gne.mu_[2], covs), row=1, col=1)

        # LDA PLOT:
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=lda._predict(X), symbol=y, size=8, line=dict(color="black", width=2),
                                   colorscale=['blue', 'yellow', 'red']),
                       ),
            row=1, col=2
        )

        # Add `X` dots specifying fitted Gaussian's' means
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers", showlegend=False,
                       marker=dict(color='black', symbol=['x', 'x', 'x'], size=13,
                                   line=dict(color="black", width=5),
                                   ),
                       ),
            row=1, col=2
        )

        # Add ellipses depicting the covariances of the fitted Gaussian's
        fig.add_trace(get_ellipse(lda.mu_[0], lda.cov_), row=1, col=2)
        fig.add_trace(get_ellipse(lda.mu_[1], lda.cov_), row=1, col=2)
        fig.add_trace(get_ellipse(lda.mu_[2], lda.cov_), row=1, col=2)

        # layout
        fig.update_layout(title_text=f"{f} Dataset",
                          title_x=0.5,
                          font=dict(
                              family="Courier New, monospace",
                              size=16,
                              color="RebeccaPurple"
                          )
                          )
        # fig.write_html(f"{f}.Fit.Gaussian.And.LDA.png")
        fig.show()

        # Add traces for data-points setting symbols and colors


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
