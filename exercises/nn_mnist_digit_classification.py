import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


def callback_func(**kwargs):
    values, weights, grads = [], [], []

    def callback(**kwargs):
        values.append(kwargs["val"])
        grads.append(np.linalg.norm(kwargs["grad"], ord=2))
        if int(kwargs["t"]) % 100 == 0:
            weights.append(kwargs["weights"])

    return callback, values, weights, grads


import plotly.offline

if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    neurons = 64
    callback, values, weights, grads_norm = callback_func()
    relu1, relu2, lr = ReLU(), ReLU(), FixedLR(0.1)
    loss = CrossEntropyLoss()
    layer_one = FullyConnectedLayer(input_dim=len(train_X[0]), output_dim=neurons, activation=relu1)
    hidden_one = FullyConnectedLayer(input_dim=neurons, output_dim=neurons, activation=relu2)
    hidden_two = FullyConnectedLayer(input_dim=neurons, output_dim=10)
    st_gradient = StochasticGradientDescent(learning_rate=lr, max_iter=10000, batch_size=256, callback=callback)
    nn = NeuralNetwork(modules=[layer_one, hidden_one, hidden_two], loss_fn=loss, solver=st_gradient)
    nn._fit(train_X, train_y)
    print("----- q5 -----")
    print(f"accuracy over test = {accuracy(test_y, nn._predict(test_X))}")

    # Plotting convergence process
    plt.title(f"loss for iteration with hidden layers with 64 neurons")
    plt.plot(list(range(len(values))), values, label="loss")
    plt.plot(list(range(len(grads_norm))), grads_norm, label="gradient norm")
    plt.grid(), plt.legend()
    plt.xlabel("iteration")
    plt.show()

    # Plotting test true- vs predicted confusion matrix
    print(confusion_matrix(test_y, nn._predict(test_X)))

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    callback2, values2, weights2, grads_norm2 = callback_func()
    layer_one = FullyConnectedLayer(input_dim=len(train_X[0]), output_dim=10, activation=ReLU(),
                                    include_intercept=True)
    st_gradient2 = StochasticGradientDescent(learning_rate=lr, max_iter=10000, batch_size=256, callback=callback)
    nn2 = NeuralNetwork(modules=[layer_one], loss_fn=loss, solver=st_gradient2)
    nn2._fit(train_X, train_y)
    print(f"accuracy over test = {accuracy(test_y, nn._predict(test_X))}")

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    sevens = test_X[test_y == 7]
    confidence = np.max(nn.compute_prediction(sevens), axis=1)
    sevens = np.c_[confidence, sevens]
    sevens = sevens[sevens[:, 0].argsort()]
    sevens = sevens[:, 1:]
    fig = plot_images_grid(sevens[:64], "low confidence")
    fig.write_image("low_cof.png")
    plotly.offline.plot(fig)
    fig = plot_images_grid(sevens[-64:], "high confidence")
    fig.write_image("high_cof.png")
    plotly.offline.plot(fig)

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    gd_time, gd_losses, sgd_time, sgd_losses = list(), list(), list(), list()
    lr = FixedLR(0.1)

    def gd_time_callback(solver, **kwargs):
        gd_time.append(time.time())
        gd_losses.append(kwargs["val"])


    def sgd_time_callback(**kwargs):
        sgd_time.append(time.time())
        sgd_losses.append(kwargs["val"])


    batch, train_size, max_iter = 64, 2500, 10000
    sgd = StochasticGradientDescent(max_iter=max_iter, tol=10 ** -10, learning_rate=lr, batch_size=batch,
                                    callback=sgd_time_callback)
    sgd_one = FullyConnectedLayer(input_dim=len(train_X[0]), output_dim=neurons, activation=ReLU())
    sgd_two = FullyConnectedLayer(input_dim=neurons, output_dim=neurons, activation=ReLU())
    sgd_three = FullyConnectedLayer(input_dim=neurons, output_dim=10)
    sgd_model = NeuralNetwork(modules=[sgd_one, sgd_two, sgd_three], loss_fn=CrossEntropyLoss(), solver=sgd)
    sgd_model.fit(X=train_X[:train_size], y=train_y[:train_size])
    gd = GradientDescent(max_iter=max_iter, tol=10 ** -10, learning_rate=lr, callback=gd_time_callback)
    gd_one = FullyConnectedLayer(input_dim=len(train_X[0]), output_dim=neurons, activation=ReLU())
    gd_two = FullyConnectedLayer(input_dim=neurons, output_dim=neurons, activation=ReLU())
    gd_three = FullyConnectedLayer(input_dim=neurons, output_dim=10)
    gd_model = NeuralNetwork(modules=[gd_one, gd_two, gd_three], loss_fn=CrossEntropyLoss(), solver=gd)
    gd_model.fit(train_X[:train_size], train_y[:train_size])
    plt.title(f"loss for time gd")
    plt.plot(gd_time, gd_losses, label="loss")
    plt.xlabel("time")
    plt.savefig("time_loss_gd.png")
    plt.title(f"loss for time sgd")
    plt.plot(gd_time, gd_losses, label="loss")
    plt.xlabel("time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gd_time, y=gd_losses,
                             mode='markers',
                             name='gd'))
    fig.add_trace(go.Scatter(x=sgd_time, y=sgd_losses,
                             mode='markers',
                             name='sgd'))
    plotly.offline.plot(fig)
