from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics.loss_functions import mean_square_error
import statsmodels.api as sm
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df["zipcode"] = df["zipcode"].astype(int)

    df["grade"] = df["grade"].astype(int)
    df = df[df["grade"] >= 1]

    # remove unimportant data fot fit
    for col in ["date", "lat", "id", "long"]:
        df = df.drop(col, axis=1)

    # only positive
    for col in ["sqft_living", "price", "sqft_above",
                "sqft_lot", "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[col] > 0]

    for col in ["bathrooms", "sqft_basement", "floors", "yr_renovated"]:
        df = df[df[col] >= 0]

    # Handling yr_renovated column
    df["is_renovated"] = df['yr_renovated'].apply(lambda y: 1 if y > 0 else 0)
    df = df.drop("yr_renovated", axis=1)

    # Decade built House
    df["Decade"] = df["yr_built"] / 10
    df["Decade"] = df["Decade"].astype(int)
    df = df.drop("yr_built", axis=1)

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='Decade_', columns=['Decade'])

    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1300000]
    df = df[df["waterfront"].isin([0, 1])]

    df.insert(0, 'intercept', 1, True)
    return df.drop("price", axis=1), df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.drop("intercept", axis=1)
    # P = ["condition"]
    for col in X:
        if "zipcode" in col or "Decade" in col:
            continue
        # calculate pearson corr
        rho = np.cov(X[col], y)[0, 1] / (np.std(X[col]) * np.std(y))

        fig = px.scatter(pd.DataFrame({'x': X[col], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {col} Values and Response <br>Pearson Correlation {rho}",
                         labels={"x": f"{col} Values", "y": "Response Values"})
        fig.write_image(f"pearson.correlation.{col}.png")


if __name__ == '__main__':
    # np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, Y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, Y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, Y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    y_plot, y_std = [], []
    frames = []
    x = []
    linear_regressor = LinearRegression(True)
    for i in range(10, 101):
        p = i / 100
        n = round(len(train_y) * p)
        loss = []
        x.append(i)
        print(i)
        # repeat 10 time over p:
        for j in range(10):
            sample_X = train_X.sample(n=n, replace=False)
            linear_regressor._fit(sample_X.to_numpy(), train_y.reindex_like(sample_X).to_numpy())
            y_pred = linear_regressor._predict(test_X.to_numpy())
            mse = mean_square_error(test_y.to_numpy(), y_pred)
            loss.append(mse)
        # save average and variance of loss over test set
        y_plot.append(np.mean(loss))
        y_std.append(np.std(loss))

    frames.append(go.Frame(
        # data=[
        #     go.Scatter(x=x, y=y_plot, mode="markers+lines",
        #                name="Real Points", marker=dict(color="black", opacity=.7))],
        layout=go.Layout(
            title="MEAN LOSS AS A FUNCTION OF P% of training set WITH CONFIDENCE INTERVAL(MEAN +- 2 * STD)",
            xaxis={"title": r"$p$"},
            yaxis={"title": r"$MEAN MSE over test set$"})))

    #
    std_pred = np.array(y_std)
    y_plot = np.array(y_plot)
    for i in range(len(frames)):
        frames[i]["data"] = (go.Scatter(x=x, y=y_plot, mode="markers+lines", name="Mean Prediction",
                                        marker=dict(color="black", opacity=.7)),
                             go.Scatter(x=x, y=y_plot - 2 * std_pred, fill=None, mode="lines",
                                        line=dict(color="lightblue"), showlegend=False),
                             go.Scatter(x=x, y=y_plot + 2 * std_pred, fill='tonexty', mode="lines",
                                        line=dict(color="lightblue"), showlegend=False),) + frames[i]["data"]

    fig = go.Figure(data=frames[0]["data"],
                    frames=frames[1:],
                    layout=go.Layout(
                        title=frames[0]["layout"]["title"],
                        xaxis=frames[0]["layout"]["xaxis"],
                        yaxis=frames[0]["layout"]["yaxis"],
                        updatemenus=[dict(visible=True,
                                          type="buttons")]))

    fig.write_image(f"mean.loss.as.p%.of.training.data.png")

    fig.show()
