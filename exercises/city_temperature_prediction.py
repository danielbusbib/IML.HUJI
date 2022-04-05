import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(str)

    df = df[1 <= df['Month']]
    df = df[12 >= df['Month']]
    df = df[df['Day'] >= 1]
    df = df[df['Day'] <= 31]
    df = df[df['Temp'] > -22]

    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')
    X, y = df.drop('Temp', axis=1), df['Temp']

    # Question 2 - Exploring data for specific country
    israel_data = df[df['Country'] == 'Israel']
    # israel_data["Year"] = israel_data["Year"].astype(str)
    fig = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                     title="Average daily temperature  change as a function of the Day of Year.")

    fig.show()

    dd = israel_data.groupby('Month')['Temp'].std()
    fig = px.bar(dd, x=list(range(1, 13)), y='Temp',
                 title=f"Standard deviation of the daily temperatures of each Month",
                 labels={"x": f"Month", "y": "std of Temperature"})
    fig.show()

    # Question 3 - Exploring differences between countries
    df_3 = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp': ['mean', 'std']})
    idx = pd.IndexSlice
    ds = pd.DataFrame()
    ds['Country'] = df_3.loc[:, idx['Country', :]]['Country']
    ds['Month'] = df_3.loc[:, idx['Month', :]]['Month']
    ds['Temp_mean'] = df_3.loc[:, idx[:, 'mean']]
    ds['Temp_std'] = df_3.loc[:, idx[:, 'std']]
    fig = px.line(ds, x='Month', y='Temp_mean',
                  color='Country', error_y='Temp_std',
                  title=f"Average monthly temperature with error bars of standard deviation",
                  labels={"x": f"Month", "y": f"Temperature mean"})
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_data['DayOfYear'], israel_data['Temp'])
    loss = []
    for k in range(1, 11):
        pol = PolynomialFitting(k)
        pol._fit(train_X.to_numpy(), train_y.to_numpy())
        mse = mean_square_error(test_y.to_numpy(), pol._predict(test_X.to_numpy()))
        loss.append(mse)
        print('k:', k, ', test error:', mse)
    fig = px.bar(x=list(range(1, 11)), y=loss,
                 title=f"Loss of the polynomial model of degree k over the test set",
                 labels={"x": f"K", "y": "Loss"})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    pol = PolynomialFitting(5)
    loss = []
    pol._fit(israel_data['DayOfYear'].to_numpy(), israel_data['Temp'].to_numpy())
    COUNTRIES = ['Israel','South Africa', 'Jordan', 'The Netherlands']
    for c in COUNTRIES:
        mse = mean_square_error(df[df['Country'] == c]['Temp'].to_numpy()
                                , pol._predict(df[df['Country'] == c]['DayOfYear'].to_numpy()))
        loss.append(mse)

    fig = px.bar(x=COUNTRIES, y=loss,
                 title=f"Loss of the polynomial model of degree 5 over the Country set",
                 labels={"x": f"COUNTRY", "y": "Loss"})
    fig.show()
