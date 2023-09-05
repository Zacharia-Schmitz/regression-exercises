"""Evaluate Linear Regression Models"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
from scipy import stats

import matplotlib as mpl

mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["axes.formatter.limits"] = (-1_000_000, 1_000_000)


def plot_res(df, x, y, yhat):
    """
    Plots the residuals of a linear regression model.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    x (str): The name of the x-axis column.
    y (str): The name of the y-axis column.
    yhat (str): The name of the predicted y-axis column.

    Returns:
    None
    """
    # Calculate the residuals
    df = df.assign(res=(df[yhat] - df[y]))

    # Create a scatter plot of the residuals
    sns.scatterplot(data=df, x=x, y=df["res"])
    plt.show()


def reg_err(df, y, yhat, result=None):
    """
    Calculates regression metrics and returns them as a DataFrame or prints them out.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    y (str): The name of the y-axis column.
    yhat (str): The name of the predicted y-axis column.
    result (str): The type of output to return. Can be 'var' for variable outputs, 'df' for a DataFrame output, or None to print the metrics.

    Returns:
    pandas.DataFrame or None: The regression metrics as a DataFrame if result is 'df', otherwise None.
    tuple or None: The regression metrics as a tuple if result is 'var', otherwise None.
    """
    # Calculate the regression metrics
    MSE = mean_squared_error(df[y], df[yhat])
    SSE = MSE * len(df)
    RMSE = MSE**0.5
    ESS = sum((df[yhat] - df[y].mean()) ** 2)
    TSS = ESS + SSE

    # Return the metrics as a tuple if result is 'var'
    if result == "var":
        return SSE, ESS, TSS, MSE, RMSE

    # Return the metrics as a DataFrame if result is 'df'
    elif result == "df":
        return pd.DataFrame(
            {"SSE": [SSE], "ESS": [ESS], "TSS": [TSS], "MSE": [MSE], "RMSE": [RMSE]}
        )

    # Print the metrics if result is None
    else:
        print("SSE = ", SSE)
        print("ESS = ", ESS)
        print("TSS = ", TSS)
        print("MSE = ", MSE)
        print("RMSE = ", RMSE)


def bl_mean_err(df, y, yhat_bl, result=None):
    """
    Calculates the baseline regression metrics and returns them as a DataFrame or prints them out.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    y (str): The name of the y-axis column.
    yhat_bl (str): The name of the baseline predicted y-axis column.
    result (str): The type of output to return. Can be 'var' for variable outputs, 'df' for a DataFrame output, or None to print the metrics.

    Returns:
    pandas.DataFrame or None: The baseline regression metrics as a DataFrame if result is 'df', otherwise None.
    tuple or None: The baseline regression metrics as a tuple if result is 'var', otherwise None.
    """
    # Calculate the baseline regression metrics
    MSE_bl = mean_squared_error(df[y], df[yhat_bl])
    SSE_bl = MSE_bl * len(df)
    RMSE_bl = MSE_bl**0.5

    # Return the metrics as a tuple if result is 'var'
    if result == "var":
        return SSE_bl, MSE_bl, RMSE_bl

    # Return the metrics as a DataFrame if result is 'df'
    elif result == "df":
        return pd.DataFrame(
            {"SSE_bl": [SSE_bl], "MSE_bl": [MSE_bl], "RMSE_bl": [RMSE_bl]}
        )

    # Print the metrics if result is None
    else:
        print("SSE_bl = ", SSE_bl)
        print("MSE_bl = ", MSE_bl)
        print("RMSE_bl = ", RMSE_bl)


def better_than_bl(df, y, yhat, yhat_bl, metric=None):
    """
    Compares the regression metrics of a model to those of a baseline model and prints out whether the model is better.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    y (str): The name of the y-axis column.
    yhat (str): The name of the predicted y-axis column for the model.
    yhat_bl (str): The name of the predicted y-axis column for the baseline model.
    metric (str): The type of metric to use for comparison. Can be 'SSE', 'MSE', 'RMSE', or None to compare all metrics.

    Returns:
    None
    """
    # Calculate the regression metrics for the model and the baseline model
    SSE, ESS, TSS, MSE, RMSE = reg_err(df, y, yhat, "var")
    SSE_bl, MSE_bl, RMSE_bl = bl_mean_err(df, y, yhat_bl, "var")

    # Compare the metrics based on the specified metric or all metrics
    if metric == "SSE":
        print("Model SSE better: ", SSE - SSE_bl < 0)
    elif metric == "MSE":
        print("Model MSE better: ", MSE - MSE_bl < 0)
    elif metric == "RMSE":
        print("Model RMSE better: ", RMSE - RMSE_bl < 0)
    else:
        print("Model SSE better: ", SSE - SSE_bl < 0)
        print("Model MSE better: ", MSE - MSE_bl < 0)
        print("Model RMSE better: ", RMSE - RMSE_bl < 0)
