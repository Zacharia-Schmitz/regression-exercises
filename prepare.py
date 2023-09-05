import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def robust_scaler(df, cols=None):
    """
    Applies a RobustScaler to all numeric columns in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The input dataframe with scaled numeric columns.
    """
    if cols == None:
        # Select only the numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        # Create a RobustScaler object
        scaler = RobustScaler()

        # Scale the numeric columns and create a new DataFrame
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        # Create a RobustScaler object
        scaler = RobustScaler()

        # Scale the numeric columns and create a new DataFrame
        df[cols] = scaler.fit_transform(df[cols])

    # Return the modified dataframe
    return df


def encode_county(df):
    """
    Encodes the 'county' column of a pandas DataFrame as one-hot columns for each county.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The input dataframe with one-hot columns for each county.
    """
    # Create one-hot columns for each county
    df = pd.concat([df, pd.get_dummies(df["county"])], axis=1)

    # Drop the original 'county' column
    df = df.drop("county", axis=1)

    df.rename(
        columns={"LA": "los_angeles", "Orange": "orange", "Ventura": "ventura"},
        inplace=True,
    )

    # Return the modified dataframe
    return df


def split_data(df):
    """Split into train, validate, test with a 60% train, 20% validate, 20% test"""
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    print(f"train: {len(train)} ({round(len(train)/len(df)*100)}% of {len(df)})")
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df)*100)}% of {len(df)})"
    )
    print(f"test: {len(test)} ({round(len(test)/len(df)*100)}% of {len(df)})")
    return train, validate, test
