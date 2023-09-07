import os
import pandas as pd
import env as e
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def wrangle_zillow():
    """
    This function reads the Zillow data from a cached CSV file if it exists,
    or from a SQL database if it doesn't exist. It then renames the columns
    to more descriptive names.

    Args:
    - None

    Returns:
    - pandas dataframe
    """
    # Name of cached CSV file
    filename = "zillow.csv"
    # If cached data exists, read from CSV file
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    # Otherwise, read from SQL database
    else:
        df = pd.read_sql(
            """SELECT yearbuilt,
                                   bedroomcnt,
                                   bathroomcnt,
                                   calculatedfinishedsquarefeet,
                                   taxvaluedollarcnt,
                                   fips
                            FROM properties_2017
                            WHERE propertylandusetypeid = 261""",  # 261 is single family residential id
            f"mysql+pymysql://{e.user}:{e.password}@{e.host}/zillow",
        )
        # Cache data locally
        df.to_csv(filename, index=False)
    # Rename columns
    df = df.rename(
        columns={
            "yearbuilt": "year",
            "bedroomcnt": "beds",
            "bathroomcnt": "baths",
            "calculatedfinishedsquarefeet": "sqfeet",
            "taxvaluedollarcnt": "prop_value",
            "taxamount": "prop_tax",
            "fips": "county",
        }
    )

    # Drop rows with missing values in specific columns
    df = df.dropna(subset=["year", "beds", "baths", "sqfeet", "prop_value", "prop_tax"])

    # Map county codes to county names
    county_map = {6037: "LA", 6059: "Orange", 6111: "Ventura"}
    df["county"] = df["county"].replace(county_map)

    # Convert columns to int data type
    df = df.astype(
        {"year": int, "beds": int, "sqfeet": int, "prop_value": int, "prop_tax": int}
    )
    return df


def check_columns(df_telco):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, the proportion of null values,
    and the data type of the column. The resulting dataframe is sorted by the
    'Number of Unique Values' column in ascending order.

    Args:
    - df_telco: pandas dataframe

    Returns:
    - pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df_telco.columns:
        # Append the column name, number of unique values, unique values, number of null values, proportion of null values, and data type to the data list
        data.append(
            [
                column,
                df_telco[column].nunique(),
                df_telco[column].unique(),
                df_telco[column].isna().sum(),
                df_telco[column].isna().mean(),
                df_telco[column].dtype,
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', 'Proportion of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "Proportion of Null Values",
            "dtype",
        ],
    )


from sklearn.preprocessing import QuantileTransformer


def quantiler(train, validate, test):
    """
    This function scales the train, validate, and test data using QuantileTransformer.
    It returns the scaled versions of each.
    Be sure to only learn the parameters for scaling from your training data!

    Parameters:
    -----------
    train : pandas DataFrame
        The training dataset to be scaled.
    validate : pandas DataFrame
        The validation dataset to be scaled.
    test : pandas DataFrame
        The test dataset to be scaled.

    Returns:
    --------
    train : pandas DataFrame
        The scaled training dataset.
    validate : pandas DataFrame
        The scaled validation dataset.
    test : pandas DataFrame
        The scaled test dataset.

    Example:
    --------
    # call the function
    train_scaled, validate_scaled, test_scaled = quantiler(train, validate, test)
    """
    # create an instance of QuantileTransformer with normal distribution
    qt = QuantileTransformer(output_distribution="normal")

    # define the columns to be scaled
    scale = ["year", "beds", "baths", "sqfeet", "prop_value", "prop_tax"]

    # fit and transform the train data
    train[scale] = qt.fit_transform(train[scale])

    # transform the validate and test data
    validate[scale] = qt.transform(validate[scale])
    test[scale] = qt.transform(test[scale])

    # return the scaled data
    return train, validate, test


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


def box_plotter(df):
    """
    Generates a box plot for all columns in a dataframe using matplotlib.
    """
    for col in df.columns:
        try:
            plt.figure(figsize=(6, 1))
            plt.boxplot(df[col], vert=False)
            plt.title(col)
            plt.show()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            print(
                f"Number of results in lower quartile: {len(df[df[col] < lower_bound])} ({(len(df[df[col] < lower_bound])/len(df))*100:.2f}%)"
            )
            print(
                f"Number of results in inner quartile: {len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)])} ({(len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)])/len(df))*100:.2f}%)"
            )
            print(
                f"Number of results in upper quartile: {len(df[df[col] > upper_bound])} ({(len(df[df[col] > upper_bound])/len(df))*100:.2f}%)"
            )
        except:
            print(
                f"Error: Could not generate box plot for column {col}. Skipping to next column..."
            )
            plt.close()
            continue


f""
