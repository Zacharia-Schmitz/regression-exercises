import os
import pandas as pd
import env as e


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
                                   taxamount,
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
    # Map county codes to county names
    county_map = {6037: "LA", 6059: "Orange", 6111: "Ventura"}
    df["county"] = df["county"].replace(county_map)

    # Convert columns to int data type
    df = df.astype(
        {"year": int, "beds": int, "sqfeet": int, "prop_value": int, "prop_tax": int}
    )

    # Drop rows with missing values in specific columns
    df = df.dropna(subset=["year", "beds", "baths", "sqfeet", "prop_value", "prop_tax"])

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
    ).sort_values(by="Number of Unique Values")
